"""Gateway streaming-TTS consumer — LLM deltas to adapter PCM audio sink.

``on_delta`` (agent worker thread) never blocks: SentenceChunker -> thread-safe queue. The
``_run`` task on the gateway loop drains, synthesises via a ``StreamingTTSProvider`` and writes
PCM so playback starts mid-generation. Outcome: success -> ``completed``; failure before audible
output -> ``suppress_whole_file=False`` (gateway falls back to whole-file TTS); failure after
partial audio -> ``partial`` + suppress (never replay the response from the beginning).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
from typing import Any, Dict, Optional

from gateway.platforms.base import AudioFormat, StreamingTTSHandle

logger = logging.getLogger("gateway.streaming_tts_consumer")

_ABORT = object()
_DONE = object()


class StreamingTTSConsumer:
    """Consumes LLM text deltas and produces streaming PCM audio for an adapter."""

    def __init__(self, adapter: Any, chat_id: str, tts_config: Dict[str, Any],
                 loop: asyncio.AbstractEventLoop, *, metadata: Optional[Dict[str, Any]] = None,
                 audio_format: Optional[AudioFormat] = None) -> None:
        from tools.tts_streaming import SentenceChunker, resolve_streaming_provider, speech_alignment_enabled
        self._adapter, self._chat_id, self._tts_config, self._loop, self._metadata = (
            adapter, chat_id, tts_config, loop, metadata)
        # Resolved once; None => inactive, gateway falls back to whole-file TTS.
        self._streamer = resolve_streaming_provider(tts_config)
        self._chunker = SentenceChunker()
        self._alignment_enabled = speech_alignment_enabled(tts_config)
        self._audio_format = audio_format or AudioFormat() if self._streamer is None else (
            AudioFormat(**{f: int(getattr(self._streamer, f, getattr(AudioFormat, f)))
                           for f in ("sample_rate", "channels", "sample_width")})
        )
        # Thread-safe queue of completed clauses plus the _DONE/_ABORT sentinels.
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=256)
        self._handle: Optional[StreamingTTSHandle] = None
        self._task: Optional[asyncio.Task] = None  # drain task, created once by start()
        self._completed = self._partial = self._aborted = False
        self._finished = self._dropped = self._suppress_whole_file = False
        self._lock, self._strip_markdown = threading.Lock(), None  # stripper lazily imported
        self._segment_index = 0
        self._audio_offset_ms = 0

    active = property(lambda self: self._streamer is not None)  # usable streaming provider
    completed = property(lambda self: self._completed)  # streaming audio fully delivered
    partial = property(lambda self: self._partial)  # some audio audible before a failure/drop
    audible = property(lambda self: bool(self._handle and self._handle.audible))  # PCM written
    dropped = property(lambda self: self._dropped)  # queue saturation dropped >= 1 clause
    suppress_whole_file = property(lambda self: self._suppress_whole_file)  # skip whole-file TTS
    done = property(lambda self: self._task is not None and self._task.done())  # drain task ended

    def _enqueue_clauses(self, clauses, full_msg: str, *, log_errors: bool) -> None:
        try:
            for clause in clauses:
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug(full_msg)
        except Exception:
            if log_errors:
                logger.debug("streaming TTS on_delta error", exc_info=True)

    def on_delta(self, text: str) -> None:
        """Receive a text delta from the agent. Non-blocking."""
        if self._aborted or not self.active or self._finished:
            return
        self._enqueue_clauses(self._chunker.feed(text), "streaming TTS queue full, dropping clause",
                              log_errors=True)

    def finish(self) -> None:
        """Signal end-of-text: flush the chunker tail, then enqueue ``_DONE`` after all flushed
        clauses so the drain loop ends deterministically without racing a late ``on_delta``."""
        if self._finished:
            return
        self._finished = True
        if self._aborted or not self.active:
            return
        self._enqueue_clauses(self._chunker.flush(), "streaming TTS queue full while flushing tail",
                              log_errors=False)
        # The load-bearing _DONE sentinel must never be lost: evict clauses until it fits.
        while not self._put_sentinel(_DONE, mark_dropped=True):
            pass

    def _put_sentinel(self, sentinel, *, mark_dropped: bool) -> bool:
        """Try to enqueue a sentinel, evicting one queued item when the queue is full. Returns True
        when the caller should stop retrying: enqueued, or (abort path) nothing left to evict."""
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(sentinel)
            return True
        try:
            self._queue.get_nowait()
        except queue.Empty:
            return not mark_dropped
        self._dropped = self._dropped or mark_dropped
        return False

    def start(self) -> asyncio.Task:
        """Create (once) and return the async drain task on the gateway loop."""
        if self._task is None:
            self._task = self._loop.create_task(self._run())
        return self._task

    def _settle(self, *, failed: bool) -> None:
        """Set outcome flags from what was audible: never report completion after a failure or a
        dropped clause; keep suppression whenever audio was audible (no replay from the start)."""
        audible, degraded = self._handle.audible, failed or self._dropped
        self._completed = audible and not degraded
        self._partial = self._partial or (audible and degraded)
        self._suppress_whole_file = audible

    async def _open_handle(self) -> bool:
        """Open the adapter's streaming-audio handle; False when unsupported or begin failed."""
        if not self.active:
            return False
        if not self._adapter.supports_streaming_tts(self._chat_id, self._audio_format):
            name = getattr(self._adapter, "name", "?")
            logger.debug("adapter %s does not support streaming TTS", name)
            return False
        try:
            self._handle = await self._adapter.begin_streaming_tts(
                self._chat_id, self._audio_format, metadata=self._metadata
            )
        except Exception as exc:
            logger.debug("begin_streaming_tts failed: %s", exc)
            self._handle = None
        return self._handle is not None

    async def _run(self) -> None:
        """Drain clauses until a sentinel/abort, synthesise + write each, then finalise the stream;
        a clause or finalise failure settles the outcome flags and aborts the adapter stream."""
        if not await self._open_handle():
            return
        self._suppress_whole_file = False
        try:
            while not self._aborted:
                try:
                    item = await asyncio.to_thread(self._queue.get, True, 0.1)
                except queue.Empty:
                    continue
                if item is _ABORT or item is _DONE or self._aborted:
                    break
                if not isinstance(item, str):
                    continue
                try:
                    await self._synthesise_and_write(item)
                except Exception as exc:
                    logger.warning("streaming TTS clause failed: %s", exc)
                    self._settle(failed=True)
                    await self._safe_abort(str(exc))
                    return
            if not self._aborted and self._handle is not None:
                try:
                    handle, interrupted = self._handle, self._aborted
                    await self._adapter.finish_streaming_tts(handle, interrupted=interrupted)
                except Exception as exc:
                    logger.debug("finish_streaming_tts error: %s", exc)
                    self._settle(failed=True)
                    await self._safe_abort("finish_streaming_tts failed")
                else:
                    self._settle(failed=False)
        except Exception as exc:
            logger.warning("streaming TTS consumer error: %s", exc)
            await self._safe_abort(str(exc))
        finally:
            with contextlib.suppress(Exception):
                while not self._queue.empty():
                    self._queue.get_nowait()

    async def _synthesise_and_write(self, clause: str) -> None:
        """Synthesise one clause via the streamer and write PCM chunks."""
        if self._handle is None or self._handle.aborted or self._streamer is None:
            return
        cleaned = self._strip_markdown_for_tts(clause)
        text = self._normalize_spoken_text(cleaned)
        if not text:
            return
        if self._streamer is None:
            return

        align = getattr(self._streamer, "align", None)
        if (
            self._alignment_enabled
            and getattr(self._streamer, "supports_alignment", False)
            and callable(align)
        ):
            await self._synthesise_aligned_and_write(text, align)
            return

        fallback_reason = "disabled" if not self._alignment_enabled else "unsupported"
        await self._synthesise_streamed_and_write(text, fallback_reason)

    async def _synthesise_streamed_and_write(
        self,
        text: str,
        fallback_reason: str,
    ) -> None:
        """Stream PCM immediately, then publish its duration fallback record."""
        pcm_length = 0
        async for chunk in self._iter_stream_chunks(text):
            if self._aborted or self._handle is None or self._handle.aborted:
                return
            if not chunk:
                continue
            chunk_bytes = bytes(chunk)
            await self._write_audio_chunk(chunk_bytes)
            pcm_length += len(chunk_bytes)

        if (
            pcm_length <= 0
            or self._aborted
            or self._handle is None
            or self._handle.aborted
        ):
            return

        await self._send_speech_timing(
            self._duration_fallback_payload(text, pcm_length, fallback_reason)
        )
        self._advance_segment(pcm_length)

    async def _synthesise_aligned_and_write(self, text: str, align: Any) -> None:
        """Buffer one sentence so timing can precede its PCM on the wire."""
        chunks: list[bytes] = []
        async for chunk in self._iter_stream_chunks(text):
            if self._aborted or self._handle is None or self._handle.aborted:
                return
            if chunk:
                chunks.append(bytes(chunk))
        if not chunks or self._aborted or self._handle is None or self._handle.aborted:
            return

        pcm = b"".join(chunks)
        timing = None
        fallback_reason = "missing"
        try:
            timing = await asyncio.wait_for(
                asyncio.to_thread(align, text, pcm),
                timeout=self._alignment_timeout_seconds(),
            )
        except asyncio.TimeoutError:
            fallback_reason = "timeout"
            logger.info("speech alignment timed out; using audio fallback")
        except Exception as exc:
            # Alignment is an experiment. A slow or broken aligner must never
            # turn a valid voice response into silence.
            fallback_reason = "error"
            logger.info("speech alignment unavailable; using audio fallback: %s", exc)

        payload = self._validated_timing_payload(timing, text, len(pcm))
        if payload is None:
            if timing is not None and fallback_reason == "missing":
                fallback_reason = "invalid"
            payload = self._duration_fallback_payload(
                text,
                len(pcm),
                fallback_reason,
            )

        if self._aborted or self._handle is None or self._handle.aborted:
            return
        await self._send_speech_timing(payload)

        for chunk in chunks:
            await self._write_audio_chunk(chunk)
        self._advance_segment(len(pcm))

    async def _send_speech_timing(self, payload: Dict[str, Any]) -> None:
        """Best-effort timing delivery; PCM remains authoritative on failure."""
        sender = getattr(self._adapter, "send_speech_timing", None)
        if not callable(sender) or self._handle is None:
            return
        try:
            sent = await sender(self._handle, payload, metadata=self._metadata)
            if sent is False:
                logger.info("speech timing event was not accepted; using audio fallback")
        except Exception as exc:
            logger.info("speech timing event failed; using audio fallback: %s", exc)

    async def _write_audio_chunk(self, chunk: bytes) -> None:
        if self._aborted or self._handle is None or self._handle.aborted or not chunk:
            return
        was_audible = self._handle.audible
        await self._adapter.write_streaming_tts(self._handle, chunk)
        if not was_audible:
            self._handle.audible = True
            self._suppress_whole_file = True

    def _validated_timing_payload(
        self,
        timing: Any,
        text: str,
        pcm_length: int,
    ) -> Optional[Dict[str, Any]]:
        """Make provider timing safe for the iOS wire contract."""
        if not isinstance(timing, dict):
            return None
        raw_words = timing.get("words")
        if not isinstance(raw_words, list) or not raw_words:
            return None

        spoken_text = self._normalize_spoken_text(text)
        reported_text = timing.get("text")
        if (
            not isinstance(reported_text, str)
            or self._normalize_spoken_text(reported_text) != spoken_text
        ):
            return None

        duration_ms = self._pcm_duration_ms(pcm_length)
        words: list[Dict[str, Any]] = []
        previous_end = 0
        expected_words = spoken_text.split()
        if len(raw_words) != len(expected_words):
            return None
        for raw_word, expected in zip(raw_words, expected_words):
            if not isinstance(raw_word, dict):
                return None
            word = str(raw_word.get("text") or "").strip()
            try:
                start_ms = float(raw_word["start_ms"])
                end_ms = float(raw_word["end_ms"])
            except (KeyError, TypeError, ValueError):
                return None
            if (
                not word
                or word.strip(".,!?;:'\"()[]{}").casefold()
                != expected.strip(".,!?;:'\"()[]{}").casefold()
                or not start_ms.is_integer()
                or not end_ms.is_integer()
                or start_ms < previous_end
                or end_ms <= start_ms
                or end_ms > duration_ms + 250
            ):
                return None
            words.append(
                {
                    "text": word,
                    "start_ms": int(start_ms) + self._audio_offset_ms,
                    "end_ms": int(end_ms) + self._audio_offset_ms,
                }
            )
            previous_end = int(end_ms)

        turn_id = self._segment_turn_id()
        return {
            "segment_id": f"{turn_id}-tts-{self._segment_index}",
            "text": spoken_text,
            "words": words,
            "timing_source": "alignment",
            "audio_offset_ms": self._audio_offset_ms,
            "duration_ms": duration_ms,
        }

    def _duration_fallback_payload(
        self,
        text: str,
        pcm_length: int,
        fallback_reason: str,
    ) -> Dict[str, Any]:
        """Build a complete segment record with no misleading word spans."""
        turn_id = self._segment_turn_id()
        return {
            "segment_id": f"{turn_id}-tts-{self._segment_index}",
            "text": self._normalize_spoken_text(text),
            "words": [],
            "timing_source": "duration_fallback",
            "fallback_reason": fallback_reason,
            "audio_offset_ms": self._audio_offset_ms,
            "duration_ms": self._pcm_duration_ms(pcm_length),
        }

    def _advance_segment(self, pcm_length: int) -> None:
        """Advance the absolute stream clock after a complete segment."""
        self._audio_offset_ms += self._pcm_duration_ms(pcm_length)
        self._segment_index += 1

    def _segment_turn_id(self) -> str:
        """Use the adapter's authoritative turn marker when metadata is absent."""
        handle_turn_id = str(getattr(self._handle, "turn_id", "") or "").strip()
        metadata_turn_id = str(
            (self._metadata or {}).get("voice_session_turn_id") or ""
        ).strip()
        return handle_turn_id or metadata_turn_id or "speech"

    def _alignment_timeout_seconds(self) -> float:
        from tools.tts_streaming import speech_alignment_timeout_seconds

        return speech_alignment_timeout_seconds(self._tts_config)

    @staticmethod
    def _normalize_spoken_text(text: str) -> str:
        """Collapse formatting whitespace into the text spoken by the provider."""
        return " ".join(str(text).split())

    def _strip_markdown_for_tts(self, text: str) -> str:
        """Lazy-import and apply the TTS markdown stripper."""
        if self._strip_markdown is None:
            try:
                from tools.tts_text_normalize import _strip_markdown_for_tts as _strip
                self._strip_markdown = _strip
            except ImportError:
                self._strip_markdown = lambda t: t  # noqa: E731
        return self._strip_markdown(text).strip()

    def _pcm_duration_ms(self, pcm_length: int) -> int:
        bytes_per_second = (
            int(getattr(self._audio_format, "sample_rate", 0))
            * int(getattr(self._audio_format, "channels", 0))
            * int(getattr(self._audio_format, "sample_width", 0))
        )
        if pcm_length <= 0 or bytes_per_second <= 0:
            return 0
        return int(round(pcm_length * 1_000 / bytes_per_second))

    async def _iter_stream_chunks(self, text: str):
        """Yield provider PCM chunks one at a time without blocking the loop."""
        if self._streamer is None:
            return
        iterator = iter(self._streamer.stream(text))
        while True:
            has_chunk, chunk = await asyncio.to_thread(self._next_stream_chunk, iterator)
            if not has_chunk:
                break
            yield chunk

    @staticmethod
    def _next_stream_chunk(iterator: Any) -> tuple[bool, Optional[bytes]]:
        try:
            return True, next(iterator)
        except StopIteration:
            return False, None


    async def _safe_abort(self, reason: str) -> None:
        """Abort the adapter stream, swallowing errors (idempotent)."""
        if self._handle is None:
            return
        try:
            with contextlib.suppress(Exception):
                await self._adapter.abort_streaming_tts(self._handle, error=reason)
        finally:
            if self._handle:
                self._handle.aborted = True

    def abort(self, reason: str = "cancelled") -> None:
        """Idempotent cancellation from any thread."""
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
        # The load-bearing _ABORT sentinel must reach the queue even when full: evict to make room.
        if not any(self._put_sentinel(_ABORT, mark_dropped=False) for _ in range(3)):
            logger.debug("streaming TTS _ABORT sentinel could not be enqueued")
        if self._handle is not None and not self._handle.aborted:
            with contextlib.suppress(Exception):
                self._loop.call_soon_threadsafe(asyncio.create_task, self._safe_abort(reason))

    async def wait_complete(self, timeout: float = 60.0) -> bool:
        """Wait for the drain task to finish. Returns True only on full success."""
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        return self._completed
