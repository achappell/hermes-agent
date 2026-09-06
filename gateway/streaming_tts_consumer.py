"""Gateway streaming-TTS consumer — LLM deltas to adapter PCM audio sink.

Bridges the synchronous agent ``stream_delta_callback`` (fired from the
worker thread) to a voice-capable platform adapter's streaming-audio
contract, so playback begins while the LLM is still generating.

Lifecycle::

    consumer = StreamingTTSConsumer(adapter, chat_id, tts_config, loop, metadata)
    agent.stream_delta_callback = consumer.on_delta   # sync, non-blocking
    ... agent runs in executor ...
    consumer.finish()            # signal end-of-text
    success = await consumer.wait_complete(timeout=60)
    if consumer.suppress_whole_file:
        # suppress whole-file auto-TTS for this turn
    consumer.abort("cancelled")  # idempotent cancellation

Design:
- ``on_delta`` is synchronous and never blocks the agent thread. It feeds
  deltas into a ``SentenceChunker`` and queues completed clauses onto a
  thread-safe ``queue.Queue``.
- An asyncio task (``run``) runs on the gateway event loop, draining the
  queue, synthesising each clause via a ``StreamingTTSProvider``, and
  writing PCM chunks to the adapter.
- Per-turn state is isolated: each consumer instance owns its own chunker,
  queue, handle, and flags. Concurrent chats cannot cross-contaminate.
- On successful completion (all clauses synthesised and written), the
  consumer reports ``completed=True`` so the gateway can suppress the
  duplicate whole-file auto-TTS.
- On failure before any audible output, the consumer reports
  ``completed=False`` and clears ``suppress_whole_file`` so the gateway can
  fall back to whole-file TTS.
- On failure after partial audible output, the consumer reports
  ``completed=False`` but keeps ``suppress_whole_file=True`` so the gateway
  does NOT replay the whole response from the beginning.
- Cancellation/abort is idempotent: late chunks are silently dropped.
"""

from __future__ import annotations

import asyncio
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

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        tts_config: Dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        from tools.tts_streaming import (
            SentenceChunker,
            resolve_streaming_provider,
            speech_alignment_enabled,
        )

        self._adapter = adapter
        self._chat_id = chat_id
        self._tts_config = tts_config
        self._loop = loop
        self._metadata = metadata

        # Resolve the streaming provider once. If unavailable, the consumer is
        # inactive and the gateway falls back to whole-file TTS.
        self._streamer = resolve_streaming_provider(tts_config)
        self._chunker = SentenceChunker()
        self._alignment_enabled = speech_alignment_enabled(tts_config)

        if self._streamer is not None:
            self._audio_format = AudioFormat(
                sample_rate=int(getattr(self._streamer, "sample_rate", AudioFormat.sample_rate)),
                channels=int(getattr(self._streamer, "channels", AudioFormat.channels)),
                sample_width=int(getattr(self._streamer, "sample_width", AudioFormat.sample_width)),
            )
        else:
            self._audio_format = audio_format or AudioFormat()

        # Thread-safe queue: completed clauses and the occasional abort sentinel.
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=256)

        # Per-turn state.
        self._handle: Optional[StreamingTTSHandle] = None
        self._started = False
        self._completed = False
        self._partial = False
        self._aborted = False
        self._finished = False
        self._dropped = False
        self._suppress_whole_file = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Pre-allocate the strip-markdown helper lazily to avoid import cycles.
        self._strip_markdown = None
        self._segment_index = 0
        self._audio_offset_ms = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True when this consumer has a usable streaming provider."""
        return self._streamer is not None

    @property
    def completed(self) -> bool:
        """True when streaming audio was fully delivered."""
        return self._completed

    @property
    def partial(self) -> bool:
        """True when some audio was audible before a failure or drop."""
        return self._partial

    @property
    def started(self) -> bool:
        """True when the adapter accepted the streaming session."""
        return self._started

    @property
    def audible(self) -> bool:
        """True once the first PCM chunk has been written."""
        return bool(self._handle and self._handle.audible)

    @property
    def dropped(self) -> bool:
        """True when queue saturation dropped at least one clause."""
        return self._dropped

    @property
    def suppress_whole_file(self) -> bool:
        """True when the gateway should skip the legacy whole-file TTS fallback."""
        return self._suppress_whole_file

    @property
    def done(self) -> bool:
        """True once the async drain task has terminated."""
        return self._task is not None and self._task.done()

    # ------------------------------------------------------------------
    # Sync callback (agent worker thread)
    # ------------------------------------------------------------------

    def on_delta(self, text: str) -> None:
        """Receive a text delta from the agent. Non-blocking."""
        if self._aborted or not self.active or self._finished:
            return
        try:
            for clause in self._chunker.feed(text):
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug("streaming TTS queue full, dropping clause")
        except Exception:
            logger.debug("streaming TTS on_delta error", exc_info=True)

    def finish(self) -> None:
        """Signal end-of-text and flush the chunker tail.

        Enqueues a ``_DONE`` sentinel after all flushed clauses so the
        drain loop has a deterministic termination signal that cannot
        race with a late ``on_delta`` or be lost when the queue is full.
        """
        if self._finished:
            return
        self._finished = True
        if self._aborted or not self.active:
            return
        try:
            for clause in self._chunker.flush():
                self._queue.put_nowait(clause)
        except queue.Full:
            self._dropped = True
            logger.debug("streaming TTS queue full while flushing tail")
        except Exception:
            pass
        # Guarantee the _DONE sentinel reaches the queue.  If the bounded
        # queue is full, drain one item to make room — the sentinel is
        # load-bearing and must not be lost (#60671 hardening).
        self._enqueue_done()

    def _enqueue_done(self) -> None:
        """Enqueue the _DONE sentinel, evicting a queued clause if necessary."""
        while True:
            try:
                self._queue.put_nowait(_DONE)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._dropped = True
                except queue.Empty:
                    continue

    # ------------------------------------------------------------------
    # Async lifecycle (gateway event loop)
    # ------------------------------------------------------------------

    def start(self) -> asyncio.Task:
        """Create and return the async drain task on the gateway loop."""
        if self._task is not None:
            return self._task
        self._task = self._loop.create_task(self._run())
        return self._task

    async def _run(self) -> None:
        """Drain clauses from the queue, synthesise, and write to the adapter."""
        if not self.active:
            return

        if not self._adapter.supports_streaming_tts(self._chat_id, self._audio_format):
            logger.debug("adapter %s does not support streaming TTS", getattr(self._adapter, "name", "?"))
            return

        try:
            self._handle = await self._adapter.begin_streaming_tts(
                self._chat_id,
                self._audio_format,
                metadata=self._metadata,
            )
        except Exception as exc:
            logger.debug("begin_streaming_tts failed: %s", exc)
            self._handle = None
            return

        if self._handle is None:
            return

        self._started = True
        self._suppress_whole_file = False

        try:
            while True:
                if self._aborted:
                    break
                try:
                    item = await asyncio.to_thread(self._queue.get, True, 0.1)
                except queue.Empty:
                    continue

                if item is _ABORT:
                    break
                if item is _DONE:
                    break
                if not isinstance(item, str):
                    continue
                if self._aborted:
                    break

                try:
                    await self._synthesise_and_write(item)
                except Exception as exc:
                    logger.warning("streaming TTS clause failed: %s", exc)
                    if self._handle and self._handle.audible:
                        self._partial = True
                        self._suppress_whole_file = True
                    else:
                        self._suppress_whole_file = False
                    self._completed = False
                    await self._safe_abort(str(exc))
                    return

            if not self._aborted and self._handle is not None:
                _finish_failed = False
                try:
                    await self._adapter.finish_streaming_tts(self._handle, interrupted=self._aborted)
                except Exception as exc:
                    logger.debug("finish_streaming_tts error: %s", exc)
                    _finish_failed = True

                if _finish_failed:
                    # finish_streaming_tts() raised — never report full
                    # completion.  If audio was already audible, report
                    # partial and preserve suppression so the gateway
                    # does not replay from the beginning.  If no audio
                    # was audible, permit whole-file fallback.
                    if self._handle.audible:
                        self._partial = True
                        self._completed = False
                        self._suppress_whole_file = True
                    else:
                        self._completed = False
                        self._suppress_whole_file = False
                    await self._safe_abort("finish_streaming_tts failed")
                elif self._handle.audible and not self._dropped:
                    self._completed = True
                    self._suppress_whole_file = True
                elif self._handle.audible and self._dropped:
                    self._partial = True
                    self._completed = False
                    self._suppress_whole_file = True
                else:
                    self._completed = False
                    self._suppress_whole_file = False
        except Exception as exc:
            logger.warning("streaming TTS consumer error: %s", exc)
            await self._safe_abort(str(exc))
        finally:
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except Exception:
                pass

    async def _synthesise_and_write(self, clause: str) -> None:
        """Synthesise one clause via the streamer and write PCM chunks."""
        if self._handle is None or self._handle.aborted:
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

    def _strip_markdown_for_tts(self, text: str) -> str:
        """Lazy-import and apply the TTS markdown stripper."""
        if self._strip_markdown is None:
            try:
                from tools.tts_tool import _strip_markdown_for_tts as _strip
                self._strip_markdown = _strip
            except ImportError:
                self._strip_markdown = lambda t: t  # noqa: E731
        return self._strip_markdown(text).strip()

    async def _safe_abort(self, reason: str) -> None:
        """Abort the adapter stream, swallowing errors (idempotent)."""
        if self._handle is None:
            return
        try:
            await self._adapter.abort_streaming_tts(self._handle, error=reason)
        except Exception:
            pass
        finally:
            if self._handle:
                self._handle.aborted = True

    # ------------------------------------------------------------------
    # Cancellation and completion
    # ------------------------------------------------------------------

    def abort(self, reason: str = "cancelled") -> None:
        """Idempotent cancellation from any thread."""
        with self._lock:
            if self._aborted:
                return
            self._aborted = True
        # Guarantee the _ABORT sentinel reaches the queue.  If the bounded
        # queue is full, drain one item to make room — the sentinel must
        # not be lost (#60671 hardening).
        for _attempt in range(3):
            try:
                self._queue.put_nowait(_ABORT)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        else:
            logger.debug("streaming TTS _ABORT sentinel could not be enqueued")
        if self._handle is not None and not self._handle.aborted:
            try:
                self._loop.call_soon_threadsafe(
                    asyncio.create_task,
                    self._safe_abort(reason),
                )
            except Exception:
                pass

    async def wait_complete(self, timeout: float = 60.0) -> bool:
        """Wait for the drain task to finish. Returns True only on full success."""
        if self._task is None:
            return self._completed
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass
        return self._completed
