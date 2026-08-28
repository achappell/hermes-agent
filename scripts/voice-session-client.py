#!/usr/bin/env python3
"""Small client for the Hermes voice-session channel.

This is intentionally a channel client, not a second agent client.  Use
``--text`` to exercise the transport, or ``--microphone`` to record on this
machine, transcribe locally with Hermes' Faster-Whisper pipeline, and send
only the transcript to the media server.  Responses are printed while any
PCM audio is saved as a WAV file.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
import threading
import uuid
import wave
from pathlib import Path
from typing import Any, Iterable, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]


class LocalMicrophone:
    """Capture one bounded, silence-ended utterance and transcribe it locally.

    The audio recorder and STT dispatcher are imported lazily.  That keeps the
    transport-only client usable on machines without PortAudio or the optional
    voice dependencies, and mirrors Hermes' own lazy voice-mode setup.
    """

    def __init__(
        self,
        *,
        max_seconds: float = 15.0,
        silence_duration: float = 3.0,
        silence_threshold: int = 200,
        model: Optional[str] = None,
        recorder_factory: Any = None,
        transcribe_fn: Any = None,
        hallucination_fn: Any = None,
    ) -> None:
        if max_seconds <= 0:
            raise ValueError("microphone max seconds must be greater than zero")
        if silence_duration <= 0:
            raise ValueError("microphone silence duration must be greater than zero")
        if silence_threshold < 0:
            raise ValueError("microphone silence threshold cannot be negative")
        self.max_seconds = float(max_seconds)
        self.silence_duration = float(silence_duration)
        self.silence_threshold = int(silence_threshold)
        self.model = model or None
        self._recorder_factory = recorder_factory
        self._transcribe_fn = transcribe_fn
        self._hallucination_fn = hallucination_fn
        self._recorder: Any = None

    def _load(self) -> None:
        if self._recorder is not None:
            return
        if (
            self._recorder_factory is None
            or self._transcribe_fn is None
            or self._hallucination_fn is None
        ):
            try:
                # The script normally lives in the Hermes checkout.  Add its
                # root explicitly because Python puts ``scripts/`` (not the
                # repository root) on sys.path for a path-invoked script.
                root = str(_REPO_ROOT)
                if root not in sys.path:
                    sys.path.insert(0, root)
                from tools.transcription_tools import (
                    transcribe_audio_local_fallback,
                )
                from tools.voice_mode import (
                    create_audio_recorder,
                    is_whisper_hallucination,
                )
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "Microphone mode needs Hermes' voice dependencies. "
                    "From the Hermes checkout, run `uv sync --extra voice` "
                    "(or install faster-whisper, sounddevice, and numpy)."
                ) from exc
            if self._recorder_factory is None:
                self._recorder_factory = create_audio_recorder
            if self._transcribe_fn is None:
                self._transcribe_fn = transcribe_audio_local_fallback
            if self._hallucination_fn is None:
                self._hallucination_fn = is_whisper_hallucination
        self._recorder = self._recorder_factory()
        if not getattr(self._recorder, "supports_silence_autostop", False):
            raise RuntimeError(
                "This microphone backend cannot detect utterance boundaries. "
                "Use a desktop audio input with Hermes' sounddevice recorder."
            )
        # AudioRecorder exposes these as configuration attributes because the
        # voice CLI wires them from config before every recording.  Keep the
        # same endpointing semantics here, with a hard cap for a stuck mic.
        self._recorder._silence_threshold = self.silence_threshold
        self._recorder._silence_duration = self.silence_duration
        self._recorder._max_recording_seconds = self.max_seconds

    def capture(self) -> str:
        """Record one utterance, return its transcript, or ``""`` for silence."""
        self._load()
        finished = threading.Event()

        try:
            self._recorder.start(on_silence_stop=finished.set)
            # The recorder normally fires at the silence endpoint or its hard
            # cap.  The extra grace protects the CLI if an audio callback dies.
            finished.wait(max(self.max_seconds + 2.0, 5.0))
            wav_path = self._recorder.stop()
        except KeyboardInterrupt:
            self._recorder.cancel()
            raise
        except Exception as exc:
            try:
                self._recorder.cancel()
            except Exception:
                pass
            raise RuntimeError(f"microphone capture failed: {exc}") from exc

        if not wav_path:
            return ""
        try:
            result = self._transcribe_fn(wav_path, model=self.model)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "local transcription failed")
        transcript = str(result.get("transcript") or "").strip()
        if self._hallucination_fn and self._hallucination_fn(transcript):
            return ""
        return transcript

    def close(self) -> None:
        if self._recorder is not None:
            try:
                self._recorder.shutdown()
            finally:
                self._recorder = None


def _connect_factory():
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        from websockets import connect  # type: ignore[no-redef]
    return connect


def _connection_kwargs(connect: Any, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        params = inspect.signature(connect).parameters
    except (TypeError, ValueError):
        params = {}
    # websockets 15 renamed extra_headers to additional_headers.
    key = "additional_headers" if "additional_headers" in params else "extra_headers"
    return {key: headers, "max_size": 256 * 1024}


async def _receive_json(ws: Any) -> dict[str, Any]:
    while True:
        frame = await ws.recv()
        if isinstance(frame, bytes):
            # The hello/turn control frames are JSON. Preserve unexpected
            # binary frames for the caller rather than trying to decode PCM.
            continue
        payload = json.loads(frame)
        if isinstance(payload, dict):
            return payload
        raise RuntimeError("server sent a non-object JSON frame")


def _output_path(base: Optional[str], index: int, turn_id: str) -> Optional[Path]:
    if not base:
        return Path(f"voice-session-{turn_id}.wav")
    path = Path(base)
    if index == 0:
        return path
    return path.with_name(f"{path.stem}-{index}{path.suffix or '.wav'}")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


async def _send_turn(
    ws: Any,
    args: argparse.Namespace,
    index: int,
    text: str,
    *,
    stt_source: str,
) -> None:
    """Send one transcript and consume its streamed text/audio response."""
    turn_id = uuid.uuid4().hex
    await ws.send(
        json.dumps({
            "type": "turn",
            "protocol_version": 1,
            "turn_id": turn_id,
            "session_id": args.session_id,
            "text": text,
            "stt_source": stt_source,
        })
    )
    audio: bytearray = bytearray()
    audio_format: Optional[tuple[int, int, int]] = None
    streamed_text = False
    rendered_preview = ""
    print("hermes: ", end="", flush=True)
    while True:
        frame = await ws.recv()
        if isinstance(frame, bytes):
            audio.extend(frame)
            continue
        payload = json.loads(frame)
        kind = payload.get("type")
        if kind == "turn_accepted":
            continue
        if kind == "text_delta":
            preview = str(payload.get("text") or "")
            if preview.startswith(rendered_preview):
                print(preview[len(rendered_preview) :], end="", flush=True)
            else:
                # Tool-status lines or a cursor can replace the accumulated
                # prefix. Keep the terminal readable instead of printing the
                # whole preview twice.
                print(f"\n{preview}", end="", flush=True)
            rendered_preview = preview
            streamed_text = True
        elif kind in {"text", "text_final"}:
            final_text = str(payload.get("text") or "")
            if kind == "text_final" and not streamed_text:
                print(final_text, end="", flush=True)
            elif kind == "text_final" and final_text != rendered_preview.rstrip("▉"):
                print(f"\n{final_text}", end="", flush=True)
            elif kind == "text":
                print(final_text, end="", flush=True)
        elif kind == "audio_start":
            audio_format = (
                int(payload.get("sample_rate", 24000)),
                int(payload.get("channels", 1)),
                int(payload.get("sample_width", 2)),
            )
        elif kind == "error":
            raise RuntimeError(payload.get("error", "voice-session error"))
        elif kind == "turn_end":
            print()
            if audio and audio_format:
                output = _output_path(args.output, index, turn_id)
                assert output is not None
                output.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output), "wb") as wav:
                    wav.setnchannels(audio_format[1])
                    wav.setsampwidth(audio_format[2])
                    wav.setframerate(audio_format[0])
                    wav.writeframes(audio)
                print(f"audio: {output} ({len(audio)} PCM bytes)")
            break


async def _run(args: argparse.Namespace) -> int:
    token = args.token or os.getenv("VOICE_SESSION_TOKEN", "")
    if not token:
        raise SystemExit("set --token or VOICE_SESSION_TOKEN")

    microphone: Optional[LocalMicrophone] = None
    if args.microphone:
        if not sys.stdin.isatty() and not args.once:
            raise SystemExit(
                "--microphone needs a terminal for repeated turns; add --once "
                "when running non-interactively"
            )
        microphone = LocalMicrophone(
            max_seconds=args.mic_max_seconds,
            silence_duration=args.mic_silence_duration,
            silence_threshold=args.mic_silence_threshold,
            model=args.stt_model,
        )

    connect = _connect_factory()
    kwargs = _connection_kwargs(connect, token)
    try:
        async with connect(args.url, **kwargs) as ws:
            await ws.send(
                json.dumps({
                    "type": "hello",
                    "protocol_version": 1,
                    "client_id": args.client_id,
                    "device_id": args.device_id,
                    "session_id": args.session_id,
                    "display_name": args.display_name,
                })
            )
            hello = await _receive_json(ws)
            if hello.get("type") != "hello_ack":
                raise RuntimeError(f"voice-session hello failed: {hello}")
            print(
                f"connected: {hello.get('chat_id')} "
                f"(protocol v{hello.get('protocol_version')})"
            )

            if microphone is not None:
                index = 0
                while True:
                    if sys.stdin.isatty():
                        try:
                            await asyncio.to_thread(
                                input, "press Enter to speak (Ctrl-D exits): "
                            )
                        except EOFError:
                            break
                    print("listening…", flush=True)
                    transcript = await asyncio.to_thread(microphone.capture)
                    if transcript:
                        print(f"you: {transcript}")
                        await _send_turn(
                            ws,
                            args,
                            index,
                            transcript,
                            stt_source="local-faster-whisper",
                        )
                        index += 1
                    else:
                        print("no speech detected.")
                    if args.once:
                        break
            else:
                texts: Iterable[str]
                if args.text:
                    texts = args.text
                else:
                    if sys.stdin.isatty():
                        print("Type a transcript and press Enter; Ctrl-D exits.")
                    texts = (line.rstrip("\n") for line in sys.stdin if line.strip())
                for index, text in enumerate(texts):
                    text = str(text).strip()
                    if text:
                        await _send_turn(ws, args, index, text, stt_source="local")
    finally:
        if microphone is not None:
            await asyncio.to_thread(microphone.close)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.getenv(
            "HERMES_VOICE_SESSION_URL", "ws://127.0.0.1:8790/voice-session"
        ),
    )
    parser.add_argument("--token", help="Bearer token (or VOICE_SESSION_TOKEN)")
    parser.add_argument(
        "--client-id", default=os.getenv("VOICE_SESSION_CLIENT_ID", "amanda-laptop")
    )
    parser.add_argument(
        "--device-id", default=os.getenv("VOICE_SESSION_DEVICE_ID", "macbook")
    )
    parser.add_argument(
        "--session-id", default=os.getenv("VOICE_SESSION_ID", "default")
    )
    parser.add_argument("--display-name", default="")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text", action="append", help="Transcript to send; repeat for multiple turns"
    )
    input_group.add_argument(
        "--microphone",
        action="store_true",
        help=(
            "record on this machine and transcribe with local Faster-Whisper "
            "before sending the transcript"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="with --microphone, capture one turn and exit",
    )
    parser.add_argument(
        "--mic-max-seconds",
        type=float,
        default=_env_float("VOICE_SESSION_MIC_MAX_SECONDS", 15.0),
        help="hard cap for one microphone utterance (default: 15 seconds)",
    )
    parser.add_argument(
        "--mic-silence-duration",
        type=float,
        default=_env_float("VOICE_SESSION_MIC_SILENCE_DURATION", 3.0),
        help="silence needed to end an utterance (default: 3 seconds)",
    )
    parser.add_argument(
        "--mic-silence-threshold",
        type=int,
        default=_env_int("VOICE_SESSION_MIC_SILENCE_THRESHOLD", 200),
        help="RMS below this is silence (default: 200)",
    )
    parser.add_argument(
        "--stt-model",
        default=os.getenv("VOICE_SESSION_STT_MODEL") or None,
        help="local Faster-Whisper model; otherwise use stt.local.model",
    )
    parser.add_argument(
        "--output", help="WAV path for the first response (later turns get a suffix)"
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
