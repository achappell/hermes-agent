#!/usr/bin/env python3
"""Small transcript-to-stream client for the Hermes voice-session channel.

This is intentionally a transport POC, not a second agent client. Feed it
text from a local STT process (or use ``--text`` to exercise the channel) and
it will print streamed text while saving any PCM response as a WAV file.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
import uuid
import wave
from pathlib import Path
from typing import Any, Iterable, Optional


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


async def _run(args: argparse.Namespace) -> int:
    token = args.token or os.getenv("VOICE_SESSION_TOKEN", "")
    if not token:
        raise SystemExit("set --token or VOICE_SESSION_TOKEN")
    texts: Iterable[str]
    if args.text:
        texts = args.text
    else:
        if sys.stdin.isatty():
            print("Type a transcript and press Enter; Ctrl-D exits.")
        texts = (line.rstrip("\n") for line in sys.stdin if line.strip())

    connect = _connect_factory()
    kwargs = _connection_kwargs(connect, token)
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
            f"connected: {hello.get('chat_id')} (protocol v{hello.get('protocol_version')})"
        )

        for index, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            turn_id = uuid.uuid4().hex
            await ws.send(
                json.dumps({
                    "type": "turn",
                    "protocol_version": 1,
                    "turn_id": turn_id,
                    "session_id": args.session_id,
                    "text": text,
                    "stt_source": "local",
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
                        # Tool-status lines or a cursor can replace the
                        # accumulated prefix. Keep the terminal readable
                        # instead of printing the whole preview twice.
                        print(f"\n{preview}", end="", flush=True)
                    rendered_preview = preview
                    streamed_text = True
                elif kind in {"text", "text_final"}:
                    final_text = str(payload.get("text") or "")
                    if kind == "text_final" and not streamed_text:
                        print(final_text, end="", flush=True)
                    elif kind == "text_final" and final_text != rendered_preview.rstrip(
                        "▉"
                    ):
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
    parser.add_argument(
        "--text", action="append", help="Transcript to send; repeat for multiple turns"
    )
    parser.add_argument(
        "--output", help="WAV path for the first response (later turns get a suffix)"
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
