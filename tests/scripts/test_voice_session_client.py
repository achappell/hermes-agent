"""Tests for the transport client’s optional microphone boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[2] / "scripts" / "voice-session-client.py"
_SPEC = importlib.util.spec_from_file_location("voice_session_client", _SCRIPT)
assert _SPEC and _SPEC.loader
client = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(client)


class _Recorder:
    supports_silence_autostop = True

    def __init__(self, wav_path: Path, *, fire_endpoint: bool = True) -> None:
        self.wav_path = wav_path
        self.fire_endpoint = fire_endpoint
        self.started = False
        self.stopped = False
        self.shutdown_called = False

    def start(self, *, on_silence_stop):
        self.started = True
        if self.fire_endpoint:
            on_silence_stop()

    def stop(self):
        self.stopped = True
        if not self.fire_endpoint:
            return None
        self.wav_path.write_bytes(b"fake wav")
        return str(self.wav_path)

    def cancel(self):
        self.stopped = True

    def shutdown(self):
        self.shutdown_called = True


def test_microphone_captures_transcribes_and_cleans_up(tmp_path):
    wav_path = tmp_path / "utterance.wav"
    recorder = _Recorder(wav_path)
    calls = []

    def transcribe(path, *, model):
        calls.append((path, model))
        assert Path(path).exists()
        return {"success": True, "transcript": " hello from the mic "}

    microphone = client.LocalMicrophone(
        max_seconds=4,
        silence_duration=0.75,
        silence_threshold=321,
        model="small.en",
        recorder_factory=lambda: recorder,
        transcribe_fn=transcribe,
        hallucination_fn=lambda text: False,
    )

    assert microphone.capture() == "hello from the mic"
    assert calls == [(str(wav_path), "small.en")]
    assert recorder.started is True
    assert recorder.stopped is True
    assert recorder._max_recording_seconds == 4
    assert recorder._silence_duration == 0.75
    assert recorder._silence_threshold == 321
    assert not wav_path.exists()

    microphone.close()
    assert recorder.shutdown_called is True


def test_microphone_discards_empty_or_hallucinated_audio(tmp_path):
    empty_recorder = _Recorder(tmp_path / "empty.wav", fire_endpoint=False)
    empty = client.LocalMicrophone(
        recorder_factory=lambda: empty_recorder,
        transcribe_fn=lambda *args, **kwargs: pytest.fail("should not transcribe"),
        hallucination_fn=lambda text: False,
    )
    assert empty.capture() == ""

    hallucinated_path = tmp_path / "hallucinated.wav"
    hallucinated_recorder = _Recorder(hallucinated_path)
    hallucinated = client.LocalMicrophone(
        recorder_factory=lambda: hallucinated_recorder,
        transcribe_fn=lambda *args, **kwargs: {
            "success": True,
            "transcript": "Thank you.",
        },
        hallucination_fn=lambda text: text == "Thank you.",
    )
    assert hallucinated.capture() == ""


def test_microphone_requires_positive_bounds():
    for kwargs in ({"max_seconds": 0}, {"silence_duration": 0}):
        try:
            client.LocalMicrophone(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion keeps the failure readable
            raise AssertionError("invalid microphone bounds were accepted")


class _WebSocket:
    def __init__(self):
        self.sent = []
        self.frames = [
            json.dumps({"type": "turn_accepted"}),
            json.dumps({"type": "audio_start", "sample_rate": 16000}),
            json.dumps({"type": "text_delta", "text": "reply"}),
            b"\x00\x01",
            json.dumps({"type": "text_final", "text": "reply"}),
            json.dumps({"type": "audio_end"}),
            json.dumps({"type": "turn_end"}),
        ]

    async def send(self, frame):
        self.sent.append(frame)

    async def recv(self):
        return self.frames.pop(0)


@pytest.mark.asyncio
async def test_send_turn_marks_local_stt_and_writes_audio(tmp_path):
    ws = _WebSocket()
    args = type(
        "Args",
        (),
        {"session_id": "session", "output": str(tmp_path / "reply.wav")},
    )()

    await client._send_turn(ws, args, 0, "hello", stt_source="local-faster-whisper")

    sent = json.loads(ws.sent[0])
    assert sent["type"] == "turn"
    assert sent["text"] == "hello"
    assert sent["stt_source"] == "local-faster-whisper"
    output = tmp_path / "reply.wav"
    assert output.exists()
