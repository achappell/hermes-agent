from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import AudioFormat, MessageType
from plugins.platforms.voice_session.adapter import (
    PROTOCOL_VERSION,
    VoiceSessionAdapter,
    _Connection,
    _bearer_token,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.closed = False
        self.json_frames: list[dict] = []
        self.binary_frames: list[bytes] = []

    async def send_json(self, payload):
        self.json_frames.append(payload)

    async def send_bytes(self, payload):
        self.binary_frames.append(payload)

    async def close(self, **kwargs):
        self.closed = True


def _adapter(monkeypatch) -> VoiceSessionAdapter:
    monkeypatch.setenv("VOICE_SESSION_TOKEN", "test-token")
    monkeypatch.setenv("VOICE_SESSION_ALLOW_ALL_USERS", "true")
    return VoiceSessionAdapter(PlatformConfig(enabled=True, extra={"port": 0}))


def _connection(adapter: VoiceSessionAdapter) -> tuple[_Connection, FakeWebSocket]:
    websocket = FakeWebSocket()
    connection = _Connection(
        websocket=websocket,
        chat_id="amanda-laptop:macbook",
        client_id="amanda-laptop",
        device_id="macbook",
        session_id="default",
    )
    adapter._connections[connection.chat_id] = connection
    return connection, websocket


async def _noop_handle_message(event):
    return None


def test_bearer_token_requires_bearer_scheme():
    assert _bearer_token({"Authorization": "Bearer abc"}) == "abc"
    assert _bearer_token({"Authorization": "Basic abc"}) == ""
    assert _bearer_token({}) == ""


@pytest.mark.asyncio
async def test_turn_becomes_voice_event_with_local_stt_metadata(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)
    events = []

    async def capture(event):
        events.append(event)

    adapter.handle_message = capture
    await adapter._handle_turn(
        connection,
        {
            "type": "turn",
            "protocol_version": PROTOCOL_VERSION,
            "turn_id": "turn-1",
            "text": "  hello Hermes  ",
            "stt_source": "local-whisper",
        },
    )

    assert len(events) == 1
    event = events[0]
    assert event.message_type is MessageType.VOICE
    assert event.text == "hello Hermes"
    assert event.media_urls == []
    assert event.source.chat_id == "amanda-laptop:macbook"
    assert event.source.user_id == "amanda-laptop"
    assert event.source.thread_id == "default"
    assert event.metadata["stt_source"] == "local-whisper"
    assert {frame["type"] for frame in websocket.json_frames} == {"turn_accepted"}


@pytest.mark.asyncio
async def test_text_draft_and_final_close_a_turn(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)
    adapter.handle_message = _noop_handle_message
    await adapter._handle_turn(
        connection, {"type": "turn", "text": "hello", "turn_id": "t1"}
    )

    draft = await adapter.send_draft(connection.chat_id, 123, "hello")
    final = await adapter.send(connection.chat_id, "hello", metadata={"notify": True})

    assert draft.success is True
    assert final.success is True
    assert [frame["type"] for frame in websocket.json_frames] == [
        "turn_accepted",
        "text_delta",
        "text_final",
        "turn_end",
    ]
    assert websocket.json_frames[1]["replace"] is True


@pytest.mark.asyncio
async def test_streaming_pcm_lifecycle_is_ordered(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)
    adapter.handle_message = _noop_handle_message
    await adapter._handle_turn(
        connection, {"type": "turn", "text": "hello", "turn_id": "t2"}
    )

    fmt = AudioFormat(sample_rate=16_000, channels=1, sample_width=2)
    assert adapter.supports_streaming_tts(connection.chat_id, fmt)
    handle = await adapter.begin_streaming_tts(connection.chat_id, fmt)
    assert handle is not None
    await adapter.write_streaming_tts(handle, b"\x01\x02")
    await adapter.finish_streaming_tts(handle)
    await adapter.send(connection.chat_id, "done", metadata={"notify": True})

    assert websocket.binary_frames == [b"\x01\x02"]
    assert [frame["type"] for frame in websocket.json_frames] == [
        "turn_accepted",
        "audio_start",
        "audio_end",
        "text_final",
        "turn_end",
    ]
    assert websocket.json_frames[1]["sample_rate"] == 16_000
    assert websocket.json_frames[-1]["interrupted"] is False


@pytest.mark.asyncio
async def test_speech_timing_is_sent_on_the_active_audio_stream(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)
    adapter.handle_message = _noop_handle_message
    await adapter._handle_turn(
        connection, {"type": "turn", "text": "hello", "turn_id": "t-timing"}
    )
    handle = await adapter.begin_streaming_tts(connection.chat_id, AudioFormat())
    assert handle is not None

    assert await adapter.send_speech_timing(
        handle,
        {
            "segment_id": "t-timing-tts-0",
            "text": "Hermes moves.",
            "words": [
                {"text": "Hermes", "start_ms": 0, "end_ms": 180},
                {"text": "moves.", "start_ms": 180, "end_ms": 390},
            ],
        },
    ) is True

    assert [frame["type"] for frame in websocket.json_frames] == [
        "turn_accepted",
        "audio_start",
        "speech_timing",
    ]
    assert websocket.json_frames[-1]["payload"]["words"][1]["end_ms"] == 390


@pytest.mark.asyncio
async def test_late_pcm_after_abort_is_dropped(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)
    adapter.handle_message = _noop_handle_message
    await adapter._handle_turn(
        connection, {"type": "turn", "text": "hello", "turn_id": "t3"}
    )
    handle = await adapter.begin_streaming_tts(connection.chat_id, AudioFormat())
    assert handle is not None

    await adapter.abort_streaming_tts(handle, "barge-in")
    await adapter.write_streaming_tts(handle, b"late")
    await adapter.abort_streaming_tts(handle, "duplicate abort")

    assert websocket.binary_frames == []
    assert [frame["type"] for frame in websocket.json_frames] == [
        "turn_accepted",
        "audio_start",
        "audio_abort",
    ]


@pytest.mark.asyncio
async def test_reconnect_cursor_suppresses_duplicate_turn(monkeypatch):
    adapter = _adapter(monkeypatch)
    adapter.handle_message = _noop_handle_message
    first_ws = FakeWebSocket()
    first = await adapter._accept_hello(
        first_ws,
        {
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "client_id": "amanda-laptop",
            "device_id": "macbook",
            "session_id": "default",
        },
    )
    await adapter._handle_turn(
        first, {"type": "turn", "turn_id": "retry-me", "text": "hello"}
    )

    second_ws = FakeWebSocket()
    second = await adapter._accept_hello(
        second_ws,
        {
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "client_id": "amanda-laptop",
            "device_id": "macbook",
            "session_id": "default",
            "last_turn_id": "retry-me",
        },
    )
    assert second.resume_turn_id == "retry-me"
    assert "retry-me" in second.recent_turns

    await adapter._handle_turn(
        second, {"type": "turn", "turn_id": "retry-me", "text": "hello again"}
    )
    assert second_ws.json_frames[-1]["type"] == "turn_duplicate"
    assert len(second_ws.json_frames) == 1
