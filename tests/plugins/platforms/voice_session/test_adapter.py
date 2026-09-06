from __future__ import annotations

import asyncio

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
async def test_command_dispatch_reuses_gateway_event_path(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)
    events = []

    async def capture(event):
        events.append(event)

    adapter.handle_message = capture
    await adapter._handle_payload(
        connection,
        {
            "type": "command",
            "command_id": "command-1",
            "command": "status",
            "args": "brief",
        },
    )

    assert len(events) == 1
    event = events[0]
    assert event.message_type is MessageType.TEXT
    assert event.text == "/status brief"
    assert event.metadata["voice_session_command_id"] == "command-1"
    assert event.metadata["voice_session_command"] == "status"
    assert [frame["type"] for frame in websocket.json_frames] == [
        "command_accepted"
    ]


@pytest.mark.asyncio
async def test_unsupported_command_is_reported_without_gateway_dispatch(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)
    adapter.handle_message = _noop_handle_message

    await adapter._handle_payload(
        connection,
        {
            "type": "command",
            "command_id": "command-unsupported",
            "command": "not-a-gateway-command",
        },
    )

    assert websocket.json_frames == [
        {
            "type": "command_result",
            "command_id": "command-unsupported",
            "command": "not-a-gateway-command",
            "status": "unsupported",
            "text": "",
            "session_id": "default",
            "error": "command is not supported by the gateway",
        }
    ]


@pytest.mark.asyncio
async def test_gateway_command_response_is_correlated_without_creating_a_turn(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)

    async def respond(event):
        assert event.text == "/status"
        return "Gateway is healthy"

    adapter.set_message_handler(respond)
    await adapter._handle_payload(
        connection,
        {
            "type": "command",
            "command_id": "command-status",
            "command": "status",
        },
    )

    for _ in range(20):
        if any(frame["type"] == "command_result" for frame in websocket.json_frames):
            break
        await asyncio.sleep(0)

    assert websocket.json_frames[-1] == {
        "type": "command_result",
        "command_id": "command-status",
        "command": "status",
        "status": "ok",
        "text": "Gateway is healthy",
        "session_id": "default",
    }
    assert connection.current_turn_id is None
    assert connection.active_command_id is None


@pytest.mark.asyncio
async def test_command_during_active_turn_returns_busy_without_dispatch(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)
    connection.current_turn_id = "turn-active"
    connection.turn_end_sent = False
    adapter.handle_message = pytest.fail

    await adapter._handle_payload(
        connection,
        {
            "type": "command",
            "command_id": "command-busy",
            "command": "status",
        },
    )

    assert websocket.json_frames == [
        {
            "type": "command_result",
            "command_id": "command-busy",
            "command": "status",
            "status": "busy",
            "text": "",
            "session_id": "default",
            "error": "a turn is already in progress",
        }
    ]


@pytest.mark.asyncio
async def test_gateway_command_without_text_still_completes(monkeypatch):
    adapter = _adapter(monkeypatch)
    connection, websocket = _connection(adapter)

    async def no_response(_event):
        return None

    adapter.set_message_handler(no_response)
    await adapter._handle_payload(
        connection,
        {
            "type": "command",
            "command_id": "command-empty",
            "command": "status",
        },
    )

    for _ in range(20):
        if any(frame["type"] == "command_result" for frame in websocket.json_frames):
            break
        await asyncio.sleep(0)

    assert websocket.json_frames[-1]["type"] == "command_result"
    assert websocket.json_frames[-1]["command_id"] == "command-empty"
    assert websocket.json_frames[-1]["status"] == "ok"
    assert websocket.json_frames[-1]["text"] == ""


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
            "timing_source": "alignment",
            "audio_offset_ms": 0,
            "duration_ms": 500,
        },
    ) is True

    assert [frame["type"] for frame in websocket.json_frames] == [
        "turn_accepted",
        "audio_start",
        "speech_timing",
    ]
    assert websocket.json_frames[-1]["payload"]["words"][1]["end_ms"] == 390
    assert websocket.json_frames[-1]["payload"]["timing_source"] == "alignment"
    assert websocket.json_frames[-1]["payload"]["duration_ms"] == 500


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
