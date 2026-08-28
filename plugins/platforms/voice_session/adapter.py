"""Authenticated, first-class voice-session transport for Hermes Agent.

The adapter deliberately keeps the wire protocol small.  A client authenticates
the WebSocket upgrade, sends local-STT transcripts as ``turn`` messages, and
receives ordinary Hermes text plus raw little-endian PCM as the agent responds.
All turns still enter Hermes through :class:`BasePlatformAdapter`; this is a
channel, not a second Chat Completions client.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret

try:  # Keep plugin discovery/import useful when optional deps are absent.
    from aiohttp import WSMsgType, web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by dependency checks
    WSMsgType = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    AudioFormat,
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    StreamingTTSHandle,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8790
DEFAULT_PATH = "/voice-session"
DEFAULT_HELLO_TIMEOUT = 5.0
MAX_FRAME_BYTES = 256 * 1024
MAX_TRANSCRIPT_CHARS = 32_000
MAX_ID_CHARS = 128
MAX_DISPLAY_NAME_CHARS = 128
MAX_RECENT_TURNS = 128
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _get_scoped_secret(name: str, default: str = "") -> str:
    """Read a profile-scoped secret, with the default-profile fallback."""

    try:
        value = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        value = os.getenv(name, default)
    return str(value if value is not None else default)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _csv_set(value: Any) -> Set[str]:
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value or "").split(",")
    return {str(item).strip() for item in items if str(item).strip()}


def _int_setting(
    value: Any, default: int, *, minimum: int = 1, maximum: int = 65535
) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if minimum <= parsed <= maximum else default


def _float_setting(
    value: Any, default: float, *, minimum: float = 0.1, maximum: float = 60.0
) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if minimum <= parsed <= maximum else default


def _safe_id(value: Any, field_name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if not text and not required:
        return ""
    if not _ID_RE.fullmatch(text):
        raise ValueError(f"invalid {field_name}")
    return text


def _bearer_token(headers: Any) -> str:
    raw = str(headers.get("Authorization", "") or "").strip()
    scheme, _, token = raw.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return ""
    return token.strip()


@dataclass
class _Connection:
    websocket: Any
    chat_id: str
    client_id: str
    device_id: str
    session_id: str
    display_name: str = ""
    current_turn_id: Optional[str] = None
    active_tts: Optional["_VoiceSessionTTSHandle"] = None
    final_text_sent: bool = False
    turn_end_sent: bool = False
    interrupted: bool = False
    closed: bool = False
    recent_turns: Set[str] = field(default_factory=set)
    recent_turn_order: list[str] = field(default_factory=list)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class _VoiceSessionTTSHandle(StreamingTTSHandle):
    connection: Any = field(default=None, repr=False)
    turn_id: str = ""
    finished: bool = False


class VoiceSessionAdapter(BasePlatformAdapter):
    """WebSocket voice channel that feeds local-STT transcripts to Hermes."""

    MAX_MESSAGE_LENGTH = MAX_TRANSCRIPT_CHARS

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("voice_session"))

        extra = getattr(config, "extra", {}) or {}
        self._host = str(
            extra.get("host") or os.getenv("VOICE_SESSION_HOST") or DEFAULT_HOST
        ).strip()
        self._port = _int_setting(
            extra.get("port") or os.getenv("VOICE_SESSION_PORT"),
            DEFAULT_PORT,
        )
        path = str(
            extra.get("path") or os.getenv("VOICE_SESSION_PATH") or DEFAULT_PATH
        ).strip()
        self._path = "/" + path.lstrip("/").rstrip("/")
        self._hello_timeout = _float_setting(
            extra.get("hello_timeout") or os.getenv("VOICE_SESSION_HELLO_TIMEOUT"),
            DEFAULT_HELLO_TIMEOUT,
        )

        self._auth_token = (
            getattr(config, "token", None)
            or getattr(config, "api_key", None)
            or _get_scoped_secret("VOICE_SESSION_TOKEN")
        ).strip()
        self._allow_all = _truthy(
            os.getenv("VOICE_SESSION_ALLOW_ALL_USERS") or extra.get("allow_all_users")
        )
        self._allowed_users = _csv_set(
            os.getenv("VOICE_SESSION_ALLOWED_USERS")
        ) | _csv_set(extra.get("allowed_users", []))

        self._app = None
        self._runner = None
        self._site = None
        # One live socket per authenticated physical device.  The session id
        # is a thread/session lane inside that device connection.
        self._connections: Dict[str, _Connection] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "missing_dep",
                "aiohttp is required for the voice-session adapter — install with `pip install aiohttp`",
                retryable=False,
            )
            return False
        if not self._auth_token:
            self._set_fatal_error(
                "config_missing",
                "VOICE_SESSION_TOKEN must be set",
                retryable=False,
            )
            return False

        self._app = web.Application(client_max_size=MAX_FRAME_BYTES)
        self._app.router.add_get(self._path, self._handle_websocket)
        self._app.router.add_get(f"{self._path}/health", self._handle_health)
        self._runner = web.AppRunner(self._app)
        try:
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
        except OSError as exc:
            self._set_fatal_error(
                "bind_failed",
                f"Could not bind voice-session listener on {self._host}:{self._port}: {exc}",
                retryable=True,
            )
            if self._runner is not None:
                await self._runner.cleanup()
            self._runner = None
            self._site = None
            self._app = None
            return False
        except Exception as exc:
            logger.error("Voice session listener failed to start: %s", exc)
            if self._runner is not None:
                await self._runner.cleanup()
            self._runner = None
            self._site = None
            self._app = None
            return False

        self._mark_connected()
        logger.info(
            "Voice session listening on %s:%s%s (protocol v%s)",
            self._host,
            self._port,
            self._path,
            PROTOCOL_VERSION,
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        for connection in list(self._connections.values()):
            await self._close_connection(connection)
        self._connections.clear()

        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                logger.debug("Voice session site cleanup failed", exc_info=True)
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                logger.debug("Voice session runner cleanup failed", exc_info=True)
            self._runner = None
        self._app = None

    # ------------------------------------------------------------------
    # WebSocket ingress
    # ------------------------------------------------------------------

    async def _handle_health(self, request: Any) -> Any:
        return web.json_response({
            "status": "ok",
            "platform": "voice_session",
            "protocol_version": PROTOCOL_VERSION,
        })

    async def _handle_websocket(self, request: Any) -> Any:
        if not hmac.compare_digest(_bearer_token(request.headers), self._auth_token):
            return web.Response(status=401, text="unauthorized")

        websocket = web.WebSocketResponse(
            heartbeat=30,
            max_msg_size=MAX_FRAME_BYTES,
        )
        await websocket.prepare(request)
        connection: Optional[_Connection] = None
        try:
            try:
                first = await asyncio.wait_for(
                    websocket.receive(), timeout=self._hello_timeout
                )
            except asyncio.TimeoutError:
                await self._protocol_close(websocket, "hello timeout")
                return websocket
            if first.type != WSMsgType.TEXT:
                await self._protocol_close(websocket, "hello must be a JSON object")
                return websocket
            try:
                hello = self._decode_payload(first.data)
                connection = await self._accept_hello(websocket, hello)
            except ValueError as exc:
                await self._send_error_raw(websocket, str(exc))
                await self._protocol_close(websocket, str(exc))
                return websocket

            await self._send_json(
                connection,
                {
                    "type": "hello_ack",
                    "protocol_version": PROTOCOL_VERSION,
                    "client_id": connection.client_id,
                    "device_id": connection.device_id,
                    "session_id": connection.session_id,
                    "chat_id": connection.chat_id,
                    "capabilities": ["text_stream", "pcm_s16le", "interrupt"],
                },
            )

            async for message in websocket:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = self._decode_payload(message.data)
                        await self._handle_payload(connection, payload)
                    except ValueError as exc:
                        await self._send_json(
                            connection,
                            {"type": "error", "error": str(exc)},
                        )
                    except Exception:
                        logger.exception("Voice session payload failed")
                        await self._send_json(
                            connection,
                            {"type": "error", "error": "internal server error"},
                        )
                elif message.type == WSMsgType.BINARY:
                    await self._send_json(
                        connection,
                        {
                            "type": "error",
                            "error": "binary ingress is not part of protocol v1; send local-STT text",
                        },
                    )
                elif message.type in {
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSED,
                    WSMsgType.ERROR,
                }:
                    break
        finally:
            if connection is not None:
                await self._close_connection(connection)
            elif not websocket.closed:
                await websocket.close()
        return websocket

    async def _accept_hello(
        self, websocket: Any, payload: Dict[str, Any]
    ) -> _Connection:
        if payload.get("type") != "hello":
            raise ValueError("first message must be hello")
        version = payload.get("protocol_version")
        if isinstance(version, bool) or int(version or 0) != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol_version (expected {PROTOCOL_VERSION})"
            )

        client_id = _safe_id(payload.get("client_id"), "client_id")
        device_id = _safe_id(payload.get("device_id") or client_id, "device_id")
        session_id = _safe_id(payload.get("session_id") or "default", "session_id")
        display_name = str(payload.get("display_name") or "").strip()[
            :MAX_DISPLAY_NAME_CHARS
        ]
        if not self._allow_all and client_id not in self._allowed_users:
            raise ValueError("client is not allowlisted")

        chat_id = f"{client_id}:{device_id}"
        old = self._connections.get(chat_id)
        if old is not None and old.websocket is not websocket:
            await self._close_connection(old)
        connection = _Connection(
            websocket=websocket,
            chat_id=chat_id,
            client_id=client_id,
            device_id=device_id,
            session_id=session_id,
            display_name=display_name,
        )
        self._connections[chat_id] = connection
        return connection

    @staticmethod
    def _decode_payload(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_FRAME_BYTES:
            raise ValueError("JSON frame is too large")
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid JSON frame") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON frame must be an object")
        return payload

    async def _handle_payload(
        self, connection: _Connection, payload: Dict[str, Any]
    ) -> None:
        kind = str(payload.get("type") or "").strip().lower()
        if kind == "ping":
            await self._send_json(connection, {"type": "pong"})
            return
        if kind == "turn":
            await self._handle_turn(connection, payload)
            return
        if kind == "interrupt":
            await self._handle_interrupt(connection, payload)
            return
        raise ValueError("unknown message type")

    async def _handle_turn(
        self, connection: _Connection, payload: Dict[str, Any]
    ) -> None:
        if connection.current_turn_id and not connection.turn_end_sent:
            raise ValueError("a turn is already in progress")

        turn_id = _safe_id(payload.get("turn_id") or uuid.uuid4().hex, "turn_id")
        if turn_id in connection.recent_turns:
            await self._send_json(
                connection,
                {
                    "type": "turn_duplicate",
                    "turn_id": turn_id,
                    "session_id": connection.session_id,
                },
            )
            return
        session_id = _safe_id(
            payload.get("session_id") or connection.session_id,
            "session_id",
        )
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("turn text is empty")
        if len(text) > MAX_TRANSCRIPT_CHARS:
            raise ValueError("turn text is too long")

        connection.session_id = session_id
        connection.current_turn_id = turn_id
        connection.final_text_sent = False
        connection.turn_end_sent = False
        connection.interrupted = False
        connection.active_tts = None
        connection.recent_turns.add(turn_id)
        connection.recent_turn_order.append(turn_id)
        while len(connection.recent_turn_order) > MAX_RECENT_TURNS:
            expired = connection.recent_turn_order.pop(0)
            connection.recent_turns.discard(expired)

        source = self._source_for(connection, turn_id)
        event = MessageEvent(
            text=text,
            message_type=MessageType.VOICE,
            user_id=connection.client_id,
            user_name=connection.display_name or connection.client_id,
            source=source,
            raw_message=payload,
            message_id=turn_id,
            metadata={
                "voice_session_protocol": PROTOCOL_VERSION,
                "voice_session_id": session_id,
                "voice_session_turn_id": turn_id,
                "voice_session_client_id": connection.client_id,
                "voice_session_device_id": connection.device_id,
                "stt_source": str(payload.get("stt_source") or "local"),
                "audio_format": payload.get("audio_format") or "pcm_s16le",
            },
            timestamp=datetime.now(timezone.utc),
            allow_gateway_control=True,
        )
        await self._send_json(
            connection,
            {
                "type": "turn_accepted",
                "turn_id": turn_id,
                "session_id": session_id,
            },
        )
        await self.handle_message(event)

    async def _handle_interrupt(
        self, connection: _Connection, payload: Dict[str, Any]
    ) -> None:
        requested = str(
            payload.get("turn_id") or connection.current_turn_id or ""
        ).strip()
        if not requested or requested != connection.current_turn_id:
            raise ValueError("turn_id does not match the active turn")
        source = self._source_for(connection, requested)
        from gateway.session import build_session_key

        session_key = build_session_key(
            source,
            group_sessions_per_user=bool(
                (getattr(self.config, "extra", {}) or {}).get(
                    "group_sessions_per_user", True
                )
            ),
            thread_sessions_per_user=bool(
                (getattr(self.config, "extra", {}) or {}).get(
                    "thread_sessions_per_user", False
                )
            ),
            profile=self._session_key_profile(source),
        )
        await self.interrupt_session_activity(
            session_key,
            connection.chat_id,
            {"thread_id": connection.session_id},
        )
        handle = connection.active_tts
        if handle is not None:
            await self.abort_streaming_tts(handle, "client interrupt")
        connection.interrupted = True
        await self._send_json(
            connection,
            {
                "type": "turn_interrupted",
                "turn_id": requested,
                "session_id": connection.session_id,
            },
        )

    def _source_for(
        self, connection: _Connection, message_id: Optional[str] = None
    ) -> Any:
        return self.build_source(
            chat_id=connection.chat_id,
            chat_name=connection.device_id,
            chat_type="dm",
            user_id=connection.client_id,
            user_name=connection.display_name or connection.client_id,
            thread_id=connection.session_id,
            message_id=message_id,
        )

    # ------------------------------------------------------------------
    # Outbound text and status
    # ------------------------------------------------------------------

    def _connection_for_chat(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[_Connection]:
        connection = self._connections.get(str(chat_id))
        if connection is None or connection.closed:
            return None
        metadata = metadata or {}
        requested_session = metadata.get("voice_session_id") or metadata.get(
            "thread_id"
        )
        if requested_session and str(requested_session) != connection.session_id:
            return None
        return connection

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        connection = self._connection_for_chat(chat_id, metadata)
        if connection is None:
            return SendResult(
                success=False,
                error="voice-session device is not connected",
                retryable=True,
            )
        text = str(content or "")
        if not text:
            return SendResult(success=True, message_id=None)
        metadata = metadata or {}
        is_final = bool(metadata.get("notify")) and not bool(
            metadata.get("_interim_send")
        )
        event = {
            "type": "text_final" if is_final else "text",
            "text": text,
            "turn_id": connection.current_turn_id,
            "session_id": connection.session_id,
        }
        if not await self._send_json(connection, event):
            return SendResult(
                success=False, error="voice-session socket closed", retryable=True
            )
        if is_final:
            connection.final_text_sent = True
            await self._maybe_end_turn(connection)
        return SendResult(success=True, message_id=connection.current_turn_id)

    def supports_draft_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
    ) -> bool:
        if chat_id is None:
            return False
        return self._connection_for_chat(chat_id, metadata) is not None

    async def send_draft(
        self,
        chat_id: str,
        draft_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        connection = self._connection_for_chat(chat_id, metadata)
        if connection is None:
            return SendResult(
                success=False,
                error="voice-session device is not connected",
                retryable=True,
            )
        ok = await self._send_json(
            connection,
            {
                "type": "text_delta",
                "draft_id": int(draft_id),
                "text": str(content or ""),
                "turn_id": connection.current_turn_id,
                "session_id": connection.session_id,
            },
        )
        return SendResult(
            success=ok,
            error=None if ok else "voice-session socket closed",
            retryable=not ok,
        )

    async def send_typing(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        connection = self._connection_for_chat(chat_id, metadata)
        if connection is not None:
            await self._send_json(
                connection,
                {
                    "type": "status",
                    "status": "thinking",
                    "text": self._status_text.get(chat_id) or "",
                    "turn_id": connection.current_turn_id,
                    "session_id": connection.session_id,
                },
            )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        connection = self._connections.get(str(chat_id))
        return {
            "name": connection.device_id if connection else str(chat_id),
            "type": "dm",
        }

    def format_message(self, content: str) -> str:
        return str(content or "")

    # ------------------------------------------------------------------
    # Streaming PCM TTS
    # ------------------------------------------------------------------

    def supports_streaming_tts(self, chat_id: str, audio_format: AudioFormat) -> bool:
        connection = self._connection_for_chat(chat_id)
        return bool(
            connection
            and connection.current_turn_id
            and not connection.turn_end_sent
            and connection.active_tts is None
        )

    async def begin_streaming_tts(
        self,
        chat_id: str,
        audio_format: AudioFormat,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamingTTSHandle]:
        connection = self._connection_for_chat(chat_id, metadata)
        if (
            connection is None
            or not connection.current_turn_id
            or connection.turn_end_sent
            or connection.active_tts is not None
        ):
            return None
        handle = _VoiceSessionTTSHandle(
            chat_id=chat_id,
            audio_format=audio_format,
            connection=connection,
            turn_id=connection.current_turn_id,
        )
        connection.active_tts = handle
        ok = await self._send_json(
            connection,
            {
                "type": "audio_start",
                "turn_id": handle.turn_id,
                "session_id": connection.session_id,
                "sample_rate": int(audio_format.sample_rate),
                "channels": int(audio_format.channels),
                "sample_width": int(audio_format.sample_width),
                "encoding": "pcm_s16le",
            },
        )
        if not ok:
            connection.active_tts = None
            handle.aborted = True
            return None
        return handle

    async def write_streaming_tts(
        self, handle: StreamingTTSHandle, chunk: bytes
    ) -> None:
        if not isinstance(handle, _VoiceSessionTTSHandle):
            return
        if handle.aborted or handle.finished or not chunk:
            return
        connection = handle.connection
        if connection is None or connection.closed:
            return
        if connection.active_tts is not handle:
            return
        ok = await self._send_bytes(connection, bytes(chunk))
        if ok:
            handle.audible = True

    async def finish_streaming_tts(
        self,
        handle: StreamingTTSHandle,
        *,
        interrupted: bool = False,
    ) -> None:
        if (
            not isinstance(handle, _VoiceSessionTTSHandle)
            or handle.finished
            or handle.aborted
        ):
            return
        handle.finished = True
        connection = handle.connection
        if connection is None:
            return
        if connection.active_tts is handle:
            connection.active_tts = None
        await self._send_json(
            connection,
            {
                "type": "audio_end",
                "turn_id": handle.turn_id,
                "session_id": connection.session_id,
                "interrupted": bool(interrupted),
            },
        )
        await self._maybe_end_turn(connection, interrupted=interrupted)

    async def abort_streaming_tts(
        self,
        handle: StreamingTTSHandle,
        error: Optional[str] = None,
    ) -> None:
        if not isinstance(handle, _VoiceSessionTTSHandle) or handle.aborted:
            return
        handle.aborted = True
        connection = handle.connection
        if connection is None:
            return
        if connection.active_tts is handle:
            connection.active_tts = None
        await self._send_json(
            connection,
            {
                "type": "audio_abort",
                "turn_id": handle.turn_id,
                "session_id": connection.session_id,
                "error": str(error or "audio stream aborted"),
            },
        )
        await self._maybe_end_turn(connection, interrupted=True)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Deliver whole-file fallback without exposing a host filesystem path."""

        connection = self._connection_for_chat(chat_id, metadata)
        if connection is None:
            return SendResult(
                success=False,
                error="voice-session device is not connected",
                retryable=True,
            )
        try:
            data = await asyncio.to_thread(Path(audio_path).read_bytes)
        except (OSError, ValueError) as exc:
            return SendResult(
                success=False, error=f"could not read audio fallback: {exc}"
            )
        content_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
        if not await self._send_json(
            connection,
            {
                "type": "audio_file_start",
                "turn_id": connection.current_turn_id,
                "session_id": connection.session_id,
                "content_type": content_type,
                "caption": caption or "",
            },
        ):
            return SendResult(
                success=False, error="voice-session socket closed", retryable=True
            )
        if not await self._send_bytes(connection, data):
            return SendResult(
                success=False, error="voice-session socket closed", retryable=True
            )
        await self._send_json(
            connection,
            {
                "type": "audio_file_end",
                "turn_id": connection.current_turn_id,
                "session_id": connection.session_id,
            },
        )
        return SendResult(success=True, message_id=connection.current_turn_id)

    # ------------------------------------------------------------------
    # Wire helpers
    # ------------------------------------------------------------------

    async def _send_json(
        self, connection: _Connection, payload: Dict[str, Any]
    ) -> bool:
        if connection.closed or connection.websocket.closed:
            return False
        try:
            async with connection.send_lock:
                if connection.closed or connection.websocket.closed:
                    return False
                await connection.websocket.send_json(payload)
            return True
        except (ConnectionError, RuntimeError, OSError):
            connection.closed = True
            return False

    async def _send_bytes(self, connection: _Connection, payload: bytes) -> bool:
        if connection.closed or connection.websocket.closed:
            return False
        try:
            async with connection.send_lock:
                if connection.closed or connection.websocket.closed:
                    return False
                await connection.websocket.send_bytes(payload)
            return True
        except (ConnectionError, RuntimeError, OSError):
            connection.closed = True
            return False

    async def _send_error_raw(self, websocket: Any, error: str) -> None:
        try:
            await websocket.send_json({"type": "error", "error": error})
        except Exception:
            pass

    async def _protocol_close(self, websocket: Any, reason: str) -> None:
        try:
            await websocket.close(code=1008, message=reason.encode("utf-8")[:120])
        except Exception:
            pass

    async def _maybe_end_turn(
        self, connection: _Connection, *, interrupted: bool = False
    ) -> None:
        if (
            connection.closed
            or connection.turn_end_sent
            or not connection.final_text_sent
            or connection.active_tts is not None
        ):
            return
        connection.turn_end_sent = True
        await self._send_json(
            connection,
            {
                "type": "turn_end",
                "turn_id": connection.current_turn_id,
                "session_id": connection.session_id,
                "interrupted": bool(interrupted or connection.interrupted),
            },
        )

    async def _close_connection(self, connection: _Connection) -> None:
        if connection.closed:
            if self._connections.get(connection.chat_id) is connection:
                self._connections.pop(connection.chat_id, None)
            return
        connection.closed = True
        if connection.active_tts is not None:
            connection.active_tts.aborted = True
            connection.active_tts = None
        if self._connections.get(connection.chat_id) is connection:
            self._connections.pop(connection.chat_id, None)
        try:
            if not connection.websocket.closed:
                await connection.websocket.close()
        except Exception:
            logger.debug("Voice session socket close failed", exc_info=True)


# ----------------------------------------------------------------------
# Plugin registration
# ----------------------------------------------------------------------


def check_requirements() -> bool:
    return AIOHTTP_AVAILABLE and bool(_get_scoped_secret("VOICE_SESSION_TOKEN").strip())


def validate_config(config: PlatformConfig) -> bool:
    token = (
        getattr(config, "token", None)
        or getattr(config, "api_key", None)
        or _get_scoped_secret("VOICE_SESSION_TOKEN")
    )
    return AIOHTTP_AVAILABLE and bool(str(token or "").strip())


def is_connected(config: PlatformConfig) -> bool:
    extra = getattr(config, "extra", {}) or {}
    token = (
        getattr(config, "token", None)
        or getattr(config, "api_key", None)
        or _get_scoped_secret("VOICE_SESSION_TOKEN")
    )
    return bool(str(token or "").strip() and (extra.get("enabled", True) is not False))


def _env_enablement() -> Optional[dict]:
    token = _get_scoped_secret("VOICE_SESSION_TOKEN").strip()
    if not token:
        return None
    seed: dict[str, Any] = {}
    host = os.getenv("VOICE_SESSION_HOST", "").strip()
    path = os.getenv("VOICE_SESSION_PATH", "").strip()
    port = os.getenv("VOICE_SESSION_PORT", "").strip()
    if host:
        seed["host"] = host
    if path:
        seed["path"] = path
    if port:
        try:
            seed["port"] = int(port)
        except ValueError:
            pass
    return seed


def register(ctx) -> None:
    ctx.register_platform(
        name="voice_session",
        label="Voice session",
        adapter_factory=lambda cfg: VoiceSessionAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["VOICE_SESSION_TOKEN"],
        install_hint="pip install aiohttp",
        env_enablement_fn=_env_enablement,
        allowed_users_env="VOICE_SESSION_ALLOWED_USERS",
        allow_all_env="VOICE_SESSION_ALLOW_ALL_USERS",
        max_message_length=MAX_TRANSCRIPT_CHARS,
        emoji="🎙️",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting through an authenticated Hermes voice session. "
            "Inbound text is a local-STT transcript. Responses may arrive as "
            "streamed text and raw PCM audio; keep spoken replies clear and "
            "avoid relying on Markdown-only presentation."
        ),
    )


__all__ = ["VoiceSessionAdapter", "register"]
