# Hermes voice-session channel

This plugin is the first transport slice for the microphone/speaker device and
phone clients. It is a normal Hermes platform adapter: local speech-to-text
produces a transcript, the adapter creates a `MessageEvent` with
`MessageType.VOICE`, and the gateway owns profile routing, history, queues,
interrupts, and TTS selection.

It does not call Qwen or `/v1/chat/completions` directly.

## Configuration

Set these on the active Hermes profile:

```text
VOICE_SESSION_TOKEN=<long-random-token>
VOICE_SESSION_ALLOWED_USERS=amanda-laptop
VOICE_SESSION_HOST=127.0.0.1
VOICE_SESSION_PORT=8790
```

The default loopback bind is intentional. Bind to a Tailscale/private address
only when the device network path and firewall are ready. Token authentication
and the client allowlist are independent gates.

When enabled, the listener exposes:

* `GET /voice-session/health` — unauthenticated readiness probe.
* `GET /voice-session` — authenticated WebSocket session.

## Protocol v1

The client supplies `Authorization: Bearer <VOICE_SESSION_TOKEN>` on the
WebSocket upgrade, then sends:

```json
{"type":"hello","protocol_version":1,"client_id":"amanda-laptop","device_id":"macbook","session_id":"default","last_turn_id":"turn-0"}
```

For each locally recognized utterance:

```json
{"type":"turn","turn_id":"turn-1","session_id":"default","text":"What is the weather?","stt_source":"local"}
```

The client may send `{"type":"interrupt","turn_id":"turn-1"}` to barge in,
or `{"type":"ping"}` for an application-level liveness check. Binary ingress
is deliberately not part of v1; microphone capture and STT stay on the client.

Clients that see the `command_dispatch` capability may send an explicit
gateway command without turning it into model input:

```json
{
  "type": "command",
  "command_id": "command-1",
  "command": "status",
  "args": "brief",
  "session_id": "default"
}
```

The relay acknowledges accepted commands with `command_accepted`, then routes
the command through the existing gateway dispatcher and returns one correlated
`command_result`:

```json
{
  "type": "command_result",
  "command_id": "command-1",
  "command": "status",
  "status": "ok",
  "text": "...",
  "session_id": "default"
}
```

`status` is `unsupported` for commands the gateway does not expose and `busy`
when a turn or another command is active. Command requests are rejected while
audio is speaking; clients should wait for `turn_end` before sending one.
Command results use the same WebSocket reader as turn events—there is no second
receive loop.

Clients that see the `structured_prompts` capability support typed interactive
operations over that same reader. The server sends a correlated request when
the gateway needs a user decision:

```json
{
  "type": "prompt_request",
  "prompt_id": "prompt-1",
  "prompt_kind": "approval",
  "turn_id": "turn-1",
  "session_id": "default",
  "text": "Command approval required",
  "options": [
    {"id": "once", "label": "Allow Once", "style": "primary"},
    {"id": "deny", "label": "Deny", "style": "danger"}
  ],
  "sensitive": false,
  "timeout_s": 300
}
```

`prompt_kind` is `approval`, `confirm`, `clarify`, `sudo`, or `secret`.
Approval and confirmation replies select an `option_id`; clarify replies use a
known option or free-text `value`; sudo and secret replies use only `value` and
must be presented as masked input. The client answers with:

```json
{
  "type": "prompt_response",
  "prompt_id": "prompt-1",
  "prompt_kind": "approval",
  "option_id": "once"
}
```

Sensitive requests have `options: []` and `sensitive: true`. Their response
value is delivered only to the waiting gateway operation. It is never included
in `prompt_resolved`, rejection frames, logs, or error messages. An empty
sensitive value cancels the operation. Clients should keep each prompt ID
pending until a matching `prompt_resolved` or `prompt_response_rejected` arrives;
duplicate, unknown, mismatched, missing, or oversized replies are rejected
without becoming ordinary turns. A disconnected session cancels its sensitive
waiters.

Server messages are JSON (`hello_ack`, `turn_accepted`, `text_delta`, `text`,
`text_final`, `status`, `audio_start`, `speech_timing`, `audio_end`, `turn_end`, `error`, and
`command_accepted`, `command_result`, `prompt_request`, `prompt_resolved`,
`prompt_response_rejected`, and interrupt events) interleaved with
binary raw little-endian PCM frames. The
`text_delta` payload is the accumulated draft preview and carries
`replace: true`; clients should replace their preview (or append only the new
suffix), not print the whole payload as a token delta. The `audio_start`
message declares sample rate, channels, sample width, and an `exclusive`
audio-focus hint; binary frames belong to that audio stream until `audio_end`.
The optional `speech_timing` event describes one completed PCM segment. Every
record has a stable `segment_id`, normalized spoken `text`, an absolute
`audio_offset_ms` from the beginning of the inbound audio stream, and the
segment's `duration_ms`. Word offsets use that same absolute stream clock.

When alignment succeeds, the event is sent after `audio_start` and before the
matching buffered PCM segment:

```json
{
  "type": "speech_timing",
  "turn_id": "turn-1",
  "session_id": "default",
  "payload": {
    "segment_id": "turn-1-tts-0",
    "text": "Hermes keeps moving.",
    "timing_source": "alignment",
    "audio_offset_ms": 0,
    "duration_ms": 920,
    "words": [
      {"text": "Hermes", "start_ms": 0, "end_ms": 280},
      {"text": "keeps", "start_ms": 280, "end_ms": 510},
      {"text": "moving.", "start_ms": 510, "end_ms": 920}
    ]
  }
}
```

If alignment is disabled, unsupported, times out, returns an error, or
returns malformed or text-mismatched metadata, the relay sends the same
segment record with no word spans:

```json
{
  "type": "speech_timing",
  "turn_id": "turn-1",
  "session_id": "default",
  "payload": {
    "segment_id": "turn-1-tts-1",
    "text": "The next segment.",
    "timing_source": "duration_fallback",
    "fallback_reason": "timeout",
    "audio_offset_ms": 920,
    "duration_ms": 640,
    "words": []
  }
}
```

An aligned record and an alignment-error fallback precede their buffered PCM.
The disabled/unsupported low-latency path streams PCM immediately and sends
its duration record after that segment completes. Valid `fallback_reason`
values are `disabled`, `unsupported`, `missing`, `error`, `timeout`, and
`invalid`; the relay never sends partial word timing or exception text.

The event is optional for clients and may be absent when an adapter does not
support timing events. Clients must continue to render and play turns when it
is absent, and must use `duration_ms` for records whose `timing_source` is
`duration_fallback`.

The stream is single-turn per device connection. Reconnects may reuse the same
`session_id` so Hermes' normal session history remains stable. A reconnect may
include `last_turn_id`; the server reports whether that cursor is known and
suppresses a duplicate retry for a recently accepted turn. Protocol v1 does
not replay the old response, so the client should retry only when it did not
receive `turn_accepted`.
