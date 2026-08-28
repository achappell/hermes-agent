# Hermes voice-session channel

This plugin is the first transport slice for the microphone/speaker device and
phone clients. It is a normal Hermes platform adapter: local speech-to-text
produces a transcript, the adapter creates a `MessageEvent` with
`MessageType.VOICE`, and the gateway owns profile routing, history, queues,
interrupts, and TTS selection.

It does not call Qwen or `/v1/chat/completions` directly.

## Configuration

Set these on the media server:

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
{"type":"hello","protocol_version":1,"client_id":"amanda-laptop","device_id":"macbook","session_id":"default"}
```

For each locally recognized utterance:

```json
{"type":"turn","turn_id":"turn-1","session_id":"default","text":"What is the weather?","stt_source":"local"}
```

The client may send `{"type":"interrupt","turn_id":"turn-1"}` to barge in,
or `{"type":"ping"}` for an application-level liveness check. Binary ingress
is deliberately not part of v1; microphone capture and STT stay on the client.

Server messages are JSON (`hello_ack`, `turn_accepted`, `text_delta`, `text`,
`text_final`, `status`, `audio_start`, `audio_end`, `turn_end`, `error`, and
interrupt events) interleaved with binary raw little-endian PCM frames. The
`audio_start` message declares sample rate, channels, and sample width; binary
frames belong to that audio stream until `audio_end`.

The stream is single-turn per device connection. Reconnects may reuse the same
`session_id` so Hermes' normal session history remains stable.
