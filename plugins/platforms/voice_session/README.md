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

## Laptop microphone POC

Run `scripts/voice-session-client.py` on the laptop that has the microphone.
Install the optional local audio/STT dependencies in that Hermes checkout first:

```bash
uv sync --extra voice
```

Then capture one turn:

```bash
VOICE_SESSION_TOKEN='…' \
  scripts/voice-session-client.py \
  --url ws://<media-server-address>:8790/voice-session \
  --microphone --once
```

Without `--once`, the client returns to a listening prompt for another turn.
The recorder ends an utterance after the configured silence duration and has a
hard cap; tune those boundaries with `--mic-silence-duration` and
`--mic-max-seconds`. `--stt-model` overrides `stt.local.model` for that run.

The microphone audio is recorded and transcribed locally through Hermes'
existing Faster-Whisper pipeline. Only the resulting transcript crosses the
WebSocket; v1 deliberately has no binary microphone-ingress path. Response
audio is still written to a WAV file by this POC rather than played locally.
Keep the token in the environment (or another secret store), never in this
README or a shell-history-friendly command you plan to commit.

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

Server messages are JSON (`hello_ack`, `turn_accepted`, `text_delta`, `text`,
`text_final`, `status`, `audio_start`, `audio_end`, `turn_end`, `error`, and
interrupt events) interleaved with binary raw little-endian PCM frames. The
`text_delta` payload is the accumulated draft preview and carries
`replace: true`; clients should replace their preview (or append only the new
suffix), not print the whole payload as a token delta. The `audio_start`
message declares sample rate, channels, sample width, and an `exclusive`
audio-focus hint; binary frames belong to that audio stream until `audio_end`.

The stream is single-turn per device connection. Reconnects may reuse the same
`session_id` so Hermes' normal session history remains stable. A reconnect may
include `last_turn_id`; the server reports whether that cursor is known and
suppresses a duplicate retry for a recently accepted turn. Protocol v1 does
not replay the old response, so the client should retry only when it did not
receive `turn_accepted`.
