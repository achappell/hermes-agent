# Relay downstream stack

This branch carries the smallest Hermes Agent downstream layer required by the
Hermes Relay iOS and TUI clients. It is intentionally based on the official
Hermes Agent release tag `v2026.8.31` (v0.21.0), not on upstream `main`.

## Included

The branch contains only the server-side pieces needed for the relay path:

* the authenticated `voice_session` WebSocket platform;
* streamed PCM TTS, including the Qwen streaming provider;
* optional provider-side word alignment and `speech_timing` events;
* the gateway fixes for streamed-TTS tail draining and interim segment
  boundaries; and
* focused tests and protocol documentation for those pieces.

The clients own microphone capture, speech recognition, playback, transcript
rendering, and interruption UX. The server therefore does not carry a laptop
microphone client or client-side playback-speed adjustment code.

## Release update procedure

1. Fetch the next official Hermes Agent release tag.
2. Create a new branch from that tag.
3. Reapply the small, ordered downstream commits from this branch:
   `voice_session`, streaming TTS/alignment, gateway correctness fixes, and
   this documentation.
4. Resolve only the files touched by those commits.
5. Run the focused tests and the authenticated `hello`/`hello_ack` smoke test
   against each gateway before changing the service checkout.

Keep the release tag and downstream commit list in the handoff or release
record. Do not rebase this branch onto upstream `main` merely to pick up
unreleased changes.

## Wire contracts

The relay depends on protocol v1: authenticated `hello`/`hello_ack`, text
`turn` messages, accumulated `text_delta` previews, optional `speech_timing`,
binary signed 16-bit PCM after `audio_start`, and `turn_end`. The protocol has
an interrupt operation for barge-in; it does not require binary microphone
ingress.

Alignment is optional. Each completed streamed TTS segment may carry a
`speech_timing` record with a stable `segment_id`, normalized spoken `text`,
absolute `audio_offset_ms`, and segment `duration_ms`. A successful record has
`timing_source: "alignment"` and a complete validated `words` list. An
alignment failure, timeout, malformed response, text mismatch, unsupported
provider, or disabled experiment has `timing_source: "duration_fallback"`, an
empty `words` list, and a bounded `fallback_reason`. Segment IDs advance with
the audio clock, so a failed segment cannot make an earlier word track
authoritative for later PCM.

Aligned records and buffered alignment fallbacks precede their PCM. The
disabled/unsupported low-latency path writes PCM immediately and sends its
duration record after the segment. The gateway bounds alignment waits and
always preserves ordinary streamed PCM and the client’s audio-clock fallback;
it never sends partial word timing or exception text.

## Validation

From this checkout, use the canonical hermetic runner:

```bash
scripts/run_tests.sh \
  tests/plugins/platforms/voice_session/test_adapter.py \
  tests/gateway/test_streaming_tts_consumer.py \
  tests/gateway/test_streaming_tts_gateway_regression.py \
  tests/gateway/test_stream_final_contract.py \
  tests/tools/test_tts_streaming.py
```

The service smoke test must not send a prompt or record audio. It only checks
that the health endpoint answers and that an authenticated client completes
`hello`/`hello_ack`.
