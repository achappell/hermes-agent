# RELAY-07 Segment-scoped speech timing and fallback contract Implementation Plan

## Goal

Make each completed streamed TTS segment independently authoritative on the
voice-session wire. Every segment reports its identity, normalized spoken
text, absolute start offset, and PCM duration. Word timing is an optional
precision layer; duration fallback remains safe when alignment is disabled,
unsupported, slow, malformed, or unavailable.

## Contract

Extend the existing `speech_timing` payload without adding a new protocol
message:

- aligned segment: `timing_source: "alignment"`, validated `words`,
  `segment_id`, `text`, `audio_offset_ms`, and `duration_ms`;
- fallback segment: `timing_source: "duration_fallback"`, `words: []`, the
  same segment fields, and a bounded `fallback_reason` (`disabled`,
  `unsupported`, `missing`, `error`, `timeout`, or `invalid`).

Alignment-enabled segments are buffered and publish metadata before their
PCM. The low-latency path streams PCM immediately and publishes its duration
record after the segment completes. No exception text or partial word list is
sent to clients.

## TDD implementation sequence

1. Add failing gateway tests for fallback metadata, normalized text, absolute
   offsets, 422/exception and malformed alignment responses, and bounded
   timeout delivery.
2. Add the shared alignment-timeout resolver and implement the segment record
   builder/send path in `gateway/streaming_tts_consumer.py`.
3. Preserve provider streaming behavior while applying the shared timeout to
   alignment calls in `tools/tts_streaming.py`'s configuration seam.
4. Update the voice-session adapter documentation and tests to describe the
   two event-ordering cases and the new payload fields.
5. Run focused tests, the full Hermes test wrapper, diff/credential review, and
   the documented authenticated smoke checks if a configured endpoint is
   available. Leave alignment disabled in local and media profiles.
