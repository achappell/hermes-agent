# RELAY-01 Structured Prompt Operations and Steering Implementation Plan

> **For Hermes:** Use the test-driven-development workflow and keep each
> protocol slice independently verifiable.

**Goal:** Give voice-session clients typed, correlated approval, confirmation,
clarify, secret, sudo, and steering operations without routing interactive
responses through ordinary model text.

**Architecture:** The voice-session adapter emits `prompt_request` frames and
consumes `prompt_response` frames in its existing WebSocket receive loop. A
bounded per-connection prompt registry maps the wire `prompt_id` to the
existing Hermes gateway wait primitive. Responses are validated, consumed once,
and acknowledged without echoing sensitive values. Steering uses a distinct
typed operation and remains separate from interrupting.

**Tech Stack:** Python, aiohttp WebSockets, `BasePlatformAdapter`, Hermes
gateway approval and clarify primitives, pytest/pytest-asyncio.

---

## Contract decision

Server → client:

```json
{
  "type": "prompt_request",
  "prompt_id": "request-1",
  "prompt_kind": "approval",
  "turn_id": "turn-1",
  "session_id": "default",
  "text": "Command approval required",
  "options": [{"id": "once", "label": "Allow once"}],
  "sensitive": false,
  "timeout_s": 300
}
```

Client → server:

```json
{
  "type": "prompt_response",
  "prompt_id": "request-1",
  "prompt_kind": "approval",
  "option_id": "once",
  "value": "",
  "reason": ""
}
```

`approval` and `confirm` use `option_id`; `clarify` uses either a known option
ID or a free-text `value`; `sudo` and `secret` use only `value` and mark the
request `sensitive`; steering uses a separate `steer` frame carrying the
active `turn_id` and replacement text. The server never includes a submitted
secret or password in an acknowledgement, log, or exception.

Rejected directions:

- Encode prompt answers as `/approve`, `/deny`, or numbered chat text. That
  loses correlation and makes a missing capability look like model input.
- Add a second WebSocket reader for prompts. That races the turn reader and
  violates the existing one-reader invariant.
- Treat steering as interrupt. Steering changes the active turn's input;
  interrupt cancels it. They need different client UX and server semantics.

## Tasks

### Task 1: Approval prompt request and correlated response

- Add a failing adapter test proving `send_exec_approval` emits a
  `prompt_request` with approval options and a stable prompt ID.
- Add a failing test proving a matching `prompt_response` resolves the
  existing gateway approval waiter and emits a non-sensitive acknowledgement.
- Implement the bounded prompt registry, request sender, and response router.
- Keep invalid option IDs pending and return a typed rejection.

### Task 2: Confirmation and clarify operations

- Cover slash confirmation options and clarify choice/free-text mapping.
- Reuse the same prompt registry and one-reader response path.
- Preserve the existing text fallback when a prompt cannot be sent.

### Task 3: Timeout, disconnect, duplicate, and malformed responses

- Consume each prompt ID at most once.
- Reject unknown prompt IDs, mismatched kinds, missing values, and oversized
  values without dispatching them as turns.
- Close or resolve pending waiters when the connection ends.

### Task 4: Sudo and secret prompt bridges

- Add explicit callback bridges for masked `sudo` and `secret` requests.
- Mark these requests sensitive and ensure response values never appear in
  outbound acknowledgements, diagnostics, or logs.
- Test cancellation and disconnect as non-successful responses.

### Task 5: Explicit steering operation

- Add a typed steer request tied to the active turn.
- Route it through the existing gateway session activity path without creating
  a new turn or confusing it with interrupt.
- Cover active, idle, stale-turn, and busy cases.

### Task 6: Protocol documentation and verification

- Document the frames and capability in
  `plugins/platforms/voice_session/README.md`.
- Run the focused voice-session and gateway relay suites, Ruff, and diff
  hygiene checks.
- Record the evidence on RELAY-01 before moving the card to Verify.
