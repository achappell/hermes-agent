"""OpenCode session affinity rides on main and auxiliary requests."""

from __future__ import annotations

import pytest

from agent import auxiliary_client as aux
from agent.chat_completion_helpers import build_api_kwargs
from agent.opencode_affinity import (
    OPENCODE_SESSION_HEADER,
    is_opencode_target,
    merge_opencode_session_headers,
    opencode_session_headers,
)
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hi"}]


def _agent(provider, model, base_url, api_mode=None):
    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        model=model,
        provider=provider,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="sess-affinity-1",
    )
    if api_mode:
        agent.api_mode = api_mode
        agent._transport = None
        agent._anthropic_base_url = base_url
    return agent


def test_opencode_target_detection_covers_provider_and_url():
    assert is_opencode_target("opencode-go", "https://example.invalid/v1")
    assert is_opencode_target("custom", "https://opencode.ai/zen/go/v1")
    assert not is_opencode_target("openrouter", "https://openrouter.ai/api/v1")


def test_affinity_uses_conversation_context_and_preserves_existing_header(monkeypatch):
    monkeypatch.setattr(
        "agent.opencode_affinity.is_opencode_target", lambda *_: True
    )
    monkeypatch.setattr(
        "agent.portal_tags.get_conversation_context", lambda: "conversation-root"
    )

    assert opencode_session_headers("custom", "https://example.invalid") == {
        OPENCODE_SESSION_HEADER: "conversation-root"
    }
    kwargs = {"extra_headers": {OPENCODE_SESSION_HEADER: "caller-pinned"}}
    assert merge_opencode_session_headers(
        kwargs, "custom", "https://example.invalid", "session-id"
    )["extra_headers"][OPENCODE_SESSION_HEADER] == "caller-pinned"


@pytest.mark.parametrize(
    "provider, model, base_url, api_mode",
    [
        ("opencode-go", "glm-5", "https://opencode.ai/zen/go/v1", None),
        ("opencode-go", "gpt-5.6-luna", "https://opencode.ai/zen/go/v1", None),
        (
            "opencode-go",
            "minimax-m2.7",
            "https://opencode.ai/zen/go/v1",
            "anthropic_messages",
        ),
        ("opencode-free", "laguna-s-2.1-free", "https://opencode.ai/zen/v1", None),
        ("custom", "glm-5", "https://opencode.ai/zen/go/v1", None),
    ],
)
def test_main_turn_sends_stable_session_header_on_every_transport(
    provider, model, base_url, api_mode
):
    agent = _agent(provider, model, base_url, api_mode)
    first = build_api_kwargs(agent, _MSGS)["extra_headers"][OPENCODE_SESSION_HEADER]
    second = build_api_kwargs(agent, _MSGS)["extra_headers"][OPENCODE_SESSION_HEADER]
    assert first == second == "sess-affinity-1"

    other = _agent(
        "openrouter", "anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1"
    )
    assert OPENCODE_SESSION_HEADER not in (
        build_api_kwargs(other, _MSGS).get("extra_headers") or {}
    )


def test_auxiliary_calls_share_the_main_turn_session_key():
    token = aux.set_runtime_main(
        "opencode-go",
        "glm-5",
        base_url="https://opencode.ai/zen/go/v1",
        session_id="sess-affinity-1",
    )
    try:
        kwargs = aux._build_call_kwargs(
            "opencode-go",
            "glm-5",
            _MSGS,
            base_url="https://opencode.ai/zen/go/v1",
        )
        assert kwargs["extra_headers"][OPENCODE_SESSION_HEADER] == "sess-affinity-1"
        other = aux._build_call_kwargs(
            "openrouter", "x", _MSGS, base_url="https://openrouter.ai/api/v1"
        )
        assert OPENCODE_SESSION_HEADER not in (other.get("extra_headers") or {})
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)
