"""Behaviour tests for the ``set_reaction`` tool.

The tool stages one emoji for the current turn's completion tapback. It takes
no chat id — the target is always the message that triggered the turn — so it
cannot become a general messaging surface however the model is prompted.
"""
from __future__ import annotations

import json

import pytest

from gateway import pending_reactions
from tools import set_reaction_tool as srt

CHAT = "any;-;+15551234"
MSG = "msg-1"


@pytest.fixture
def photon_session(monkeypatch):
    """Pretend this turn came from a Photon message."""
    env = {
        "HERMES_SESSION_PLATFORM": "photon",
        "HERMES_SESSION_CHAT_ID": CHAT,
        "HERMES_SESSION_MESSAGE_ID": MSG,
    }
    monkeypatch.setattr(srt, "get_session_env", lambda k, d="": env.get(k, d))
    pending_reactions.clear_all()
    yield env
    pending_reactions.clear_all()


def _ok(raw: str) -> bool:
    return json.loads(raw).get("success") is True


def test_stages_the_emoji_for_this_message(photon_session):
    assert _ok(srt.set_reaction_tool("🔧"))
    key = pending_reactions.make_key(CHAT, MSG)
    assert pending_reactions.pop_pending(key) == "🔧"


def test_empty_string_clears_a_previous_pick(photon_session):
    srt.set_reaction_tool("🎉")
    assert _ok(srt.set_reaction_tool(""))
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT, MSG)) is None


def test_rejects_a_word(photon_session):
    raw = srt.set_reaction_tool("nope")
    assert not _ok(raw) if raw.strip().startswith("{") else True
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT, MSG)) is None


def test_rejects_several_emoji(photon_session):
    srt.set_reaction_tool("🔧 🎉")
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT, MSG)) is None


def test_accepts_a_zwj_sequence(photon_session):
    """👩‍👩‍👧‍👧 is one emoji made of four joined by U+200D — not four emoji."""
    assert _ok(srt.set_reaction_tool("👩‍👩‍👧‍👧"))
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT, MSG))


def test_accepts_a_variation_selector(photon_session):
    """❤️ is U+2764 plus U+FE0F; naive validation would reject it."""
    assert _ok(srt.set_reaction_tool("❤️"))


def test_refuses_on_a_platform_that_cannot_consume_it(monkeypatch):
    """Telegram has a chat and a message, but nothing reads the staged pick."""
    env = {
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_ID": "123",
        "HERMES_SESSION_MESSAGE_ID": "456",
    }
    monkeypatch.setattr(srt, "get_session_env", lambda k, d="": env.get(k, d))
    pending_reactions.clear_all()
    srt.set_reaction_tool("🔧")
    assert pending_reactions.pop_pending(pending_reactions.make_key("123", "456")) is None


def test_refuses_when_there_is_no_message(monkeypatch):
    """A bare CLI run has no bubble to tapback."""
    env = {"HERMES_SESSION_PLATFORM": "photon"}
    monkeypatch.setattr(srt, "get_session_env", lambda k, d="": env.get(k, d))
    raw = srt.set_reaction_tool("🔧")
    assert "No message to react to" in raw
