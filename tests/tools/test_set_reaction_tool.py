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
    key = pending_reactions.make_key(CHAT)
    assert pending_reactions.pop_pending(key) == "🔧"


def test_empty_string_clears_a_previous_pick(photon_session):
    srt.set_reaction_tool("🎉")
    assert _ok(srt.set_reaction_tool(""))
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT)) is None


def test_rejects_a_word(photon_session):
    raw = srt.set_reaction_tool("nope")
    assert not _ok(raw) if raw.strip().startswith("{") else True
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT)) is None


def test_rejects_several_emoji(photon_session):
    srt.set_reaction_tool("🔧 🎉")
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT)) is None


def test_accepts_a_zwj_sequence(photon_session):
    """👩‍👩‍👧‍👧 is one emoji made of four joined by U+200D — not four emoji."""
    assert _ok(srt.set_reaction_tool("👩‍👩‍👧‍👧"))
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT))


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
    assert pending_reactions.pop_pending(pending_reactions.make_key("123")) is None


def test_refuses_when_there_is_no_chat(monkeypatch):
    """A bare CLI run has no bubble to tapback."""
    env = {"HERMES_SESSION_PLATFORM": "photon"}
    monkeypatch.setattr(srt, "get_session_env", lambda k, d="": env.get(k, d))
    raw = srt.set_reaction_tool("🔧")
    assert "No chat to react in" in raw


def test_works_without_a_session_message_id(monkeypatch):
    """The regression this fix exists for.

    Photon puts message_id on the MessageEvent, never on the MessageSource, so
    HERMES_SESSION_MESSAGE_ID is always empty here. Keying on it meant the tool
    could never build a key and refused every single call.
    """
    env = {
        "HERMES_SESSION_PLATFORM": "photon",
        "HERMES_SESSION_CHAT_ID": CHAT,
        "HERMES_SESSION_MESSAGE_ID": "",
    }
    monkeypatch.setattr(srt, "get_session_env", lambda k, d="": env.get(k, d))
    pending_reactions.clear_all()
    assert _ok(srt.set_reaction_tool("🔧"))
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT)) == "🔧"


def test_accepts_slack(monkeypatch):
    """Slack consumes staged picks too (translated to a shortcode by the adapter)."""
    env = {
        "HERMES_SESSION_PLATFORM": "slack",
        "HERMES_SESSION_CHAT_ID": "D0BRN0WUXK5",
    }
    monkeypatch.setattr(srt, "get_session_env", lambda k, d="": env.get(k, d))
    pending_reactions.clear_all()
    assert _ok(srt.set_reaction_tool("🔧"))
    assert pending_reactions.pop_pending(
        pending_reactions.make_key("D0BRN0WUXK5")
    ) == "🔧"


def test_refuses_the_progress_emoji(photon_session):
    """The regression that read as 'it did not react'.

    👀 is the in-progress reaction. Staged as the FINAL one it leaves the
    bubble looking untouched — indistinguishable from a turn that never
    finished. The model picked it three times running, so refuse it and let it
    choose again rather than staging something that cannot show.
    """
    raw = srt.set_reaction_tool("👀")
    assert "already means" in raw
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT)) is None


def test_refuses_the_progress_emoji_with_variation_selector(photon_session):
    srt.set_reaction_tool("👀\ufe0f")
    assert pending_reactions.pop_pending(pending_reactions.make_key(CHAT)) is None
