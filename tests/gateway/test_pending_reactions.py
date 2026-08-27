"""Behaviour tests for the agent-chosen completion tapback mailbox.

``gateway/pending_reactions.py`` is a one-slot-per-session mailbox: the agent
drops an emoji in during its turn (via the ``set_reaction`` tool) and the
platform adapter's ``on_processing_complete`` hook pops it when the turn ends,
using it INSTEAD of the static ``_OK_EMOJI``.

Keyed by session so two concurrent chats cannot read each other's pick, and
bounded so a long-lived gateway cannot grow the dict without limit.
"""
from __future__ import annotations

import pytest

from gateway import pending_reactions as pr


@pytest.fixture(autouse=True)
def _clear():
    pr.clear_all()
    yield
    pr.clear_all()


def test_pop_returns_what_was_set():
    pr.set_pending("sess-a", "🔧")
    assert pr.pop_pending("sess-a") == "🔧"


def test_pop_is_one_shot():
    """A pick applies to one turn only — a later turn must not reuse it."""
    pr.set_pending("sess-a", "🔧")
    assert pr.pop_pending("sess-a") == "🔧"
    assert pr.pop_pending("sess-a") is None


def test_pop_unknown_session_is_none():
    assert pr.pop_pending("never-set") is None


def test_sessions_are_isolated():
    """Two chats in flight at once must not see each other's emoji."""
    pr.set_pending("sess-a", "🎹")
    pr.set_pending("sess-b", "☕")
    assert pr.pop_pending("sess-b") == "☕"
    assert pr.pop_pending("sess-a") == "🎹"


def test_latest_set_wins():
    """The agent may change its mind mid-turn; last call before the hook wins."""
    pr.set_pending("sess-a", "🤔")
    pr.set_pending("sess-a", "🎉")
    assert pr.pop_pending("sess-a") == "🎉"


def test_empty_emoji_clears_the_slot():
    """Passing an empty string retracts a pick, falling back to the default."""
    pr.set_pending("sess-a", "🎉")
    pr.set_pending("sess-a", "")
    assert pr.pop_pending("sess-a") is None


def test_blank_key_is_ignored():
    """No message to react to (e.g. a CLI run) must never write a global slot."""
    pr.set_pending("", "🔧")
    assert pr.pop_pending("") is None


def test_make_key_needs_a_chat():
    """No chat means no address — must not produce a usable global key."""
    assert pr.make_key("any;-;+15551234")
    assert pr.make_key("") == ""


def test_make_key_distinguishes_chats():
    """Photon chat ids are punctuation-heavy; they must not collide."""
    assert pr.make_key("any;-;+1555") != pr.make_key("any;-;+15551")


def test_mailbox_is_bounded():
    """A long-lived gateway must not grow this dict without limit."""
    for i in range(pr._MAX_PENDING + 50):
        pr.set_pending(f"sess-{i}", "🔧")
    assert len(pr._pending) <= pr._MAX_PENDING


def test_eviction_drops_oldest_first():
    """Under pressure the newest picks are the ones worth keeping."""
    pr.set_pending("oldest", "🥇")
    for i in range(pr._MAX_PENDING):
        pr.set_pending(f"filler-{i}", "🔧")
    assert pr.pop_pending("oldest") is None
    assert pr.pop_pending(f"filler-{pr._MAX_PENDING - 1}") == "🔧"
