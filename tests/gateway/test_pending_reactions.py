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


def test_make_key_needs_both_halves():
    """A half-known message is not a key — it would collide across chats."""
    assert pr.make_key("chat", "msg")
    assert pr.make_key("chat", "") == ""
    assert pr.make_key("", "msg") == ""


def test_make_key_is_unambiguous():
    """Chat and message ids are separated so they cannot run together.

    Photon chat ids look like ``any;-;+15551234`` — punctuation-heavy — so the
    separator must be something they cannot contain.
    """
    a = pr.make_key("any;-;+1555", "12")
    b = pr.make_key("any;-;+15551", "2")
    assert a != b


def test_key_binds_the_pick_to_one_message():
    """Two turns in the SAME chat must not inherit each other's emoji."""
    first = pr.make_key("chat-1", "msg-1")
    second = pr.make_key("chat-1", "msg-2")
    pr.set_pending(first, "🔧")
    assert pr.pop_pending(second) is None
    assert pr.pop_pending(first) == "🔧"


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
