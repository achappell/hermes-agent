"""Behaviour tests for the context-aware completion reaction on Slack.

Slack already swaps :eyes: for :white_check_mark: when a turn finishes. If the
agent staged an emoji via ``set_reaction``, that shortcode is used instead.

The important property is that this can only match or improve on the previous
behaviour: no staged pick, or one with no shortcode mapping, must still produce
:white_check_mark: rather than nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway import pending_reactions
from gateway.platforms.base import ProcessingOutcome
from plugins.platforms.slack.adapter import SlackAdapter

CHANNEL = "D0BRN0WUXK5"
TS = "1787850000.000100"
TEAM = "T0BRS6ZKSP3"


def _event(channel: str = CHANNEL, ts: str = TS):
    return SimpleNamespace(
        source=SimpleNamespace(chat_id=channel, scope_id=TEAM), message_id=ts
    )


@pytest.fixture
def adapter():
    """A SlackAdapter with the reaction primitives recorded, not sent."""
    a = SlackAdapter.__new__(SlackAdapter)
    calls: list[tuple] = []

    async def fake_add(channel, timestamp, emoji, team_id=""):
        calls.append(("add", emoji))
        return True

    async def fake_remove(channel, timestamp, emoji, team_id=""):
        calls.append(("remove", emoji))
        return True

    a._add_reaction = fake_add
    a._remove_reaction = fake_remove
    a._reactions_enabled = lambda: True
    a._workspace_message_marker = lambda team_id, ts: f"{team_id}:{ts}"
    a._reacting_message_ids = {f"{TEAM}:{TS}"}
    a.calls = calls
    pending_reactions.clear_all()
    yield a
    pending_reactions.clear_all()


def _added(adapter):
    return [name for kind, name in adapter.calls if kind == "add"]


@pytest.mark.asyncio
async def test_uses_the_staged_emoji_as_a_shortcode(adapter):
    pending_reactions.set_pending(pending_reactions.make_key(CHANNEL), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _added(adapter) == ["wrench"]


@pytest.mark.asyncio
async def test_removes_the_eyes_first(adapter):
    pending_reactions.set_pending(pending_reactions.make_key(CHANNEL), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert adapter.calls[0] == ("remove", "eyes")


@pytest.mark.asyncio
async def test_no_pick_keeps_the_check_mark(adapter):
    """Unchanged behaviour when the agent stages nothing."""
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _added(adapter) == ["white_check_mark"]


@pytest.mark.asyncio
async def test_unmappable_emoji_falls_back_not_silent(adapter):
    """A gap in the table must degrade to the old default, never to nothing."""
    pending_reactions.set_pending(pending_reactions.make_key(CHANNEL), "🦑")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _added(adapter) == ["white_check_mark"]


@pytest.mark.asyncio
async def test_failure_ignores_the_staged_emoji(adapter):
    pending_reactions.set_pending(pending_reactions.make_key(CHANNEL), "🎉")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.FAILURE)
    assert _added(adapter) == ["x"]


@pytest.mark.asyncio
async def test_pick_is_consumed_not_reused(adapter):
    pending_reactions.set_pending(pending_reactions.make_key(CHANNEL), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    adapter.calls.clear()
    adapter._reacting_message_ids = {f"{TEAM}:{TS}"}
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _added(adapter) == ["white_check_mark"]


@pytest.mark.asyncio
async def test_pick_does_not_leak_from_another_channel(adapter):
    pending_reactions.set_pending(pending_reactions.make_key("C-OTHER"), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _added(adapter) == ["white_check_mark"]


@pytest.mark.asyncio
async def test_reactions_disabled_stays_silent(adapter):
    adapter._reactions_enabled = lambda: False
    pending_reactions.set_pending(pending_reactions.make_key(CHANNEL), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert adapter.calls == []
