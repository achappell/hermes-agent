"""Behaviour tests for the context-aware completion tapback on Photon.

Photon tapbacks every message: 👀 while the agent works, 👍 when it finishes.
That second one is a status light. ``set_reaction`` lets the agent stage a
better emoji during its turn; ``PhotonAdapter.on_processing_complete`` uses it
instead — but only on success, only when reactions are enabled, and only on the
message the pick was made for.

These call the real ``on_processing_complete`` with the reaction primitives
stubbed, so they assert the decision, not the sidecar.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway import pending_reactions
from gateway.platforms.base import ProcessingOutcome
from plugins.platforms.photon.adapter import PhotonAdapter

CHAT = "any;-;+15551234"
MSG = "msg-1"


def _event(chat_id: str = CHAT, message_id: str = MSG):
    return SimpleNamespace(
        source=SimpleNamespace(chat_id=chat_id), message_id=message_id
    )


@pytest.fixture
def adapter(monkeypatch):
    """A PhotonAdapter with the reaction primitives recorded, not sent."""
    a = PhotonAdapter.__new__(PhotonAdapter)
    calls: list[tuple] = []

    async def fake_add(chat_id, message_id, emoji):
        calls.append(("add", chat_id, message_id, emoji))
        return True

    async def fake_remove(chat_id, message_id):
        calls.append(("remove", chat_id, message_id))
        return True

    a._add_reaction = fake_add
    a._remove_reaction = fake_remove
    a._reactions_enabled = lambda: True
    a.calls = calls
    pending_reactions.clear_all()
    yield a
    pending_reactions.clear_all()


def _added(adapter):
    return [c[3] for c in adapter.calls if c[0] == "add"]


@pytest.mark.asyncio
async def test_uses_the_agents_emoji_on_success(adapter):
    pending_reactions.set_pending(pending_reactions.make_key(CHAT), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _added(adapter) == ["🔧"]


@pytest.mark.asyncio
async def test_removes_the_watching_eyes_first(adapter):
    """The 👀 must come off before the new tapback goes on."""
    pending_reactions.set_pending(pending_reactions.make_key(CHAT), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert [c[0] for c in adapter.calls] == ["remove", "add"]


@pytest.mark.asyncio
async def test_falls_back_to_thumbs_up_when_agent_did_not_choose(adapter):
    """Forgetting to call set_reaction must not regress today's behaviour."""
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _added(adapter) == [PhotonAdapter._OK_EMOJI]


@pytest.mark.asyncio
async def test_failure_ignores_the_agents_emoji(adapter):
    """A cheerful emoji on a crashed turn would be a lie — 👎 wins."""
    pending_reactions.set_pending(pending_reactions.make_key(CHAT), "🎉")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.FAILURE)
    assert _added(adapter) == [PhotonAdapter._FAIL_EMOJI]


@pytest.mark.asyncio
async def test_pick_is_consumed_not_reused(adapter):
    """The next turn in the same chat must start from the default again."""
    pending_reactions.set_pending(pending_reactions.make_key(CHAT), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    adapter.calls.clear()
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _added(adapter) == [PhotonAdapter._OK_EMOJI]


@pytest.mark.asyncio
async def test_pick_does_not_leak_to_another_chat(adapter):
    """A pick staged in one chat must never land in a different one."""
    pending_reactions.set_pending(pending_reactions.make_key("other-chat"), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert _added(adapter) == [PhotonAdapter._OK_EMOJI]


@pytest.mark.asyncio
async def test_reactions_disabled_stays_silent(adapter):
    """With PHOTON_REACTIONS off the adapter must not tapback at all."""
    adapter._reactions_enabled = lambda: False
    pending_reactions.set_pending(pending_reactions.make_key(CHAT), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_cancelled_leaves_the_message_unreacted(adapter):
    pending_reactions.set_pending(pending_reactions.make_key(CHAT), "🔧")
    await adapter.on_processing_complete(_event(), ProcessingOutcome.CANCELLED)
    assert _added(adapter) == []
