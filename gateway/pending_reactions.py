"""One-slot-per-session mailbox for the agent's chosen completion tapback.

LOCAL CARRY — not upstream. See ``tools/set_reaction_tool.py``.

The platform lifecycle hooks tapback a fixed ``_OK_EMOJI`` (👍) when a turn
finishes. That is a status light, not a reaction: it says "the machine
finished", never "this was funny" or "that config was a mess". This module is
the seam that lets the agent say the latter.

Flow, all inside one gateway process::

    agent turn      set_reaction("🔧")  ->  set_pending(session_key, "🔧")
    turn completes  adapter hook        ->  pop_pending(session_key) -> "🔧"
                                            (None -> fall back to 👍)

Keyed by ``chat_id``. Both sides derive that without coordinating: the tool
from ``HERMES_SESSION_CHAT_ID``, the adapter from ``event.source.chat_id``.
``pop_pending`` is one-shot, so a pick cannot leak into a later turn, and
separate chats cannot see each other's.

Bounded because a gateway runs for weeks: the dict is insertion-ordered and the
oldest entries are evicted first, matching the ``_sent_message_ids`` pattern
already used in the Photon adapter.

Deliberately dependency-free — the tool layer and the adapter layer both
import it, and neither should have to import the other.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

#: Plenty for concurrent conversations while staying trivially bounded.
_MAX_PENDING = 256

_pending: Dict[str, str] = {}
_lock = threading.Lock()


def make_key(chat_id: str) -> str:
    """Build the mailbox key for a chat, or "" when there is no chat.

    Keyed by CHAT, not by message. Photon sets ``message_id`` on the
    MessageEvent but never on the MessageSource, and the gateway populates
    ``HERMES_SESSION_MESSAGE_ID`` from ``context.source.message_id`` — so the
    tool side always saw an empty message id and could never build a key.

    Chat-scoped matches what this adapter already does natively:
    ``PhotonAdapter.add_reaction`` defaults to "the chat's most recent inbound
    message". A turn is answering that message, so the chat is a sufficient
    address. ``pop_pending`` stays one-shot, so a pick still cannot survive
    into a later turn.
    """
    return chat_id or ""


def set_pending(key: str, emoji: str) -> None:
    """Record the emoji to tapback onto the message identified by ``key``.

    An empty ``emoji`` retracts a previous pick, so the turn falls back to the
    adapter's default. A blank ``key`` is ignored rather than writing a shared
    slot — a surface with no message to react to (a bare CLI run) must not
    scribble on another conversation's.
    """
    if not key:
        return
    with _lock:
        _pending.pop(key, None)  # refresh insertion order
        if not emoji:
            return
        _pending[key] = emoji
        while len(_pending) > _MAX_PENDING:
            _pending.pop(next(iter(_pending)))


def pop_pending(key: str) -> Optional[str]:
    """Return and clear the pending emoji for ``key``, or None if unset.

    One-shot by design: a pick belongs to the turn that made it.
    """
    if not key:
        return None
    with _lock:
        return _pending.pop(key, None)


def clear_all() -> None:
    """Drop every pending pick (tests, and gateway shutdown)."""
    with _lock:
        _pending.clear()
