"""Let the agent choose the tapback that lands when its turn finishes.

LOCAL CARRY — not upstream. Re-apply after every ``hermes update``.

Photon already tapbacks every message: 👀 while working, 👍 when done. That
second one is a status light — it says "the machine finished", never "this was
funny" or "that config was a disgrace". This tool replaces it, per turn, with
something the agent actually chose.

Deliberately NOT ``send_message(action="react")``. That tool is withheld from
the model on purpose (see the note at the foot of ``tools/send_message_tool.py``)
because it can message any target on any platform. This one takes no chat id at
all: it writes a single emoji into the current session's slot, and the adapter
that was already going to tapback reads it. There is no reachable target other
than the message the agent is replying to, so it cannot become a messaging
surface however the model is prompted.
"""
from __future__ import annotations

import json

from gateway import pending_reactions
from gateway.session_context import get_session_env
from tools.registry import registry, tool_error
from utils import env_var_enabled

#: Emoji are graphemes, not codepoints: ❤️ carries a variation selector and
#: 👩‍👩‍👧‍👧 is four people joined by U+200D. Counting codepoints would reject
#: perfectly ordinary emoji, so cap the cluster generously and reject only what
#: is obviously not a single reaction — whitespace or plain text.
_MAX_CODEPOINTS = 12


def _looks_like_one_emoji(value: str) -> bool:
    if len(value) > _MAX_CODEPOINTS:
        return False
    if any(ch.isspace() for ch in value):
        return False
    # Any ASCII alphanumeric means we were handed a word, not a reaction.
    return not any(ch.isascii() and ch.isalnum() for ch in value)


#: Platforms whose completion hook actually consumes a staged emoji. Photon and
#: CLI resolve to the same toolset list, so the tool is advertised more widely
#: than it works — say so plainly here rather than accepting a silent no-op on
#: Telegram, where a pick would be staged and then never read.
_CONSUMING_PLATFORMS = frozenset({"photon"})


def set_reaction_tool(emoji: str) -> str:
    """Stage ``emoji`` as this turn's completion tapback."""
    emoji = (emoji or "").strip()

    platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or "").lower()
    if platform not in _CONSUMING_PLATFORMS:
        return tool_error(
            f"Reactions aren't wired up on {platform or 'this surface'} — "
            "this only does anything on iMessage (Photon)."
        )

    key = pending_reactions.make_key(
        get_session_env("HERMES_SESSION_CHAT_ID", ""),
        get_session_env("HERMES_SESSION_MESSAGE_ID", ""),
    )
    if not key:
        return tool_error(
            "No message to react to on this surface."
        )

    if emoji and not _looks_like_one_emoji(emoji):
        return tool_error(
            "Pass a single emoji (e.g. '🔧'), or an empty string to fall back "
            "to the default tapback."
        )

    pending_reactions.set_pending(key, emoji)
    return json.dumps({"success": True, "emoji": emoji or None}, ensure_ascii=False)


def check_set_reaction_requirements() -> bool:
    """Only advertise the tool where the tapback will actually land.

    The completion hook this rides on is gated by ``PHOTON_REACTIONS``; with it
    off the adapter never tapbacks at all, so a staged emoji would silently go
    nowhere. Don't offer the model a tool that cannot do anything.
    """
    return env_var_enabled("PHOTON_REACTIONS")


SET_REACTION_SCHEMA = {
    "name": "set_reaction",
    "description": (
        "Choose the tapback that lands on the user's message when you finish. "
        "You are ALREADY reacting to every message — the only question is "
        "whether it's a generic 👍 or something that shows you were paying "
        "attention. So call this on essentially every turn, and be opinionated: "
        "🔧 for a config you fixed, 🎹 for practice notes, ⏰ for a reminder, "
        "🍽️ for anything about food, 💥 when something broke, 🎉 when it "
        "finally worked, ☕ at 1am, 🙄 for the third time the same API failed. "
        "Match the content, not the outcome — 'it succeeded' is what 👍 already "
        "said, and it said it badly. NEVER mention that you reacted or explain "
        "the choice: the emoji appearing on the bubble is the entire point, and "
        "narrating it kills the joke. Pass an empty string to fall back to the "
        "default. Failed turns always get 👎 regardless."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": (
                    "A single emoji, e.g. '🔧'. Empty string falls back to the "
                    "default tapback."
                ),
            },
        },
        "required": ["emoji"],
    },
}


registry.register(
    name="set_reaction",
    toolset="clarify",
    schema=SET_REACTION_SCHEMA,
    handler=lambda args, **kw: set_reaction_tool(emoji=args.get("emoji", "")),
    check_fn=check_set_reaction_requirements,
    emoji="💛",
)
