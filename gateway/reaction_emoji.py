"""Translate a staged unicode emoji into what each platform will accept.

LOCAL CARRY — not upstream. See ``tools/set_reaction_tool.py``.

The agent stages one unicode emoji (``set_reaction("🔧")``). No two platforms
take the same thing:

* Photon / iMessage — any unicode emoji. No translation needed.
* Slack — ``reactions.add`` takes a SHORTCODE (``name="wrench"``), not the
  character. An unknown name is rejected by the API.
* Telegram — a fixed Bot API allowlist of ~70 emoji; anything else fails and
  leaves the message with no reaction at all. Not wired up yet.

Every lookup that misses returns ``None`` so the caller can fall back to the
platform's existing default. On Slack that default is ``white_check_mark``,
i.e. exactly what shipped before this feature — a missing mapping degrades to
current behaviour rather than to nothing.

The table is deliberately hand-written and small: it covers the vocabulary the
persona actually reaches for, plus the obvious neighbours. There is no ``emoji``
package in this venv, and pulling one in to convert thirty strings would be a
poor trade.
"""
from __future__ import annotations

from typing import Optional

#: Variation selector-16. ❤️ is U+2764 + U+FE0F; the bare U+2764 is the same
#: reaction as far as anyone reading the thread is concerned, so normalise it
#: away before lookup instead of carrying two keys for every emoji.
_VS16 = "️"

_SLACK_SHORTCODES: dict[str, str] = {
    # The persona's named vocabulary
    "🔧": "wrench",
    "🎹": "musical_keyboard",
    "⏰": "alarm_clock",
    "🍽": "knife_fork_plate",
    "💥": "boom",
    "🎉": "tada",
    "☕": "coffee",
    "🙄": "face_with_rolling_eyes",
    # Work / debugging
    "🔍": "mag",
    "🐛": "bug",
    "🚀": "rocket",
    "📝": "memo",
    "🧹": "broom",
    "📦": "package",
    "🗑": "wastebasket",
    "⚡": "zap",
    "💡": "bulb",
    "🎯": "dart",
    "🧠": "brain",
    "🚨": "rotating_light",
    "🔥": "fire",
    "✨": "sparkles",
    # Reactions proper
    "👍": "+1",
    "👎": "-1",
    "✅": "white_check_mark",
    "❌": "x",
    "❤": "heart",
    "😂": "joy",
    "🤔": "thinking_face",
    "🙈": "see_no_evil",
    "👀": "eyes",
    "🤯": "exploding_head",
    "🥱": "yawning_face",
    "💤": "zzz",
    # Life
    "🎂": "birthday",
    "🌙": "crescent_moon",
    "🎸": "guitar",
    "🏆": "trophy",
    "🍕": "pizza",
    "🛒": "shopping_trolley",
    "🩺": "stethoscope",
    "📚": "books",
}


def to_slack_shortcode(emoji: str) -> Optional[str]:
    """Return the Slack reaction name for ``emoji``, or None if unmapped.

    None means "no opinion" — the caller keeps whatever it would have sent.
    """
    if not emoji:
        return None
    return _SLACK_SHORTCODES.get(emoji.replace(_VS16, ""))
