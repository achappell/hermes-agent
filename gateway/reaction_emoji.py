"""Translate a staged unicode emoji into what each platform will accept.

LOCAL CARRY — not upstream. See ``tools/set_reaction_tool.py``.

The agent stages one unicode emoji (``set_reaction("🔧")``). No two platforms
take the same thing:

* Photon / iMessage — any unicode emoji. No translation needed.
* Slack — ``reactions.add`` takes a SHORTCODE (``name="wrench"``), not the
  character. An unknown name is rejected by the API.
* Telegram — a fixed Bot API allowlist of ~70 emoji; anything else fails and
  leaves the message with no reaction at all. Not wired up yet.

Slack resolution is three layers, because a hand-written table cannot keep up
with a model choosing freely from all of Unicode:

1. ``_SLACK_SHORTCODES`` — only for emoji where Slack's name DIFFERS from the
   Unicode name (🎉 is ``tada``, not ``party_popper``). Small on purpose.
2. Otherwise derive from ``unicodedata.name()``. Slack's names track the
   Unicode ones closely, so this covers the long tail for free: 🧪 →
   ``test_tube``, 🤦 → ``face_palm``, 🦑 → ``squid``.
3. The CALLER must treat a failed ``reactions.add`` as "wrong guess" and fall
   back to its default. Layer 2 is a heuristic; without that guard a bad name
   means NO reaction at all, which is worse than a generic one.

There is no ``emoji`` package in this venv, and ``unicodedata`` is stdlib.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

#: Variation selector-16. ❤️ is U+2764 + U+FE0F; the bare U+2764 is the same
#: reaction as far as anyone reading the thread is concerned, so normalise it
#: away before lookup instead of carrying two keys for every emoji.
_VS16 = "️"

#: ONLY emoji whose Slack name differs from the derived Unicode name. Anything
#: that derives correctly must NOT be listed here — a redundant entry is one
#: more thing to get wrong.
_SLACK_SHORTCODES: dict[str, str] = {
    "🎉": "tada",              # party_popper
    "💥": "boom",              # collision_symbol
    "☕": "coffee",            # hot_beverage
    "🍽": "knife_fork_plate",  # fork_and_knife_with_plate
    "✅": "white_check_mark",  # white_heavy_check_mark
    "❌": "x",                 # cross_mark
    "👍": "+1",                # thumbs_up
    "👎": "-1",                # thumbs_down
    "❤": "heart",             # heavy_black_heart
    "😂": "joy",               # face_with_tears_of_joy
    "🔍": "mag",               # left_pointing_magnifying_glass
    "⚡": "zap",               # high_voltage_sign
    "💡": "bulb",              # electric_light_bulb
    "🎯": "dart",              # direct_hit
    "🚨": "rotating_light",    # police_cars_revolving_light
    "🤯": "exploding_head",    # shocked_face_with_exploding_head
    "💤": "zzz",               # sleeping_symbol
}



def _derive_shortcode(emoji: str) -> Optional[str]:
    """Slack-style name from the Unicode name, or None if there isn't one."""
    if not emoji or emoji[0].isascii():
        # ASCII has Unicode names too — " " is SPACE, "a" is LATIN SMALL
        # LETTER A. Deriving a reaction name from those would send nonsense to
        # Slack. Every emoji is non-ASCII, so this costs nothing real.
        return None
    try:
        name = unicodedata.name(emoji[0])
    except (ValueError, IndexError):
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or None


def to_slack_shortcode(emoji: str) -> Optional[str]:
    """Return a best-effort Slack reaction name for ``emoji``, or None.

    None means "no idea" — the caller keeps its default. A NON-None result is
    still only a best guess for anything outside the override table, so the
    caller must also handle ``reactions.add`` rejecting it.
    """
    if not emoji:
        return None
    bare = emoji.replace(_VS16, "")
    if not bare:
        return None
    return _SLACK_SHORTCODES.get(bare) or _derive_shortcode(bare)
