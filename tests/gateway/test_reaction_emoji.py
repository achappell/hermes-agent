"""Behaviour tests for the staged-emoji → Slack shortcode resolution.

Three layers: an override table for names Slack spells differently, a
unicodedata-derived name for the long tail, and (in the caller) a fallback when
Slack rejects the guess. These cover the first two.
"""
from __future__ import annotations

import pytest

from gateway.reaction_emoji import _SLACK_SHORTCODES, to_slack_shortcode


def test_override_wins_where_slack_differs():
    """🎉 derives as party_popper; Slack calls it tada."""
    assert to_slack_shortcode("🎉") == "tada"
    assert to_slack_shortcode("💥") == "boom"
    assert to_slack_shortcode("☕") == "coffee"


def test_derives_the_long_tail():
    """The regression this rewrite exists for.

    A hand table cannot keep up with a model choosing freely. 🧪 and 🤦 were
    both picked in real turns and both fell through to the generic check.
    """
    assert to_slack_shortcode("🧪") == "test_tube"
    assert to_slack_shortcode("🤦") == "face_palm"
    assert to_slack_shortcode("🦑") == "squid"


def test_maps_the_personas_vocabulary():
    for emoji in ["🔧", "🎹", "⏰", "💥", "🎉", "☕", "🙄", "🍽️"]:
        assert to_slack_shortcode(emoji), f"{emoji} resolves to nothing"


def test_variation_selector_is_normalised():
    assert to_slack_shortcode("❤️") == "heart"
    assert to_slack_shortcode("❤") == "heart"


def test_plate_uses_the_slack_spelling():
    assert to_slack_shortcode("🍽️") == "knife_fork_plate"


def test_empty_returns_none():
    assert to_slack_shortcode("") is None
    assert to_slack_shortcode("\ufe0f") is None


def test_non_emoji_returns_none():
    """A space has no Unicode-derived reaction name worth sending."""
    assert to_slack_shortcode(" ") is None


def test_no_override_key_carries_a_variation_selector():
    for key in _SLACK_SHORTCODES:
        assert "\ufe0f" not in key, f"{key!r} still has U+FE0F"


def test_overrides_are_not_redundant():
    """An override that matches the derived name is dead weight — and a lie.

    The table's contract is 'only where Slack differs'. A redundant entry means
    the next person cannot tell which entries are load-bearing.
    """
    from gateway.reaction_emoji import _derive_shortcode

    for emoji, name in _SLACK_SHORTCODES.items():
        assert _derive_shortcode(emoji) != name, (
            f"{emoji} -> {name!r} derives identically; remove the override"
        )


def test_results_look_like_slack_names():
    import re

    for emoji in ["🔧", "🧪", "🤦", "🎉", "❤️", "🍽️", "👍"]:
        name = to_slack_shortcode(emoji)
        assert re.fullmatch(r"[a-z0-9_+-]+", name), f"{emoji} -> {name!r}"
