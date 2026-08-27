"""Behaviour tests for the staged-emoji → platform translation table.

Slack's ``reactions.add`` takes a shortcode name, not the emoji character, so a
staged 🔧 has to become ``"wrench"`` before it can land. Anything unmapped
returns None, which the caller reads as "keep your existing default" — on Slack
that default is ``white_check_mark``, so a gap degrades to the behaviour that
shipped before this feature rather than to no reaction at all.
"""
from __future__ import annotations

import pytest

from gateway.reaction_emoji import _SLACK_SHORTCODES, to_slack_shortcode


def test_maps_the_personas_vocabulary():
    """Every emoji the persona names by hand must resolve."""
    for emoji in ["🔧", "🎹", "⏰", "💥", "🎉", "☕", "🙄"]:
        assert to_slack_shortcode(emoji), f"{emoji} has no Slack mapping"


def test_maps_a_known_emoji():
    assert to_slack_shortcode("🔧") == "wrench"


def test_unmapped_emoji_returns_none():
    """None is 'no opinion' — the caller keeps its default."""
    assert to_slack_shortcode("🦑") is None


def test_empty_returns_none():
    assert to_slack_shortcode("") is None


def test_variation_selector_is_normalised():
    """❤️ is U+2764 + U+FE0F. Both spellings are the same reaction."""
    assert to_slack_shortcode("❤️") == "heart"
    assert to_slack_shortcode("❤") == "heart"


def test_plate_with_cutlery_both_spellings():
    """🍽️ is commonly typed with and without the variation selector."""
    assert to_slack_shortcode("🍽️") == "knife_fork_plate"
    assert to_slack_shortcode("🍽") == "knife_fork_plate"


def test_no_key_carries_a_variation_selector():
    """Keys must be pre-normalised or the lookup silently misses."""
    for key in _SLACK_SHORTCODES:
        assert "️" not in key, f"{key!r} still has U+FE0F"


def test_shortcodes_look_like_slack_names():
    """Slack names are lowercase alphanumerics, underscores, + and -."""
    import re

    for emoji, name in _SLACK_SHORTCODES.items():
        assert re.fullmatch(r"[a-z0-9_+-]+", name), f"{emoji} -> {name!r}"


def test_defaults_round_trip():
    """The emoji Slack already uses must map to the names it already sends."""
    assert to_slack_shortcode("✅") == "white_check_mark"
    assert to_slack_shortcode("❌") == "x"
    assert to_slack_shortcode("👀") == "eyes"
