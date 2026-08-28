"""Tests for the opt-in Hermes-local TTS timing controls."""

import json
from pathlib import Path

from tools import tts_tool


def test_audio_tags_are_disabled_by_default_and_explicitly_opt_in():
    assert tts_tool._tts_audio_tags_enabled({}) is False
    assert tts_tool._tts_audio_tags_enabled({"audio_tags": False}) is False
    assert tts_tool._tts_audio_tags_enabled({"audio_tags": "off"}) is False
    assert tts_tool._tts_audio_tags_enabled({"audio_tags": True}) is True
    assert tts_tool._tts_audio_tags_enabled(
        {"audio_tags": {"enabled": True}}
    ) is True


def test_disabled_audio_tags_pass_through_to_provider(tmp_path, monkeypatch):
    raw = "Hello [pause:300ms] world."
    captured = []

    def fake_single(*, text, output_path, **_kwargs):
        captured.append(text)
        Path(output_path).write_bytes(b"audio")
        return json.dumps({
            "success": True,
            "file_path": output_path,
            "provider": "edge",
            "voice_compatible": False,
        })

    def fake_build(paths, output_path, _profile, **_kwargs):
        source = Path(paths[0])
        destination = Path(output_path)
        if source != destination:
            destination.write_bytes(source.read_bytes())
        return [str(destination)], False

    monkeypatch.setattr(
        tts_tool, "_load_tts_config",
        lambda: {"provider": "edge", "audio_tags": False},
    )
    monkeypatch.setattr(tts_tool, "_text_to_speech_single", fake_single)
    monkeypatch.setattr(tts_tool, "_build_audio_delivery_files", fake_build)

    result = json.loads(
        tts_tool.text_to_speech_tool(
            text=raw,
            output_path=str(tmp_path / "out.mp3"),
            provider="edge",
        )
    )

    assert result["success"] is True
    assert captured == [raw]
