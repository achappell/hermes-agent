from __future__ import annotations

from tools import skills_tool, terminal_tool


def test_sudo_password_context_callback_is_scoped():
    callback = lambda: "masked-value"

    terminal_tool.set_sudo_password_callback(lambda: "stale-value")
    token = terminal_tool.set_sudo_password_context_callback(callback)
    try:
        assert terminal_tool._get_sudo_password_callback() is callback
    finally:
        terminal_tool.reset_sudo_password_context_callback(token)
        terminal_tool.set_sudo_password_callback(None)

    assert terminal_tool._get_sudo_password_callback() is None


def test_secret_capture_context_callback_is_scoped():
    callback = lambda *_args: {"success": True}

    skills_tool.set_secret_capture_callback(lambda *_args: {"success": False})
    token = skills_tool.set_secret_capture_context_callback(callback)
    try:
        assert skills_tool._get_secret_capture_callback() is callback
    finally:
        skills_tool.reset_secret_capture_context_callback(token)
        skills_tool.set_secret_capture_callback(None)

    assert skills_tool._get_secret_capture_callback() is None
