"""OpenCode relay session-affinity headers.

OpenCode pins requests that share an ``x-opencode-session`` value to one
upstream backend.  Keeping that value stable across a Hermes conversation
keeps the provider's prompt cache warm without exposing prompt content.
"""

from __future__ import annotations

from typing import Any, Optional

OPENCODE_SESSION_HEADER = "x-opencode-session"


def is_opencode_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """Return whether a provider or URL addresses the OpenCode relay."""
    try:
        from hermes_cli.models import opencode_provider_family

        if opencode_provider_family(provider) is not None:
            return True
    except Exception:
        pass

    try:
        from agent.anthropic_endpoints import _is_opencode_endpoint

        return _is_opencode_endpoint(str(base_url or ""))
    except Exception:
        return False


def opencode_session_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Return the stable OpenCode affinity header, or an empty mapping."""
    if not is_opencode_target(provider, base_url):
        return {}

    try:
        from agent.portal_tags import get_conversation_context
        from agent.transports.codex import _cache_scope_from_session_id

        key = _cache_scope_from_session_id(
            get_conversation_context() or session_id
        )
    except Exception:
        key = str(session_id or "")
    return {OPENCODE_SESSION_HEADER: key} if key else {}


def merge_opencode_session_headers(
    kwargs: dict[str, Any],
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Merge the affinity header into request kwargs without overwriting one."""
    headers = opencode_session_headers(provider, base_url, session_id)
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs
