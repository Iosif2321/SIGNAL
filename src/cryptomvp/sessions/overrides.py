"""Session-specific config overrides."""

from __future__ import annotations

from typing import Any, Dict

from cryptomvp.config import SessionConfig


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def apply_session_overrides(
    base_cfg: Dict[str, Any],
    session_cfg: SessionConfig | None,
    session_id: str,
) -> Dict[str, Any]:
    """Return config dict with session-specific overrides applied."""
    if session_cfg is None:
        return dict(base_cfg)
    session = session_cfg.sessions.get(session_id)
    if session is None:
        return dict(base_cfg)
    if not session.overrides:
        return dict(base_cfg)
    return _deep_merge(dict(base_cfg), session.overrides)
