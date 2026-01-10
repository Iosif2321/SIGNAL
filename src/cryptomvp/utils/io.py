"""Filesystem path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import os


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_root() -> Path:
    env_root = os.environ.get("CRYPTOMVP_RUN_DIR")
    if env_root:
        return Path(env_root)
    return project_root()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(*parts: str) -> Path:
    return ensure_dir(run_root() / "data" / Path(*parts))


def reports_dir(test_name: Optional[str] = None) -> Path:
    base = ensure_dir(run_root() / "reports")
    if test_name:
        return ensure_dir(base / test_name)
    return base


def checkpoints_dir() -> Path:
    return ensure_dir(run_root() / "checkpoints")


def sessions_dir(session_id: Optional[str] = None) -> Path:
    base = ensure_dir(run_root() / "sessions")
    if session_id:
        return ensure_dir(base / session_id)
    return base
