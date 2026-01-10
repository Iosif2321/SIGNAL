"""Run directory initialization and metadata logging."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import json
import os
import shutil
import subprocess
import sys

import importlib.metadata as metadata

from cryptomvp.utils.io import ensure_dir, project_root


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _safe_version(pkg: str) -> Optional[str]:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return None


def _gpu_metadata() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda_available": False, "cuda_version": None}
    try:
        import torch
    except Exception:
        return info
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def _package_versions() -> Dict[str, Optional[str]]:
    packages = [
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "httpx",
        "websockets",
        "pyarrow",
    ]
    return {pkg: _safe_version(pkg) for pkg in packages}


def init_run_dir(run_dir: Optional[str | Path], config_path: Optional[str | Path]) -> Path:
    """Initialize a run directory and write metadata/config copies."""
    if run_dir is None:
        env_root = os.environ.get("CRYPTOMVP_RUN_DIR")
        if env_root:
            run_dir = env_root
        else:
            run_dir = project_root() / "runs" / _timestamp()
    run_path = Path(run_dir)

    ensure_dir(run_path)
    ensure_dir(run_path / "data" / "raw")
    ensure_dir(run_path / "data" / "processed")
    ensure_dir(run_path / "reports")
    ensure_dir(run_path / "checkpoints")
    ensure_dir(run_path / "sessions")

    os.environ["CRYPTOMVP_RUN_DIR"] = str(run_path)

    if config_path:
        cfg_path = Path(config_path)
        if cfg_path.exists() and not (run_path / "config.yaml").exists():
            shutil.copy2(cfg_path, run_path / "config.yaml")

    metadata_path = run_path / "metadata.json"
    if not metadata_path.exists():
        metadata_payload: Dict[str, Any] = {
            "run_id": run_path.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path) if config_path else None,
            "git_commit": _git_commit(),
            "python_version": sys.version.split()[0],
            "packages": _package_versions(),
            "gpu": _gpu_metadata(),
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return run_path
