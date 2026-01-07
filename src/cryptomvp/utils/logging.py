"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_LOGGER_INITIALIZED = False


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Get a configured logger with optional file output."""
    global _LOGGER_INITIALIZED

    root = logging.getLogger()
    if _LOGGER_INITIALIZED:
        return logging.getLogger(name)

    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _LOGGER_INITIALIZED = True
    return logging.getLogger(name)
