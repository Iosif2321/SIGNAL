"""CUDA device checks."""

from __future__ import annotations

import logging

import torch


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA is required but not available."""


def require_cuda() -> torch.device:
    """Return CUDA device or raise if unavailable."""
    if not torch.cuda.is_available():
        raise CudaUnavailableError(
            "CUDA is required for training/evaluation, but torch.cuda.is_available() is False. "
            "Check with: python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\""
        )
    name = torch.cuda.get_device_name(0)
    logging.getLogger(__name__).info(
        "CUDA available: %s | device: %s | cuda_version: %s",
        torch.cuda.is_available(),
        name,
        torch.version.cuda,
    )
    return torch.device("cuda")


def get_device() -> torch.device:
    """Alias for require_cuda for clarity."""
    return require_cuda()
