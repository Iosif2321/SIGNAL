"""CUDA device checks."""

from __future__ import annotations

import torch


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA is required but not available."""


def require_cuda() -> torch.device:
    """Return CUDA device or raise if unavailable."""
    if not torch.cuda.is_available():
        raise CudaUnavailableError(
            "CUDA is required for training/evaluation, but torch.cuda.is_available() is False."
        )
    return torch.device("cuda")


def get_device() -> torch.device:
    """Alias for require_cuda for clarity."""
    return require_cuda()