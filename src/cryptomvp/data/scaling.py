"""Feature scaling utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_standard_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean/std for standardization."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standard_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply standard scaling using provided mean/std."""
    return (X - mean) / std
