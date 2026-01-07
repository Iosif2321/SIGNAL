"""Shared plotting style."""

from __future__ import annotations

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def apply_style() -> None:
    """Apply shared matplotlib style settings."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )