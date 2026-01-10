"""Plotting helpers with moving mean band."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cryptomvp.viz.style import apply_style

import matplotlib.pyplot as plt


def _ensure_array(y: Sequence[float]) -> np.ndarray:
    return np.asarray(y, dtype=float)


def _rolling_mean_std(y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    series = pd.Series(y)
    mean = series.rolling(window=window, min_periods=1).mean().to_numpy()
    std = series.rolling(window=window, min_periods=1).std().fillna(0.0).to_numpy()
    return mean, std


def _mean_std_across_runs(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    return mean, std


def save_figure(fig: plt.Figure, out_base: Path, formats: Iterable[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(str(out_base.with_suffix(f".{fmt}")), bbox_inches="tight")


def plot_series_with_band(
    x: Sequence[float],
    y: Sequence[float],
    window: int,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    out_base: Path,
    formats: Iterable[str],
) -> None:
    apply_style()
    x_arr = _ensure_array(x)
    y_arr = _ensure_array(y)
    mean, std = _rolling_mean_std(y_arr, window)
    fig, ax = plt.subplots()
    ax.plot(x_arr, y_arr, label=label, alpha=0.6)
    ax.plot(x_arr, mean, label=f"{label} mean")
    ax.fill_between(x_arr, mean - std, mean + std, alpha=0.2, label="mean +/- std")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_series_with_mean_band(
    x: Sequence[float],
    y: Sequence[float],
    window: int,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    out_base: Path,
    formats: Iterable[str],
) -> None:
    """Alias for plot_series_with_band to keep naming consistent."""
    plot_series_with_band(
        x,
        y,
        window=window,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        label=label,
        out_base=out_base,
        formats=formats,
    )


def plot_runs_with_band(
    x: Sequence[float],
    y_runs: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_base: Path,
    formats: Iterable[str],
    labels: Optional[List[str]] = None,
) -> None:
    apply_style()
    x_arr = _ensure_array(x)
    fig, ax = plt.subplots()
    if y_runs.ndim == 1:
        mean, std = _rolling_mean_std(y_runs, window=min(20, len(y_runs)))
        ax.plot(x_arr, y_runs, label="run", alpha=0.6)
        ax.plot(x_arr, mean, label="mean")
        ax.fill_between(x_arr, mean - std, mean + std, alpha=0.2, label="mean +/- std")
    else:
        mean, std = _mean_std_across_runs(y_runs)
        for idx, run in enumerate(y_runs):
            label = labels[idx] if labels and idx < len(labels) else f"run_{idx}"
            ax.plot(x_arr, run, label=label, alpha=0.4)
        ax.plot(x_arr, mean, label="mean")
        ax.fill_between(x_arr, mean - std, mean + std, alpha=0.2, label="mean +/- std")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_histogram(
    y: Sequence[float],
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    out_base: Path,
    formats: Iterable[str],
) -> None:
    apply_style()
    y_arr = _ensure_array(y)
    fig, ax = plt.subplots()
    ax.hist(y_arr, bins=bins, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_bar(
    categories: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_base: Path,
    formats: Iterable[str],
) -> None:
    apply_style()
    fig, ax = plt.subplots()
    ax.bar(categories, values, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    title: str,
    out_base: Path,
    formats: Iterable[str],
) -> None:
    apply_style()
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_threshold_scan(
    thresholds: Sequence[float],
    hold_rates: Sequence[float],
    window: int,
    title: str,
    out_base: Path,
    formats: Iterable[str],
    ylabel: str = "Hold rate",
    label: str = "hold_rate",
) -> None:
    plot_series_with_band(
        thresholds,
        hold_rates,
        window=window,
        title=title,
        xlabel="Threshold",
        ylabel=ylabel,
        label=label,
        out_base=out_base,
        formats=formats,
    )


def plot_heatmap(
    values: np.ndarray,
    x_labels: Sequence[float],
    y_labels: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_base: Path,
    formats: Iterable[str],
    fmt: str = ".3f",
) -> None:
    apply_style()
    fig, ax = plt.subplots()
    im = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels([f"{val:.3f}" for val in x_labels])
    ax.set_yticklabels([f"{val:.3f}" for val in y_labels])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    save_figure(fig, out_base, formats)
    plt.close(fig)


def plot_reliability_curve(
    mean_prob: Sequence[float],
    accuracy: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_base: Path,
    formats: Iterable[str],
    label: str = "reliability",
) -> None:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(mean_prob, accuracy, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save_figure(fig, out_base, formats)
    plt.close(fig)
