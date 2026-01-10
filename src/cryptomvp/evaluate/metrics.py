"""Metrics and reporting helpers for directional decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


@dataclass
class CoverageBin:
    bin_start: float
    bin_end: float
    count: int
    action_rate: float
    accuracy_non_hold: float


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, label: str
) -> Tuple[float, float, float]:
    pred_pos = y_pred == label
    true_pos = y_true == label
    tp = int(np.sum(pred_pos & true_pos))
    fp = int(np.sum(pred_pos & ~true_pos))
    fn = int(np.sum(~pred_pos & true_pos))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _confusion_matrix_up_down(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    mask = np.isin(y_true, ["UP", "DOWN"]) & np.isin(y_pred, ["UP", "DOWN"])
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    labels = ["UP", "DOWN"]
    matrix = np.zeros((2, 2), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = int(np.sum((y_true_f == true_label) & (y_pred_f == pred_label)))
    return matrix


def _balanced_accuracy(recall_up: float, recall_down: float) -> float:
    return 0.5 * (recall_up + recall_down)


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isin(y_true, ["UP", "DOWN"]) & np.isin(y_pred, ["UP", "DOWN"])
    if mask.sum() < 2:
        return 0.0
    y_true_bin = np.where(y_true[mask] == "UP", 1, 0)
    y_pred_bin = np.where(y_pred[mask] == "UP", 1, 0)
    return float(matthews_corrcoef(y_true_bin, y_pred_bin))


def _brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prob - target) ** 2))


def _ece(prob: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(prob)
    for i in range(bins):
        mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if not mask.any():
            continue
        bin_prob = np.mean(prob[mask])
        bin_acc = np.mean(target[mask])
        ece += (mask.sum() / n) * abs(bin_prob - bin_acc)
    return float(ece)


def _coverage_bins(
    confidence: np.ndarray,
    decisions: np.ndarray,
    y_true: np.ndarray,
    bins: int = 10,
) -> List[CoverageBin]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bins_out: List[CoverageBin] = []
    for i in range(bins):
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        if not mask.any():
            bins_out.append(CoverageBin(edges[i], edges[i + 1], 0, 0.0, 0.0))
            continue
        decisions_bin = decisions[mask]
        action_mask = decisions_bin != "HOLD"
        action_rate = float(np.mean(action_mask))
        if action_mask.any():
            acc = float(np.mean(y_true[mask][action_mask] == decisions_bin[action_mask]))
        else:
            acc = 0.0
        bins_out.append(
            CoverageBin(edges[i], edges[i + 1], int(mask.sum()), action_rate, acc)
        )
    return bins_out


def _reliability_bins(
    prob: np.ndarray, target: np.ndarray, bins: int = 10
) -> List[Dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: List[Dict[str, float]] = []
    for i in range(bins):
        mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if not mask.any():
            out.append(
                {
                    "bin_start": float(edges[i]),
                    "bin_end": float(edges[i + 1]),
                    "count": 0,
                    "mean_prob": 0.0,
                    "accuracy": 0.0,
                }
            )
            continue
        out.append(
            {
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "count": int(mask.sum()),
                "mean_prob": float(np.mean(prob[mask])),
                "accuracy": float(np.mean(target[mask])),
            }
        )
    return out


def compute_metrics(
    df: pd.DataFrame,
    bootstrap_samples: int = 0,
    seed: int = 7,
) -> Dict[str, object]:
    """Compute global metrics from a decision log."""
    y_true = df["y_true"].astype(str).to_numpy()
    decisions = df["decision"].astype(str).to_numpy()
    hold_mask = decisions == "HOLD"
    action_mask = ~hold_mask
    action_rate = float(np.mean(action_mask))
    hold_rate = float(np.mean(hold_mask))
    conflict_rate = float(df["conflict"].mean()) if "conflict" in df.columns else 0.0

    accuracy_non_hold = (
        float(np.mean(y_true[action_mask] == decisions[action_mask])) if action_mask.any() else 0.0
    )
    precision_up, recall_up, f1_up = _precision_recall_f1(y_true, decisions, "UP")
    precision_down, recall_down, f1_down = _precision_recall_f1(y_true, decisions, "DOWN")
    balanced_acc = _balanced_accuracy(recall_up, recall_down)
    mcc = _mcc(y_true, decisions)
    cm = _confusion_matrix_up_down(y_true, decisions)

    confidence = None
    if "p_up" in df.columns and "p_down" in df.columns:
        p_up = df["p_up"].to_numpy(dtype=float)
        p_down = df["p_down"].to_numpy(dtype=float)
        confidence = np.maximum(p_up, p_down)
        y_up = (y_true == "UP").astype(float)
        y_down = (y_true == "DOWN").astype(float)
        brier_up = _brier_score(p_up, y_up)
        brier_down = _brier_score(p_down, y_down)
        ece_up = _ece(p_up, y_up)
        ece_down = _ece(p_down, y_down)
        reliability_up = _reliability_bins(p_up, y_up)
        reliability_down = _reliability_bins(p_down, y_down)
    else:
        brier_up = brier_down = ece_up = ece_down = 0.0
        reliability_up = []
        reliability_down = []

    coverage_bins: List[CoverageBin] = []
    if confidence is not None:
        coverage_bins = _coverage_bins(confidence, decisions, y_true, bins=10)

    metrics: Dict[str, object] = {
        "n": int(len(df)),
        "accuracy_non_hold": accuracy_non_hold,
        "precision_up": precision_up,
        "recall_up": recall_up,
        "f1_up": f1_up,
        "precision_down": precision_down,
        "recall_down": recall_down,
        "f1_down": f1_down,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "hold_rate": hold_rate,
        "action_rate": action_rate,
        "conflict_rate": conflict_rate,
        "brier_up": brier_up,
        "brier_down": brier_down,
        "ece_up": ece_up,
        "ece_down": ece_down,
        "reliability_up": reliability_up,
        "reliability_down": reliability_down,
        "confusion_matrix": cm,
        "coverage_bins": [bin.__dict__ for bin in coverage_bins],
    }

    if bootstrap_samples > 0 and len(df) > 0:
        rng = np.random.default_rng(seed)
        accs = []
        prec_u = []
        prec_d = []
        for _ in range(bootstrap_samples):
            idx = rng.integers(0, len(df), size=len(df))
            sample = df.iloc[idx]
            sample_metrics = compute_metrics(sample, bootstrap_samples=0)
            accs.append(sample_metrics["accuracy_non_hold"])
            prec_u.append(sample_metrics["precision_up"])
            prec_d.append(sample_metrics["precision_down"])
        metrics["ci_accuracy_non_hold"] = _ci_bounds(accs)
        metrics["ci_precision_up"] = _ci_bounds(prec_u)
        metrics["ci_precision_down"] = _ci_bounds(prec_d)

    return metrics


def _ci_bounds(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    arr = np.asarray(values)
    return {"low": float(np.quantile(arr, 0.025)), "high": float(np.quantile(arr, 0.975))}


def compute_metrics_by_session(
    df: pd.DataFrame,
    bootstrap_samples: int = 0,
    seed: int = 7,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for session_id, group in df.groupby("session_id"):
        out[str(session_id)] = compute_metrics(group, bootstrap_samples=bootstrap_samples, seed=seed)
    return out


def compute_metrics_by_regime(
    df: pd.DataFrame,
    bootstrap_samples: int = 0,
    seed: int = 7,
) -> Dict[str, Dict[str, object]]:
    if "regime" not in df.columns:
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for regime_id, group in df.groupby("regime"):
        out[str(regime_id)] = compute_metrics(group, bootstrap_samples=bootstrap_samples, seed=seed)
    return out
