"""Monitoring utilities for rolling metrics and drift detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from cryptomvp.data.features import compute_features


@dataclass(frozen=True)
class DriftScore:
    feature: str
    psi: float
    kl: float


def _hist_counts(values: np.ndarray, bins: int) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    counts = counts.astype(float)
    counts = counts / (counts.sum() + 1e-9)
    return counts


def compute_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref_counts = _hist_counts(ref, bins)
    cur_counts = _hist_counts(cur, bins)
    diff = cur_counts - ref_counts
    return float(np.sum(diff * np.log((cur_counts + 1e-9) / (ref_counts + 1e-9))))


def compute_kl(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref_counts = _hist_counts(ref, bins)
    cur_counts = _hist_counts(cur, bins)
    return float(np.sum(ref_counts * np.log((ref_counts + 1e-9) / (cur_counts + 1e-9))))


def drift_report(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    feature_list: Iterable[str],
    bins: int = 10,
) -> List[DriftScore]:
    features_ref = compute_features(df_ref, list(feature_list))
    features_cur = compute_features(df_cur, list(feature_list))
    cols = [c for c in features_ref.columns if c != "open_time_ms"]
    scores: List[DriftScore] = []
    for col in cols:
        if col not in features_cur.columns:
            continue
        ref_vals = features_ref[col].to_numpy(dtype=float)
        cur_vals = features_cur[col].to_numpy(dtype=float)
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        scores.append(
            DriftScore(
                feature=col,
                psi=compute_psi(ref_vals, cur_vals, bins=bins),
                kl=compute_kl(ref_vals, cur_vals, bins=bins),
            )
        )
    return scores


def rolling_metrics(decision_log: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling metrics from decision logs."""
    df = decision_log.copy()
    decisions = df["decision"].astype(str)
    hold = (decisions == "HOLD").astype(int)
    conflict = df["conflict"].astype(int) if "conflict" in df.columns else 0
    action = (decisions != "HOLD").astype(int)
    if "correct_direction" in df.columns:
        correct = df["correct_direction"].astype(int)
    else:
        true_dir = df.get("true_direction")
        if true_dir is None:
            correct = pd.Series(np.zeros(len(df), dtype=int))
        else:
            correct = (decisions == true_dir.astype(str)).astype(int)

    action_sum = action.rolling(window=window, min_periods=1).sum()
    correct_sum = (correct * action).rolling(window=window, min_periods=1).sum()
    accuracy_non_hold = (correct_sum / action_sum.replace(0, np.nan)).fillna(0.0)

    out = pd.DataFrame(
        {
            "open_time_ms": df["open_time_ms"].to_numpy(),
            "hold_rate": hold.rolling(window=window, min_periods=1).mean().to_numpy(),
            "conflict_rate": pd.Series(conflict).rolling(window=window, min_periods=1).mean().to_numpy(),
            "action_rate": action.rolling(window=window, min_periods=1).mean().to_numpy(),
            "accuracy_non_hold": accuracy_non_hold.to_numpy(),
        }
    )
    return out


def rolling_metrics_by_session(
    decision_log: pd.DataFrame, window: int
) -> Dict[str, pd.DataFrame]:
    """Compute rolling metrics per session."""
    if "session_id" not in decision_log.columns:
        return {}
    results: Dict[str, pd.DataFrame] = {}
    for session_id, sdf in decision_log.groupby("session_id"):
        results[str(session_id)] = rolling_metrics(sdf, window)
    return results
