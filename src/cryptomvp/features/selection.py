"""Staged feature selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from cryptomvp.data.features import compute_features
from cryptomvp.data.labels import make_up_down_labels
from cryptomvp.data.scaling import apply_standard_scaler, fit_standard_scaler
from cryptomvp.data.windowing import make_windows
from cryptomvp.features.registry import load_feature_sets


@dataclass(frozen=True)
class FeatureSetScore:
    feature_set_id: str
    score: float
    accuracy: float
    roc_auc: float | None
    n_features: int


def _time_split(n: int) -> Tuple[int, int]:
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return train_end, val_end


def _direction_labels(y_up: np.ndarray, y_down: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_dir = np.where(y_up == 1, 1, np.where(y_down == 1, 0, -1))
    mask = y_dir >= 0
    return y_dir[mask], mask


def cheap_filter(
    features_df: pd.DataFrame,
    corr_threshold: float = 0.98,
    var_threshold: float = 1e-12,
) -> List[str]:
    """Remove constant and highly correlated features."""
    feature_cols = [c for c in features_df.columns if c != "open_time_ms"]
    df = features_df[feature_cols].copy()
    variances = df.var().fillna(0.0)
    keep = variances[variances > var_threshold].index.tolist()
    df = df[keep]
    if df.shape[1] <= 1:
        return df.columns.tolist()
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    return [c for c in df.columns if c not in to_drop]


def score_feature_set(
    df: pd.DataFrame,
    window_size: int,
    features_df: pd.DataFrame | None = None,
) -> FeatureSetScore:
    """Score features with a quick logistic regression proxy."""
    if features_df is None:
        raise ValueError("features_df is required for scoring.")
    windows, window_times, feature_cols = make_windows(features_df, window_size)
    if len(windows) == 0:
        return FeatureSetScore("unknown", 0.0, 0.0, None, 0)
    y_up, y_down = make_up_down_labels(df, window_times)
    y_dir, mask = _direction_labels(y_up, y_down)
    if y_dir.size == 0:
        return FeatureSetScore("unknown", 0.0, 0.0, None, len(feature_cols))

    X = windows.reshape(len(windows), -1)[mask]
    train_end, val_end = _time_split(len(X))
    if val_end <= train_end:
        return FeatureSetScore("unknown", 0.0, 0.0, None, len(feature_cols))

    mean, std = fit_standard_scaler(X[:train_end])
    X_scaled = apply_standard_scaler(X, mean, std)

    model = LogisticRegression(max_iter=200, n_jobs=1)
    model.fit(X_scaled[:train_end], y_dir[:train_end])
    preds = model.predict(X_scaled[train_end:val_end])
    probs = model.predict_proba(X_scaled[train_end:val_end])[:, 1]
    acc = float(accuracy_score(y_dir[train_end:val_end], preds))
    roc = None
    if len(np.unique(y_dir[train_end:val_end])) > 1:
        roc = float(roc_auc_score(y_dir[train_end:val_end], probs))
    score = acc if roc is None else (acc + roc) / 2
    return FeatureSetScore("unknown", score, acc, roc, len(feature_cols))


def staged_feature_selection(
    df: pd.DataFrame,
    window_size: int,
    feature_sets_path: Path | None = None,
    top_n: int = 10,
    corr_threshold: float = 0.98,
    var_threshold: float = 1e-12,
) -> List[FeatureSetScore]:
    """Run staged selection and return top-N feature sets."""
    sets = load_feature_sets(feature_sets_path)
    scores: List[FeatureSetScore] = []
    for feature_set_id, features in sets.items():
        features_df = compute_features(df, list(features))
        filtered_cols = cheap_filter(
            features_df, corr_threshold=corr_threshold, var_threshold=var_threshold
        )
        keep_cols = filtered_cols + ["open_time_ms"]
        filtered_df = features_df[keep_cols]
        score = score_feature_set(df, window_size, features_df=filtered_df)
        scores.append(
            FeatureSetScore(
                feature_set_id=feature_set_id,
                score=score.score,
                accuracy=score.accuracy,
                roc_auc=score.roc_auc,
                n_features=len(filtered_features),
            )
        )
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores[:top_n]
