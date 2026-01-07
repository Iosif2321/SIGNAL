"""Windowing helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def make_windows(features_df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create sliding windows of features and corresponding timestamps."""
    feature_cols = [c for c in features_df.columns if c != "open_time_ms"]
    values = features_df[feature_cols].to_numpy(dtype=np.float32)
    times = features_df["open_time_ms"].to_numpy(dtype=np.int64)

    if len(values) < window_size:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), times, feature_cols

    windows = []
    window_times = []
    for idx in range(window_size - 1, len(values)):
        windows.append(values[idx - window_size + 1 : idx + 1])
        window_times.append(times[idx])

    return np.stack(windows), np.asarray(window_times), feature_cols