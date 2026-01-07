"""Label construction for UP and DOWN."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def make_up_down_labels(df: pd.DataFrame, window_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create y_up and y_down aligned to window end times."""
    base = df[["open_time_ms", "close"]].copy()
    base = base.sort_values("open_time_ms").reset_index(drop=True)
    base["next_close"] = base["close"].shift(-1)
    base = base.dropna().reset_index(drop=True)

    lookup = dict(zip(base["open_time_ms"].to_numpy(), base["next_close"].to_numpy()))
    curr_lookup = dict(zip(base["open_time_ms"].to_numpy(), base["close"].to_numpy()))

    y_up = []
    y_down = []
    for t in window_times:
        if t not in lookup:
            continue
        close_t = curr_lookup[t]
        next_close = lookup[t]
        y_up.append(1 if next_close > close_t else 0)
        y_down.append(1 if next_close < close_t else 0)

    return np.asarray(y_up, dtype=np.int64), np.asarray(y_down, dtype=np.int64)