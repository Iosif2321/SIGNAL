"""Feature engineering without future leakage."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Compute feature columns and drop rows with NaNs."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    if "returns" in feature_list:
        out["returns"] = close.pct_change()
    if "log_returns" in feature_list:
        out["log_returns"] = np.log(close).diff()
    if "volatility" in feature_list:
        returns = close.pct_change()
        out["volatility"] = returns.rolling(window=10, min_periods=1).std()
    if "range" in feature_list:
        out["range"] = (high - low) / close.replace(0, np.nan)
    if "volume" in feature_list:
        out["volume"] = np.log(volume + 1.0)
    if "turnover" in feature_list and "turnover" in df.columns:
        out["turnover"] = np.log(df["turnover"] + 1.0)

    out = out.replace([np.inf, -np.inf], np.nan)
    out["open_time_ms"] = df["open_time_ms"].values
    out = out.dropna().reset_index(drop=True)
    return out