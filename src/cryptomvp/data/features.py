"""Feature engineering without future leakage."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def compute_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Compute feature columns and drop rows with NaNs."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    safe_close = close.replace(0, np.nan)

    if "returns" in feature_list:
        out["returns"] = close.pct_change()
    if "log_returns" in feature_list:
        out["log_returns"] = np.log(close).diff()
    if "volatility" in feature_list:
        returns = close.pct_change()
        out["volatility"] = returns.rolling(window=10, min_periods=1).std()
    if "volatility_20" in feature_list:
        returns = close.pct_change()
        out["volatility_20"] = returns.rolling(window=20, min_periods=1).std()
    if "range" in feature_list:
        out["range"] = (high - low) / safe_close
    if "body" in feature_list:
        out["body"] = (close - open_) / safe_close
    if "upper_wick" in feature_list:
        out["upper_wick"] = (high - np.maximum(open_, close)) / safe_close
    if "lower_wick" in feature_list:
        out["lower_wick"] = (np.minimum(open_, close) - low) / safe_close
    if "volume" in feature_list:
        out["volume"] = np.log(volume + 1.0)
    if "volume_change" in feature_list:
        out["volume_change"] = volume.pct_change()
    if "volume_zscore_20" in feature_list:
        vol_mean = volume.rolling(window=20, min_periods=1).mean()
        vol_std = volume.rolling(window=20, min_periods=1).std()
        out["volume_zscore_20"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    if "turnover" in feature_list and "turnover" in df.columns:
        out["turnover"] = np.log(df["turnover"] + 1.0)
    if "sma_5" in feature_list:
        sma_5 = close.rolling(window=5, min_periods=1).mean()
        out["sma_5"] = _safe_div(close, sma_5) - 1.0
    if "sma_20" in feature_list:
        sma_20 = close.rolling(window=20, min_periods=1).mean()
        out["sma_20"] = _safe_div(close, sma_20) - 1.0
    if "ema_12" in feature_list:
        ema_12 = close.ewm(span=12, adjust=False).mean()
        out["ema_12"] = _safe_div(close, ema_12) - 1.0
    if "ema_26" in feature_list:
        ema_26 = close.ewm(span=26, adjust=False).mean()
        out["ema_26"] = _safe_div(close, ema_26) - 1.0
    if "ema_diff" in feature_list:
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        out["ema_diff"] = (ema_12 - ema_26) / safe_close
    if "rsi_14" in feature_list:
        out["rsi_14"] = _rsi(close, window=14) / 100.0
    if "roc_5" in feature_list:
        out["roc_5"] = close.pct_change(periods=5)
    if "momentum_10" in feature_list:
        out["momentum_10"] = close.pct_change(periods=10)
    if "atr_14" in feature_list:
        atr_14 = _atr(high, low, close, window=14)
        out["atr_14"] = atr_14 / safe_close

    out = out.replace([np.inf, -np.inf], np.nan)
    out["open_time_ms"] = df["open_time_ms"].values
    out = out.dropna().reset_index(drop=True)
    return out
