"""Feature engineering without future leakage."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100 - (100 / (1 + rs))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(window=window, min_periods=1).mean()


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def _slope(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr), dtype=np.float32)
        coef = np.polyfit(x, arr.astype(np.float32), 1)
        return float(coef[0])

    return series.rolling(window=window, min_periods=2).apply(_slope, raw=True)


def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    def _autocorr(arr: np.ndarray) -> float:
        if len(arr) <= lag:
            return 0.0
        x = arr[:-lag]
        y = arr[lag:]
        if np.std(x) < EPS or np.std(y) < EPS:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    return series.rolling(window=window, min_periods=lag + 2).apply(_autocorr, raw=True)


def _rolling_entropy(series: pd.Series, window: int, bins: int = 10) -> pd.Series:
    def _entropy(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        hist, _ = np.histogram(arr, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        return float(-(hist * np.log(hist + EPS)).sum())

    return series.rolling(window=window, min_periods=2).apply(_entropy, raw=True)


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    lowest = low.rolling(window=window, min_periods=1).min()
    highest = high.rolling(window=window, min_periods=1).max()
    return _safe_div(close - lowest, highest - lowest + EPS)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=window, min_periods=1).mean()

    def _mad(arr: np.ndarray) -> float:
        mean = np.mean(arr)
        return float(np.mean(np.abs(arr - mean)))

    mad = tp.rolling(window=window, min_periods=1).apply(_mad, raw=True)
    return (tp - sma_tp) / (0.015 * (mad + EPS))


def _aroon(high: pd.Series, low: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    def _aroon_up(arr: np.ndarray) -> float:
        idx = np.argmax(arr)
        periods_since = len(arr) - 1 - idx
        return float(100 * (len(arr) - periods_since) / max(len(arr), 1))

    def _aroon_down(arr: np.ndarray) -> float:
        idx = np.argmin(arr)
        periods_since = len(arr) - 1 - idx
        return float(100 * (len(arr) - periods_since) / max(len(arr), 1))

    aroon_up = high.rolling(window=window, min_periods=1).apply(_aroon_up, raw=True)
    aroon_down = low.rolling(window=window, min_periods=1).apply(_aroon_down, raw=True)
    return aroon_up, aroon_down


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(high, low, close)
    atr = tr.rolling(window=window, min_periods=1).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(window=window, min_periods=1).sum() / (
        atr + EPS
    )
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(window=window, min_periods=1).sum() / (
        atr + EPS
    )
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + EPS) * 100
    adx = dx.rolling(window=window, min_periods=1).mean()
    return adx, plus_di, minus_di


def _trix(close: pd.Series, window: int) -> pd.Series:
    ema1 = _ema(close, window)
    ema2 = _ema(ema1, window)
    ema3 = _ema(ema2, window)
    return ema3.pct_change()


def _parse_int_suffix(name: str, prefix: str) -> Optional[int]:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _tf_to_rule(tf: str) -> str:
    if tf.endswith("m"):
        return tf.replace("m", "min")
    if tf.endswith("h"):
        return tf.replace("h", "H")
    raise ValueError(f"Unsupported timeframe: {tf}")


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    times = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    resampled = (
        df.set_index(times)
        .resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "turnover": "sum",
            }
        )
        .dropna()
    )
    resampled["open_time_ms"] = (resampled.index.view("int64") // 1_000_000).astype("int64")
    return resampled.reset_index(drop=True)


def _streak(series: pd.Series, positive: bool = True) -> pd.Series:
    flags = series > 0 if positive else series < 0
    counts = np.zeros(len(series), dtype=np.float32)
    streak = 0
    for idx, flag in enumerate(flags.fillna(False).to_numpy()):
        if flag:
            streak += 1
        else:
            streak = 0
        counts[idx] = streak
    return pd.Series(counts, index=series.index)


def compute_features(df: pd.DataFrame, feature_list: List[str], allow_mtf: bool = True) -> pd.DataFrame:
    """Compute feature columns and drop rows with NaNs."""
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    safe_close = close.replace(0, np.nan)
    returns = close.pct_change()
    log_returns = np.log(safe_close).diff()

    mtf_cache: dict[str, pd.DataFrame] = {}
    mtf_feature_cache: dict[tuple[str, str], pd.Series] = {}

    for feature in feature_list:
        if feature.startswith("mtf_"):
            if not allow_mtf:
                continue
            parts = feature.split("_", 2)
            if len(parts) < 3:
                raise ValueError(f"Invalid mtf feature name: {feature}")
            tf = parts[1]
            base_feature = parts[2]
            if tf not in mtf_cache:
                mtf_cache[tf] = _resample_ohlcv(df, _tf_to_rule(tf))
            cache_key = (tf, base_feature)
            if cache_key not in mtf_feature_cache:
                base_df = compute_features(mtf_cache[tf], [base_feature], allow_mtf=False)
                base_series = base_df.set_index("open_time_ms")[base_feature]
                mtf_feature_cache[cache_key] = base_series
            aligned = mtf_feature_cache[cache_key].reindex(
                df["open_time_ms"].to_numpy(), method="ffill"
            )
            out[feature] = aligned.to_numpy()
            continue

        if feature == "returns":
            out[feature] = returns
            continue
        if feature == "log_returns":
            out[feature] = log_returns
            continue
        if feature == "return_sign":
            out[feature] = np.sign(returns)
            continue
        if feature == "gap":
            out[feature] = open_ - close.shift(1)
            continue
        if feature == "gap_pct":
            out[feature] = _safe_div(open_ - close.shift(1), close.shift(1))
            continue
        if feature == "close_pos_range":
            out[feature] = _safe_div(close - low, (high - low) + EPS)
            continue
        if feature == "typical_price":
            out[feature] = (high + low + close) / 3.0
            continue
        if feature == "ohlc4":
            out[feature] = (open_ + high + low + close) / 4.0
            continue
        if feature == "hlc3":
            out[feature] = (high + low + close) / 3.0
            continue

        window = _parse_int_suffix(feature, "return_")
        if window is not None:
            out[feature] = close.pct_change(periods=window)
            continue
        window = _parse_int_suffix(feature, "cum_return_")
        if window is not None:
            out[feature] = close.pct_change(periods=window)
            continue

        if feature == "body":
            out[feature] = (close - open_) / safe_close
            continue
        if feature == "abs_body":
            out[feature] = (close - open_).abs() / safe_close
            continue
        if feature == "body_pct":
            out[feature] = _safe_div(close - open_, open_)
            continue
        if feature == "range":
            out[feature] = (high - low) / safe_close
            continue
        if feature == "range_pct":
            out[feature] = _safe_div(high - low, open_)
            continue
        if feature == "upper_wick":
            out[feature] = (high - np.maximum(open_, close)) / safe_close
            continue
        if feature == "lower_wick":
            out[feature] = (np.minimum(open_, close) - low) / safe_close
            continue
        if feature == "body_to_range":
            out[feature] = _safe_div((close - open_).abs(), (high - low) + EPS)
            continue
        if feature == "wick_to_range":
            wick = (high - np.maximum(open_, close)) + (np.minimum(open_, close) - low)
            out[feature] = _safe_div(wick.abs(), (high - low) + EPS)
            continue
        if feature == "doji_ratio":
            out[feature] = _safe_div((close - open_).abs(), (high - low) + EPS)
            continue
        if feature == "engulfing_strength":
            prev_body = (close - open_).shift(1)
            body = close - open_
            opposite = np.sign(body) != np.sign(prev_body)
            strength = (body.abs() - prev_body.abs()).clip(lower=0.0) / ((high - low) + EPS)
            out[feature] = strength.where(opposite, 0.0)
            continue
        window = _parse_int_suffix(feature, "up_count_")
        if window is not None:
            out[feature] = (returns > 0).rolling(window=window, min_periods=1).sum()
            continue
        window = _parse_int_suffix(feature, "down_count_")
        if window is not None:
            out[feature] = (returns < 0).rolling(window=window, min_periods=1).sum()
            continue
        if feature == "up_streak":
            out[feature] = _streak(returns, positive=True)
            continue
        if feature == "down_streak":
            out[feature] = _streak(returns, positive=False)
            continue

        window = _parse_int_suffix(feature, "sma_")
        if window is not None:
            sma = _sma(close, window)
            out[feature] = _safe_div(close, sma) - 1.0
            continue
        window = _parse_int_suffix(feature, "ema_")
        if window is not None:
            ema = _ema(close, window)
            out[feature] = _safe_div(close, ema) - 1.0
            continue
        if feature == "ema_diff":
            ema_fast = _ema(close, 12)
            ema_slow = _ema(close, 26)
            out[feature] = (ema_fast - ema_slow) / safe_close
            continue
        window = _parse_int_suffix(feature, "sma_slope_")
        if window is not None:
            out[feature] = _rolling_slope(_sma(close, window), window)
            continue
        window = _parse_int_suffix(feature, "ema_slope_")
        if window is not None:
            out[feature] = _rolling_slope(_ema(close, window), window)
            continue
        if feature == "macd":
            out[feature] = _ema(close, 12) - _ema(close, 26)
            continue
        if feature == "macd_signal":
            macd = _ema(close, 12) - _ema(close, 26)
            out[feature] = _ema(macd, 9)
            continue
        if feature == "macd_hist":
            macd = _ema(close, 12) - _ema(close, 26)
            signal = _ema(macd, 9)
            out[feature] = macd - signal
            continue
        window = _parse_int_suffix(feature, "rsi_")
        if window is not None:
            out[feature] = _rsi(close, window) / 100.0
            continue
        if feature == "stoch_k_14":
            out[feature] = _stoch_k(high, low, close, window=14)
            continue
        if feature == "stoch_d_3":
            k = _stoch_k(high, low, close, window=14)
            out[feature] = k.rolling(window=3, min_periods=1).mean()
            continue
        window = _parse_int_suffix(feature, "roc_")
        if window is not None:
            out[feature] = close.pct_change(periods=window)
            continue
        window = _parse_int_suffix(feature, "momentum_")
        if window is not None:
            out[feature] = close.pct_change(periods=window)
            continue
        if feature == "cci_20":
            out[feature] = _cci(high, low, close, window=20)
            continue
        if feature == "williams_r_14":
            highest = high.rolling(window=14, min_periods=1).max()
            lowest = low.rolling(window=14, min_periods=1).min()
            out[feature] = -100 * _safe_div(highest - close, (highest - lowest) + EPS)
            continue
        if feature == "aroon_up_25" or feature == "aroon_down_25":
            aroon_up, aroon_down = _aroon(high, low, window=25)
            if feature == "aroon_up_25":
                out[feature] = aroon_up / 100.0
            else:
                out[feature] = aroon_down / 100.0
            continue
        if feature in {"adx_14", "plus_di_14", "minus_di_14"}:
            adx, plus_di, minus_di = _adx(high, low, close, window=14)
            if feature == "adx_14":
                out[feature] = adx / 100.0
            elif feature == "plus_di_14":
                out[feature] = plus_di / 100.0
            else:
                out[feature] = minus_di / 100.0
            continue
        if feature == "trix_15":
            out[feature] = _trix(close, window=15)
            continue
        if feature == "linreg_slope_20":
            out[feature] = _rolling_slope(close, window=20)
            continue

        window = _parse_int_suffix(feature, "volatility_")
        if window is not None:
            out[feature] = returns.rolling(window=window, min_periods=1).std()
            continue
        if feature == "volatility":
            out[feature] = returns.rolling(window=10, min_periods=1).std()
            continue
        if feature == "true_range":
            out[feature] = _true_range(high, low, close) / safe_close
            continue
        if feature == "atr_14":
            out[feature] = _atr(high, low, close, window=14) / safe_close
            continue
        if feature == "vol_parkinson_20":
            log_hl = np.log(_safe_div(high, low).abs() + EPS)
            out[feature] = (
                (log_hl**2).rolling(window=20, min_periods=1).mean() / (4 * np.log(2))
            ) ** 0.5
            continue
        if feature == "vol_gk_20":
            log_hl = np.log(_safe_div(high, low).abs() + EPS)
            log_co = np.log(_safe_div(close, open_).abs() + EPS)
            out[feature] = (
                0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)
            ).rolling(window=20, min_periods=1).mean().abs()
            continue
        if feature == "vol_rs_20":
            term1 = np.log(_safe_div(high, close).abs() + EPS) * np.log(
                _safe_div(high, open_).abs() + EPS
            )
            term2 = np.log(_safe_div(low, close).abs() + EPS) * np.log(
                _safe_div(low, open_).abs() + EPS
            )
            out[feature] = (term1 + term2).rolling(window=20, min_periods=1).mean().abs()
            continue
        if feature == "bb_width_20":
            sma = _sma(close, 20)
            std = close.rolling(window=20, min_periods=1).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            out[feature] = _safe_div(upper - lower, sma)
            continue
        if feature == "keltner_width_20":
            ema = _ema(close, 20)
            atr = _atr(high, low, close, window=20)
            upper = ema + 2 * atr
            lower = ema - 2 * atr
            out[feature] = _safe_div(upper - lower, ema)
            continue
        if feature == "vol_zscore_20":
            vol = returns.rolling(window=20, min_periods=1).std()
            mean = vol.rolling(window=20, min_periods=1).mean()
            std = vol.rolling(window=20, min_periods=1).std()
            out[feature] = _safe_div(vol - mean, std + EPS)
            continue

        if feature == "volume":
            out[feature] = np.log(volume + 1.0)
            continue
        if feature == "log_volume":
            out[feature] = np.log(volume + 1.0)
            continue
        if feature == "volume_change":
            out[feature] = volume.pct_change()
            continue
        if feature == "volume_zscore_20":
            vol_mean = volume.rolling(window=20, min_periods=1).mean()
            vol_std = volume.rolling(window=20, min_periods=1).std()
            out[feature] = _safe_div(volume - vol_mean, vol_std + EPS)
            continue
        if feature == "obv":
            direction = np.sign(close.diff()).fillna(0.0)
            out[feature] = (direction * volume).cumsum()
            continue
        if feature == "vwap_20":
            tp = (high + low + close) / 3.0
            vwap = (tp * volume).rolling(window=20, min_periods=1).sum() / (
                volume.rolling(window=20, min_periods=1).sum() + EPS
            )
            out[feature] = _safe_div(close, vwap) - 1.0
            continue
        if feature == "mfi_14":
            tp = (high + low + close) / 3.0
            mf = tp * volume
            pos = mf.where(tp > tp.shift(1), 0.0)
            neg = mf.where(tp < tp.shift(1), 0.0)
            pos_sum = pos.rolling(window=14, min_periods=1).sum()
            neg_sum = neg.rolling(window=14, min_periods=1).sum()
            mfi = 100 - (100 / (1 + _safe_div(pos_sum, neg_sum + EPS)))
            out[feature] = mfi / 100.0
            continue
        if feature == "adl":
            mfm = _safe_div((close - low) - (high - close), (high - low) + EPS)
            out[feature] = (mfm * volume).cumsum()
            continue
        if feature == "cmf_20":
            mfm = _safe_div((close - low) - (high - close), (high - low) + EPS)
            mfv = mfm * volume
            out[feature] = mfv.rolling(window=20, min_periods=1).sum() / (
                volume.rolling(window=20, min_periods=1).sum() + EPS
            )
            continue
        if feature == "volume_weighted_return_20":
            vw = (returns * volume).rolling(window=20, min_periods=1).sum()
            out[feature] = vw / (volume.rolling(window=20, min_periods=1).sum() + EPS)
            continue
        if feature == "amihud_20":
            out[feature] = returns.abs().rolling(window=20, min_periods=1).mean() / (
                volume.rolling(window=20, min_periods=1).mean() + EPS
            )
            continue

        if feature == "bb_upper_20" or feature == "bb_lower_20" or feature == "bb_percent_b_20":
            sma = _sma(close, 20)
            std = close.rolling(window=20, min_periods=1).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            if feature == "bb_upper_20":
                out[feature] = _safe_div(upper, close) - 1.0
            elif feature == "bb_lower_20":
                out[feature] = _safe_div(lower, close) - 1.0
            else:
                out[feature] = _safe_div(close - lower, (upper - lower) + EPS)
            continue
        if feature in {"donchian_high_20", "donchian_low_20", "donchian_width_20", "donchian_pos_20"}:
            high_roll = high.rolling(window=20, min_periods=1).max()
            low_roll = low.rolling(window=20, min_periods=1).min()
            if feature == "donchian_high_20":
                out[feature] = _safe_div(high_roll, close) - 1.0
            elif feature == "donchian_low_20":
                out[feature] = _safe_div(low_roll, close) - 1.0
            elif feature == "donchian_width_20":
                out[feature] = _safe_div(high_roll - low_roll, close)
            else:
                out[feature] = _safe_div(close - low_roll, (high_roll - low_roll) + EPS)
            continue
        if feature in {"keltner_upper_20", "keltner_lower_20"}:
            ema = _ema(close, 20)
            atr = _atr(high, low, close, window=20)
            upper = ema + 2 * atr
            lower = ema - 2 * atr
            if feature == "keltner_upper_20":
                out[feature] = _safe_div(upper, close) - 1.0
            else:
                out[feature] = _safe_div(lower, close) - 1.0
            continue
        if feature == "dist_to_support_20":
            low_roll = low.rolling(window=20, min_periods=1).min()
            out[feature] = _safe_div(close - low_roll, close)
            continue
        if feature == "dist_to_resistance_20":
            high_roll = high.rolling(window=20, min_periods=1).max()
            out[feature] = _safe_div(high_roll - close, close)
            continue
        if feature == "pivot_distance":
            pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3.0
            out[feature] = _safe_div(close - pivot, close)
            continue

        if feature == "returns_skew_20":
            out[feature] = returns.rolling(window=20, min_periods=2).skew()
            continue
        if feature == "returns_kurt_20":
            out[feature] = returns.rolling(window=20, min_periods=2).kurt()
            continue
        if feature == "autocorr_1_20":
            out[feature] = _rolling_autocorr(returns.fillna(0.0), window=20, lag=1)
            continue
        if feature == "entropy_20":
            out[feature] = _rolling_entropy(returns.fillna(0.0), window=20, bins=10)
            continue
        if feature == "abs_return_zscore_20":
            abs_ret = returns.abs()
            mean = abs_ret.rolling(window=20, min_periods=1).mean()
            std = abs_ret.rolling(window=20, min_periods=1).std()
            out[feature] = _safe_div(abs_ret - mean, std + EPS)
            continue

        if feature == "hour_sin" or feature == "hour_cos" or feature == "dow_sin" or feature == "dow_cos" or feature == "is_weekend" or feature == "minute_sin" or feature == "minute_cos":
            times = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
            hour = times.dt.hour + times.dt.minute / 60.0
            dow = times.dt.dayofweek
            minute = times.dt.hour * 60 + times.dt.minute
            if feature == "hour_sin":
                out[feature] = np.sin(2 * np.pi * hour / 24.0)
            elif feature == "hour_cos":
                out[feature] = np.cos(2 * np.pi * hour / 24.0)
            elif feature == "dow_sin":
                out[feature] = np.sin(2 * np.pi * dow / 7.0)
            elif feature == "dow_cos":
                out[feature] = np.cos(2 * np.pi * dow / 7.0)
            elif feature == "minute_sin":
                out[feature] = np.sin(2 * np.pi * minute / (24 * 60.0))
            elif feature == "minute_cos":
                out[feature] = np.cos(2 * np.pi * minute / (24 * 60.0))
            else:
                out[feature] = (dow >= 5).astype(float)
            continue
        if feature == "session_onehot":
            if "session_id" not in df.columns:
                out[feature] = 0.0
            else:
                session_ids = df["session_id"].astype(str)
                for sid in sorted(session_ids.unique()):
                    out[f"session_{sid}"] = (session_ids == sid).astype(float)
            continue
        if feature == "minutes_since_session_start":
            if "minutes_since_session_start" in df.columns:
                out[feature] = df["minutes_since_session_start"].astype(float)
            else:
                out[feature] = 0.0
            continue
        if feature == "minutes_to_session_end":
            if "minutes_to_session_end" in df.columns:
                out[feature] = df["minutes_to_session_end"].astype(float)
            else:
                out[feature] = 0.0
            continue
        if feature == "session_overlap_flag":
            if "session_overlap" in df.columns:
                out[feature] = df["session_overlap"].astype(float)
            else:
                out[feature] = 0.0
            continue

        if feature == "vol_regime_20":
            vol = returns.rolling(window=20, min_periods=1).std()
            vmin = vol.rolling(window=20, min_periods=1).min()
            vmax = vol.rolling(window=20, min_periods=1).max()
            out[feature] = _safe_div(vol - vmin, (vmax - vmin) + EPS)
            continue
        if feature == "trend_strength_20":
            out[feature] = _rolling_slope(close, window=20).abs()
            continue
        window = _parse_int_suffix(feature, "regime_trend_")
        if window is not None:
            sma = _sma(close, window)
            slope = _rolling_slope(sma, window)
            vol = returns.rolling(window=window, min_periods=1).std()
            strength = slope.abs() / (vol + EPS)
            out[feature] = (strength > 1.0).astype(float)
            continue
        if feature == "mean_reversion_score":
            mean = close.rolling(window=20, min_periods=1).mean()
            std = close.rolling(window=20, min_periods=1).std()
            z = _safe_div(close - mean, std + EPS)
            out[feature] = -z.abs()
            continue
        if feature == "range_bound_score":
            sma = _sma(close, 20)
            std = close.rolling(window=20, min_periods=1).std()
            width = _safe_div((sma + 2 * std) - (sma - 2 * std), sma)
            out[feature] = 1.0 / (1.0 + width.abs())
            continue

        if feature == "turnover" and "turnover" in df.columns:
            out[feature] = np.log(df["turnover"].astype(float) + 1.0)
            continue

        raise ValueError(f"Unknown feature: {feature}")

    out = out.replace([np.inf, -np.inf], np.nan)
    out["open_time_ms"] = df["open_time_ms"].values
    out = out.dropna().reset_index(drop=True)
    return out


def compute_regime_labels(
    df: pd.DataFrame,
    window_times: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Compute trend/flat regime labels aligned to window timestamps."""
    feature_name = f"regime_trend_{window}"
    regime_df = compute_features(df, [feature_name])
    series = regime_df.set_index("open_time_ms")[feature_name]
    aligned = series.reindex(window_times, method="ffill").fillna(0.0)
    return np.where(aligned.to_numpy(dtype=float) >= 0.5, "trend", "flat")
