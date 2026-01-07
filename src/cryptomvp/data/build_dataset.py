"""Dataset building from Bybit REST or synthetic data."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from cryptomvp.bybit.rest import BybitRestClient
from cryptomvp.bybit.schemas import KlineCandle
from cryptomvp.utils.io import data_dir


def klines_to_dataframe(candles: Iterable[KlineCandle]) -> pd.DataFrame:
    rows = [asdict(c) for c in candles]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop(columns=["confirm", "source"], errors="ignore")
    df = df.sort_values("open_time_ms").reset_index(drop=True)
    return df


def fetch_klines_range(
    client: BybitRestClient,
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> List[KlineCandle]:
    """Fetch klines in a time range using fixed-size window paging."""
    all_candles: List[KlineCandle] = []
    cursor = start_ms
    interval_ms = _interval_to_ms(interval)

    while cursor < end_ms:
        chunk_end = min(cursor + limit * interval_ms, end_ms)
        candles = client.get_klines(
            category=category,
            symbol=symbol,
            interval=interval,
            start_ms=cursor,
            end_ms=chunk_end,
            limit=limit,
        )
        if candles:
            all_candles.extend(candles)
        cursor = chunk_end

    return all_candles


def _interval_to_ms(interval: str) -> int:
    if interval.isdigit():
        return int(interval) * 60_000
    raise ValueError(f"Unsupported interval: {interval}")


def build_synthetic_dataset(
    start_ms: int,
    end_ms: int,
    seed: int = 7,
    interval_ms: int = 60_000,
) -> pd.DataFrame:
    """Generate a synthetic minute-level kline dataset."""
    rng = np.random.default_rng(seed)
    times = np.arange(start_ms, end_ms, interval_ms)
    n = len(times)
    price = 20_000 + np.cumsum(rng.normal(0, 10, size=n))
    high = price + rng.normal(5, 2, size=n)
    low = price - rng.normal(5, 2, size=n)
    open_ = price + rng.normal(0, 1, size=n)
    close = price + rng.normal(0, 1, size=n)
    volume = rng.uniform(1, 100, size=n)
    turnover = volume * close

    df = pd.DataFrame(
        {
            "open_time_ms": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "turnover": turnover,
        }
    )
    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
    except Exception:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
    return output_path


def build_dataset_from_rest(
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> Tuple[pd.DataFrame, Path]:
    client = BybitRestClient()
    try:
        candles = fetch_klines_range(
            client=client,
            category=category,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
        )
    finally:
        client.close()

    df = klines_to_dataframe(candles)
    output_path = data_dir("processed") / f"{symbol.lower()}_{category}_{interval}m.parquet"
    out = save_dataset(df, output_path)
    return df, out
