"""Dataset validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    total_rows: int
    monotonic: bool
    duplicate_count: int
    missing_count: int
    missing_times: List[int]


def validate_dataset(df: pd.DataFrame, interval_ms: int = 60_000) -> ValidationReport:
    if df.empty:
        return ValidationReport(0, True, 0, 0, [])

    times = df["open_time_ms"].to_numpy()
    monotonic = bool(np.all(np.diff(times) > 0))
    duplicate_count = int(pd.Series(times).duplicated().sum())

    expected = np.arange(times.min(), times.max() + interval_ms, interval_ms)
    missing = np.setdiff1d(expected, times)

    return ValidationReport(
        total_rows=len(df),
        monotonic=monotonic,
        duplicate_count=duplicate_count,
        missing_count=int(len(missing)),
        missing_times=missing.tolist(),
    )