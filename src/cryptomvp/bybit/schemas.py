"""Bybit message schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KlineCandle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    confirm: Optional[bool] = None
    source: Optional[str] = None