"""Decision rule for UP/DOWN/HOLD."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def decide(p_up: float, p_down: float, t_min: float, delta_min: float = 0.0) -> str:
    max_p = max(p_up, p_down)
    if max_p >= t_min and abs(p_up - p_down) >= delta_min:
        return "UP" if p_up >= p_down else "DOWN"
    return "HOLD"


def batch_decide(
    p_up: np.ndarray, p_down: np.ndarray, t_min: float, delta_min: float = 0.0
) -> List[str]:
    return [decide(u, d, t_min, delta_min=delta_min) for u, d in zip(p_up, p_down)]


def hold_rate(decisions: Iterable[str]) -> float:
    decisions = list(decisions)
    if not decisions:
        return 0.0
    return float(sum(1 for d in decisions if d == "HOLD") / len(decisions))


def scan_thresholds(
    p_up: np.ndarray,
    p_down: np.ndarray,
    thresholds: Iterable[float],
    delta_min: float = 0.0,
) -> List[float]:
    rates = []
    for t in thresholds:
        decisions = batch_decide(p_up, p_down, t, delta_min=delta_min)
        rates.append(hold_rate(decisions))
    return rates
