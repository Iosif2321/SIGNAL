"""Adaptation criteria checks."""

from __future__ import annotations

from typing import Dict, List, Tuple

from cryptomvp.config import AdaptationConfig


def assess_adaptation(metrics: Dict[str, float], cfg: AdaptationConfig) -> Tuple[bool, List[str]]:
    """Check whether metrics meet adaptation criteria."""
    failures: List[str] = []

    def _check_min(key: str, threshold: float | None, label: str) -> None:
        if threshold is None:
            return
        value = metrics.get(key)
        if value is None:
            failures.append(f"{label} missing")
        elif value < threshold:
            failures.append(f"{label} < {threshold:.4f} (got {value:.4f})")

    def _check_max(key: str, threshold: float | None, label: str) -> None:
        if threshold is None:
            return
        value = metrics.get(key)
        if value is None:
            failures.append(f"{label} missing")
        elif value > threshold:
            failures.append(f"{label} > {threshold:.4f} (got {value:.4f})")

    _check_min("action_accuracy_non_hold", cfg.min_action_accuracy, "action_accuracy_non_hold")
    _check_min("action_rate", cfg.min_action_rate, "action_rate")
    _check_max("hold_rate", cfg.max_hold_rate, "hold_rate")
    _check_max("conflict_rate", cfg.max_conflict_rate, "conflict_rate")
    _check_min("precision_up", cfg.min_precision_up, "precision_up")
    _check_min("precision_down", cfg.min_precision_down, "precision_down")
    _check_min("rl_up_accuracy", cfg.min_rl_up_accuracy, "rl_up_accuracy")
    _check_min("rl_down_accuracy", cfg.min_rl_down_accuracy, "rl_down_accuracy")
    _check_max("rl_up_hold_rate", cfg.max_rl_up_hold_rate, "rl_up_hold_rate")
    _check_max("rl_down_hold_rate", cfg.max_rl_down_hold_rate, "rl_down_hold_rate")

    return (len(failures) == 0), failures
