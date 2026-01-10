from __future__ import annotations

import pandas as pd

from cryptomvp.evaluate.metrics import compute_metrics


def test_compute_metrics_basic() -> None:
    df = pd.DataFrame(
        {
            "y_true": ["UP", "UP", "DOWN", "DOWN"],
            "decision": ["UP", "HOLD", "DOWN", "UP"],
            "p_up": [0.9, 0.6, 0.1, 0.7],
            "p_down": [0.1, 0.4, 0.9, 0.3],
            "conflict": [0, 0, 0, 1],
            "session_id": ["ASIA", "ASIA", "US", "US"],
        }
    )
    metrics = compute_metrics(df)
    assert metrics["hold_rate"] == 0.25
    assert metrics["action_rate"] == 0.75
    assert abs(metrics["accuracy_non_hold"] - (2 / 3)) < 1e-6
    assert abs(metrics["precision_up"] - 0.5) < 1e-6
    assert abs(metrics["recall_up"] - 0.5) < 1e-6
    assert metrics["conflict_rate"] == 0.25
