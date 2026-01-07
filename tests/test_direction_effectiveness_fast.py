from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cryptomvp.analysis.direction_effectiveness import AnalysisInputs, analyze_run


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_direction_effectiveness_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    reports_dir = run_dir / "reports"

    n = 40
    open_time_ms = np.arange(n) * 60_000
    prob = np.linspace(0.1, 0.9, n)
    true = (prob > 0.5).astype(int)
    pred = (prob > 0.55).astype(int)

    supervised = pd.DataFrame(
        {
            "open_time_ms": open_time_ms,
            "prob": prob,
            "true": true,
            "pred": pred,
        }
    )
    _write_parquet(reports_dir / "supervised_up/decision_log.parquet", supervised)
    _write_parquet(reports_dir / "supervised_down/decision_log.parquet", supervised)

    p_up = np.linspace(0.4, 0.8, n)
    p_down = 1.0 - p_up * 0.9
    decision = np.where(np.maximum(p_up, p_down) >= 0.55, np.where(p_up >= p_down, "UP", "DOWN"), "HOLD")
    true_dir = np.where(p_up >= p_down, "UP", "DOWN")
    true_dir[::7] = "FLAT"
    decision_rule = pd.DataFrame(
        {
            "open_time_ms": open_time_ms,
            "p_up": p_up,
            "p_down": p_down,
            "decision": decision,
            "true_direction": true_dir,
        }
    )
    _write_parquet(reports_dir / "decision_rule/decision_log.parquet", decision_rule)

    step_rows = []
    for i in range(12):
        action = 0 if i % 3 else 1
        is_hold = 1.0 if action == 1 else 0.0
        correct = 1.0 if (action == 0 and i % 2 == 0) else 0.0
        step_rows.append(
            {
                "episode": 1,
                "step": i + 1,
                "index": i,
                "time_ms": open_time_ms[i % n],
                "action": action,
                "p_direction": 0.6,
                "p_hold": 0.4,
                "reward": 1.0 if correct else -0.1,
                "is_hold": is_hold,
                "correct": correct,
            }
        )
    step_df = pd.DataFrame(step_rows)
    _write_parquet(reports_dir / "rl_up/step_log.parquet", step_df)
    _write_parquet(reports_dir / "rl_down/step_log.parquet", step_df)

    episode_df = pd.DataFrame(
        {
            "episode": [1, 2],
            "reward": [0.1, -0.05],
            "hold_rate": [0.3, 0.4],
            "accuracy": [0.6, 0.5],
            "entropy": [0.8, 0.7],
        }
    )
    (reports_dir / "rl_up").mkdir(parents=True, exist_ok=True)
    (reports_dir / "rl_down").mkdir(parents=True, exist_ok=True)
    episode_df.to_csv(reports_dir / "rl_up/episode_metrics.csv", index=False)
    episode_df.to_csv(reports_dir / "rl_down/episode_metrics.csv", index=False)

    out_dir = run_dir / "reports/direction_effectiveness"
    inputs = AnalysisInputs(
        run_dir=run_dir,
        out_dir=out_dir,
        threshold=0.55,
        rolling_window=3,
        formats=["png"],
        scan_thresholds=True,
        t_min=0.5,
        t_max=0.6,
        t_step=0.05,
        config_path=Path("configs/mvp.yaml"),
    )
    analyze_run(inputs)

    summary_csv = out_dir / "summary.csv"
    assert summary_csv.exists()
    pngs = list((out_dir / "figures").glob("*.png"))
    assert len(pngs) >= 3
