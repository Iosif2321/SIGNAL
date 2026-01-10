"""Rolling monitoring report and drift alerts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.analysis.adaptation import assess_adaptation  # noqa: E402
from cryptomvp.analysis.monitoring import (  # noqa: E402
    rolling_drift_report,
    rolling_metrics,
    rolling_metrics_by_session,
)
from cryptomvp.config import load_config  # noqa: E402
from cryptomvp.evaluate.metrics import compute_metrics, compute_metrics_by_session  # noqa: E402
from cryptomvp.utils.io import reports_dir  # noqa: E402
from cryptomvp.utils.run_dir import init_run_dir  # noqa: E402
from cryptomvp.viz.plotting import plot_bar, plot_series_with_band  # noqa: E402


def run_monitoring(
    config_path: str,
    run_dir: Path | None,
    decision_log_path: Path | None,
    window: int,
) -> None:
    init_run_dir(run_dir, config_path)
    cfg = load_config(config_path)

    report_dir = reports_dir("monitoring")
    figures_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if decision_log_path is None:
        decision_log_path = Path("reports/decision_rule/decision_log.parquet")
        if run_dir is not None:
            decision_log_path = Path(run_dir) / "reports" / "decision_rule" / "decision_log.parquet"
    if not decision_log_path.exists():
        raise RuntimeError(f"Decision log not found at {decision_log_path}")

    decision_log = pd.read_parquet(decision_log_path)
    roll = rolling_metrics(decision_log, window=window)
    roll.to_csv(report_dir / "metrics_over_time.csv", index=False)

    x = roll["open_time_ms"].to_numpy()
    plot_series_with_band(
        x,
        roll["accuracy_non_hold"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Rolling Accuracy (non-hold)",
        xlabel="open_time_ms",
        ylabel="Accuracy",
        label="accuracy_non_hold",
        out_base=figures_dir / "rolling_accuracy",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        x,
        roll["hold_rate"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Rolling Hold Rate",
        xlabel="open_time_ms",
        ylabel="Hold rate",
        label="hold_rate",
        out_base=figures_dir / "rolling_hold_rate",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        x,
        roll["conflict_rate"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Rolling Conflict Rate",
        xlabel="open_time_ms",
        ylabel="Conflict rate",
        label="conflict_rate",
        out_base=figures_dir / "rolling_conflict_rate",
        formats=cfg.viz.save_formats,
    )

    session_rolls = rolling_metrics_by_session(decision_log, window=window)
    for session_id, sroll in session_rolls.items():
        sroll.to_csv(report_dir / f"metrics_over_time_{session_id}.csv", index=False)

    metrics = compute_metrics(decision_log)
    metrics_by_session = compute_metrics_by_session(decision_log)

    drift_scores = []
    drift_score = None
    drift_score_max = None
    drift_ks_score = None
    drift_ks_score_max = None
    dataset_path = Path(cfg.dataset.output_path)
    if dataset_path.exists():
        df = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_csv(dataset_path)
        drift_scores = rolling_drift_report(
            df,
            feature_list=["returns", "volatility_20", "volume_change"],
            window=window,
            bins=10,
        )
        if drift_scores:
            drift_df = pd.DataFrame([d.__dict__ for d in drift_scores])
            drift_df.to_csv(report_dir / "drift_scores.csv", index=False)
            drift_summary = (
                drift_df.groupby("open_time_ms")[["psi", "ks"]].mean().reset_index()
            )
            drift_summary.to_csv(report_dir / "drift_over_time.csv", index=False)
            drift_score = float(drift_summary["psi"].iloc[-1])
            drift_score_max = float(drift_summary["psi"].max())
            drift_ks_score = float(drift_summary["ks"].iloc[-1])
            drift_ks_score_max = float(drift_summary["ks"].max())
            plot_series_with_band(
                drift_summary["open_time_ms"].to_numpy(),
                drift_summary["psi"].to_numpy(),
                window=cfg.viz.moving_window,
                title="Rolling Drift PSI (mean)",
                xlabel="open_time_ms",
                ylabel="PSI",
                label="psi_mean",
                out_base=figures_dir / "rolling_drift_psi",
                formats=cfg.viz.save_formats,
            )
            plot_series_with_band(
                drift_summary["open_time_ms"].to_numpy(),
                drift_summary["ks"].to_numpy(),
                window=cfg.viz.moving_window,
                title="Rolling Drift KS (mean)",
                xlabel="open_time_ms",
                ylabel="KS",
                label="ks_mean",
                out_base=figures_dir / "rolling_drift_ks",
                formats=cfg.viz.save_formats,
            )
            latest = drift_df[drift_df["open_time_ms"] == drift_summary["open_time_ms"].iloc[-1]]
            if not latest.empty:
                plot_bar(
                    latest["feature"].tolist(),
                    latest["psi"].tolist(),
                    title="Latest Window PSI Drift",
                    xlabel="Feature",
                    ylabel="PSI",
                    out_base=figures_dir / "drift_psi_latest",
                    formats=cfg.viz.save_formats,
                )
                plot_bar(
                    latest["feature"].tolist(),
                    latest["ks"].tolist(),
                    title="Latest Window KS Drift",
                    xlabel="Feature",
                    ylabel="KS",
                    out_base=figures_dir / "drift_ks_latest",
                    formats=cfg.viz.save_formats,
                )

    alert = {
        "triggered": False,
        "actions": [],
        "drift_score": drift_score,
        "drift_score_max": drift_score_max,
        "drift_ks_score": drift_ks_score,
        "drift_ks_score_max": drift_ks_score_max,
    }
    if cfg.adaptation is not None:
        if drift_score is not None:
            metrics["drift_score"] = drift_score
        good, failures = assess_adaptation(metrics, cfg.adaptation)
        alert["adaptation_good"] = good
        alert["failures"] = failures
        if not good:
            alert["triggered"] = True
            alert["actions"].append("rescan_thresholds")
        if cfg.adaptation.max_drift_score is not None and drift_score is not None:
            if drift_score > cfg.adaptation.max_drift_score:
                alert["triggered"] = True
                alert["actions"].extend(["switch_strategy", "retune_params"])

    (report_dir / "alert.json").write_text(json.dumps(alert, indent=2), encoding="utf-8")
    (report_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Monitoring Summary",
                f"Window: {window}",
                f"Samples: {len(decision_log)}",
                f"Accuracy (non-hold): {metrics['accuracy_non_hold']:.4f}",
                f"Hold rate: {metrics['hold_rate']:.4f}",
                f"Conflict rate: {metrics['conflict_rate']:.4f}",
                f"Drift PSI (latest mean): {drift_score if drift_score is not None else 'n/a'}",
                f"Drift KS (latest mean): {drift_ks_score if drift_ks_score is not None else 'n/a'}",
                f"Drift KS (max mean): {drift_ks_score_max if drift_ks_score_max is not None else 'n/a'}",
                f"Drift PSI (max mean): {drift_score_max if drift_score_max is not None else 'n/a'}",
                f"Alert triggered: {alert['triggered']}",
                f"Actions: {', '.join(alert['actions']) if alert['actions'] else 'none'}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--decision-log", default=None)
    parser.add_argument("--window", type=int, default=200)
    args = parser.parse_args()
    run_monitoring(
        config_path=args.config,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        decision_log_path=Path(args.decision_log) if args.decision_log else None,
        window=args.window,
    )
