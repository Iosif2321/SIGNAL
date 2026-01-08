"""Test 3: Supervised baseline training for UP/DOWN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from cryptomvp.config import load_config
from cryptomvp.data.build_dataset import build_synthetic_dataset
from cryptomvp.data.features import compute_features
from cryptomvp.data.labels import make_up_down_labels
from cryptomvp.data.windowing import make_windows
from cryptomvp.decision.rule import batch_decide, hold_rate, scan_thresholds
from cryptomvp.models.down_model import DownModel
from cryptomvp.models.up_model import UpModel
from cryptomvp.train.eval_supervised import evaluate_supervised
from cryptomvp.train.feature_importance import compute_feature_importance
from cryptomvp.train.supervised import train_supervised
from cryptomvp.utils.io import reports_dir
from cryptomvp.utils.logging import get_logger
from cryptomvp.utils.run_dir import init_run_dir
from cryptomvp.utils.seed import set_seed
from cryptomvp.viz.plotting import (
    plot_bar,
    plot_confusion_matrix,
    plot_histogram,
    plot_series_with_band,
    plot_threshold_scan,
)


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _build_feature_columns(feature_cols: list[str], window_size: int) -> list[str]:
    cols = []
    for lag in range(window_size):
        lag_idx = window_size - 1 - lag
        for feature in feature_cols:
            cols.append(f"{feature}_t-{lag_idx}")
    return cols


def _save_decision_log(
    out_path: Path,
    times: np.ndarray,
    X_window: np.ndarray,
    feature_cols: list[str],
    window_size: int,
    probs: np.ndarray,
    preds: np.ndarray,
    y_true: np.ndarray,
) -> None:
    X_flat = X_window.reshape(len(X_window), -1)
    columns = _build_feature_columns(feature_cols, window_size)
    df = pd.DataFrame(X_flat, columns=columns)
    df.insert(0, "open_time_ms", times)
    df["prob"] = probs.astype(float)
    df["pred"] = preds.astype(int)
    df["true"] = y_true.astype(int)
    df["y_prob"] = probs.astype(float)
    df["y_pred"] = preds.astype(int)
    df["y_true"] = y_true.astype(int)
    df["correct"] = (df["pred"] == df["true"]).astype(int)
    df.to_parquet(out_path, index=False)


def _save_training_metrics(out_path: Path, history) -> None:
    df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.train_loss) + 1),
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_acc": history.train_acc,
            "val_acc": history.val_acc,
            "weight_norm": history.weight_norms,
            "delta_weight_norm": history.delta_weight_norms,
        }
    )
    df.to_csv(out_path, index=False)


def run_supervised(config_path: str, fast: bool, run_dir: Path | None = None) -> None:
    init_run_dir(run_dir, config_path)
    cfg = load_config(config_path)
    logger = get_logger("supervised")
    set_seed(cfg.seed)

    if fast:
        interval_ms = int(cfg.interval) * 60_000
        df = build_synthetic_dataset(0, interval_ms * 800, seed=17, interval_ms=interval_ms)
    else:
        df = _load_dataset(Path(cfg.dataset.output_path))
    start_ms = int(df["open_time_ms"].min())
    end_ms = int(df["open_time_ms"].max())

    features = compute_features(df, cfg.features.list_of_features)
    X, window_times, feature_cols = make_windows(features, cfg.features.window_size_K)
    y_up, y_down = make_up_down_labels(df, window_times)

    n = min(len(X), len(y_up))
    X = X[:n]
    y_up = y_up[:n]
    y_down = y_down[:n]
    window_times = window_times[:n]

    X_flat = X.reshape(len(X), -1)

    epochs = 2 if fast else cfg.supervised.epochs
    batch_size = 32 if fast else cfg.supervised.batch_size

    n_total = len(X_flat)
    train_end = int(n_total * 0.7)
    val_end = int(n_total * 0.85)

    y_up_test = y_up[val_end:]
    y_down_test = y_down[val_end:]
    up_rate = float(np.mean(y_up_test))
    down_rate = float(np.mean(y_down_test))
    flat_rate = max(0.0, 1.0 - up_rate - down_rate)
    Xtr, ytr = X_flat[:train_end], y_up[:train_end]
    Xv, yv = X_flat[train_end:val_end], y_up[train_end:val_end]
    Xte, yte = X_flat[val_end:], y_up_test
    times_te = window_times[val_end:]
    X_window_te = X[val_end:]

    # UP model
    up_model = UpModel(input_dim=Xtr.shape[1])
    up_hist = train_supervised(
        up_model,
        Xtr,
        ytr,
        Xv,
        yv,
        epochs=epochs,
        batch_size=batch_size,
        lr=cfg.supervised.lr,
        patience=cfg.supervised.early_stopping_patience,
        model_name="baseline_up",
        track_weights=True,
    )
    up_eval = evaluate_supervised(up_model, Xte, yte)

    up_report_dir = reports_dir("supervised_up")
    up_fig_dir = up_report_dir / "figures"
    up_fig_dir.mkdir(parents=True, exist_ok=True)

    epoch_idx = np.arange(1, len(up_hist.train_loss) + 1)
    plot_series_with_band(
        epoch_idx,
        up_hist.train_loss,
        window=cfg.viz.moving_window,
        title="UP Train Loss",
        xlabel="Epoch",
        ylabel="Loss",
        label="train_loss",
        out_base=up_fig_dir / "train_loss",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        epoch_idx,
        up_hist.val_loss,
        window=cfg.viz.moving_window,
        title="UP Val Loss",
        xlabel="Epoch",
        ylabel="Loss",
        label="val_loss",
        out_base=up_fig_dir / "val_loss",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        epoch_idx,
        up_hist.val_acc,
        window=cfg.viz.moving_window,
        title="UP Val Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        label="val_acc",
        out_base=up_fig_dir / "val_accuracy",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        epoch_idx,
        up_hist.weight_norms,
        window=cfg.viz.moving_window,
        title="UP Weight Norm",
        xlabel="Epoch",
        ylabel="Weight norm",
        label="weight_norm",
        out_base=up_fig_dir / "weight_norm",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        epoch_idx,
        up_hist.delta_weight_norms,
        window=cfg.viz.moving_window,
        title="UP Delta Weight Norm",
        xlabel="Epoch",
        ylabel="Delta norm",
        label="delta_weight_norm",
        out_base=up_fig_dir / "delta_weight_norm",
        formats=cfg.viz.save_formats,
    )
    plot_confusion_matrix(
        up_eval["confusion_matrix"],
        labels=["0", "1"],
        title="UP Confusion Matrix",
        out_base=up_fig_dir / "confusion_matrix",
        formats=cfg.viz.save_formats,
    )
    plot_histogram(
        up_eval["probs"][:, 1],
        bins=30,
        title="UP Probabilities",
        xlabel="P(up)",
        ylabel="Count",
        out_base=up_fig_dir / "prob_hist",
        formats=cfg.viz.save_formats,
    )

    (up_report_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Supervised UP",
                f"Symbol: {cfg.symbol}",
                f"Interval: {cfg.interval}",
                f"Seed: {cfg.seed}",
                f"Start ms: {start_ms}",
                f"End ms: {end_ms}",
                f"Test samples: {len(y_up_test)}",
                f"Class rates (UP/DOWN/FLAT): {up_rate:.4f}/{down_rate:.4f}/{flat_rate:.4f}",
                f"Threshold: {cfg.decision_rule.T_min}",
                f"Accuracy: {up_eval['accuracy']:.4f}",
                f"F1: {up_eval['f1']:.4f}",
                f"Final weight_norm: {up_hist.weight_norms[-1]:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(up_hist.layer_weight_norms).to_csv(
        up_report_dir / "layer_weight_norms.csv", index=False
    )
    _save_training_metrics(up_report_dir / "training_metrics.csv", up_hist)
    _save_decision_log(
        up_report_dir / "decision_log.parquet",
        times=times_te,
        X_window=X_window_te,
        feature_cols=feature_cols,
        window_size=cfg.features.window_size_K,
        probs=up_eval["probs"][:, 1],
        preds=up_eval["preds"],
        y_true=yte,
    )
    up_imp, up_imp_agg = compute_feature_importance(
        up_model, feature_cols, cfg.features.window_size_K
    )
    up_imp.to_csv(up_report_dir / "feature_importance.csv", index=False)
    up_imp_agg.to_csv(up_report_dir / "feature_importance_by_feature.csv", index=False)
    plot_bar(
        up_imp_agg["feature"].tolist(),
        up_imp_agg["importance"].tolist(),
        title="UP Feature Importance (mean abs weight)",
        xlabel="Feature",
        ylabel="Importance",
        out_base=up_fig_dir / "feature_importance",
        formats=cfg.viz.save_formats,
    )

    # DOWN model
    Xtr, ytr = X_flat[:train_end], y_down[:train_end]
    Xv, yv = X_flat[train_end:val_end], y_down[train_end:val_end]
    Xte, yte = X_flat[val_end:], y_down_test
    down_model = DownModel(input_dim=Xtr.shape[1])
    down_hist = train_supervised(
        down_model,
        Xtr,
        ytr,
        Xv,
        yv,
        epochs=epochs,
        batch_size=batch_size,
        lr=cfg.supervised.lr,
        patience=cfg.supervised.early_stopping_patience,
        model_name="baseline_down",
        track_weights=True,
    )
    down_eval = evaluate_supervised(down_model, Xte, yte)

    down_report_dir = reports_dir("supervised_down")
    down_fig_dir = down_report_dir / "figures"
    down_fig_dir.mkdir(parents=True, exist_ok=True)

    epoch_idx = np.arange(1, len(down_hist.train_loss) + 1)
    plot_series_with_band(
        epoch_idx,
        down_hist.train_loss,
        window=cfg.viz.moving_window,
        title="DOWN Train Loss",
        xlabel="Epoch",
        ylabel="Loss",
        label="train_loss",
        out_base=down_fig_dir / "train_loss",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        epoch_idx,
        down_hist.val_loss,
        window=cfg.viz.moving_window,
        title="DOWN Val Loss",
        xlabel="Epoch",
        ylabel="Loss",
        label="val_loss",
        out_base=down_fig_dir / "val_loss",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        epoch_idx,
        down_hist.val_acc,
        window=cfg.viz.moving_window,
        title="DOWN Val Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        label="val_acc",
        out_base=down_fig_dir / "val_accuracy",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        epoch_idx,
        down_hist.weight_norms,
        window=cfg.viz.moving_window,
        title="DOWN Weight Norm",
        xlabel="Epoch",
        ylabel="Weight norm",
        label="weight_norm",
        out_base=down_fig_dir / "weight_norm",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        epoch_idx,
        down_hist.delta_weight_norms,
        window=cfg.viz.moving_window,
        title="DOWN Delta Weight Norm",
        xlabel="Epoch",
        ylabel="Delta norm",
        label="delta_weight_norm",
        out_base=down_fig_dir / "delta_weight_norm",
        formats=cfg.viz.save_formats,
    )
    plot_confusion_matrix(
        down_eval["confusion_matrix"],
        labels=["0", "1"],
        title="DOWN Confusion Matrix",
        out_base=down_fig_dir / "confusion_matrix",
        formats=cfg.viz.save_formats,
    )
    plot_histogram(
        down_eval["probs"][:, 1],
        bins=30,
        title="DOWN Probabilities",
        xlabel="P(down)",
        ylabel="Count",
        out_base=down_fig_dir / "prob_hist",
        formats=cfg.viz.save_formats,
    )

    (down_report_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Supervised DOWN",
                f"Symbol: {cfg.symbol}",
                f"Interval: {cfg.interval}",
                f"Seed: {cfg.seed}",
                f"Start ms: {start_ms}",
                f"End ms: {end_ms}",
                f"Test samples: {len(y_down_test)}",
                f"Class rates (UP/DOWN/FLAT): {up_rate:.4f}/{down_rate:.4f}/{flat_rate:.4f}",
                f"Threshold: {cfg.decision_rule.T_min}",
                f"Accuracy: {down_eval['accuracy']:.4f}",
                f"F1: {down_eval['f1']:.4f}",
                f"Final weight_norm: {down_hist.weight_norms[-1]:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(down_hist.layer_weight_norms).to_csv(
        down_report_dir / "layer_weight_norms.csv", index=False
    )
    _save_training_metrics(down_report_dir / "training_metrics.csv", down_hist)
    _save_decision_log(
        down_report_dir / "decision_log.parquet",
        times=times_te,
        X_window=X_window_te,
        feature_cols=feature_cols,
        window_size=cfg.features.window_size_K,
        probs=down_eval["probs"][:, 1],
        preds=down_eval["preds"],
        y_true=yte,
    )
    down_imp, down_imp_agg = compute_feature_importance(
        down_model, feature_cols, cfg.features.window_size_K
    )
    down_imp.to_csv(down_report_dir / "feature_importance.csv", index=False)
    down_imp_agg.to_csv(down_report_dir / "feature_importance_by_feature.csv", index=False)
    plot_bar(
        down_imp_agg["feature"].tolist(),
        down_imp_agg["importance"].tolist(),
        title="DOWN Feature Importance (mean abs weight)",
        xlabel="Feature",
        ylabel="Importance",
        out_base=down_fig_dir / "feature_importance",
        formats=cfg.viz.save_formats,
    )

    # Decision rule scan on shared test segment
    n_test = min(len(up_eval["probs"]), len(down_eval["probs"]))
    p_up = up_eval["probs"][:n_test, 1]
    p_down = down_eval["probs"][:n_test, 1]
    thresholds = np.arange(
        cfg.decision_rule.scan_min,
        cfg.decision_rule.scan_max + cfg.decision_rule.scan_step / 2,
        cfg.decision_rule.scan_step,
    )
    hold_rates = scan_thresholds(p_up, p_down, thresholds)
    default_decisions = batch_decide(p_up, p_down, cfg.decision_rule.T_min)
    default_hold_rate = hold_rate(default_decisions)
    conflict_rate = float(np.mean((p_up >= cfg.decision_rule.T_min) & (p_down >= cfg.decision_rule.T_min)))
    rule_dir = reports_dir("decision_rule")
    rule_fig_dir = rule_dir / "figures"
    rule_fig_dir.mkdir(parents=True, exist_ok=True)
    plot_threshold_scan(
        thresholds,
        hold_rates,
        window=cfg.viz.moving_window,
        title="Hold Rate vs Threshold",
        out_base=rule_fig_dir / "hold_rate_threshold",
        formats=cfg.viz.save_formats,
    )

    decisions = batch_decide(p_up, p_down, cfg.decision_rule.T_min)
    true_dir = []
    for up_val, down_val in zip(y_up_test[:n_test], y_down_test[:n_test]):
        if up_val == 1:
            true_dir.append("UP")
        elif down_val == 1:
            true_dir.append("DOWN")
        else:
            true_dir.append("FLAT")
    correct_dir = [int(d == t) if d != "HOLD" else 0 for d, t in zip(decisions, true_dir)]
    action_mask = np.array([d != "HOLD" for d in decisions])
    action_rate = float(np.mean(action_mask))
    action_accuracy = float(np.mean([c for c, d in zip(correct_dir, decisions) if d != "HOLD"])) if action_mask.any() else 0.0
    precision_up = float(
        np.mean([t == "UP" for t, d in zip(true_dir, decisions) if d == "UP"])
    ) if any(d == "UP" for d in decisions) else 0.0
    precision_down = float(
        np.mean([t == "DOWN" for t, d in zip(true_dir, decisions) if d == "DOWN"])
    ) if any(d == "DOWN" for d in decisions) else 0.0

    accuracy_by_threshold = []
    conflict_rates = []
    for t in thresholds:
        d_list = batch_decide(p_up, p_down, float(t))
        mask = np.array([d != "HOLD" for d in d_list])
        conflict_rates.append(float(np.mean((p_up >= t) & (p_down >= t))))
        if mask.any():
            accuracy_by_threshold.append(
                float(np.mean([d == tr for d, tr in zip(d_list, true_dir) if d != "HOLD"]))
            )
        else:
            accuracy_by_threshold.append(0.0)

    plot_threshold_scan(
        thresholds,
        accuracy_by_threshold,
        window=cfg.viz.moving_window,
        title="Action Accuracy vs Threshold",
        out_base=rule_fig_dir / "accuracy_threshold",
        formats=cfg.viz.save_formats,
        ylabel="Accuracy (non-hold)",
        label="accuracy_non_hold",
    )

    plot_threshold_scan(
        thresholds,
        conflict_rates,
        window=cfg.viz.moving_window,
        title="Conflict Rate vs Threshold",
        out_base=rule_fig_dir / "conflict_rate_threshold",
        formats=cfg.viz.save_formats,
        ylabel="Conflict rate",
        label="conflict_rate",
    )

    threshold_scan_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "hold_rate": hold_rates,
            "action_accuracy_non_hold": accuracy_by_threshold,
            "conflict_rate": conflict_rates,
        }
    )
    threshold_scan_df.to_csv(rule_dir / "threshold_scan.csv", index=False)

    (rule_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Decision Rule",
                f"Symbol: {cfg.symbol}",
                f"Interval: {cfg.interval}",
                f"Seed: {cfg.seed}",
                f"Samples: {n_test}",
                f"Class rates (UP/DOWN/FLAT): {up_rate:.4f}/{down_rate:.4f}/{flat_rate:.4f}",
                f"Default T_min: {cfg.decision_rule.T_min}",
                f"Default hold_rate: {default_hold_rate:.4f}",
                f"Action rate: {action_rate:.4f}",
                f"Action accuracy (non-hold): {action_accuracy:.4f}",
                f"Conflict rate: {conflict_rate:.4f}",
                f"Precision UP (system): {precision_up:.4f}",
                f"Precision DOWN (system): {precision_down:.4f}",
                f"Threshold scan rows: {len(thresholds)}",
            ]
        ),
        encoding="utf-8",
    )
    decision_log = pd.DataFrame(
        {
            "open_time_ms": times_te[:n_test],
            "p_up": p_up.astype(float),
            "p_down": p_down.astype(float),
            "p_max": np.maximum(p_up, p_down).astype(float),
            "decision": decisions,
            "true_direction": true_dir,
            "correct_direction": correct_dir,
            "conflict": ((p_up >= cfg.decision_rule.T_min) & (p_down >= cfg.decision_rule.T_min)).astype(int),
        }
    )
    for idx, feature in enumerate(feature_cols):
        decision_log[f"{feature}_t0"] = X_window_te[:n_test, -1, idx]
    decision_log.to_parquet(rule_dir / "decision_log.parquet", index=False)

    logger.info("Supervised training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_supervised(args.config, fast=args.fast, run_dir=Path(args.run_dir) if args.run_dir else None)
