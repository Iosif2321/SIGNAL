"""Fixed-period evaluation with session-aware reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from cryptomvp.bybit.rest import BybitRestClient
from cryptomvp.config import load_config
from cryptomvp.data.build_dataset import fetch_klines_range, klines_to_dataframe, save_dataset
from cryptomvp.data.features import compute_features
from cryptomvp.data.labels import make_up_down_labels
from cryptomvp.data.scaling import apply_standard_scaler, fit_standard_scaler
from cryptomvp.data.windowing import make_windows
from cryptomvp.decision.rule import batch_decide
from cryptomvp.evaluate.metrics import compute_metrics, compute_metrics_by_session
from cryptomvp.models.down_model import DownModel
from cryptomvp.models.up_model import UpModel
from cryptomvp.features.registry import resolve_feature_list
from cryptomvp.sessions import SessionRouter, apply_session_overrides, assign_session_features
from cryptomvp.utils.io import data_dir, ensure_dir, reports_dir, run_root, sessions_dir
from cryptomvp.utils.run_dir import init_run_dir
from cryptomvp.utils.seed import set_seed
from cryptomvp.utils.gpu import resolve_device
from cryptomvp.viz.plotting import (
    plot_confusion_matrix,
    plot_reliability_curve,
    plot_series_with_band,
)


def _parse_date(value: str) -> int:
    ts = pd.Timestamp(value, tz="UTC")
    return int(ts.value // 1_000_000)


def _load_or_build_dataset(
    base_cfg: Dict[str, Any],
    start_ms: int,
    end_ms: int,
    interval: str,
) -> pd.DataFrame:
    output_path = Path(base_cfg["dataset"]["output_path"])
    if output_path.exists():
        df = pd.read_parquet(output_path) if output_path.suffix == ".parquet" else pd.read_csv(output_path)
        return df[(df["open_time_ms"] >= start_ms) & (df["open_time_ms"] <= end_ms)].reset_index(drop=True)

    client = BybitRestClient()
    try:
        candles = fetch_klines_range(
            client=client,
            category=base_cfg["category"],
            symbol=base_cfg["symbol"],
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=int(base_cfg["dataset"]["limit_per_call"]),
        )
    finally:
        client.close()

    df = klines_to_dataframe(candles)
    run_path = data_dir("processed")
    out_name = f"{base_cfg['symbol'].lower()}_{base_cfg['category']}_{interval}m_{start_ms}_{end_ms}.parquet"
    save_dataset(df, run_path / out_name)
    return df


def _time_split(n: int) -> Tuple[int, int]:
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return train_end, val_end


def _get_cfg_value(data: Dict[str, Any], dotted_key: str, default: Any) -> Any:
    node = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _resolve_thresholds(cfg_dict: Dict[str, Any]) -> Tuple[float, float]:
    t_min = float(_get_cfg_value(cfg_dict, "decision_rule.T_min", 0.5))
    delta_min = float(_get_cfg_value(cfg_dict, "decision_rule.delta_min", 0.0))
    return t_min, delta_min


def _train_and_predict(
    X: np.ndarray,
    y_up: np.ndarray,
    y_down: np.ndarray,
    cfg_dict: Dict[str, Any],
    fast: bool,
    device: "torch.device",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    train_end, val_end = _time_split(n)
    epochs = 2 if fast else int(_get_cfg_value(cfg_dict, "supervised.epochs", 50))
    batch_size = 32 if fast else int(_get_cfg_value(cfg_dict, "supervised.batch_size", 64))
    lr = float(_get_cfg_value(cfg_dict, "supervised.lr", 0.001))
    weight_decay = float(_get_cfg_value(cfg_dict, "supervised.weight_decay", 0.0))
    patience = int(_get_cfg_value(cfg_dict, "supervised.early_stopping_patience", 10))
    hidden_dim = int(_get_cfg_value(cfg_dict, "supervised.hidden_dim", 64))

    X_flat = X.reshape(len(X), -1)
    scaler_mean, scaler_std = fit_standard_scaler(X_flat[:train_end])
    X_scaled = apply_standard_scaler(X_flat, scaler_mean, scaler_std)

    Xtr, Xv, Xte = X_scaled[:train_end], X_scaled[train_end:val_end], X_scaled[val_end:]
    y_up_tr, y_up_v = y_up[:train_end], y_up[train_end:val_end]
    y_down_tr, y_down_v = y_down[:train_end], y_down[train_end:val_end]

    up_model = UpModel(input_dim=Xtr.shape[1], hidden_dim=hidden_dim)
    down_model = DownModel(input_dim=Xtr.shape[1], hidden_dim=hidden_dim)

    from cryptomvp.train.supervised import train_supervised
    from cryptomvp.train.eval_supervised import evaluate_supervised

    train_supervised(
        up_model,
        Xtr,
        y_up_tr,
        Xv,
        y_up_v,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        weight_decay=weight_decay,
        model_name="eval_up",
        device=device,
    )
    train_supervised(
        down_model,
        Xtr,
        y_down_tr,
        Xv,
        y_down_v,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        weight_decay=weight_decay,
        model_name="eval_down",
        device=device,
    )

    up_eval = evaluate_supervised(up_model, Xte, y_up[val_end:], device=device)
    down_eval = evaluate_supervised(down_model, Xte, y_down[val_end:], device=device)
    return (
        up_eval["probs"][:, 1],
        down_eval["probs"][:, 1],
        y_up[val_end:],
        y_down[val_end:],
        scaler_mean,
        scaler_std,
    )


def _build_decision_log(
    times: np.ndarray,
    p_up: np.ndarray,
    p_down: np.ndarray,
    y_up: np.ndarray,
    y_down: np.ndarray,
    t_min: float,
    delta_min: float,
    session_id: np.ndarray | None,
) -> pd.DataFrame:
    true_dir = np.where(y_up == 1, "UP", np.where(y_down == 1, "DOWN", "FLAT"))
    decisions = np.array(batch_decide(p_up, p_down, t_min, delta_min=delta_min))
    conflict = (p_up >= t_min) & (p_down >= t_min)
    df = pd.DataFrame(
        {
            "open_time_ms": times,
            "p_up": p_up.astype(float),
            "p_down": p_down.astype(float),
            "decision": decisions,
            "y_true": true_dir,
            "conflict": conflict.astype(int),
        }
    )
    if session_id is not None:
        df["session_id"] = session_id
    return df


def _write_metrics_artifacts(
    report_dir: Path,
    metrics: Dict[str, object],
    metrics_by_session: Dict[str, Dict[str, object]],
    figures_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (report_dir / "metrics_by_session.json").write_text(
        json.dumps(metrics_by_session, indent=2), encoding="utf-8"
    )
    rows: List[Dict[str, object]] = []
    for key, value in metrics.items():
        if key in {"confusion_matrix", "coverage_bins"}:
            continue
        rows.append({"scope": "overall", "session_id": "ALL", "metric": key, "value": value})
    for session_id, sess_metrics in metrics_by_session.items():
        for key, value in sess_metrics.items():
            if key in {"confusion_matrix", "coverage_bins"}:
                continue
            rows.append(
                {"scope": "session", "session_id": session_id, "metric": key, "value": value}
            )
    pd.DataFrame(rows).to_csv(report_dir / "metrics.csv", index=False)

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        labels=["UP", "DOWN"],
        title="Confusion Matrix (non-hold, UP/DOWN)",
        out_base=figures_dir / "confusion_matrix",
        formats=["png", "svg"],
    )

    if "coverage_bins" in metrics:
        cov_df = pd.DataFrame(metrics["coverage_bins"])
        plot_series_with_band(
            cov_df["bin_start"].to_numpy(),
            cov_df["action_rate"].to_numpy(),
            window=1,
            title="Action Rate by Confidence Bin",
            xlabel="Confidence bin start",
            ylabel="Action rate",
            label="action_rate",
            out_base=figures_dir / "coverage_action_rate",
            formats=["png", "svg"],
        )
        plot_series_with_band(
            cov_df["bin_start"].to_numpy(),
            cov_df["accuracy_non_hold"].to_numpy(),
            window=1,
            title="Accuracy (non-hold) by Confidence Bin",
            xlabel="Confidence bin start",
            ylabel="Accuracy",
            label="accuracy_non_hold",
            out_base=figures_dir / "coverage_accuracy",
            formats=["png", "svg"],
        )

    rel_up = metrics.get("reliability_up", [])
    if rel_up:
        rel_df = pd.DataFrame(rel_up)
        plot_reliability_curve(
            rel_df["mean_prob"].to_numpy(),
            rel_df["accuracy"].to_numpy(),
            title="Reliability (UP)",
            xlabel="Mean predicted probability",
            ylabel="Empirical accuracy",
            out_base=figures_dir / "reliability_up",
            formats=["png", "svg"],
            label="up",
        )
    rel_down = metrics.get("reliability_down", [])
    if rel_down:
        rel_df = pd.DataFrame(rel_down)
        plot_reliability_curve(
            rel_df["mean_prob"].to_numpy(),
            rel_df["accuracy"].to_numpy(),
            title="Reliability (DOWN)",
            xlabel="Mean predicted probability",
            ylabel="Empirical accuracy",
            out_base=figures_dir / "reliability_down",
            formats=["png", "svg"],
            label="down",
        )


def run_fixed_period_eval(
    config_path: str,
    start: str | None,
    end: str | None,
    session_mode: str | None,
    session_strategy: str | None,
    fast: bool,
    run_dir: Path | None,
) -> Path:
    run_path = init_run_dir(run_dir, config_path)
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device, cfg.allow_cpu_fallback)

    base_cfg = yaml.safe_load(Path(config_path).read_text())
    start_ms = _parse_date(start) if start else int(cfg.dataset.start_ms or 0)
    end_ms = _parse_date(end) if end else int(cfg.dataset.end_ms or 0)
    interval = str(base_cfg.get("interval", cfg.interval))

    df = _load_or_build_dataset(base_cfg, start_ms, end_ms, interval)
    router = SessionRouter(
        mode=session_mode or (cfg.session.mode if cfg.session else "fixed_utc_partitions"),
        overlap_policy=(cfg.session.overlap_policy if cfg.session else "priority"),
        priority_order=(cfg.session.priority_order if cfg.session else None),
        sessions=None,
    )
    df = assign_session_features(df, router)

    strategy = session_strategy or (cfg.session.strategy if cfg.session else "experts_per_session")

    report_dir = reports_dir("fixed_period")
    figures_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    decision_logs: List[pd.DataFrame] = []

    if strategy == "experts_per_session":
        for session_id in sorted(set(df["session_id"].tolist())):
            sess_cfg = apply_session_overrides(base_cfg, cfg.session, session_id)
            feature_sets_path = sess_cfg["features"].get("feature_sets_path")
            feature_list = resolve_feature_list(
                sess_cfg["features"]["list_of_features"],
                sess_cfg["features"].get("feature_set_id"),
                feature_sets_path=Path(feature_sets_path) if feature_sets_path else None,
            )
            df_sess = df[df["session_id"] == session_id].reset_index(drop=True)
            features = compute_features(df_sess, feature_list)
            X, window_times, _feature_cols = make_windows(
                features, int(sess_cfg["features"]["window_size_K"])
            )
            y_up, y_down = make_up_down_labels(df_sess, window_times)
            n = min(len(X), len(y_up))
            X = X[:n]
            y_up = y_up[:n]
            y_down = y_down[:n]
            window_times = window_times[:n]
            if len(X) < 100:
                continue
            p_up, p_down, y_up_t, y_down_t, _, _ = _train_and_predict(
                X, y_up, y_down, sess_cfg, fast=fast, device=device
            )
            t_min, delta_min = _resolve_thresholds(sess_cfg)
            times = window_times[-len(p_up) :]
            sess_id = np.array([session_id] * len(p_up))
            log = _build_decision_log(
                times, p_up, p_down, y_up_t, y_down_t, t_min, delta_min, sess_id
            )
            decision_logs.append(log)

            sess_root = sessions_dir(session_id)
            ensure_dir(sess_root / "reports")
            log.to_parquet(sess_root / "decisions.parquet", index=False)

    else:
        sess_cfg = base_cfg
        feature_sets_path = sess_cfg["features"].get("feature_sets_path")
        feature_list = resolve_feature_list(
            sess_cfg["features"]["list_of_features"],
            sess_cfg["features"].get("feature_set_id"),
            feature_sets_path=Path(feature_sets_path) if feature_sets_path else None,
        )
        features = compute_features(df, feature_list)
        X, window_times, _feature_cols = make_windows(
            features, int(sess_cfg["features"]["window_size_K"])
        )
        y_up, y_down = make_up_down_labels(df, window_times)
        n = min(len(X), len(y_up))
        X = X[:n]
        y_up = y_up[:n]
        y_down = y_down[:n]
        window_times = window_times[:n]
        session_ids = df.set_index("open_time_ms").loc[window_times, "session_id"].to_numpy()
        p_up, p_down, y_up_t, y_down_t, _, _ = _train_and_predict(
            X, y_up, y_down, sess_cfg, fast=fast, device=device
        )
        # align test portion
        test_times = window_times[len(window_times) - len(p_up) :]
        test_sessions = session_ids[len(session_ids) - len(p_up) :]
        logs = []
        for session_id in sorted(set(test_sessions)):
            sess_mask = test_sessions == session_id
            sess_cfg = apply_session_overrides(base_cfg, cfg.session, session_id)
            t_min, delta_min = _resolve_thresholds(sess_cfg)
            log = _build_decision_log(
                test_times[sess_mask],
                p_up[sess_mask],
                p_down[sess_mask],
                y_up_t[sess_mask],
                y_down_t[sess_mask],
                t_min,
                delta_min,
                np.array([session_id] * int(np.sum(sess_mask))),
            )
            logs.append(log)
            sess_root = sessions_dir(session_id)
            ensure_dir(sess_root / "reports")
            log.to_parquet(sess_root / "decisions.parquet", index=False)
        decision_logs = logs

    full_log = pd.concat(decision_logs, ignore_index=True)
    metrics = compute_metrics(full_log, bootstrap_samples=200, seed=cfg.seed)
    metrics_by_session = compute_metrics_by_session(full_log, bootstrap_samples=200, seed=cfg.seed)
    _write_metrics_artifacts(report_dir, metrics, metrics_by_session, figures_dir)

    full_log.to_parquet(report_dir / "decisions.parquet", index=False)
    (report_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Fixed Period Evaluation",
                f"Symbol: {cfg.symbol}",
                f"Interval: {cfg.interval}",
                f"Start ms: {start_ms}",
                f"End ms: {end_ms}",
                f"Session mode: {router.mode}",
                f"Session strategy: {strategy}",
                f"Samples: {len(full_log)}",
                f"Accuracy (non-hold): {metrics['accuracy_non_hold']:.4f}",
                f"Hold rate: {metrics['hold_rate']:.4f}",
                f"Conflict rate: {metrics['conflict_rate']:.4f}",
            ]
        ),
        encoding="utf-8",
    )

    for session_id, sess_metrics in metrics_by_session.items():
        sess_root = sessions_dir(session_id)
        sess_report = ensure_dir(sess_root / "reports" / "fixed_period")
        (sess_report / "metrics.json").write_text(
            json.dumps(sess_metrics, indent=2), encoding="utf-8"
        )

    return run_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--session-mode", default=None)
    parser.add_argument("--session-strategy", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_fixed_period_eval(
        config_path=args.config,
        start=args.start,
        end=args.end,
        session_mode=args.session_mode,
        session_strategy=args.session_strategy,
        fast=args.fast,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )


if __name__ == "__main__":
    main()
