"""Walk-forward evaluation harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from cryptomvp.config import load_config
from cryptomvp.data.features import compute_features, compute_regime_labels
from cryptomvp.data.labels import make_up_down_labels
from cryptomvp.data.scaling import apply_standard_scaler, fit_standard_scaler
from cryptomvp.data.windowing import make_windows
from cryptomvp.decision.rule import batch_decide
from cryptomvp.evaluate.metrics import (
    compute_metrics,
    compute_metrics_by_regime,
    compute_metrics_by_session,
)
from cryptomvp.models.down_model import DownModel
from cryptomvp.models.up_model import UpModel
from cryptomvp.features.registry import resolve_feature_list
from cryptomvp.sessions import SessionRouter, apply_session_overrides, assign_session_features
from cryptomvp.utils.io import ensure_dir, reports_dir, run_root, sessions_dir
from cryptomvp.utils.run_dir import init_run_dir
from cryptomvp.utils.seed import set_seed
from cryptomvp.utils.gpu import resolve_device
from cryptomvp.viz.plotting import plot_series_with_band


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


def _train_models(
    X_train: np.ndarray,
    y_up_train: np.ndarray,
    y_down_train: np.ndarray,
    X_val: np.ndarray,
    y_up_val: np.ndarray,
    y_down_val: np.ndarray,
    cfg_dict: Dict[str, Any],
    fast: bool,
    device: "torch.device",
) -> Tuple[UpModel, DownModel]:
    epochs = 2 if fast else int(_get_cfg_value(cfg_dict, "supervised.epochs", 50))
    batch_size = 32 if fast else int(_get_cfg_value(cfg_dict, "supervised.batch_size", 64))
    lr = float(_get_cfg_value(cfg_dict, "supervised.lr", 0.001))
    weight_decay = float(_get_cfg_value(cfg_dict, "supervised.weight_decay", 0.0))
    patience = int(_get_cfg_value(cfg_dict, "supervised.early_stopping_patience", 10))
    hidden_dim = int(_get_cfg_value(cfg_dict, "supervised.hidden_dim", 64))

    up_model = UpModel(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
    down_model = DownModel(input_dim=X_train.shape[1], hidden_dim=hidden_dim)

    from cryptomvp.train.supervised import train_supervised

    train_supervised(
        up_model,
        X_train,
        y_up_train,
        X_val,
        y_up_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        weight_decay=weight_decay,
        model_name="wf_up",
        device=device,
    )
    train_supervised(
        down_model,
        X_train,
        y_down_train,
        X_val,
        y_down_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        weight_decay=weight_decay,
        model_name="wf_down",
        device=device,
    )
    return up_model, down_model


def _predict_probs(
    model: UpModel | DownModel, X: np.ndarray, device: "torch.device"
) -> np.ndarray:
    from cryptomvp.train.eval_supervised import evaluate_supervised

    eval_out = evaluate_supervised(model, X, np.zeros(len(X), dtype=int), device=device)
    return eval_out["probs"][:, 1]


def _build_decision_log(
    times: np.ndarray,
    p_up: np.ndarray,
    p_down: np.ndarray,
    y_up: np.ndarray,
    y_down: np.ndarray,
    t_min: float,
    delta_min: float,
    session_id: np.ndarray,
    regime: np.ndarray | None = None,
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
            "session_id": session_id,
        }
    )
    if regime is not None:
        df["regime"] = regime
    return df


def _build_folds(
    times: pd.Series,
    train_days: int,
    test_days: int,
    step_days: int,
    expanding: bool,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    start = times.min()
    train_end = start + pd.Timedelta(days=train_days)
    test_end = train_end + pd.Timedelta(days=test_days)
    while test_end <= times.max():
        folds.append((start, train_end, train_end, test_end))
        if expanding:
            train_end += pd.Timedelta(days=step_days)
        else:
            start += pd.Timedelta(days=step_days)
            train_end = start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
    return folds


def run_walk_forward(
    config_path: str,
    train_days: int,
    test_days: int,
    step_days: int,
    expanding: bool,
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

    dataset_path = Path(cfg.dataset.output_path)
    df = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_csv(dataset_path)

    router = SessionRouter(
        mode=session_mode or (cfg.session.mode if cfg.session else "fixed_utc_partitions"),
        overlap_policy=(cfg.session.overlap_policy if cfg.session else "priority"),
        priority_order=(cfg.session.priority_order if cfg.session else None),
        sessions=None,
    )
    df = assign_session_features(df, router)

    strategy = session_strategy or (cfg.session.strategy if cfg.session else "experts_per_session")
    times_all = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    folds = _build_folds(times_all, train_days, test_days, step_days, expanding)

    session_payload: Dict[str, Dict[str, object]] = {}
    global_payload: Dict[str, object] = {}

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
            features_sess = compute_features(df_sess, feature_list)
            X_sess, window_times_sess, _ = make_windows(
                features_sess, int(sess_cfg["features"]["window_size_K"])
            )
            y_up_sess, y_down_sess = make_up_down_labels(df_sess, window_times_sess)
            regime_sess = compute_regime_labels(df_sess, window_times_sess)
            n = min(len(X_sess), len(y_up_sess))
            if n == 0:
                continue
            X_sess = X_sess[:n]
            y_up_sess = y_up_sess[:n]
            y_down_sess = y_down_sess[:n]
            window_times_sess = window_times_sess[:n]
            regime_sess = regime_sess[:n]
            times_sess = pd.to_datetime(window_times_sess, unit="ms", utc=True)
            session_payload[session_id] = {
                "X": X_sess,
                "y_up": y_up_sess,
                "y_down": y_down_sess,
                "times": times_sess,
                "window_times": window_times_sess,
                "regime": regime_sess,
                "cfg": sess_cfg,
            }
    else:
        feature_sets_path = base_cfg["features"].get("feature_sets_path")
        feature_list = resolve_feature_list(
            base_cfg["features"]["list_of_features"],
            base_cfg["features"].get("feature_set_id"),
            feature_sets_path=Path(feature_sets_path) if feature_sets_path else None,
        )
        features = compute_features(df, feature_list)
        X, window_times, _ = make_windows(features, int(base_cfg["features"]["window_size_K"]))
        y_up, y_down = make_up_down_labels(df, window_times)
        regime_labels = compute_regime_labels(df, window_times)
        n = min(len(X), len(y_up))
        X = X[:n]
        y_up = y_up[:n]
        y_down = y_down[:n]
        window_times = window_times[:n]
        regime_labels = regime_labels[:n]
        session_ids = df.set_index("open_time_ms").loc[window_times, "session_id"].to_numpy()
        times = pd.to_datetime(window_times, unit="ms", utc=True)
        global_payload = {
            "X": X,
            "y_up": y_up,
            "y_down": y_down,
            "window_times": window_times,
            "session_ids": session_ids,
            "times": times,
            "regime": regime_labels,
        }

    fold_metrics = []
    fold_session_metrics: List[Dict[str, Dict[str, object]]] = []
    fold_dir = ensure_dir(run_root() / "folds")

    for idx, (train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
        fold_name = f"fold_{idx:03d}"
        fold_root = ensure_dir(fold_dir / fold_name)
        decision_logs: List[pd.DataFrame] = []

        if strategy == "experts_per_session":
            for session_id, payload in session_payload.items():
                times_sess = payload["times"]
                train_mask = (times_sess >= train_start) & (times_sess < train_end)
                test_mask = (times_sess >= test_start) & (times_sess < test_end)
                if train_mask.sum() < 50 or test_mask.sum() < 20:
                    continue
                X_sess = payload["X"]
                y_up_sess = payload["y_up"]
                y_down_sess = payload["y_down"]
                scaler_mean, scaler_std = fit_standard_scaler(
                    X_sess[train_mask].reshape(np.sum(train_mask), -1)
                )
                X_scaled = apply_standard_scaler(
                    X_sess.reshape(len(X_sess), -1), scaler_mean, scaler_std
                )
                X_train_s = X_scaled[train_mask]
                X_val_s = X_train_s
                y_up_train_s = y_up_sess[train_mask]
                y_down_train_s = y_down_sess[train_mask]
                sess_cfg = payload["cfg"]
                up_model, down_model = _train_models(
                    X_train_s,
                    y_up_train_s,
                    y_down_train_s,
                    X_val_s,
                    y_up_train_s,
                    y_down_train_s,
                    sess_cfg,
                    fast=fast,
                    device=device,
                )
                p_up = _predict_probs(up_model, X_scaled[test_mask], device=device)
                p_down = _predict_probs(down_model, X_scaled[test_mask], device=device)
                t_min, delta_min = _resolve_thresholds(sess_cfg)
                window_times_sess = payload["window_times"]
                regime_sess = payload["regime"]
                log = _build_decision_log(
                    window_times_sess[test_mask],
                    p_up,
                    p_down,
                    y_up_sess[test_mask],
                    y_down_sess[test_mask],
                    t_min,
                    delta_min,
                    np.array([session_id] * len(p_up)),
                    regime_sess[test_mask],
                )
                decision_logs.append(log)
        else:
            times = global_payload["times"]
            train_mask = (times >= train_start) & (times < train_end)
            test_mask = (times >= test_start) & (times < test_end)
            if not train_mask.any() or not test_mask.any():
                continue
            X = global_payload["X"]
            y_up = global_payload["y_up"]
            y_down = global_payload["y_down"]
            window_times = global_payload["window_times"]
            session_ids = global_payload["session_ids"]
            regime_labels = global_payload["regime"]
            scaler_mean, scaler_std = fit_standard_scaler(
                X[train_mask].reshape(np.sum(train_mask), -1)
            )
            X_scaled = apply_standard_scaler(X.reshape(len(X), -1), scaler_mean, scaler_std)
            X_train = X_scaled[train_mask]
            X_test = X_scaled[test_mask]
            y_up_train = y_up[train_mask]
            y_down_train = y_down[train_mask]
            y_up_test = y_up[test_mask]
            y_down_test = y_down[test_mask]
            sess_test = session_ids[test_mask]
            up_model, down_model = _train_models(
                X_train,
                y_up_train,
                y_down_train,
                X_train,
                y_up_train,
                y_down_train,
                base_cfg,
                fast=fast,
                device=device,
            )
            p_up = _predict_probs(up_model, X_test, device=device)
            p_down = _predict_probs(down_model, X_test, device=device)
            for session_id in sorted(set(sess_test)):
                sess_mask = sess_test == session_id
                sess_cfg = apply_session_overrides(base_cfg, cfg.session, session_id)
                t_min, delta_min = _resolve_thresholds(sess_cfg)
                log = _build_decision_log(
                    window_times[test_mask][sess_mask],
                    p_up[sess_mask],
                    p_down[sess_mask],
                    y_up_test[sess_mask],
                    y_down_test[sess_mask],
                    t_min,
                    delta_min,
                    np.array([session_id] * int(np.sum(sess_mask))),
                    regime_labels[test_mask][sess_mask],
                )
                decision_logs.append(log)

        if not decision_logs:
            continue
        fold_log = pd.concat(decision_logs, ignore_index=True)
        fold_metrics_out = compute_metrics(fold_log)
        fold_metrics_sessions = compute_metrics_by_session(fold_log)
        fold_metrics_regime = compute_metrics_by_regime(fold_log)
        fold_metrics.append(fold_metrics_out)
        fold_session_metrics.append(fold_metrics_sessions)
        fold_log.to_parquet(fold_root / "decisions.parquet", index=False)
        (fold_root / "metrics.json").write_text(
            json.dumps(fold_metrics_out, indent=2), encoding="utf-8"
        )
        (fold_root / "metrics_by_session.json").write_text(
            json.dumps(fold_metrics_sessions, indent=2), encoding="utf-8"
        )
        (fold_root / "metrics_by_regime.json").write_text(
            json.dumps(fold_metrics_regime, indent=2), encoding="utf-8"
        )

    if not fold_metrics:
        return run_path

    summary_dir = reports_dir("walk_forward")
    summary_dir.mkdir(parents=True, exist_ok=True)
    accs = [m["accuracy_non_hold"] for m in fold_metrics]
    hold_rates = [m["hold_rate"] for m in fold_metrics]
    steps = np.arange(1, len(accs) + 1)
    plot_series_with_band(
        steps,
        accs,
        window=min(5, len(accs)),
        title="Walk-forward Accuracy (non-hold)",
        xlabel="Fold",
        ylabel="Accuracy",
        label="accuracy_non_hold",
        out_base=summary_dir / "accuracy_over_folds",
        formats=["png", "svg"],
    )
    plot_series_with_band(
        steps,
        hold_rates,
        window=min(5, len(hold_rates)),
        title="Walk-forward Hold Rate",
        xlabel="Fold",
        ylabel="Hold rate",
        label="hold_rate",
        out_base=summary_dir / "hold_rate_over_folds",
        formats=["png", "svg"],
    )

    worst_session_id = None
    worst_session_acc = None
    for fold_metrics_by_session in fold_session_metrics:
        for session_id, sess_metrics in fold_metrics_by_session.items():
            acc = sess_metrics.get("accuracy_non_hold")
            if acc is None:
                continue
            if worst_session_acc is None or acc < worst_session_acc:
                worst_session_acc = acc
                worst_session_id = session_id

    summary = {
        "folds": len(fold_metrics),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "hold_rate_mean": float(np.mean(hold_rates)),
        "hold_rate_std": float(np.std(hold_rates)),
        "worst_session_id": worst_session_id,
        "worst_session_accuracy": float(worst_session_acc) if worst_session_acc is not None else None,
    }
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--expanding", action="store_true")
    parser.add_argument("--session-mode", default=None)
    parser.add_argument("--session-strategy", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_walk_forward(
        config_path=args.config,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        expanding=args.expanding,
        session_mode=args.session_mode,
        session_strategy=args.session_strategy,
        fast=args.fast,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )


if __name__ == "__main__":
    main()
