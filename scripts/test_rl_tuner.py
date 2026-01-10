"""Test RL tuner with full pipeline per episode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.analysis.adaptation import assess_adaptation  # noqa: E402
from cryptomvp.config import load_config  # noqa: E402
from cryptomvp.data.features import compute_features, compute_regime_labels  # noqa: E402
from cryptomvp.data.labels import make_directional_labels  # noqa: E402
from cryptomvp.data.scaling import apply_standard_scaler, fit_standard_scaler  # noqa: E402
from cryptomvp.data.windowing import make_windows  # noqa: E402
from cryptomvp.decision.rule import batch_decide  # noqa: E402
from cryptomvp.features.registry import default_feature_sets_path, load_feature_sets  # noqa: E402
from cryptomvp.features.registry import resolve_feature_list  # noqa: E402
from cryptomvp.features.selection import staged_feature_selection  # noqa: E402
from cryptomvp.sessions import SessionRouter, assign_session_features  # noqa: E402
from cryptomvp.train.rl_policy import PolicyNet  # noqa: E402
from cryptomvp.train.rl_env import RewardConfig  # noqa: E402
from cryptomvp.train.rl_train import train_reinforce  # noqa: E402
from cryptomvp.utils.gpu import require_cuda, resolve_device  # noqa: E402
from cryptomvp.utils.io import checkpoints_dir, reports_dir, run_root  # noqa: E402
from cryptomvp.utils.logging import get_logger  # noqa: E402
from cryptomvp.utils.run_dir import init_run_dir  # noqa: E402
from cryptomvp.utils.seed import set_seed  # noqa: E402
from cryptomvp.viz.plotting import plot_series_with_band  # noqa: E402


def _apply_override(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _load_decision_metrics(decision_log_path: Path) -> Dict[str, float]:
    df = pd.read_parquet(decision_log_path)
    decisions = df["decision"].astype(str)
    hold_mask = decisions == "HOLD"
    action_mask = ~hold_mask
    hold_rate = float(hold_mask.mean())
    action_rate = float(action_mask.mean())
    if action_mask.any():
        action_accuracy = float(df.loc[action_mask, "correct_direction"].mean())
    else:
        action_accuracy = 0.0
    conflict_rate = float(df["conflict"].mean()) if "conflict" in df.columns else 0.0
    precision_up = 0.0
    precision_down = 0.0
    up_rate = 0.0
    down_rate = 0.0
    down_share = 0.0
    if "true_direction" in df.columns:
        true_dir = df["true_direction"].astype(str)
        if (decisions == "UP").any():
            precision_up = float((true_dir[decisions == "UP"] == "UP").mean())
        if (decisions == "DOWN").any():
            precision_down = float((true_dir[decisions == "DOWN"] == "DOWN").mean())
    if action_mask.any():
        up_rate = float((decisions[action_mask] == "UP").mean())
        down_rate = float((decisions[action_mask] == "DOWN").mean())
        down_share = down_rate / max(1e-9, (up_rate + down_rate))
    metrics = {
        "decision_hold_rate": hold_rate,
        "decision_action_rate": action_rate,
        "decision_action_accuracy_non_hold": action_accuracy,
        "decision_conflict_rate": conflict_rate,
        "decision_precision_up": precision_up,
        "decision_precision_down": precision_down,
        "decision_up_rate": up_rate,
        "decision_down_rate": down_rate,
        "decision_down_share": down_share,
    }
    if "session_id" in df.columns:
        session_metrics = []
        for session_id, sdf in df.groupby("session_id"):
            s_decisions = sdf["decision"].astype(str)
            s_hold = s_decisions == "HOLD"
            s_action = ~s_hold
            s_hold_rate = float(s_hold.mean())
            s_action_rate = float(s_action.mean())
            if s_action.any():
                s_action_acc = float(sdf.loc[s_action, "correct_direction"].mean())
            else:
                s_action_acc = 0.0
            s_conflict = float(sdf["conflict"].mean()) if "conflict" in sdf.columns else 0.0
            s_precision_up = 0.0
            s_precision_down = 0.0
            if "true_direction" in sdf.columns:
                s_true = sdf["true_direction"].astype(str)
                if (s_decisions == "UP").any():
                    s_precision_up = float((s_true[s_decisions == "UP"] == "UP").mean())
                if (s_decisions == "DOWN").any():
                    s_precision_down = float((s_true[s_decisions == "DOWN"] == "DOWN").mean())
            session_metrics.append(
                {
                    "session_id": session_id,
                    "action_accuracy": s_action_acc,
                    "precision_up": s_precision_up,
                    "precision_down": s_precision_down,
                    "hold_rate": s_hold_rate,
                    "conflict_rate": s_conflict,
                    "action_rate": s_action_rate,
                }
            )
        if session_metrics:
            metrics["decision_min_session_accuracy"] = min(
                m["action_accuracy"] for m in session_metrics
            )
            metrics["decision_min_session_precision_up"] = min(
                m["precision_up"] for m in session_metrics
            )
            metrics["decision_min_session_precision_down"] = min(
                m["precision_down"] for m in session_metrics
            )
            metrics["decision_max_session_hold_rate"] = max(
                m["hold_rate"] for m in session_metrics
            )
            metrics["decision_max_session_conflict_rate"] = max(
                m["conflict_rate"] for m in session_metrics
            )
    metrics.update(
        {
            "hold_rate": hold_rate,
            "action_rate": action_rate,
            "action_accuracy_non_hold": action_accuracy,
            "conflict_rate": conflict_rate,
            "precision_up": precision_up,
            "precision_down": precision_down,
        }
    )
    return metrics


def _decision_metrics_from_probs(
    p_up: np.ndarray,
    p_down: np.ndarray,
    true_dir: np.ndarray,
    threshold: float,
    delta_min: float,
    regime: np.ndarray | None = None,
) -> Dict[str, float]:
    decisions = batch_decide(p_up, p_down, threshold, delta_min=delta_min)
    decisions_arr = np.array(decisions, dtype=object)
    true_str = np.where(true_dir > 0, "UP", np.where(true_dir < 0, "DOWN", "HOLD"))

    def _summarize(
        decisions_slice: np.ndarray,
        p_up_slice: np.ndarray,
        p_down_slice: np.ndarray,
        true_slice: np.ndarray,
        suffix: str = "",
    ) -> Dict[str, float]:
        hold_mask = decisions_slice == "HOLD"
        action_mask = ~hold_mask
        hold_rate = float(hold_mask.mean()) if len(decisions_slice) else 0.0
        action_rate = float(action_mask.mean()) if len(decisions_slice) else 0.0
        conflict_rate = (
            float(np.mean((p_up_slice >= threshold) & (p_down_slice >= threshold)))
            if len(p_up_slice)
            else 0.0
        )
        action_accuracy = 0.0
        precision_up = 0.0
        precision_down = 0.0
        up_rate = 0.0
        down_rate = 0.0
        down_share = 0.0
        if len(decisions_slice):
            if action_mask.any():
                action_accuracy = float(np.mean(decisions_slice[action_mask] == true_slice[action_mask]))
            if (decisions_slice == "UP").any():
                precision_up = float(np.mean(true_slice[decisions_slice == "UP"] == "UP"))
            if (decisions_slice == "DOWN").any():
                precision_down = float(np.mean(true_slice[decisions_slice == "DOWN"] == "DOWN"))
            if action_mask.any():
                up_rate = float(np.mean(decisions_slice[action_mask] == "UP"))
                down_rate = float(np.mean(decisions_slice[action_mask] == "DOWN"))
                down_share = down_rate / max(1e-9, (up_rate + down_rate))
        suffix_key = f"_{suffix}" if suffix else ""
        return {
            f"decision_hold_rate{suffix_key}": hold_rate,
            f"decision_action_rate{suffix_key}": action_rate,
            f"decision_action_accuracy_non_hold{suffix_key}": action_accuracy,
            f"decision_conflict_rate{suffix_key}": conflict_rate,
            f"decision_precision_up{suffix_key}": precision_up,
            f"decision_precision_down{suffix_key}": precision_down,
            f"decision_up_rate{suffix_key}": up_rate,
            f"decision_down_rate{suffix_key}": down_rate,
            f"decision_down_share{suffix_key}": down_share,
            f"hold_rate{suffix_key}": hold_rate,
            f"action_rate{suffix_key}": action_rate,
            f"action_accuracy_non_hold{suffix_key}": action_accuracy,
            f"conflict_rate{suffix_key}": conflict_rate,
            f"precision_up{suffix_key}": precision_up,
            f"precision_down{suffix_key}": precision_down,
            f"up_rate{suffix_key}": up_rate,
            f"down_rate{suffix_key}": down_rate,
            f"down_share{suffix_key}": down_share,
        }

    metrics = _summarize(decisions_arr, p_up, p_down, true_str)
    if regime is not None and len(regime) == len(decisions_arr):
        regime_arr = np.asarray(regime)
        if regime_arr.dtype.kind in {"f", "i", "u", "b"}:
            regime_labels = np.where(regime_arr.astype(float) >= 0.5, "trend", "flat")
        else:
            regime_labels = regime_arr.astype(str)
        for label in ("trend", "flat"):
            mask = regime_labels == label
            metrics.update(
                _summarize(
                    decisions_arr[mask],
                    p_up[mask],
                    p_down[mask],
                    true_str[mask],
                    suffix=label,
                )
            )
    return metrics


def _load_rl_metrics(metrics_path: Path, prefix: str) -> Dict[str, float]:
    if not metrics_path.exists():
        return {
            f"{prefix}_accuracy": 0.0,
            f"{prefix}_hold_rate": 0.0,
        }
    df = pd.read_csv(metrics_path)
    return {
        f"{prefix}_accuracy": float(df["accuracy"].mean()),
        f"{prefix}_hold_rate": float(df["hold_rate"].mean()),
    }


def _load_supervised_metrics(path: Path, prefix: str) -> Dict[str, float]:
    if not path.exists():
        return {f"{prefix}_accuracy": 0.0, f"{prefix}_f1": 0.0}
    df = pd.read_parquet(path)
    if "y_true" in df.columns:
        y_true = df["y_true"].to_numpy()
    else:
        y_true = df["true"].to_numpy()
    if "y_pred" in df.columns:
        y_pred = df["y_pred"].to_numpy()
    else:
        y_pred = df["pred"].to_numpy()
    acc = float((y_true == y_pred).mean())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom > 0 else 0.0
    return {f"{prefix}_accuracy": acc, f"{prefix}_f1": f1}


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _time_split(n: int) -> Tuple[int, int]:
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return train_end, val_end


def _time_split_with_online(n: int, online_ratio: float) -> Tuple[int, int, int]:
    if online_ratio < 0 or online_ratio >= 1:
        raise ValueError("online_split_ratio must be in [0, 1).")
    online_size = int(n * online_ratio)
    if online_ratio > 0 and online_size == 0 and n > 0:
        online_size = 1
    online_start = n - online_size if online_size > 0 else n
    train_end, val_end = _time_split(online_start)
    return train_end, val_end, online_start


def _window_split_indices(
    train_size: int,
    val_size: int,
    online_start: int,
    window_start: int,
) -> Tuple[int, int, int, int]:
    max_start = max(0, online_start - (train_size + val_size))
    start = min(max(window_start, 0), max_start)
    train_start = start
    train_end = train_start + train_size
    val_end = train_end + val_size
    return train_start, train_end, val_end, max_start


def _load_feature_scaler(scaler_path: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if not scaler_path.exists():
        raise RuntimeError(f"Missing RL feature scaler at {scaler_path}.")
    data = np.load(scaler_path, allow_pickle=True)
    scaler_cols = list(data["feature_cols"].tolist())
    if scaler_cols != feature_cols:
        raise RuntimeError(
            "Feature columns mismatch between RL scaler and current evaluation features."
        )
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)


def _evaluate_policy_state(
    state_dict: Dict[str, torch.Tensor] | None,
    X_val: np.ndarray,
    labels: np.ndarray,
    positive_action: int,
    device: torch.device,
    hidden_dim: int,
    margin_threshold: float,
) -> Tuple[float, float, float, np.ndarray]:
    if state_dict is None or len(X_val) == 0:
        return 0.0, 0.0, 0.0, np.zeros((len(X_val), 2), dtype=np.float32)
    policy = PolicyNet(input_dim=X_val.shape[1], hidden_dim=hidden_dim).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    with torch.no_grad():
        logits = policy(torch.from_numpy(X_val).float().to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        actions = np.argmax(probs, axis=1)
    valid_mask = labels != 0
    if not valid_mask.any():
        return 0.0, 0.0, 0.0, probs
    labels_valid = labels[valid_mask]
    actions_valid = actions[valid_mask]
    expected_actions = np.where(labels_valid == 1, positive_action, 1 - positive_action)
    accuracy = float(np.mean(actions_valid == expected_actions))
    predicted_positive = actions_valid == positive_action
    fp = int(np.sum(predicted_positive & (labels_valid != 1)))
    fn = int(np.sum(~predicted_positive & (labels_valid == 1)))
    error_balance = float(abs(fp - fn) / max(1, fp + fn))
    margin = np.abs(probs[:, 0] - probs[:, 1])
    low_margin_rate = float(np.mean(margin < margin_threshold)) if margin_threshold > 0 else 0.0
    return accuracy, error_balance, low_margin_rate, probs


def _evaluate_rl_policy(
    policy_path: Path,
    X_val: np.ndarray,
    labels: np.ndarray,
    positive_action: int,
    device: torch.device,
    hidden_dim: int,
) -> Tuple[float, float]:
    if not policy_path.exists() or len(X_val) == 0:
        return 0.0, 0.0
    policy = PolicyNet(input_dim=X_val.shape[1], hidden_dim=hidden_dim).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    with torch.no_grad():
        logits = policy(torch.from_numpy(X_val).float().to(device))
        actions = torch.argmax(logits, dim=1).cpu().numpy()
    valid_mask = labels != 0
    if not valid_mask.any():
        return 0.0, 0.0
    labels_valid = labels[valid_mask]
    actions_valid = actions[valid_mask]
    expected_actions = np.where(labels_valid == 1, positive_action, 1 - positive_action)
    accuracy = float(np.mean(actions_valid == expected_actions))
    predicted_positive = actions_valid == positive_action
    fp = int(np.sum(predicted_positive & (labels_valid != 1)))
    fn = int(np.sum(~predicted_positive & (labels_valid == 1)))
    error_balance = float(abs(fp - fn) / max(1, fp + fn))
    return accuracy, error_balance


def _prepare_rl_arrays(cfg: Any, dataset_path: Path) -> Dict[str, Any]:
    df = _load_dataset(dataset_path)
    router = SessionRouter(
        mode=cfg.session.mode if cfg.session else "fixed_utc_partitions",
        overlap_policy=cfg.session.overlap_policy if cfg.session else "priority",
        priority_order=cfg.session.priority_order if cfg.session else None,
        sessions=None,
    )
    df = assign_session_features(df, router)
    feature_sets_path = (
        Path(cfg.features.feature_sets_path)
        if cfg.features.feature_sets_path is not None
        else None
    )
    feature_list = resolve_feature_list(
        cfg.features.list_of_features,
        cfg.features.feature_set_id,
        feature_sets_path=feature_sets_path,
    )
    features = compute_features(df, feature_list)
    X, window_times, feature_cols = make_windows(features, cfg.features.window_size_K)
    y_up_dir, y_down_dir = make_directional_labels(df, window_times)
    regime_labels = compute_regime_labels(df, window_times)
    n = min(len(X), len(y_up_dir))
    X = X[:n]
    y_up_dir = y_up_dir[:n]
    y_down_dir = y_down_dir[:n]
    regime_labels = regime_labels[:n]
    X_flat = X.reshape(len(X), -1)
    scaler_path = run_root() / "reports" / "rl_tuner" / "feature_scaler.npz"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    if scaler_path.exists():
        mean, std = _load_feature_scaler(scaler_path, feature_cols)
    else:
        mean, std = fit_standard_scaler(X_flat)
        np.savez(
            scaler_path,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            feature_cols=np.array(feature_cols, dtype=object),
        )
    X_scaled = apply_standard_scaler(X_flat, mean, std)
    train_end, val_end = _time_split(len(X_scaled))
    if val_end <= train_end:
        raise RuntimeError("Not enough samples for RL tuner split.")
    return {
        "X_train": X_scaled[:train_end],
        "X_val": X_scaled[train_end:val_end],
        "y_up_train": y_up_dir[:train_end],
        "y_up_val": y_up_dir[train_end:val_end],
        "y_down_train": y_down_dir[:train_end],
        "y_down_val": y_down_dir[train_end:val_end],
        "regime_val": regime_labels[train_end:val_end],
    }


def _evaluate_rl_validation(
    cfg: Any,
    dataset_path: Path,
    run_dir: Path,
) -> Dict[str, float]:
    df = _load_dataset(dataset_path)
    router = SessionRouter(
        mode=cfg.session.mode if cfg.session else "fixed_utc_partitions",
        overlap_policy=cfg.session.overlap_policy if cfg.session else "priority",
        priority_order=cfg.session.priority_order if cfg.session else None,
        sessions=None,
    )
    df = assign_session_features(df, router)
    feature_sets_path = (
        Path(cfg.features.feature_sets_path)
        if cfg.features.feature_sets_path is not None
        else None
    )
    feature_list = resolve_feature_list(
        cfg.features.list_of_features,
        cfg.features.feature_set_id,
        feature_sets_path=feature_sets_path,
    )
    features = compute_features(df, feature_list)
    X, window_times, feature_cols = make_windows(features, cfg.features.window_size_K)
    y_up_dir, y_down_dir = make_directional_labels(df, window_times)

    n = min(len(X), len(y_up_dir))
    X = X[:n]
    y_up_dir = y_up_dir[:n]
    y_down_dir = y_down_dir[:n]
    X_flat = X.reshape(len(X), -1)

    scaler_path = run_dir / "reports" / "rl_up" / "feature_scaler.npz"
    mean, std = _load_feature_scaler(scaler_path, feature_cols)
    X_scaled = apply_standard_scaler(X_flat, mean, std)
    train_end, val_end = _time_split(len(X_scaled))
    if val_end <= train_end:
        return {
            "rl_up_accuracy": 0.0,
            "rl_down_accuracy": 0.0,
            "rl_up_error_balance": 0.0,
            "rl_down_error_balance": 0.0,
        }

    X_val = X_scaled[train_end:val_end]
    y_up_val = y_up_dir[train_end:val_end]
    y_down_val = y_down_dir[train_end:val_end]

    device = resolve_device(cfg.device, cfg.allow_cpu_fallback)
    rl_up_acc, rl_up_balance = _evaluate_rl_policy(
        run_dir / "checkpoints" / "rl_up.pt",
        X_val,
        y_up_val,
        positive_action=0,
        device=device,
        hidden_dim=cfg.rl.policy_hidden_dim,
    )
    rl_down_acc, rl_down_balance = _evaluate_rl_policy(
        run_dir / "checkpoints" / "rl_down.pt",
        X_val,
        y_down_val,
        positive_action=1,
        device=device,
        hidden_dim=cfg.rl.policy_hidden_dim,
    )

    return {
        "rl_up_accuracy": rl_up_acc,
        "rl_down_accuracy": rl_down_acc,
        "rl_up_error_balance": rl_up_balance,
        "rl_down_error_balance": rl_down_balance,
    }


def run_rl_tuner_agent(
    cfg: Any,
    config_path: str,
    fast: bool,
    run_dir: Path | None,
    episodes_override: int | None,
) -> None:
    logger = get_logger("rl_tuner")
    dataset_path = Path(cfg.dataset.output_path).resolve()
    if not dataset_path.exists():
        logger.info("Dataset not found; building dataset once before tuning.")
        _run_script("scripts/test_build_dataset.py", Path(config_path), run_root(), fast=False)
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset still missing at {dataset_path}.")

    report_dir = reports_dir("rl_tuner")
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    arrays = _prepare_rl_arrays(cfg, dataset_path)
    X_train = arrays["X_train"]
    X_val = arrays["X_val"]
    y_up_train = arrays["y_up_train"]
    y_up_val = arrays["y_up_val"]
    y_down_train = arrays["y_down_train"]
    y_down_val = arrays["y_down_val"]
    regime_val = arrays["regime_val"]

    reward_cfg = RewardConfig(
        R_correct=cfg.rl.reward.R_correct,
        R_wrong=cfg.rl.reward.R_wrong,
        R_opposite=cfg.rl.reward.R_opposite,
        margin_threshold=cfg.rl.reward.margin_threshold,
        margin_penalty=cfg.rl.reward.margin_penalty,
    )

    if episodes_override is not None:
        episodes = int(episodes_override)
    else:
        episodes = 2 if fast else cfg.tuner.episodes
    train_episodes = 1 if fast else cfg.tuner.agent_train_episodes
    steps_per_episode = 50 if fast else cfg.tuner.agent_steps_per_episode
    device = require_cuda()

    up_state: Dict[str, torch.Tensor] | None = None
    down_state: Dict[str, torch.Tensor] | None = None
    best_up_state: Dict[str, torch.Tensor] | None = None
    best_down_state: Dict[str, torch.Tensor] | None = None
    prev_metrics: Dict[str, float] | None = None
    best_reward = -float("inf")
    best_state: Dict[str, Any] | None = None
    results: List[Dict[str, Any]] = []
    interrupted = False

    try:
        for ep in range(1, episodes + 1):
            up_policy, up_hist = train_reinforce(
                X_train,
                y_up_train,
                episodes=train_episodes,
                steps_per_episode=steps_per_episode,
                lr=cfg.rl.lr,
                gamma=cfg.rl.gamma,
                reward_cfg=reward_cfg,
                policy_hidden_dim=cfg.rl.policy_hidden_dim,
                entropy_bonus=cfg.rl.entropy_bonus,
                seed=cfg.seed + ep,
                track_diagnostics=True,
                track_steps=False,
                model_name="tuner_up",
                device=device,
                init_state=up_state,
            )
            up_state = up_policy.state_dict()

            down_policy, down_hist = train_reinforce(
                X_train,
                y_down_train,
                episodes=train_episodes,
                steps_per_episode=steps_per_episode,
                lr=cfg.rl.lr,
                gamma=cfg.rl.gamma,
                reward_cfg=reward_cfg,
                policy_hidden_dim=cfg.rl.policy_hidden_dim,
                entropy_bonus=cfg.rl.entropy_bonus,
                seed=cfg.seed + ep + 100,
                track_diagnostics=True,
                track_steps=False,
                model_name="tuner_down",
                label_action_map={1: 1, -1: 0},
                device=device,
                init_state=down_state,
            )
            down_state = down_policy.state_dict()

            rl_up_acc, rl_up_balance, rl_up_low_margin, up_probs = _evaluate_policy_state(
                up_state,
                X_val,
                y_up_val,
                positive_action=0,
                device=device,
                hidden_dim=cfg.rl.policy_hidden_dim,
                margin_threshold=reward_cfg.margin_threshold,
            )
            rl_down_acc, rl_down_balance, rl_down_low_margin, down_probs = _evaluate_policy_state(
                down_state,
                X_val,
                y_down_val,
                positive_action=1,
                device=device,
                hidden_dim=cfg.rl.policy_hidden_dim,
                margin_threshold=reward_cfg.margin_threshold,
            )

            p_up = up_probs[:, 0] if len(up_probs) else np.array([])
            p_down = down_probs[:, 1] if len(down_probs) else np.array([])
            decision_metrics = _decision_metrics_from_probs(
                p_up,
                p_down,
                y_up_val,
                threshold=cfg.decision_rule.T_min,
                delta_min=cfg.decision_rule.delta_min,
                regime=regime_val,
            )

            metrics: Dict[str, float] = {}
            metrics.update(decision_metrics)
            metrics.update(
                {
                    "rl_up_accuracy": rl_up_acc,
                    "rl_down_accuracy": rl_down_acc,
                    "rl_up_error_balance": rl_up_balance,
                    "rl_down_error_balance": rl_down_balance,
                    "rl_up_hold_rate": rl_up_low_margin,
                    "rl_down_hold_rate": rl_down_low_margin,
                }
            )

            reward_base = (
                cfg.tuner.reward.decision_accuracy_weight * metrics["decision_action_accuracy_non_hold"]
                + cfg.tuner.reward.decision_action_rate_weight * metrics["decision_action_rate"]
                - cfg.tuner.reward.decision_conflict_penalty * metrics["decision_conflict_rate"]
                - cfg.tuner.reward.decision_hold_penalty * metrics["decision_hold_rate"]
                - cfg.tuner.reward.decision_balance_penalty
                * abs(metrics["decision_down_share"] - cfg.tuner.reward.decision_down_target)
                + cfg.tuner.reward.rl_up_accuracy_weight * metrics["rl_up_accuracy"]
                + cfg.tuner.reward.rl_down_accuracy_weight * metrics["rl_down_accuracy"]
                - cfg.tuner.reward.rl_up_error_balance_penalty * metrics["rl_up_error_balance"]
                - cfg.tuner.reward.rl_down_error_balance_penalty * metrics["rl_down_error_balance"]
                - cfg.tuner.reward.rl_up_hold_penalty * metrics["rl_up_hold_rate"]
                - cfg.tuner.reward.rl_down_hold_penalty * metrics["rl_down_hold_rate"]
            )
            metrics["score_base"] = float(reward_base)

            prev_up = prev_metrics.get("rl_up_accuracy", 0.0) if prev_metrics else 0.0
            prev_down = prev_metrics.get("rl_down_accuracy", 0.0) if prev_metrics else 0.0
            delta_up = float(metrics["rl_up_accuracy"] - prev_up)
            delta_down = float(metrics["rl_down_accuracy"] - prev_down)
            improve_up = max(0.0, delta_up)
            improve_down = max(0.0, delta_down)
            reward_improve = (
                cfg.tuner.reward.improve_up_accuracy_weight * improve_up
                + cfg.tuner.reward.improve_down_accuracy_weight * improve_down
            )
            reward_total = float(reward_base + reward_improve)
            metrics["score_improve"] = float(reward_improve)
            metrics["score"] = reward_total
            metrics["delta_up_accuracy"] = float(delta_up)
            metrics["delta_down_accuracy"] = float(delta_down)
            metrics["improve_up_accuracy"] = float(improve_up)
            metrics["improve_down_accuracy"] = float(improve_down)

            adaptation_good = None
            adaptation_failures: list[str] = []
            if cfg.adaptation is not None:
                good, failures = assess_adaptation(metrics, cfg.adaptation)
                adaptation_good = int(good)
                adaptation_failures = failures
                metrics["adaptation_good"] = float(adaptation_good)

            was_best = reward_total > best_reward
            if was_best:
                best_reward = reward_total
                best_state = {
                    "episode": ep,
                    "reward": reward_total,
                    "metrics": dict(metrics),
                }
                best_up_state = dict(up_state)
                best_down_state = dict(down_state)
            elif cfg.tuner.agent_reset_to_best and best_up_state is not None and best_down_state is not None:
                up_state = dict(best_up_state)
                down_state = dict(best_down_state)

            results.append(
                {
                    "episode": ep,
                    "reward": reward_total,
                    "best_reward": best_reward,
                    "adaptation_good": adaptation_good if adaptation_good is not None else -1,
                    "adaptation_failures": ";".join(adaptation_failures) if adaptation_failures else "",
                    "up_last_reward": up_hist.rewards[-1] if up_hist.rewards else 0.0,
                    "down_last_reward": down_hist.rewards[-1] if down_hist.rewards else 0.0,
                    "reset_to_best": int(
                        cfg.tuner.agent_reset_to_best
                        and best_up_state is not None
                        and best_down_state is not None
                        and not was_best
                    ),
                    **metrics,
                }
            )
            prev_metrics = metrics
            logger.info(
                "Agent episode %s/%s - reward=%.4f best=%.4f",
                ep,
                episodes,
                reward_total,
                best_reward,
            )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("RL tuner agent interrupted; saving partial results.")

    save_up = best_up_state if best_up_state is not None else up_state
    save_down = best_down_state if best_down_state is not None else down_state
    if save_up is not None:
        torch.save(save_up, checkpoints_dir() / "rl_up.pt")
    if save_down is not None:
        torch.save(save_down, checkpoints_dir() / "rl_down.pt")

    history_df = pd.DataFrame(results)
    history_df.to_csv(report_dir / "episode_metrics.csv", index=False)
    if not history_df.empty:
        episodes_idx = history_df["episode"].to_numpy()
        plot_series_with_band(
            episodes_idx,
            history_df["reward"].to_numpy(),
            window=cfg.viz.moving_window,
            title="Agent Reward per Episode (RL UP/DOWN)",
            xlabel="Episode",
            ylabel="Reward",
            label="reward",
            out_base=fig_dir / "reward_per_episode",
            formats=cfg.viz.save_formats,
        )
        plot_series_with_band(
            episodes_idx,
            history_df["rl_up_accuracy"].to_numpy(),
            window=cfg.viz.moving_window,
            title="RL UP Accuracy (val)",
            xlabel="Episode",
            ylabel="Accuracy",
            label="rl_up_accuracy",
            out_base=fig_dir / "rl_up_accuracy_per_episode",
            formats=cfg.viz.save_formats,
        )
        plot_series_with_band(
            episodes_idx,
            history_df["rl_down_accuracy"].to_numpy(),
            window=cfg.viz.moving_window,
            title="RL DOWN Accuracy (val)",
            xlabel="Episode",
            ylabel="Accuracy",
            label="rl_down_accuracy",
            out_base=fig_dir / "rl_down_accuracy_per_episode",
            formats=cfg.viz.save_formats,
        )
        plot_series_with_band(
            episodes_idx,
            history_df["decision_action_accuracy_non_hold"].to_numpy(),
            window=cfg.viz.moving_window,
            title="Decision Accuracy (non-hold)",
            xlabel="Episode",
            ylabel="Accuracy",
            label="decision_accuracy",
            out_base=fig_dir / "decision_accuracy_per_episode",
            formats=cfg.viz.save_formats,
        )
        plot_series_with_band(
            episodes_idx,
            history_df["decision_hold_rate"].to_numpy(),
            window=cfg.viz.moving_window,
            title="Decision Hold Rate",
            xlabel="Episode",
            ylabel="Hold rate",
            label="decision_hold_rate",
            out_base=fig_dir / "decision_hold_rate_per_episode",
            formats=cfg.viz.save_formats,
        )
        if "decision_precision_down" in history_df.columns:
            plot_series_with_band(
                episodes_idx,
                history_df["decision_precision_down"].to_numpy(),
                window=cfg.viz.moving_window,
                title="Decision Precision DOWN",
                xlabel="Episode",
                ylabel="Precision",
                label="decision_precision_down",
                out_base=fig_dir / "decision_precision_down_per_episode",
                formats=cfg.viz.save_formats,
            )
        if "decision_down_share" in history_df.columns:
            plot_series_with_band(
                episodes_idx,
                history_df["decision_down_share"].to_numpy(),
                window=cfg.viz.moving_window,
                title="Decision DOWN Share (non-hold)",
                xlabel="Episode",
                ylabel="Share",
                label="decision_down_share",
                out_base=fig_dir / "decision_down_share_per_episode",
                formats=cfg.viz.save_formats,
            )

    adaptation_line = "Adaptation good rate: n/a"
    if cfg.adaptation is not None and "adaptation_good" in history_df.columns and not history_df.empty:
        good_rate = float((history_df["adaptation_good"] == 1).mean())
        adaptation_line = f"Adaptation good rate: {good_rate:.4f}"

    best_state = best_state or {"episode": 0, "reward": best_reward, "metrics": {}}
    summary_lines = [
        "# RL Tuner Summary (Agent Mode)",
        f"Symbol: {cfg.symbol}",
        f"Interval: {cfg.interval}",
        f"Seed: {cfg.seed}",
        f"Episodes: {episodes}",
        f"Agent train episodes per step: {train_episodes}",
        f"Agent steps per episode: {steps_per_episode}",
        f"Agent reset to best: {cfg.tuner.agent_reset_to_best}",
        f"Interrupted: {interrupted}",
        f"Decision weights: acc={cfg.tuner.reward.decision_accuracy_weight}, "
        f"action={cfg.tuner.reward.decision_action_rate_weight}, "
        f"conflict_penalty={cfg.tuner.reward.decision_conflict_penalty}, "
        f"hold_penalty={cfg.tuner.reward.decision_hold_penalty}, "
        f"balance_penalty={cfg.tuner.reward.decision_balance_penalty}, "
        f"down_target={cfg.tuner.reward.decision_down_target}",
        f"RL weights: up_acc={cfg.tuner.reward.rl_up_accuracy_weight}, "
        f"down_acc={cfg.tuner.reward.rl_down_accuracy_weight}, "
        f"up_error_balance_penalty={cfg.tuner.reward.rl_up_error_balance_penalty}, "
        f"down_error_balance_penalty={cfg.tuner.reward.rl_down_error_balance_penalty}, "
        f"up_hold_penalty={cfg.tuner.reward.rl_up_hold_penalty}, "
        f"down_hold_penalty={cfg.tuner.reward.rl_down_hold_penalty}",
        f"Improve weights: up={cfg.tuner.reward.improve_up_accuracy_weight}, "
        f"down={cfg.tuner.reward.improve_down_accuracy_weight}",
        f"Best reward: {best_state['reward']:.4f}",
        adaptation_line,
        "",
        "## RL policy effectiveness",
        f"- RL UP accuracy (val): {best_state['metrics'].get('rl_up_accuracy', 0.0):.4f}",
        f"- RL DOWN accuracy (val): {best_state['metrics'].get('rl_down_accuracy', 0.0):.4f}",
        f"- RL UP error balance: {best_state['metrics'].get('rl_up_error_balance', 0.0):.4f}",
        f"- RL DOWN error balance: {best_state['metrics'].get('rl_down_error_balance', 0.0):.4f}",
    ]
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("RL tuner agent report written to %s", report_dir)


def _is_numeric_list(values: List[Any]) -> bool:
    return all(isinstance(v, (int, float)) for v in values)


def _choose_alternative(
    values: List[Any], current: Any, rng: np.random.Generator, neighbor_only: bool
) -> Any:
    if len(values) <= 1:
        return current
    if neighbor_only and _is_numeric_list(values):
        ordered = sorted(values)
        if current not in ordered:
            return ordered[int(rng.integers(0, len(ordered)))]
        idx = ordered.index(current)
        candidates = []
        if idx > 0:
            candidates.append(ordered[idx - 1])
        if idx < len(ordered) - 1:
            candidates.append(ordered[idx + 1])
        if candidates:
            return candidates[int(rng.integers(0, len(candidates)))]
    alternatives = [v for v in values if v != current]
    return alternatives[int(rng.integers(0, len(alternatives)))]


def _propose_params(
    param_space: Dict[str, List[Any]],
    best_params: Dict[str, Any] | None,
    rng: np.random.Generator,
    explore_prob: float,
    mutate_prob: float,
    max_mutations: int,
    neighbor_only: bool,
    always_mutate: bool,
) -> Tuple[Dict[str, Any], List[str], bool]:
    keys = list(param_space.keys())
    if best_params is None:
        params = {key: param_space[key][int(rng.integers(0, len(param_space[key])))] for key in keys}
        return params, keys, True

    params = dict(best_params)
    rng.shuffle(keys)
    mutated = []
    explore = rng.random() < explore_prob
    local_mutate_prob = 1.0 if explore else mutate_prob
    local_max_mutations = max(max_mutations, 1)
    if explore:
        local_max_mutations = max(local_max_mutations, max(2, len(keys) // 2))
    for key in keys:
        if len(mutated) >= local_max_mutations:
            break
        if rng.random() < local_mutate_prob:
            new_val = _choose_alternative(param_space[key], params[key], rng, neighbor_only)
            if new_val != params[key]:
                params[key] = new_val
                mutated.append(key)

    if not mutated and always_mutate and keys:
        key = rng.choice(keys)
        new_val = _choose_alternative(param_space[key], params[key], rng, neighbor_only)
        if new_val != params[key]:
            params[key] = new_val
            mutated.append(key)

    return params, mutated, explore


def _run_script(script: str, config_path: Path, run_dir: Path, fast: bool) -> None:
    cmd = [
        sys.executable,
        script,
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
    ]
    if fast:
        cmd.append("--fast")
    subprocess.run(cmd, check=True)


def run_rl_tuner(
    config_path: str,
    fast: bool,
    run_dir: Path | None = None,
    episodes_override: int | None = None,
) -> None:
    init_run_dir(run_dir, config_path)
    cfg = load_config(config_path)
    if cfg.tuner is None:
        raise RuntimeError("Missing tuner config in YAML.")
    logger = get_logger("rl_tuner")
    set_seed(cfg.seed)
    if cfg.tuner.mode == "agent":
        run_rl_tuner_agent(cfg, config_path, fast, run_dir, episodes_override)
        return

    base_cfg = yaml.safe_load(Path(config_path).read_text())
    param_space = cfg.tuner.param_space
    if not param_space:
        raise RuntimeError("tuner.param_space is empty; provide parameter ranges to tune.")

    dataset_path = Path(cfg.dataset.output_path).resolve()
    if not dataset_path.exists():
        logger.info("Dataset not found; building dataset once before tuning.")
        _run_script("scripts/test_build_dataset.py", Path(config_path), run_root(), fast=False)
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset still missing at {dataset_path}.")

    tuner_root = run_root() / "tuner"
    tuner_root.mkdir(parents=True, exist_ok=True)
    report_dir = reports_dir("rl_tuner")
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if cfg.tuner.feature_selection is not None and cfg.tuner.feature_selection.enabled:
        df = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_csv(dataset_path)
        feature_sets_path = (
            Path(cfg.features.feature_sets_path)
            if cfg.features.feature_sets_path is not None
            else default_feature_sets_path()
        )
        scores = staged_feature_selection(
            df,
            window_size=cfg.features.window_size_K,
            feature_sets_path=feature_sets_path,
            top_n=cfg.tuner.feature_selection.top_n,
            corr_threshold=cfg.tuner.feature_selection.corr_threshold,
            var_threshold=cfg.tuner.feature_selection.var_threshold,
        )
        top_sets = [score.feature_set_id for score in scores]
        param_space.pop("features.list_of_features", None)
        param_space["features.feature_set_id"] = top_sets

        fs_rows = [
            {
                "feature_set_id": score.feature_set_id,
                "score": score.score,
                "accuracy": score.accuracy,
                "roc_auc": score.roc_auc,
                "n_features": score.n_features,
            }
            for score in scores
        ]
        pd.DataFrame(fs_rows).to_csv(report_dir / "feature_selection.csv", index=False)

    param_space_path = report_dir / "param_space.json"
    param_space_path.write_text(json.dumps(param_space, indent=2), encoding="utf-8")

    episode_counter = {"value": 0}

    def evaluate_sample(params: Dict[str, Any]) -> Dict[str, float]:
        episode_counter["value"] += 1
        episode_idx = episode_counter["value"]
        episode_dir = tuner_root / f"episode_{episode_idx:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        episode_cfg = deepcopy(base_cfg)
        for key, value in params.items():
            _apply_override(episode_cfg, key, value)
        episode_cfg["dataset"]["output_path"] = str(dataset_path)

        episode_cfg_path = episode_dir / "config.yaml"
        episode_cfg_path.write_text(
            yaml.safe_dump(episode_cfg, sort_keys=False),
            encoding="utf-8",
        )

        _run_script("scripts/test_train_baseline.py", episode_cfg_path, episode_dir, fast=fast)
        _run_script("scripts/test_train_rl.py", episode_cfg_path, episode_dir, fast=fast)

        episode_loaded = load_config(episode_cfg_path)
        decision_metrics = _load_decision_metrics(
            episode_dir / "reports" / "decision_rule" / "decision_log.parquet"
        )
        sup_up_metrics = _load_supervised_metrics(
            episode_dir / "reports" / "supervised_up" / "decision_log.parquet",
            prefix="sup_up",
        )
        sup_down_metrics = _load_supervised_metrics(
            episode_dir / "reports" / "supervised_down" / "decision_log.parquet",
            prefix="sup_down",
        )
        rl_up_metrics = _load_rl_metrics(
            episode_dir / "reports" / "rl_up" / "episode_metrics.csv",
            prefix="rl_up",
        )
        rl_down_metrics = _load_rl_metrics(
            episode_dir / "reports" / "rl_down" / "episode_metrics.csv",
            prefix="rl_down",
        )
        rl_eval_metrics = _evaluate_rl_validation(
            episode_loaded, dataset_path, episode_dir
        )

        metrics: Dict[str, float] = {}
        metrics.update(decision_metrics)
        metrics.update(sup_up_metrics)
        metrics.update(sup_down_metrics)
        metrics.update(rl_up_metrics)
        metrics.update(rl_down_metrics)
        metrics.update(rl_eval_metrics)

        reward_base = (
            cfg.tuner.reward.decision_accuracy_weight * metrics["decision_action_accuracy_non_hold"]
            + cfg.tuner.reward.decision_action_rate_weight * metrics["decision_action_rate"]
            - cfg.tuner.reward.decision_conflict_penalty * metrics["decision_conflict_rate"]
            - cfg.tuner.reward.decision_hold_penalty * metrics["decision_hold_rate"]
            - cfg.tuner.reward.decision_balance_penalty
            * abs(metrics["decision_down_share"] - cfg.tuner.reward.decision_down_target)
            + cfg.tuner.reward.decision_min_session_accuracy_weight
            * metrics.get("decision_min_session_accuracy", 0.0)
            + cfg.tuner.reward.decision_min_session_precision_up_weight
            * metrics.get("decision_min_session_precision_up", 0.0)
            + cfg.tuner.reward.decision_min_session_precision_down_weight
            * metrics.get("decision_min_session_precision_down", 0.0)
            - cfg.tuner.reward.decision_max_session_hold_penalty
            * metrics.get("decision_max_session_hold_rate", 0.0)
            - cfg.tuner.reward.decision_max_session_conflict_penalty
            * metrics.get("decision_max_session_conflict_rate", 0.0)
            + cfg.tuner.reward.rl_up_accuracy_weight * metrics["rl_up_accuracy"]
            + cfg.tuner.reward.rl_down_accuracy_weight * metrics["rl_down_accuracy"]
            - cfg.tuner.reward.rl_up_error_balance_penalty * metrics["rl_up_error_balance"]
            - cfg.tuner.reward.rl_down_error_balance_penalty * metrics["rl_down_error_balance"]
            - cfg.tuner.reward.rl_up_hold_penalty * metrics["rl_up_hold_rate"]
            - cfg.tuner.reward.rl_down_hold_penalty * metrics["rl_down_hold_rate"]
        )
        metrics["score_base"] = float(reward_base)
        return metrics

    if episodes_override is not None:
        episodes = int(episodes_override)
    else:
        episodes = 2 if fast else cfg.tuner.episodes
    rng = np.random.default_rng(cfg.seed)
    best_params: Dict[str, Any] | None = None
    best_reward = -float("inf")
    best_state: Dict[str, Any] | None = None
    prev_metrics: Dict[str, float] | None = None
    results: List[Dict[str, Any]] = []
    param_count = max(1, len(param_space))
    stability_window = max(1, cfg.viz.moving_window)
    reward_history: List[float] = []
    accuracy_history: List[float] = []

    def rolling_variance(history: List[float], current: float, window: int) -> float:
        values = (history + [current])[-window:]
        if len(values) < 2:
            return 0.0
        return float(np.var(values))

    for ep in range(1, episodes + 1):
        params, mutated_keys, explored = _propose_params(
            param_space,
            best_params,
            rng,
            explore_prob=cfg.tuner.search.explore_prob,
            mutate_prob=cfg.tuner.search.mutate_prob,
            max_mutations=cfg.tuner.search.max_mutations,
            neighbor_only=cfg.tuner.search.neighbor_only,
            always_mutate=cfg.tuner.search.always_mutate,
        )

        metrics = evaluate_sample(params)
        reward_base_raw = float(metrics.get("score_base", 0.0))
        variance_reward = rolling_variance(reward_history, reward_base_raw, stability_window)
        variance_accuracy = rolling_variance(
            accuracy_history,
            float(metrics.get("decision_action_accuracy_non_hold", 0.0)),
            stability_window,
        )
        reward_base = reward_base_raw - cfg.tuner.reward.stability_penalty * variance_reward
        metrics["score_base"] = float(reward_base)
        metrics["reward_variance"] = float(variance_reward)
        metrics["accuracy_variance"] = float(variance_accuracy)

        prev_up = prev_metrics.get("sup_up_accuracy", 0.0) if prev_metrics else 0.0
        prev_down = prev_metrics.get("sup_down_accuracy", 0.0) if prev_metrics else 0.0
        delta_up = float(metrics.get("sup_up_accuracy", 0.0) - prev_up)
        delta_down = float(metrics.get("sup_down_accuracy", 0.0) - prev_down)
        improve_up = max(0.0, delta_up)
        improve_down = max(0.0, delta_down)

        reward_improve = (
            cfg.tuner.reward.improve_up_accuracy_weight * improve_up
            + cfg.tuner.reward.improve_down_accuracy_weight * improve_down
        )
        reward_total = float(reward_base + reward_improve)
        metrics["score_improve"] = float(reward_improve)
        metrics["score"] = reward_total
        metrics["delta_up_accuracy"] = float(delta_up)
        metrics["delta_down_accuracy"] = float(delta_down)
        metrics["improve_up_accuracy"] = float(improve_up)
        metrics["improve_down_accuracy"] = float(improve_down)

        adaptation_good = None
        adaptation_failures: list[str] = []
        if cfg.adaptation is not None:
            good, failures = assess_adaptation(metrics, cfg.adaptation)
            adaptation_good = int(good)
            adaptation_failures = failures
            metrics["adaptation_good"] = float(adaptation_good)

        if reward_total > best_reward:
            best_reward = reward_total
            best_params = dict(params)
            best_state = {
                "episode": ep,
                "reward": reward_total,
                "metrics": dict(metrics),
            }

        reward_history.append(float(reward_base_raw))
        accuracy_history.append(float(metrics.get("decision_action_accuracy_non_hold", 0.0)))
        results.append(
            {
                "episode": ep,
                "reward": reward_total,
                "best_reward": best_reward,
                "mutation_count": len(mutated_keys),
                "mutation_rate": len(mutated_keys) / param_count,
                "mutated_keys": ",".join(mutated_keys),
                "explore": int(explored),
                "adaptation_good": adaptation_good if adaptation_good is not None else -1,
                "adaptation_failures": ";".join(adaptation_failures) if adaptation_failures else "",
                **metrics,
                **{f"param.{key}": value for key, value in params.items()},
            }
        )

        prev_metrics = metrics
        logger.info(
            "Episode %s/%s - reward=%.4f best=%.4f mutations=%s",
            ep,
            episodes,
            reward_total,
            best_reward,
            len(mutated_keys),
        )

    history_df = pd.DataFrame(results)
    history_df.to_csv(report_dir / "episode_metrics.csv", index=False)
    leaderboard = history_df.sort_values("reward", ascending=False).head(20)
    leaderboard.to_csv(report_dir / "leaderboard.csv", index=False)
    if best_params is None:
        best_params = {}
    if best_state is None:
        best_state = {"episode": 0, "reward": best_reward, "metrics": {}}

    (report_dir / "best_params.json").write_text(
        json.dumps(
            {
                "reward": best_state["reward"],
                "episode": best_state["episode"],
                "params": best_params,
                "metrics": best_state.get("metrics", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (report_dir / "param_logits.json").write_text(
        json.dumps({"method": "best_anchor", "note": "no logits in progressive tuner"}, indent=2),
        encoding="utf-8",
    )

    episodes_idx = history_df["episode"].to_numpy()
    plot_series_with_band(
        episodes_idx,
        history_df["reward"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Tuner Reward per Episode (full pipeline)",
        xlabel="Episode",
        ylabel="Reward",
        label="reward",
        out_base=fig_dir / "reward_per_episode",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episodes_idx,
        history_df["score_improve"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Improvement Reward per Episode",
        xlabel="Episode",
        ylabel="Reward (improvement)",
        label="reward_improve",
        out_base=fig_dir / "improvement_reward_per_episode",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episodes_idx,
        history_df["decision_action_accuracy_non_hold"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Decision Accuracy (non-hold)",
        xlabel="Episode",
        ylabel="Accuracy",
        label="decision_accuracy",
        out_base=fig_dir / "decision_accuracy_per_episode",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episodes_idx,
        history_df["decision_hold_rate"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Decision Hold Rate",
        xlabel="Episode",
        ylabel="Hold rate",
        label="decision_hold_rate",
        out_base=fig_dir / "decision_hold_rate_per_episode",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episodes_idx,
        history_df["decision_conflict_rate"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Decision Conflict Rate",
        xlabel="Episode",
        ylabel="Conflict rate",
        label="decision_conflict_rate",
        out_base=fig_dir / "decision_conflict_rate_per_episode",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episodes_idx,
        history_df["rl_up_accuracy"].to_numpy(),
        window=cfg.viz.moving_window,
        title="RL UP Accuracy",
        xlabel="Episode",
        ylabel="Accuracy",
        label="rl_up_accuracy",
        out_base=fig_dir / "rl_up_accuracy_per_episode",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episodes_idx,
        history_df["rl_down_accuracy"].to_numpy(),
        window=cfg.viz.moving_window,
        title="RL DOWN Accuracy",
        xlabel="Episode",
        ylabel="Accuracy",
        label="rl_down_accuracy",
        out_base=fig_dir / "rl_down_accuracy_per_episode",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episodes_idx,
        history_df["mutation_rate"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Mutation Rate per Episode",
        xlabel="Episode",
        ylabel="Mutation rate",
        label="mutation_rate",
        out_base=fig_dir / "mutation_rate_per_episode",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episodes_idx,
        history_df["best_reward"].to_numpy(),
        window=cfg.viz.moving_window,
        title="Best Score So Far",
        xlabel="Episode",
        ylabel="Score",
        label="best_reward",
        out_base=fig_dir / "best_score",
        formats=cfg.viz.save_formats,
    )

    adaptation_line = "Adaptation good rate: n/a"
    if cfg.adaptation is not None and "adaptation_good" in history_df.columns:
        good_rate = float((history_df["adaptation_good"] == 1).mean())
        adaptation_line = f"Adaptation good rate: {good_rate:.4f}"

    summary_lines = [
        "# RL Tuner Summary (Full Pipeline)",
        f"Symbol: {cfg.symbol}",
        f"Interval: {cfg.interval}",
        f"Seed: {cfg.seed}",
        f"Episodes: {episodes}",
        f"Online split ratio: {cfg.tuner.online_split_ratio}",
        f"Param space entries: {len(param_space)}",
        f"Reward variance (last {stability_window}): "
        f"{best_state['metrics'].get('reward_variance', 0.0):.4f}",
        f"Accuracy variance (last {stability_window}): "
        f"{best_state['metrics'].get('accuracy_variance', 0.0):.4f}",
        f"Search explore_prob: {cfg.tuner.search.explore_prob}",
        f"Search mutate_prob: {cfg.tuner.search.mutate_prob}",
        f"Search max_mutations: {cfg.tuner.search.max_mutations}",
        f"Search neighbor_only: {cfg.tuner.search.neighbor_only}",
        f"Search always_mutate: {cfg.tuner.search.always_mutate}",
        "Search method: best_anchor",
        f"Decision weights: acc={cfg.tuner.reward.decision_accuracy_weight}, "
        f"action={cfg.tuner.reward.decision_action_rate_weight}, "
        f"conflict_penalty={cfg.tuner.reward.decision_conflict_penalty}, "
        f"hold_penalty={cfg.tuner.reward.decision_hold_penalty}",
        f"RL weights: up_acc={cfg.tuner.reward.rl_up_accuracy_weight}, "
        f"down_acc={cfg.tuner.reward.rl_down_accuracy_weight}, "
        f"online_up_acc={cfg.tuner.reward.online_rl_up_accuracy_weight}, "
        f"online_down_acc={cfg.tuner.reward.online_rl_down_accuracy_weight}, "
        f"up_error_balance_penalty={cfg.tuner.reward.rl_up_error_balance_penalty}, "
        f"down_error_balance_penalty={cfg.tuner.reward.rl_down_error_balance_penalty}, "
        f"up_hold_penalty={cfg.tuner.reward.rl_up_hold_penalty}, "
        f"down_hold_penalty={cfg.tuner.reward.rl_down_hold_penalty}",
        f"Improve weights: up={cfg.tuner.reward.improve_up_accuracy_weight}, "
        f"down={cfg.tuner.reward.improve_down_accuracy_weight}",
        f"Best reward: {best_state['reward']:.4f}",
        adaptation_line,
        "",
        "## RL policy effectiveness",
        f"- RL UP accuracy (val): {best_state['metrics'].get('rl_up_accuracy', 0.0):.4f}",
        f"- RL DOWN accuracy (val): {best_state['metrics'].get('rl_down_accuracy', 0.0):.4f}",
        f"- RL UP accuracy (online): {best_state['metrics'].get('online_rl_up_accuracy', 0.0):.4f}",
        f"- RL DOWN accuracy (online): {best_state['metrics'].get('online_rl_down_accuracy', 0.0):.4f}",
        f"- RL UP error balance: {best_state['metrics'].get('rl_up_error_balance', 0.0):.4f}",
        f"- RL DOWN error balance: {best_state['metrics'].get('rl_down_error_balance', 0.0):.4f}",
        "",
        "Best params:",
        json.dumps(best_params, indent=2),
    ]
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("RL tuner report written to %s", report_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    run_rl_tuner(
        args.config,
        fast=args.fast,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        episodes_override=args.episodes,
    )
