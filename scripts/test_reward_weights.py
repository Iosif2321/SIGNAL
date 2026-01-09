"""Test 5: Reward/penalty impact on weights and gradients."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.config import load_config
from cryptomvp.data.features import compute_features
from cryptomvp.data.labels import make_directional_labels, make_up_down_labels
from cryptomvp.data.scaling import apply_standard_scaler, fit_standard_scaler
from cryptomvp.data.windowing import make_windows
from cryptomvp.train.rl_env import RewardConfig
from cryptomvp.train.rl_train import train_reinforce
from cryptomvp.utils.logging import get_logger
from cryptomvp.utils.io import reports_dir
from cryptomvp.utils.run_dir import init_run_dir
from cryptomvp.utils.seed import set_seed
from cryptomvp.viz.plotting import plot_runs_with_band


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def run_reward_weights(config_path: str, fast: bool, run_dir: Path | None = None) -> None:
    init_run_dir(run_dir, config_path)
    cfg = load_config(config_path)
    logger = get_logger("reward_weights")
    set_seed(cfg.seed)

    dataset_path = Path(cfg.dataset.output_path)
    if not dataset_path.exists():
        raise RuntimeError(
            f"Dataset not found at {dataset_path}. Run scripts/test_build_dataset.py first."
        )
    df = _load_dataset(dataset_path)
    start_ms = int(df["open_time_ms"].min())
    end_ms = int(df["open_time_ms"].max())

    features = compute_features(df, cfg.features.list_of_features)
    X, window_times, feature_cols = make_windows(features, cfg.features.window_size_K)
    y_up, y_down = make_up_down_labels(df, window_times)
    y_up_dir, _ = make_directional_labels(df, window_times)

    n = min(len(X), len(y_up_dir))
    X = X[:n]
    y_up = y_up[:n]
    y_down = y_down[:n]
    y_up_dir = y_up_dir[:n]
    X_flat = X.reshape(len(X), -1)
    scaler_mean, scaler_std = fit_standard_scaler(X_flat)
    X_flat = apply_standard_scaler(X_flat, scaler_mean, scaler_std)
    up_rate = float(np.mean(y_up))
    down_rate = float(np.mean(y_down))
    flat_rate = max(0.0, 1.0 - up_rate - down_rate)

    num_episodes = 3 if fast else cfg.rl.episodes
    steps_per_episode = 50 if fast else cfg.rl.steps_per_episode
    hold_penalties = [0.01, 0.1, 0.2]
    histories = []
    labels = []
    reward_rows = []

    for rh in hold_penalties:
        reward_cfg = RewardConfig(
            R_correct=cfg.rl.reward.R_correct,
            R_wrong=cfg.rl.reward.R_wrong,
            R_opposite=cfg.rl.reward.R_opposite,
            R_hold=rh,
        )
        _, hist = train_reinforce(
            X_flat,
            y_up_dir,
            episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            lr=cfg.rl.lr,
            gamma=cfg.rl.gamma,
            reward_cfg=reward_cfg,
            policy_hidden_dim=cfg.rl.policy_hidden_dim,
            entropy_bonus=cfg.rl.entropy_bonus,
            seed=7,
            track_diagnostics=True,
            model_name=f"reward_hold_{rh}",
        )
        histories.append(hist)
        labels.append(f"R_hold={rh}")
        for idx, (reward, hold_rate, grad, delta) in enumerate(
            zip(hist.rewards, hist.hold_rates, hist.grad_norms, hist.delta_weight_norms)
        ):
            reward_rows.append(
                {
                    "R_hold": rh,
                    "episode": idx + 1,
                    "reward": reward,
                    "hold_rate": hold_rate,
                    "grad_norm": grad,
                    "delta_weight_norm": delta,
                }
            )

    episodes = np.arange(1, num_episodes + 1)
    report_dir = reports_dir("reward_weights")
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        report_dir / "feature_scaler.npz",
        mean=scaler_mean,
        std=scaler_std,
        feature_cols=np.array(feature_cols, dtype=object),
    )

    grad_runs = np.vstack([h.grad_norms for h in histories])
    delta_runs = np.vstack([h.delta_weight_norms for h in histories])
    hold_runs = np.vstack([h.hold_rates for h in histories])

    plot_runs_with_band(
        episodes,
        grad_runs,
        title="Gradient Norm vs Episode",
        xlabel="Episode",
        ylabel="Grad norm",
        out_base=fig_dir / "grad_norm",
        formats=cfg.viz.save_formats,
        labels=labels,
    )
    plot_runs_with_band(
        episodes,
        delta_runs,
        title="Delta Weight Norm vs Episode",
        xlabel="Episode",
        ylabel="Delta weight norm",
        out_base=fig_dir / "delta_weight_norm",
        formats=cfg.viz.save_formats,
        labels=labels,
    )
    plot_runs_with_band(
        episodes,
        hold_runs,
        title="Hold Rate vs Episode (R_hold sweep)",
        xlabel="Episode",
        ylabel="Hold rate",
        out_base=fig_dir / "hold_rate_sweep",
        formats=cfg.viz.save_formats,
        labels=labels,
    )

    pd.DataFrame(reward_rows).to_csv(report_dir / "reward_weights_metrics.csv", index=False)

    summary_lines = [
        "# Reward vs Weights Summary",
        f"Symbol: {cfg.symbol}",
        f"Interval: {cfg.interval}",
        f"Seed: {cfg.seed}",
        f"Start ms: {start_ms}",
        f"End ms: {end_ms}",
        f"Samples: {len(y_up)}",
        f"Class rates (UP/DOWN/FLAT): {up_rate:.4f}/{down_rate:.4f}/{flat_rate:.4f}",
        "Feature scaling: standard (global mean/std)",
        f"R_correct: {cfg.rl.reward.R_correct}",
        f"R_wrong: {cfg.rl.reward.R_wrong}",
        f"R_opposite: {cfg.rl.reward.R_opposite}",
        f"Policy hidden_dim: {cfg.rl.policy_hidden_dim}",
        "",
    ]
    for rh, hist in zip(hold_penalties, histories):
        summary_lines.append(
            f"R_hold={rh}: final reward={hist.rewards[-1]:.4f}, "
            f"final hold_rate={hist.hold_rates[-1]:.4f}, "
            f"final grad_norm={hist.grad_norms[-1]:.4f}, "
            f"final delta_weight_norm={hist.delta_weight_norms[-1]:.4f}"
        )

    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Reward weights report written to %s", report_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_reward_weights(args.config, fast=args.fast, run_dir=Path(args.run_dir) if args.run_dir else None)
