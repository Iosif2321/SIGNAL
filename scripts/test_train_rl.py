"""Test 4: RL training for UP/DOWN policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.config import load_config
from cryptomvp.data.build_dataset import build_synthetic_dataset
from cryptomvp.data.features import compute_features
from cryptomvp.data.labels import make_up_down_labels
from cryptomvp.data.windowing import make_windows
from cryptomvp.train.feature_importance import compute_feature_importance
from cryptomvp.train.rl_env import RewardConfig
from cryptomvp.train.rl_train import train_reinforce
from cryptomvp.utils.io import checkpoints_dir, reports_dir
from cryptomvp.utils.logging import get_logger
from cryptomvp.utils.run_dir import init_run_dir
from cryptomvp.utils.seed import set_seed
from cryptomvp.viz.plotting import plot_bar, plot_series_with_band


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


def run_rl(config_path: str, fast: bool, run_dir: Path | None = None) -> None:
    init_run_dir(run_dir, config_path)
    cfg = load_config(config_path)
    logger = get_logger("rl")
    set_seed(cfg.seed)

    if fast:
        interval_ms = int(cfg.interval) * 60_000
        df = build_synthetic_dataset(0, interval_ms * 800, seed=21, interval_ms=interval_ms)
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
    up_rate = float(np.mean(y_up))
    down_rate = float(np.mean(y_down))
    flat_rate = max(0.0, 1.0 - up_rate - down_rate)

    X_flat = X.reshape(len(X), -1)
    times = window_times[:n]
    num_episodes = 3 if fast else cfg.rl.episodes
    steps_per_episode = 50 if fast else cfg.rl.steps_per_episode

    reward_cfg = RewardConfig(
        R_correct=cfg.rl.reward.R_correct,
        R_wrong=cfg.rl.reward.R_wrong,
        R_hold=cfg.rl.reward.R_hold,
    )

    # UP policy
    up_policy, up_hist = train_reinforce(
        X_flat,
        y_up,
        episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        lr=cfg.rl.lr,
        gamma=cfg.rl.gamma,
        reward_cfg=reward_cfg,
        entropy_bonus=cfg.rl.entropy_bonus,
        seed=7,
        track_diagnostics=True,
        track_steps=True,
        times=times,
        model_name="up",
    )
    torch.save(up_policy.state_dict(), checkpoints_dir() / "rl_up.pt")

    up_report = reports_dir("rl_up")
    up_fig = up_report / "figures"
    up_fig.mkdir(parents=True, exist_ok=True)
    episode_idx = np.arange(1, len(up_hist.rewards) + 1)

    plot_series_with_band(
        episode_idx,
        up_hist.rewards,
        window=cfg.viz.moving_window,
        title="UP Reward per Episode",
        xlabel="Episode",
        ylabel="Reward",
        label="reward",
        out_base=up_fig / "reward",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episode_idx,
        up_hist.hold_rates,
        window=cfg.viz.moving_window,
        title="UP Hold Rate",
        xlabel="Episode",
        ylabel="Hold rate",
        label="hold_rate",
        out_base=up_fig / "hold_rate",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episode_idx,
        up_hist.entropies,
        window=cfg.viz.moving_window,
        title="UP Policy Entropy",
        xlabel="Episode",
        ylabel="Entropy",
        label="entropy",
        out_base=up_fig / "entropy",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episode_idx,
        up_hist.accuracies,
        window=cfg.viz.moving_window,
        title="UP Accuracy (non-hold)",
        xlabel="Episode",
        ylabel="Accuracy",
        label="accuracy",
        out_base=up_fig / "accuracy",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episode_idx,
        up_hist.weight_norms,
        window=cfg.viz.moving_window,
        title="UP Weight Norm",
        xlabel="Episode",
        ylabel="Weight norm",
        label="weight_norm",
        out_base=up_fig / "weight_norm",
        formats=cfg.viz.save_formats,
    )
    if up_hist.step_logs:
        step_df = pd.DataFrame(up_hist.step_logs)
        state_cols = _build_feature_columns(feature_cols, cfg.features.window_size_K)
        state_values = np.vstack(step_df.pop("state"))
        state_df = pd.DataFrame(state_values, columns=state_cols)
        step_df = pd.concat([step_df, state_df], axis=1)
        step_df.to_parquet(up_report / "step_log.parquet", index=False)
    up_imp, up_imp_agg = compute_feature_importance(
        up_policy, feature_cols, cfg.features.window_size_K
    )
    up_imp.to_csv(up_report / "feature_importance.csv", index=False)
    up_imp_agg.to_csv(up_report / "feature_importance_by_feature.csv", index=False)
    plot_bar(
        up_imp_agg["feature"].tolist(),
        up_imp_agg["importance"].tolist(),
        title="UP Feature Importance (mean abs weight)",
        xlabel="Feature",
        ylabel="Importance",
        out_base=up_fig / "feature_importance",
        formats=cfg.viz.save_formats,
    )

    (up_report / "summary.md").write_text(
        "\n".join(
            [
                "# RL UP",
                f"Symbol: {cfg.symbol}",
                f"Interval: {cfg.interval}",
                f"Seed: {cfg.seed}",
                f"Start ms: {start_ms}",
                f"End ms: {end_ms}",
                f"Samples: {len(y_up)}",
                f"Class rates (UP/DOWN/FLAT): {up_rate:.4f}/{down_rate:.4f}/{flat_rate:.4f}",
                f"R_correct: {cfg.rl.reward.R_correct}",
                f"R_wrong: {cfg.rl.reward.R_wrong}",
                f"R_hold: {cfg.rl.reward.R_hold}",
                f"Entropy bonus: {cfg.rl.entropy_bonus}",
                f"Final reward: {up_hist.rewards[-1]:.4f}",
                f"Final hold_rate: {up_hist.hold_rates[-1]:.4f}",
                f"Final accuracy: {up_hist.accuracies[-1]:.4f}",
                f"Final weight_norm: {up_hist.weight_norms[-1]:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "episode": np.arange(1, len(up_hist.rewards) + 1),
            "reward": up_hist.rewards,
            "hold_rate": up_hist.hold_rates,
            "accuracy": up_hist.accuracies,
            "entropy": up_hist.entropies,
            "grad_norm": up_hist.grad_norms,
            "delta_weight_norm": up_hist.delta_weight_norms,
            "weight_norm": up_hist.weight_norms,
        }
    ).to_csv(up_report / "episode_metrics.csv", index=False)

    # DOWN policy
    down_policy, down_hist = train_reinforce(
        X_flat,
        y_down,
        episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        lr=cfg.rl.lr,
        gamma=cfg.rl.gamma,
        reward_cfg=reward_cfg,
        entropy_bonus=cfg.rl.entropy_bonus,
        seed=13,
        track_diagnostics=True,
        track_steps=True,
        times=times,
        model_name="down",
    )
    torch.save(down_policy.state_dict(), checkpoints_dir() / "rl_down.pt")

    down_report = reports_dir("rl_down")
    down_fig = down_report / "figures"
    down_fig.mkdir(parents=True, exist_ok=True)
    episode_idx = np.arange(1, len(down_hist.rewards) + 1)

    plot_series_with_band(
        episode_idx,
        down_hist.rewards,
        window=cfg.viz.moving_window,
        title="DOWN Reward per Episode",
        xlabel="Episode",
        ylabel="Reward",
        label="reward",
        out_base=down_fig / "reward",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episode_idx,
        down_hist.hold_rates,
        window=cfg.viz.moving_window,
        title="DOWN Hold Rate",
        xlabel="Episode",
        ylabel="Hold rate",
        label="hold_rate",
        out_base=down_fig / "hold_rate",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episode_idx,
        down_hist.entropies,
        window=cfg.viz.moving_window,
        title="DOWN Policy Entropy",
        xlabel="Episode",
        ylabel="Entropy",
        label="entropy",
        out_base=down_fig / "entropy",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episode_idx,
        down_hist.accuracies,
        window=cfg.viz.moving_window,
        title="DOWN Accuracy (non-hold)",
        xlabel="Episode",
        ylabel="Accuracy",
        label="accuracy",
        out_base=down_fig / "accuracy",
        formats=cfg.viz.save_formats,
    )
    plot_series_with_band(
        episode_idx,
        down_hist.weight_norms,
        window=cfg.viz.moving_window,
        title="DOWN Weight Norm",
        xlabel="Episode",
        ylabel="Weight norm",
        label="weight_norm",
        out_base=down_fig / "weight_norm",
        formats=cfg.viz.save_formats,
    )
    if down_hist.step_logs:
        step_df = pd.DataFrame(down_hist.step_logs)
        state_cols = _build_feature_columns(feature_cols, cfg.features.window_size_K)
        state_values = np.vstack(step_df.pop("state"))
        state_df = pd.DataFrame(state_values, columns=state_cols)
        step_df = pd.concat([step_df, state_df], axis=1)
        step_df.to_parquet(down_report / "step_log.parquet", index=False)
    down_imp, down_imp_agg = compute_feature_importance(
        down_policy, feature_cols, cfg.features.window_size_K
    )
    down_imp.to_csv(down_report / "feature_importance.csv", index=False)
    down_imp_agg.to_csv(down_report / "feature_importance_by_feature.csv", index=False)
    plot_bar(
        down_imp_agg["feature"].tolist(),
        down_imp_agg["importance"].tolist(),
        title="DOWN Feature Importance (mean abs weight)",
        xlabel="Feature",
        ylabel="Importance",
        out_base=down_fig / "feature_importance",
        formats=cfg.viz.save_formats,
    )

    (down_report / "summary.md").write_text(
        "\n".join(
            [
                "# RL DOWN",
                f"Symbol: {cfg.symbol}",
                f"Interval: {cfg.interval}",
                f"Seed: {cfg.seed}",
                f"Start ms: {start_ms}",
                f"End ms: {end_ms}",
                f"Samples: {len(y_down)}",
                f"Class rates (UP/DOWN/FLAT): {up_rate:.4f}/{down_rate:.4f}/{flat_rate:.4f}",
                f"R_correct: {cfg.rl.reward.R_correct}",
                f"R_wrong: {cfg.rl.reward.R_wrong}",
                f"R_hold: {cfg.rl.reward.R_hold}",
                f"Entropy bonus: {cfg.rl.entropy_bonus}",
                f"Final reward: {down_hist.rewards[-1]:.4f}",
                f"Final hold_rate: {down_hist.hold_rates[-1]:.4f}",
                f"Final accuracy: {down_hist.accuracies[-1]:.4f}",
                f"Final weight_norm: {down_hist.weight_norms[-1]:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "episode": np.arange(1, len(down_hist.rewards) + 1),
            "reward": down_hist.rewards,
            "hold_rate": down_hist.hold_rates,
            "accuracy": down_hist.accuracies,
            "entropy": down_hist.entropies,
            "grad_norm": down_hist.grad_norms,
            "delta_weight_norm": down_hist.delta_weight_norms,
            "weight_norm": down_hist.weight_norms,
        }
    ).to_csv(down_report / "episode_metrics.csv", index=False)

    logger.info("RL training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_rl(args.config, fast=args.fast, run_dir=Path(args.run_dir) if args.run_dir else None)
