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
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.analysis.adaptation import assess_adaptation  # noqa: E402
from cryptomvp.config import load_config  # noqa: E402
from cryptomvp.utils.io import reports_dir, run_root  # noqa: E402
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
    if "true_direction" in df.columns:
        true_dir = df["true_direction"].astype(str)
        if (decisions == "UP").any():
            precision_up = float((true_dir[decisions == "UP"] == "UP").mean())
        if (decisions == "DOWN").any():
            precision_down = float((true_dir[decisions == "DOWN"] == "DOWN").mean())
    metrics = {
        "decision_hold_rate": hold_rate,
        "decision_action_rate": action_rate,
        "decision_action_accuracy_non_hold": action_accuracy,
        "decision_conflict_rate": conflict_rate,
        "decision_precision_up": precision_up,
        "decision_precision_down": precision_down,
    }
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

        metrics: Dict[str, float] = {}
        metrics.update(decision_metrics)
        metrics.update(sup_up_metrics)
        metrics.update(sup_down_metrics)
        metrics.update(rl_up_metrics)
        metrics.update(rl_down_metrics)

        reward_base = (
            cfg.tuner.reward.decision_accuracy_weight * metrics["decision_action_accuracy_non_hold"]
            + cfg.tuner.reward.decision_action_rate_weight * metrics["decision_action_rate"]
            - cfg.tuner.reward.decision_conflict_penalty * metrics["decision_conflict_rate"]
            - cfg.tuner.reward.decision_hold_penalty * metrics["decision_hold_rate"]
            + cfg.tuner.reward.rl_up_accuracy_weight * metrics["rl_up_accuracy"]
            + cfg.tuner.reward.rl_down_accuracy_weight * metrics["rl_down_accuracy"]
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
        reward_total = float(metrics.get("score_base", 0.0) + reward_improve)
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
        f"Param space entries: {len(param_space)}",
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
        f"up_hold_penalty={cfg.tuner.reward.rl_up_hold_penalty}, "
        f"down_hold_penalty={cfg.tuner.reward.rl_down_hold_penalty}",
        f"Improve weights: up={cfg.tuner.reward.improve_up_accuracy_weight}, "
        f"down={cfg.tuner.reward.improve_down_accuracy_weight}",
        f"Best reward: {best_state['reward']:.4f}",
        adaptation_line,
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
