"""Staged feature selection (cheap filter -> top-N)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.config import load_config  # noqa: E402
from cryptomvp.features.registry import default_feature_sets_path  # noqa: E402
from cryptomvp.features.selection import staged_feature_selection  # noqa: E402
from cryptomvp.utils.io import reports_dir  # noqa: E402
from cryptomvp.utils.run_dir import init_run_dir  # noqa: E402
from cryptomvp.viz.plotting import plot_bar  # noqa: E402


def run_selection(config_path: str, run_dir: Path | None, top_n: int) -> None:
    init_run_dir(run_dir, config_path)
    cfg = load_config(config_path)

    dataset_path = Path(cfg.dataset.output_path)
    if not dataset_path.exists():
        raise RuntimeError(
            f"Dataset not found at {dataset_path}. Run scripts/test_build_dataset.py first."
        )
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
        top_n=top_n,
    )

    report_dir = reports_dir("feature_selection")
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "feature_set_id": score.feature_set_id,
            "score": score.score,
            "accuracy": score.accuracy,
            "roc_auc": score.roc_auc,
            "n_features": score.n_features,
        }
        for score in scores
    ]
    df_scores = pd.DataFrame(rows)
    df_scores.to_csv(report_dir / "leaderboard.csv", index=False)

    plot_bar(
        df_scores["feature_set_id"].tolist(),
        df_scores["score"].tolist(),
        title="Top Feature Set Scores",
        xlabel="Feature set",
        ylabel="Score",
        out_base=figures_dir / "feature_set_scores",
        formats=cfg.viz.save_formats,
    )

    summary_lines = [
        "# Feature Selection Summary",
        f"Top-N: {top_n}",
        f"Window size: {cfg.features.window_size_K}",
        f"Feature sets path: {feature_sets_path}",
        "",
        "Top feature sets:",
    ]
    for row in rows:
        summary_lines.append(
            f"- {row['feature_set_id']}: score={row['score']:.4f}, "
            f"accuracy={row['accuracy']:.4f}, n_features={row['n_features']}"
        )
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    run_selection(args.config, Path(args.run_dir) if args.run_dir else None, args.top_n)
