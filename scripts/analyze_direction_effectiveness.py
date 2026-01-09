"""Analyze direction effectiveness for a specific run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.analysis.direction_effectiveness import (  # noqa: E402
    AnalysisInputs,
    analyze_run,
    list_run_artifacts,
    load_dataframe,
)
from cryptomvp.config import load_config  # noqa: E402
from cryptomvp.utils.logging import get_logger  # noqa: E402


def _print_df_info(logger, name: str, path: Path) -> None:
    if not path.exists():
        logger.info("%s: missing (%s)", name, path)
        return
    df = load_dataframe(path)
    logger.info("%s columns: %s", name, list(df.columns))
    logger.info("%s head(3):\n%s", name, df.head(3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--delta-min", type=float, default=None)
    parser.add_argument("--rolling-window", type=int, default=None)
    parser.add_argument("--scan-thresholds", action="store_true")
    parser.add_argument("--t-min", type=float, default=0.45)
    parser.add_argument("--t-max", type=float, default=0.75)
    parser.add_argument("--t-step", type=float, default=0.01)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logger = get_logger("analysis.cli")
    run_dir = Path(args.run)
    cfg = load_config(args.config)

    threshold = args.threshold if args.threshold is not None else cfg.decision_rule.T_min
    delta_min = args.delta_min if args.delta_min is not None else cfg.decision_rule.delta_min
    rolling_window = args.rolling_window if args.rolling_window is not None else cfg.viz.moving_window
    out_dir = Path(args.out) if args.out else run_dir / "reports/direction_effectiveness"

    artifacts = list_run_artifacts(run_dir)
    logger.info("Run artifacts:")
    for name, path in artifacts.items():
        logger.info("- %s: %s (exists=%s)", name, path, path.exists())
    logger.info("Dataframe previews (columns + head(3)):")
    for name, path in artifacts.items():
        _print_df_info(logger, name, path)

    inputs = AnalysisInputs(
        run_dir=run_dir,
        out_dir=out_dir,
        threshold=threshold,
        delta_min=delta_min,
        rolling_window=rolling_window,
        formats=cfg.viz.save_formats,
        scan_thresholds=args.scan_thresholds,
        t_min=args.t_min,
        t_max=args.t_max,
        t_step=args.t_step,
        config_path=Path(args.config),
    )

    analyze_run(inputs)
    logger.info("Analysis complete. Output: %s", out_dir)


if __name__ == "__main__":
    main()
