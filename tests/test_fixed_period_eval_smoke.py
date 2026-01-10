from pathlib import Path

import pytest
import yaml

from cryptomvp.data.build_dataset import build_synthetic_dataset, save_dataset
from cryptomvp.evaluate_fixed_period import run_fixed_period_eval


def test_fixed_period_eval_smoke(tmp_path: Path):
    start_ms = 1_700_000_000_000
    end_ms = start_ms + 5 * 24 * 60 * 60 * 1000
    df = build_synthetic_dataset(start_ms=start_ms, end_ms=end_ms, interval_ms=300_000)
    dataset_path = tmp_path / "synthetic.parquet"
    save_dataset(df, dataset_path)

    cfg = yaml.safe_load(Path("configs/mvp.yaml").read_text())
    cfg["device"] = "auto"
    cfg["allow_cpu_fallback"] = True
    cfg["dataset"]["start_ms"] = int(df["open_time_ms"].min())
    cfg["dataset"]["end_ms"] = int(df["open_time_ms"].max())
    cfg["dataset"]["output_path"] = str(dataset_path)
    cfg["supervised"]["epochs"] = 2
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    run_dir = tmp_path / "run"
    run_fixed_period_eval(
        config_path=str(cfg_path),
        start=None,
        end=None,
        session_mode="fixed_utc_partitions",
        session_strategy="global_model_session_thresholds",
        fast=True,
        run_dir=run_dir,
    )

    report_dir = run_dir / "reports" / "fixed_period"
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "decisions.parquet").exists()
