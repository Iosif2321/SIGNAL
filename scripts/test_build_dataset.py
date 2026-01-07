"""Test 2: Build and validate dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from cryptomvp.config import load_config
from cryptomvp.data.build_dataset import (
    build_synthetic_dataset,
    fetch_klines_range,
    save_dataset,
)
from cryptomvp.data.validate_dataset import validate_dataset
from cryptomvp.utils.io import reports_dir
from cryptomvp.utils.logging import get_logger
from cryptomvp.utils.seed import set_seed
from cryptomvp.viz.plotting import plot_histogram, plot_series_with_band


def run_build_dataset(config_path: str, fast: bool) -> Path:
    cfg = load_config(config_path)
    logger = get_logger("dataset")
    set_seed(42)

    interval_ms = int(cfg.interval) * 60_000
    if fast:
        start_ms = 0
        end_ms = interval_ms * 500
        df = build_synthetic_dataset(start_ms, end_ms, seed=11, interval_ms=interval_ms)
        output_path = Path("data/processed/synthetic.parquet")
        out = save_dataset(df, output_path)
    else:
        if cfg.dataset.start_ms is None or cfg.dataset.end_ms is None:
            raise RuntimeError("Dataset start_ms/end_ms are required in config.")

        from cryptomvp.bybit.rest import BybitRestClient
        interval_ms = int(cfg.interval) * 60_000

        client = BybitRestClient()
        try:
            candles = fetch_klines_range(
                client=client,
                category=cfg.category,
                symbol=cfg.symbol,
                interval=cfg.interval,
                start_ms=cfg.dataset.start_ms - interval_ms,
                end_ms=cfg.dataset.end_ms + interval_ms,
                limit=cfg.dataset.limit_per_call,
            )
        finally:
            client.close()
        df = pd.DataFrame([c.__dict__ for c in candles])
        df = df.drop(columns=["confirm", "source"], errors="ignore")
        df = df.sort_values("open_time_ms").drop_duplicates("open_time_ms").reset_index(drop=True)
        df = df[
            (df["open_time_ms"] >= cfg.dataset.start_ms)
            & (df["open_time_ms"] < cfg.dataset.end_ms)
        ].reset_index(drop=True)
        out = save_dataset(df, Path(cfg.dataset.output_path))

    report = validate_dataset(df, interval_ms=interval_ms)
    report_dir = reports_dir("dataset")
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    times = df["open_time_ms"].to_numpy()
    closes = df["close"].to_numpy()
    returns = pd.Series(closes).pct_change().fillna(0.0).to_numpy()
    vol = pd.Series(returns).rolling(window=10, min_periods=1).std().to_numpy()

    plot_series_with_band(
        times,
        closes,
        window=cfg.viz.moving_window,
        title="Close Price",
        xlabel="Time (ms)",
        ylabel="Close",
        label="close",
        out_base=figures_dir / "close_price",
        formats=cfg.viz.save_formats,
    )

    plot_histogram(
        returns,
        bins=30,
        title="Returns Histogram",
        xlabel="Return",
        ylabel="Count",
        out_base=figures_dir / "returns_hist",
        formats=cfg.viz.save_formats,
    )

    plot_series_with_band(
        times,
        vol,
        window=cfg.viz.moving_window,
        title="Rolling Volatility",
        xlabel="Time (ms)",
        ylabel="Volatility",
        label="volatility",
        out_base=figures_dir / "volatility",
        formats=cfg.viz.save_formats,
    )

    summary = [
        "# Dataset Validation",
        f"Rows: {report.total_rows}",
        f"Monotonic: {report.monotonic}",
        f"Duplicates: {report.duplicate_count}",
        f"Missing: {report.missing_count}",
    ]
    (report_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    logger.info("Dataset saved to %s", out)
    return Path(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run_build_dataset(args.config, fast=args.fast)
