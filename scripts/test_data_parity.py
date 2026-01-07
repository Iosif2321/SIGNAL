"""Test 1: WS vs REST parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from cryptomvp.bybit.schemas import KlineCandle
from cryptomvp.config import load_config
from cryptomvp.data.build_dataset import build_synthetic_dataset, klines_to_dataframe
from cryptomvp.utils.io import data_dir, reports_dir
from cryptomvp.utils.logging import get_logger
from cryptomvp.utils.seed import set_seed
from cryptomvp.viz.plotting import plot_bar, plot_histogram, plot_series_with_band


def _write_jsonl(path: Path, candles: List[KlineCandle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for candle in candles:
            f.write(json.dumps(candle.__dict__) + "\n")


def _compare_frames(ws_df: pd.DataFrame, rest_df: pd.DataFrame) -> pd.DataFrame:
    merged = ws_df.merge(rest_df, on="open_time_ms", suffixes=("_ws", "_rest"), how="outer")
    return merged


def run_parity(config_path: str, fast: bool) -> None:
    cfg = load_config(config_path)
    logger = get_logger("parity")
    set_seed(42)

    if fast:
        interval_ms = int(cfg.interval) * 60_000
        start_ms = 0
        end_ms = interval_ms * 200
        ws_df = build_synthetic_dataset(start_ms, end_ms, seed=7, interval_ms=interval_ms)
        rest_df = ws_df.copy()
        rest_df["close"] = rest_df["close"] + np.random.normal(0, 0.5, size=len(rest_df))
        ws_candles = [
            KlineCandle(
                open_time_ms=int(row.open_time_ms),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                turnover=float(row.turnover),
                confirm=True,
                source="ws",
            )
            for row in ws_df.itertuples()
        ]
        rest_candles = [
            KlineCandle(
                open_time_ms=int(row.open_time_ms),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                turnover=float(row.turnover),
                source="rest",
            )
            for row in rest_df.itertuples()
        ]
    else:
        topic = cfg.parity.ws_topic
        from cryptomvp.bybit.ws import collect_klines_sync
        ws_candles = collect_klines_sync(topic=topic, duration_sec=cfg.parity.duration_sec)
        if not ws_candles:
            raise RuntimeError("No WS candles collected.")
        ws_df = klines_to_dataframe(ws_candles)
        start_ms = int(ws_df["open_time_ms"].min())
        end_ms = int(ws_df["open_time_ms"].max()) + 60_000

        from cryptomvp.bybit.rest import BybitRestClient

        client = BybitRestClient()
        try:
            rest_candles = client.get_klines(
                category=cfg.category,
                symbol=cfg.symbol,
                interval=cfg.interval,
                start_ms=start_ms,
                end_ms=end_ms,
                limit=cfg.dataset.limit_per_call,
            )
        finally:
            client.close()
        rest_df = klines_to_dataframe(rest_candles)

    data_dir("parity")
    _write_jsonl(data_dir("parity") / "ws_klines.jsonl", ws_candles)
    _write_jsonl(data_dir("parity") / "rest_klines.jsonl", rest_candles)

    merged = _compare_frames(ws_df, rest_df)
    merged = merged.sort_values("open_time_ms").reset_index(drop=True)

    diff_cols = ["open", "high", "low", "close", "volume", "turnover"]
    diffs = {}
    mismatch_counts = {}
    for col in diff_cols:
        a = merged[f"{col}_ws"]
        b = merged[f"{col}_rest"]
        diff = a - b
        diffs[col] = diff
        mismatch_counts[col] = int(np.sum(~np.isclose(a, b, equal_nan=False)))

    report_dir = reports_dir("parity")
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    x = merged["open_time_ms"].to_numpy()
    plot_series_with_band(
        x,
        diffs["close"].fillna(0.0),
        window=cfg.viz.moving_window,
        title="WS-REST Close Diff",
        xlabel="Time (ms)",
        ylabel="Diff",
        label="close_diff",
        out_base=figures_dir / "close_diff",
        formats=cfg.viz.save_formats,
    )

    plot_histogram(
        diffs["close"].fillna(0.0),
        bins=30,
        title="Close Diff Histogram",
        xlabel="Diff",
        ylabel="Count",
        out_base=figures_dir / "close_diff_hist",
        formats=cfg.viz.save_formats,
    )

    plot_bar(
        list(mismatch_counts.keys()),
        list(mismatch_counts.values()),
        title="Mismatch Counts",
        xlabel="Field",
        ylabel="Count",
        out_base=figures_dir / "mismatch_counts",
        formats=cfg.viz.save_formats,
    )

    summary_path = report_dir / "summary.md"
    summary_lines = [
        "# Parity Summary",
        f"WS candles: {len(ws_df)}",
        f"REST candles: {len(rest_df)}",
        f"Merged rows: {len(merged)}",
        "",
        "Mismatch counts:",
    ]
    for k, v in mismatch_counts.items():
        summary_lines.append(f"- {k}: {v}")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Parity report written to %s", report_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run_parity(args.config, fast=args.fast)
