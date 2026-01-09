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
from cryptomvp.data.build_dataset import klines_to_dataframe
from cryptomvp.utils.io import data_dir, reports_dir
from cryptomvp.utils.logging import get_logger
from cryptomvp.utils.run_dir import init_run_dir
from cryptomvp.utils.seed import set_seed
from cryptomvp.viz.plotting import plot_bar, plot_histogram, plot_series_with_band, save_figure
from cryptomvp.viz.style import apply_style


def _write_jsonl(path: Path, candles: List[KlineCandle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for candle in candles:
            f.write(json.dumps(candle.__dict__) + "\n")


def _compare_frames(ws_df: pd.DataFrame, rest_df: pd.DataFrame) -> pd.DataFrame:
    merged = ws_df.merge(rest_df, on="open_time_ms", suffixes=("_ws", "_rest"), how="outer")
    return merged


def _plot_ecdf(values: np.ndarray, out_base: Path, formats: list[str]) -> None:
    apply_style()
    import matplotlib.pyplot as plt
    vals = np.sort(values)
    y = np.arange(1, len(vals) + 1) / len(vals)
    fig, ax = plt.subplots()
    ax.plot(vals, y, label="ECDF")
    ax.set_title("Close Diff ECDF")
    ax.set_xlabel("Diff")
    ax.set_ylabel("ECDF")
    ax.legend()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def run_parity(config_path: str, fast: bool, run_dir: Path | None = None) -> None:
    init_run_dir(run_dir, config_path)
    cfg = load_config(config_path)
    logger = get_logger("parity")
    set_seed(cfg.seed)

    interval_ms = int(cfg.interval) * 60_000
    target_candles = cfg.parity.target_closed_candles
    max_wait_sec = cfg.parity.max_wait_sec

    if fast:
        target_candles = min(target_candles, 3)

    topic = cfg.parity.ws_topic
    from cryptomvp.bybit.ws import collect_klines_sync

    ws_candles = collect_klines_sync(
        topic=topic,
        target_candles=target_candles,
        max_wait_sec=max_wait_sec,
    )
    if not ws_candles:
        raise RuntimeError("No WS candles collected.")
    ws_df = klines_to_dataframe(ws_candles)
    start_ms = int(ws_df["open_time_ms"].min())
    end_ms = int(ws_df["open_time_ms"].max()) + interval_ms

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

    data_dir("raw", "parity")
    _write_jsonl(data_dir("raw", "parity") / "ws_klines.jsonl", ws_candles)
    _write_jsonl(data_dir("raw", "parity") / "rest_klines.jsonl", rest_candles)

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
    _plot_ecdf(
        diffs["close"].fillna(0.0).to_numpy(),
        out_base=figures_dir / "close_diff_ecdf",
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
        f"Symbol: {cfg.symbol}",
        f"Interval: {cfg.interval}",
        f"Seed: {cfg.seed}",
        f"WS candles: {len(ws_df)}",
        f"REST candles: {len(rest_df)}",
        f"Merged rows: {len(merged)}",
        f"Target closed candles: {target_candles}",
        f"Max wait sec: {max_wait_sec}",
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
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_parity(args.config, fast=args.fast, run_dir=Path(args.run_dir) if args.run_dir else None)
