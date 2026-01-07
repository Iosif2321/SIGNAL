"""Ping comparison between WS and REST for BTCUSDT spot."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.bybit.rest import BybitRestClient  # noqa: E402
from cryptomvp.bybit.ws import collect_klines_sync  # noqa: E402
from cryptomvp.config import load_config  # noqa: E402
from cryptomvp.utils.io import data_dir, reports_dir  # noqa: E402
from cryptomvp.utils.logging import get_logger  # noqa: E402
from cryptomvp.utils.seed import set_seed  # noqa: E402
from cryptomvp.viz.plotting import plot_bar, plot_histogram  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_ping_compare(config_path: str, duration_sec: int, rest_delay_sec: int) -> None:
    cfg = load_config(config_path)
    logger = get_logger("ping_compare")
    set_seed(42)

    topic = f"kline.{cfg.interval}.{cfg.symbol}"
    ws_candles = collect_klines_sync(topic=topic, duration_sec=duration_sec, max_candles=1)
    if not ws_candles:
        raise RuntimeError("No WS candle collected within duration.")
    ws_candle = ws_candles[0]

    if rest_delay_sec > 0:
        logger.info("Waiting %s sec before REST ping...", rest_delay_sec)
        time.sleep(rest_delay_sec)

    client = BybitRestClient()
    try:
        rest_candles = client.get_klines(
            category=cfg.category,
            symbol=cfg.symbol,
            interval=cfg.interval,
            start_ms=ws_candle.open_time_ms,
            end_ms=ws_candle.open_time_ms + 60_000,
            limit=1,
        )
    finally:
        client.close()

    rest_candle = None
    for candle in rest_candles:
        if candle.open_time_ms == ws_candle.open_time_ms:
            rest_candle = candle
            break
    if rest_candle is None and rest_candles:
        rest_candle = rest_candles[0]

    if rest_candle is None:
        raise RuntimeError("No REST candle returned.")

    data_dir("ping")
    _write_json(data_dir("ping") / "ws_candle.json", ws_candle.__dict__)
    _write_json(data_dir("ping") / "rest_candle.json", rest_candle.__dict__)

    fields = ["open", "high", "low", "close", "volume", "turnover"]
    diffs = {}
    rel_diffs = {}
    for field in fields:
        ws_val = getattr(ws_candle, field)
        rest_val = getattr(rest_candle, field)
        diff = ws_val - rest_val
        diffs[field] = diff
        denom = rest_val if rest_val != 0 else 1.0
        rel_diffs[field] = diff / denom

    report_dir = reports_dir("ping_compare")
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_bar(
        list(diffs.keys()),
        [abs(d) for d in diffs.values()],
        title="ABS Diff WS vs REST",
        xlabel="Field",
        ylabel="Abs diff",
        out_base=fig_dir / "abs_diff",
        formats=cfg.viz.save_formats,
    )

    plot_histogram(
        list(rel_diffs.values()),
        bins=10,
        title="Relative Diff Histogram",
        xlabel="Relative diff",
        ylabel="Count",
        out_base=fig_dir / "rel_diff_hist",
        formats=cfg.viz.save_formats,
    )

    summary_lines = [
        "# Ping Compare Summary",
        f"WS open_time_ms: {ws_candle.open_time_ms}",
        f"REST open_time_ms: {rest_candle.open_time_ms}",
        f"REST delay sec: {rest_delay_sec}",
        "",
        "Diffs (WS-REST):",
    ]
    for field in fields:
        summary_lines.append(
            f"- {field}: diff={diffs[field]:.8f} rel={rel_diffs[field]:.8f}"
        )
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Ping compare report written to %s", report_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--duration-sec", type=int, default=90)
    parser.add_argument("--rest-delay-sec", type=int, default=60)
    args = parser.parse_args()
    run_ping_compare(args.config, duration_sec=args.duration_sec, rest_delay_sec=args.rest_delay_sec)
