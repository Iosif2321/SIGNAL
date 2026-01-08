"""Raw WS vs REST ping comparison (string/byte-level)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.config import load_config  # noqa: E402
from cryptomvp.utils.io import data_dir, reports_dir  # noqa: E402
from cryptomvp.utils.logging import get_logger  # noqa: E402
from cryptomvp.utils.run_dir import init_run_dir  # noqa: E402
from cryptomvp.utils.seed import set_seed  # noqa: E402
from cryptomvp.viz.plotting import plot_bar, plot_histogram  # noqa: E402

WS_URL = "wss://stream.bybit.com/v5/public/spot"
REST_URL = "https://api.bybit.com/v5/market/kline"


async def _collect_ws_candle(topic: str, timeout_sec: int) -> Dict[str, str]:
    start_time = time.time()
    async with websockets.connect(
        WS_URL,
        ping_interval=15,
        ping_timeout=10,
        max_queue=1000,
    ) as ws:
        sub_msg = {"op": "subscribe", "args": [topic]}
        await ws.send(json.dumps(sub_msg))
        while time.time() - start_time < timeout_sec:
            msg = await ws.recv()
            payload = json.loads(msg)
            if payload.get("topic") != topic:
                continue
            items = payload.get("data", []) or []
            for item in items:
                if not item.get("confirm"):
                    continue
                return {
                    "open_time_ms": str(item.get("start")),
                    "open": str(item.get("open")),
                    "high": str(item.get("high")),
                    "low": str(item.get("low")),
                    "close": str(item.get("close")),
                    "volume": str(item.get("volume")),
                    "turnover": str(item.get("turnover")),
                    "confirm": str(item.get("confirm")),
                    "source": "ws",
                }
    raise RuntimeError("WS timeout waiting for confirmed candle.")


def _fetch_rest_candle(
    category: str,
    symbol: str,
    interval: str,
    open_time_ms: str,
) -> Tuple[Optional[Dict[str, str]], bool]:
    start_ms = int(open_time_ms) - 60_000
    end_ms = int(open_time_ms) + 60_000
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "end": end_ms,
        "limit": 200,
    }
    try:
        resp = httpx.get(REST_URL, params=params, timeout=10.0)
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPError:
        return None, False
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit REST error: {payload}")
    rows = payload.get("result", {}).get("list", []) or []
    if not rows:
        return None, False
    match = None
    for row in rows:
        if str(row[0]) == open_time_ms:
            match = row
            break
    row = match or rows[0]
    return {
        "open_time_ms": str(row[0]),
        "open": str(row[1]),
        "high": str(row[2]),
        "low": str(row[3]),
        "close": str(row[4]),
        "volume": str(row[5]),
        "turnover": str(row[6]),
        "confirm": "null",
        "source": "rest",
    }, match is not None


def _compare_strings(left: str, right: str) -> Dict[str, object]:
    left_b = left.encode("utf-8")
    right_b = right.encode("utf-8")
    min_len = min(len(left_b), len(right_b))
    mismatch_idx = None
    for i in range(min_len):
        if left_b[i] != right_b[i]:
            mismatch_idx = i
            break
    if mismatch_idx is None and len(left_b) != len(right_b):
        mismatch_idx = min_len
    return {
        "equal": left_b == right_b,
        "left_len": len(left_b),
        "right_len": len(right_b),
        "mismatch_idx": mismatch_idx,
        "left": left,
        "right": right,
    }


def run_ping_compare(
    config_path: str,
    duration_sec: int,
    rest_delay_sec: int,
    max_retries: int,
    retry_delay_sec: int,
    run_dir: Path | None = None,
) -> None:
    init_run_dir(run_dir, config_path)
    cfg = load_config(config_path)
    logger = get_logger("ping_compare_raw")
    set_seed(cfg.seed)

    topic = f"kline.{cfg.interval}.{cfg.symbol}"
    ws_candle = asyncio_run(_collect_ws_candle(topic, timeout_sec=duration_sec))

    if rest_delay_sec > 0:
        logger.info("Waiting %s sec before REST ping...", rest_delay_sec)
        time.sleep(rest_delay_sec)

    rest_candle = None
    rest_match = False
    for attempt in range(max_retries):
        rest_candle, rest_match = _fetch_rest_candle(
            cfg.category,
            cfg.symbol,
            cfg.interval,
            ws_candle["open_time_ms"],
        )
        if rest_candle and rest_match:
            break
        rest_match = False
        if attempt < max_retries - 1:
            time.sleep(retry_delay_sec)

    if rest_candle is None:
        raise RuntimeError("REST candle with matching open_time_ms not found.")

    out_dir = data_dir("raw", "ping_raw")
    (out_dir / "ws_candle.json").write_text(json.dumps(ws_candle, indent=2), encoding="utf-8")
    (out_dir / "rest_candle.json").write_text(json.dumps(rest_candle, indent=2), encoding="utf-8")

    fields = ["open_time_ms", "open", "high", "low", "close", "volume", "turnover"]
    comparisons = {field: _compare_strings(ws_candle[field], rest_candle[field]) for field in fields}

    report_dir = reports_dir("ping_compare_raw")
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    abs_len_diff = [abs(comparisons[f]["left_len"] - comparisons[f]["right_len"]) for f in fields]
    plot_bar(
        fields,
        abs_len_diff,
        title="Length Diff (bytes) WS vs REST",
        xlabel="Field",
        ylabel="Abs length diff",
        out_base=fig_dir / "len_diff",
        formats=cfg.viz.save_formats,
    )

    mismatch_positions = [
        float(comparisons[f]["mismatch_idx"]) if comparisons[f]["mismatch_idx"] is not None else 0.0
        for f in fields
    ]
    plot_histogram(
        mismatch_positions,
        bins=10,
        title="Mismatch Index Histogram",
        xlabel="First mismatch index (byte)",
        ylabel="Count",
        out_base=fig_dir / "mismatch_idx_hist",
        formats=cfg.viz.save_formats,
    )

    summary_lines = [
        "# Raw Ping Compare Summary",
        f"REST delay sec: {rest_delay_sec}",
        f"REST exact match: {rest_match}",
        f"WS open_time_ms: {ws_candle['open_time_ms']}",
        f"REST open_time_ms: {rest_candle['open_time_ms']}",
        "",
        "Field comparisons:",
    ]
    for field in fields:
        info = comparisons[field]
        summary_lines.append(
            f"- {field}: equal={info['equal']}, left_len={info['left_len']}, "
            f"right_len={info['right_len']}, mismatch_idx={info['mismatch_idx']}"
        )
        summary_lines.append(f"  ws={info['left']}")
        summary_lines.append(f"  rest={info['right']}")

    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Raw ping compare report written to %s", report_dir)


def asyncio_run(coro):
    try:
        import asyncio

        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--duration-sec", type=int, default=120)
    parser.add_argument("--rest-delay-sec", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-delay-sec", type=int, default=10)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_ping_compare(
        args.config,
        duration_sec=args.duration_sec,
        rest_delay_sec=args.rest_delay_sec,
        max_retries=args.max_retries,
        retry_delay_sec=args.retry_delay_sec,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )
