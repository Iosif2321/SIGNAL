"""Bybit V5 WebSocket client for public spot data."""

from __future__ import annotations

import asyncio
import json
import time
from typing import List, Optional

import websockets

from cryptomvp.bybit.schemas import KlineCandle


WS_URL = "wss://stream.bybit.com/v5/public/spot"


def _parse_kline(item: dict) -> Optional[KlineCandle]:
    if not item.get("confirm"):
        return None
    return KlineCandle(
        open_time_ms=int(item["start"]),
        open=float(item["open"]),
        high=float(item["high"]),
        low=float(item["low"]),
        close=float(item["close"]),
        volume=float(item["volume"]),
        turnover=float(item["turnover"]),
        confirm=bool(item.get("confirm")),
        source="ws",
    )


async def collect_klines(
    topic: str,
    duration_sec: Optional[int] = None,
    max_candles: Optional[int] = None,
    target_candles: Optional[int] = None,
    max_wait_sec: Optional[int] = None,
) -> List[KlineCandle]:
    """Collect closed klines from Bybit WS with count or time limits."""
    candles: List[KlineCandle] = []
    start_time = time.time()
    target_count = target_candles or max_candles
    wait_limit = max_wait_sec or duration_sec or 60

    async with websockets.connect(
        WS_URL,
        ping_interval=15,
        ping_timeout=10,
        max_queue=1000,
    ) as ws:
        sub_msg = {"op": "subscribe", "args": [topic]}
        await ws.send(json.dumps(sub_msg))

        while time.time() - start_time < wait_limit:
            try:
                remaining = wait_limit - (time.time() - start_time)
                timeout = max(1.0, min(5.0, remaining))
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
            data = json.loads(msg)
            if data.get("topic") != topic:
                continue
            items = data.get("data", []) or []
            for item in items:
                candle = _parse_kline(item)
                if candle:
                    candles.append(candle)
                    if target_count and len(candles) >= target_count:
                        return candles
    return candles


def collect_klines_sync(
    topic: str,
    duration_sec: Optional[int] = None,
    max_candles: Optional[int] = None,
    target_candles: Optional[int] = None,
    max_wait_sec: Optional[int] = None,
) -> List[KlineCandle]:
    return asyncio.run(
        collect_klines(
            topic,
            duration_sec=duration_sec,
            max_candles=max_candles,
            target_candles=target_candles,
            max_wait_sec=max_wait_sec,
        )
    )
