"""Bybit V5 REST client."""

from __future__ import annotations

from typing import List, Optional

import httpx

from cryptomvp.bybit.schemas import KlineCandle


BASE_URL = "https://api.bybit.com"


class BybitRestClient:
    """Minimal Bybit REST client for public market data."""

    def __init__(self, base_url: str = BASE_URL, timeout: float = 10.0) -> None:
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def get_klines(
        self,
        category: str,
        symbol: str,
        interval: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> List[KlineCandle]:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)

        resp = self._client.get("/v5/market/kline", params=params)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("retCode") != 0:
            raise RuntimeError(f"Bybit REST error: {payload}")
        rows = payload.get("result", {}).get("list", []) or []

        candles: List[KlineCandle] = []
        for row in rows:
            open_time_ms = int(row[0])
            candle = KlineCandle(
                open_time_ms=open_time_ms,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                turnover=float(row[6]),
                source="rest",
            )
            candles.append(candle)

        candles.sort(key=lambda c: c.open_time_ms)
        return candles