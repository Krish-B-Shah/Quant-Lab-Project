from __future__ import annotations

import asyncio
import os
from typing import Optional

import aiohttp
import pandas as pd

from .fetcher import BaseFetcher


class PolygonAioFetcher(BaseFetcher):
    """Async aiohttp-based Polygon fetcher for daily aggregates.

    This implementation follows next_url pagination and uses retries with exponential backoff.
    """

    BASE = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self._session = session

    async def _get_json(self, url: str, params: dict, session: aiohttp.ClientSession):
        attempts = 0
        while True:
            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception:
                attempts += 1
                if attempts >= 4:
                    raise
                await asyncio.sleep(2 ** attempts)

    async def fetch(self, ticker: str, start: Optional[str], end: Optional[str], interval: str) -> pd.DataFrame:
        if interval not in ("1d", "daily", "day"):
            raise NotImplementedError("PolygonAioFetcher currently supports daily intervals only")
        frm = start or "1900-01-01"
        to = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        url = f"{self.BASE}/v2/aggs/ticker/{ticker}/range/1/day/{frm}/{to}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}
        session = self._session
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        results = []
        try:
            next_url = url
            next_params = params
            while next_url:
                data = await self._get_json(next_url, next_params, session)
                batch = data.get("results", [])
                if batch:
                    results.extend(batch)
                next_url = data.get("next_url")
                if next_url:
                    next_params = {}
            if not results:
                return pd.DataFrame()
            df = pd.DataFrame(results)
            df["datetime"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df["adj_close"] = df["close"].astype(float)
            df["ticker"] = ticker
            return df[["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"]]
        finally:
            if close_session:
                await session.close()
