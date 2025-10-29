from __future__ import annotations

import asyncio
import os
from typing import Optional

import pandas as pd
import requests

from .fetcher import BaseFetcher


class PolygonFetcher(BaseFetcher):
    """Simple Polygon.io fetcher using the aggregates endpoint.

    This implementation uses blocking requests wrapped in asyncio.to_thread so it
    works without adding aiohttp as a dependency. For production use you may
    want a full aiohttp implementation with retries and pagination handling.
    """

    BASE = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")

    def _fetch_sync(self, ticker: str, start: Optional[str], end: Optional[str], interval: str):
        # support daily aggregates for now
        if interval not in ("1d", "daily", "day"):
            raise NotImplementedError("PolygonFetcher currently supports daily intervals only")
        frm = start or "1900-01-01"
        to = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")

        # basic retry/backoff
        url = f"{self.BASE}/v2/aggs/ticker/{ticker}/range/1/day/{frm}/{to}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}
        attempts = 0
        results = []
        next_url = url
        next_params = params
        while next_url:
            attempts = 0
            while attempts < 4:
                try:
                    r = requests.get(next_url, params=next_params, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    batch = data.get("results", [])
                    if batch:
                        results.extend(batch)
                    # polygon may include a 'next_url' in the response for pagination
                    next_url = data.get("next_url")
                    # if next_url is present do not pass params again
                    if next_url:
                        next_params = {}
                    else:
                        next_url = None
                    break
                except Exception as exc:
                    attempts += 1
                    wait = 2 ** attempts
                    time_msg = f"(attempt {attempts}) sleeping {wait}s"
                    # simple backoff
                    import time as _t

                    _t.sleep(wait)
                    if attempts >= 4:
                        raise

        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        # polygon returns 't' (ms since epoch), 'o','h','l','c','v'
        df["datetime"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["adj_close"] = df["close"].astype(float)
        df["ticker"] = ticker
        return df[["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"]]

    async def fetch(self, ticker: str, start: Optional[str], end: Optional[str], interval: str) -> pd.DataFrame:
        return await asyncio.to_thread(self._fetch_sync, ticker, start, end, interval)
