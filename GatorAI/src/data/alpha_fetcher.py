from __future__ import annotations

import asyncio
import os
from typing import Optional

import pandas as pd
import requests

from .fetcher import BaseFetcher


class AlphaVantageFetcher(BaseFetcher):
    """Alpha Vantage fetcher (daily) using TIME_SERIES_DAILY_ADJUSTED.

    This implementation is simple and uses requests in a thread to avoid
    adding an async HTTP dependency. Alpha Vantage has strict rate limits;
    consider adding a per-minute throttle if you fetch many tickers.
    """

    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")

    def _fetch_sync(self, ticker: str, start: Optional[str], end: Optional[str], interval: str):
        if interval not in ("1d", "daily", "day"):
            raise NotImplementedError("AlphaVantageFetcher currently supports daily intervals only")
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker, "outputsize": "full", "apikey": self.api_key}
        attempts = 0
        while True:
            try:
                r = requests.get(self.BASE, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                attempts += 1
                if attempts >= 4:
                    raise
                import time as _t

                _t.sleep(2 ** attempts)
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return pd.DataFrame()
        rows = []
        for dt_str, vals in ts.items():
            rows.append({
                "datetime": pd.to_datetime(dt_str),
                "open": float(vals.get("1. open")),
                "high": float(vals.get("2. high")),
                "low": float(vals.get("3. low")),
                "close": float(vals.get("4. close")),
                "adj_close": float(vals.get("5. adjusted close", vals.get("4. close"))),
                "volume": int(vals.get("6. volume", 0)),
            })
        df = pd.DataFrame(rows)
        if start:
            df = df[df["datetime"] >= pd.to_datetime(start)]
        if end:
            df = df[df["datetime"] <= pd.to_datetime(end)]
        if df.empty:
            return pd.DataFrame()
        df["ticker"] = ticker
        # ensure ordering asc
        df = df.sort_values("datetime")
        return df[["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"]]

    async def fetch(self, ticker: str, start: Optional[str], end: Optional[str], interval: str) -> pd.DataFrame:
        return await asyncio.to_thread(self._fetch_sync, ticker, start, end, interval)
