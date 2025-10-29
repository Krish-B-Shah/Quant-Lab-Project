from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf


class BaseFetcher(ABC):
    @abstractmethod
    async def fetch(self, ticker: str, start: Optional[str], end: Optional[str], interval: str) -> pd.DataFrame:
        """Fetch raw OHLCV dataframe for a single ticker."""


class YahooFetcher(BaseFetcher):
    async def fetch(self, ticker: str, start: Optional[str], end: Optional[str], interval: str) -> pd.DataFrame:
        def _sync():
            return yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)

        df = await asyncio.to_thread(_sync)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={
            "Date": "datetime",
            "Adj Close": "adj_close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df["ticker"] = ticker
        return df


def get_fetcher(name: Optional[str] = None, **kwargs) -> BaseFetcher:
    """Factory: return a fetcher instance by name. Defaults to YahooFetcher.

    Supported names: 'yahoo', 'polygon', 'alpha_vantage' (case-insensitive)
    """
    if not name:
        return YahooFetcher()
    n = name.lower()
    if n in ("yahoo", "yfinance"):
        return YahooFetcher()
    if n in ("polygon", "polygonio", "polygon.io"):
        # prefer aiohttp async implementation if requested
        if kwargs.get("async") or n == "polygon_async":
            from .polygon_aio_fetcher import PolygonAioFetcher

            return PolygonAioFetcher(api_key=kwargs.get("api_key"))
        from .polygon_fetcher import PolygonFetcher

        return PolygonFetcher(api_key=kwargs.get("api_key"))
    if n in ("alpha", "alpha_vantage", "alphavantage"):
        from .alpha_fetcher import AlphaVantageFetcher

        return AlphaVantageFetcher(api_key=kwargs.get("api_key"))
    # fallback
    return YahooFetcher()
