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
