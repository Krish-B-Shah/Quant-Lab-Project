from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd

from .fetcher import BaseFetcher, YahooFetcher

logger = logging.getLogger(__name__)


class DataManager:
    """Responsible for fetching price data, running feature generation, and storing results.

    - Uses asyncio.to_thread to run blocking yfinance downloads in parallel.
    - Works with a storage adapter that implements get_latest_timestamp and upsert_price_data.
    - Feature generation functions are passed in (callable that takes a price df and returns df).
    """

    def __init__(self, storage_adapter, fetcher: BaseFetcher | None = None, features: Optional[List] = None, max_concurrency: int = 8):
        self.storage = storage_adapter
        self.fetcher = fetcher or YahooFetcher()
        self.features = features or []
        # semaphore for bounded concurrency (useful for rate-limited APIs)
        self._sem = asyncio.Semaphore(max_concurrency)

    async def _download_one(self, ticker: str, start: Optional[str], end: Optional[str], interval: str) -> pd.DataFrame:
        # delegate to fetcher
        try:
            async with self._sem:
                df = await self.fetcher.fetch(ticker, start, end, interval)
        except Exception:
            logger.exception("fetcher failed for %s", ticker)
            return pd.DataFrame()
        if df is None or df.empty:
            logger.info("no data downloaded for %s (start=%s end=%s)", ticker, start, end)
            return pd.DataFrame()
        return df

    async def fetch(self, tickers: Iterable[str], interval: str = "1d", start: Optional[str] = None, end: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
        tasks = []
        for t in tickers:
            # If not refresh, determine latest timestamp and set start accordingly
            s = start
            if not refresh:
                latest = self.storage.get_latest_timestamp(t)
                if latest is not None:
                    # add one day beyond latest to avoid overlap
                    s = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            tasks.append(self._download_one(t, s, end, interval))

        results = await asyncio.gather(*tasks)
        frames = [r for r in results if not r.empty]
        if not frames:
            logger.warning("no frames fetched for tickers: %s", list(tickers))
            return pd.DataFrame()
        combined = pd.concat(frames, axis=0, ignore_index=True)
        # conform columns
        combined = combined[["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"]]
        # store raw price rows
        for t in combined["ticker"].unique():
            tdf = combined[combined["ticker"] == t].copy()
            # convert datetime
            tdf["datetime"] = pd.to_datetime(tdf["datetime"])
            try:
                # pass source information when available
                src = getattr(self.fetcher, "name", None) or self.fetcher.__class__.__name__
                # attach source column for adapter to pick up
                tdf["source"] = src
                self.storage.upsert_price_data(t, tdf)
            except Exception:
                logger.exception("failed to upsert price data for %s", t)

        return combined

    def generate_and_store_features(self, tickers: Iterable[str], feature_funcs: List):
        for t in tickers:
            price_df = self.storage.read_price_data(t)
            if price_df is None or price_df.empty:
                logger.warning("no price data for %s; skipping features", t)
                continue
            df = price_df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")
            feat_df = df[["datetime"]].copy()
            for fn in feature_funcs:
                res = fn(df)
                # fn may return a DataFrame with datetime index or column; align on datetime
                if isinstance(res, pd.Series):
                    feat_df[res.name] = res.values
                elif isinstance(res, pd.DataFrame):
                    for c in res.columns:
                        feat_df[c] = res[c].values
            # store features
            self.storage.upsert_feature_data(t, feat_df)
