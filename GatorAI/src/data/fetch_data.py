from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from .utils import RAW_DIR, ensure_data_directories


def fetch_ohlc(
	tickers: Iterable[str],
	start: Optional[str] = None,
	end: Optional[str] = None,
	interval: str = "1d",
) -> pd.DataFrame:
	"""Fetch OHLCV data for tickers and save individual CSVs under data/raw.

	Returns a long-format DataFrame with columns:
	[ticker, datetime, open, high, low, close, adj_close, volume]
	"""
	ensure_data_directories()

	if start is None:
		start = "2005-01-01"
	if end is None:
		end = datetime.utcnow().strftime("%Y-%m-%d")

	frames = []
	for t in tickers:
		df = yf.download(t, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
		if df.empty:
			continue
		df = df.rename(
			columns={
				"Open": "open",
				"High": "high",
				"Low": "low",
				"Close": "close",
				"Adj Close": "adj_close",
				"Volume": "volume",
			}
		).reset_index().rename(columns={"Date": "datetime"})
		df["ticker"] = t
		# Persist per-ticker csv
		out = RAW_DIR / f"{t}_{interval}.csv"
		df.to_csv(out, index=False)
		frames.append(df)

	if not frames:
		return pd.DataFrame(columns=["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"])

	combined = pd.concat(frames, axis=0, ignore_index=True)
	return combined[["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"]]
