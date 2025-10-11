from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from pathlib import Path

# Define the processed directory
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def ensure_processed_directory():
    """Ensure that the processed data directory exists."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlc(
    tickers: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data for tickers and save individual CSVs under data/processed.

    Returns a long-format DataFrame with columns:
    [ticker, datetime, open, high, low, close, adj_close, volume]
    """
    ensure_processed_directory()

    if start is None:
        start = "2005-01-01"
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            print(f"No data found for ticker: {t}")
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

        # Round all numerical columns to 2 decimal places
        df = df.round(2)

        # Persist per-ticker csv in the processed folder
        out = PROCESSED_DIR / f"{t}_{interval}.csv"
        df.to_csv(out, index=False)
        print(f"Saved data for {t} to {out}")
        frames.append(df)

    if not frames:
        print("No data was fetched for any ticker.")
        return pd.DataFrame(columns=["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"])

    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined[["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"]]


# Fetch data for QQQ, IWM, and SPY
if __name__ == "__main__":
    tickers = ["QQQ", "IWM", "SPY"]
    fetch_ohlc(tickers)