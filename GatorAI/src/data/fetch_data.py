from __future__ import annotations
from datetime import datetime, timezone
from typing import Iterable, Optional, List
import time
import logging

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the processed directory
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def ensure_processed_directory():
    """Ensure that the processed data directory exists."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def validate_tickers(tickers: Iterable[str]) -> List[str]:
    """Validate and clean ticker symbols.
    
    Args:
        tickers: Iterable of ticker symbols to validate
        
    Returns:
        List of valid, cleaned ticker symbols
        
    Raises:
        ValueError: If no valid tickers are found
    """
    if not tickers:
        raise ValueError("No tickers provided")
    
    valid_tickers = []
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.strip():
            logger.warning(f"Invalid ticker: {ticker}")
            continue
        valid_tickers.append(ticker.strip().upper())
    
    if not valid_tickers:
        raise ValueError("No valid tickers found")
    
    return valid_tickers


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


@retry_on_failure(max_retries=3, delay=1.0)
def _download_ticker_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Download data for a single ticker with retry logic."""
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            logger.warning(f"No data found for ticker: {ticker}")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        raise e


def fetch_ohlc(
    tickers: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    decimal_places: int = 2,
) -> pd.DataFrame:
    """Fetch OHLCV data for tickers and save individual CSVs under data/processed.

    Args:
        tickers: Iterable of ticker symbols to fetch
        start: Start date in YYYY-MM-DD format (default: 2005-01-01)
        end: End date in YYYY-MM-DD format (default: current UTC date)
        interval: Data interval (1d, 1h, 5m, etc.)
        decimal_places: Number of decimal places for rounding (default: 2)

    Returns:
        Long-format DataFrame with columns:
        [ticker, datetime, open, high, low, close, adj_close, volume]
        
    Raises:
        ValueError: If no valid tickers provided or no data fetched
    """
    # Validate inputs
    try:
        valid_tickers = validate_tickers(tickers)
    except ValueError as e:
        logger.error(f"Ticker validation failed: {e}")
        raise e
    
    ensure_processed_directory()

    # Set default dates
    if start is None:
        start = "2005-01-01"
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info(f"Fetching data for {len(valid_tickers)} tickers from {start} to {end}")
    
    frames = []
    failed_tickers = []
    
    # Process tickers with progress bar
    for ticker in tqdm(valid_tickers, desc="Fetching data"):
        try:
            df = _download_ticker_data(ticker, start, end, interval)
            if df.empty:
                failed_tickers.append(ticker)
                continue
                
            # Clean and format data
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
            
            df["ticker"] = ticker
            
            # Round numerical columns
            numeric_cols = ["open", "high", "low", "close", "adj_close"]
            df[numeric_cols] = df[numeric_cols].round(decimal_places)
            
            # Save individual CSV
            output_path = PROCESSED_DIR / f"{ticker}_{interval}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved data for {ticker} to {output_path}")
            
            frames.append(df)
            
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            failed_tickers.append(ticker)
            continue

    # Report results
    if failed_tickers:
        logger.warning(f"Failed to fetch data for: {', '.join(failed_tickers)}")
    
    if not frames:
        logger.error("No data was fetched for any ticker.")
        return pd.DataFrame(columns=["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"])

    # Combine all data
    combined = pd.concat(frames, axis=0, ignore_index=True)
    logger.info(f"Successfully fetched data for {len(frames)} tickers, {len(combined)} total records")
    
    return combined[["ticker", "datetime", "open", "high", "low", "close", "adj_close", "volume"]]


# Fetch data for QQQ, IWM, and SPY
if __name__ == "__main__":
    tickers = ["QQQ", "IWM", "SPY"]
    fetch_ohlc(tickers)