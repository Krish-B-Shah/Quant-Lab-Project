"""
Data preparation for ML models - creates features and next-day returns target.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from ..data.storage.sqlite_adapter import SQLiteAdapter
from ..data import features as feat

logger = logging.getLogger(__name__)


def fetch_vix_data(storage: SQLiteAdapter) -> Optional[pd.DataFrame]:
    """Fetch VIX data for regime indicators."""
    try:
        # Try different VIX ticker formats
        vix_tickers = ["^VIX", "VIX", "VIX.X"]
        vix_df = None
        
        for ticker in vix_tickers:
            try:
                vix_df = storage.read_price_data(ticker)
                if not vix_df.empty:
                    logger.info(f"Found VIX data for {ticker}")
                    break
            except Exception:
                continue
        
        if vix_df is None or vix_df.empty:
            logger.warning("No VIX data found in storage, VIX features will be skipped")
            return None
        
        return vix_df
    except Exception as e:
        logger.warning(f"Failed to fetch VIX data: {e}")
        return None


def add_regime_features(df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add regime indicators: VIX buckets and day-of-week effects."""
    result_df = df.copy()
    
    # Day-of-week features (0 = Monday, 4 = Friday)
    result_df["day_of_week"] = result_df["datetime"].dt.dayofweek
    result_df["is_monday"] = (result_df["day_of_week"] == 0).astype(int)
    result_df["is_friday"] = (result_df["day_of_week"] == 4).astype(int)
    result_df["is_weekend"] = (result_df["day_of_week"] >= 5).astype(int)
    
    # Month of year (seasonality)
    result_df["month"] = result_df["datetime"].dt.month
    
    # VIX buckets (if VIX data available)
    if vix_df is not None and not vix_df.empty:
        try:
            # Merge VIX data
            vix_merged = vix_df[["datetime", "close"]].copy()
            vix_merged["datetime"] = pd.to_datetime(vix_merged["datetime"])
            vix_merged = vix_merged.rename(columns={"close": "vix"})
            result_df = result_df.merge(vix_merged, on="datetime", how="left")
            
            # Create VIX buckets: low (< 15), medium (15-25), high (> 25)
            vix_median = result_df["vix"].median() if result_df["vix"].notna().any() else 20.0
            result_df["vix"] = result_df["vix"].fillna(vix_median)
            result_df["vix_low"] = (result_df["vix"] < 15).astype(int)
            result_df["vix_medium"] = ((result_df["vix"] >= 15) & (result_df["vix"] <= 25)).astype(int)
            result_df["vix_high"] = (result_df["vix"] > 25).astype(int)
            result_df["vix_level"] = result_df["vix"]
        except Exception as e:
            logger.warning(f"Failed to merge VIX data: {e}, using default values")
            result_df["vix_low"] = 0
            result_df["vix_medium"] = 1  # Default to medium
            result_df["vix_high"] = 0
            result_df["vix_level"] = 20.0
    else:
        # Fill with zeros if no VIX data
        result_df["vix_low"] = 0
        result_df["vix_medium"] = 1  # Default to medium
        result_df["vix_high"] = 0
        result_df["vix_level"] = 20.0  # Default medium VIX
    
    return result_df


def prepare_ml_data(
    ticker: str,
    storage: SQLiteAdapter,
    feature_list: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare ML-ready data with features and next-day returns target.
    
    Args:
        ticker: Stock ticker symbol
        storage: SQLiteAdapter instance
        feature_list: List of features to include (default: all available)
        start_date: Start date for data (optional)
        end_date: End date for data (optional)
    
    Returns:
        DataFrame with features and target (next_day_return)
    """
    # Default feature list
    if feature_list is None:
        feature_list = ["rsi", "macd", "bollinger", "ema_cross", "sharpe", "vol", "atr"]
    
    # Read price data
    price_df = storage.read_price_data(ticker)
    if price_df.empty:
        raise ValueError(f"No price data found for {ticker}")
    
    # Filter by date if provided
    if start_date:
        price_df = price_df[price_df["datetime"] >= pd.to_datetime(start_date)]
    if end_date:
        price_df = price_df[price_df["datetime"] <= pd.to_datetime(end_date)]
    
    if price_df.empty:
        raise ValueError(f"No price data found for {ticker} in date range")
    
    # Sort by datetime
    price_df = price_df.sort_values("datetime").reset_index(drop=True)
    price_df["datetime"] = pd.to_datetime(price_df["datetime"])
    
    # Prepare dataframe for features
    df = price_df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    
    # Generate technical indicators
    feature_map = {
        "rsi": lambda d: feat.rsi(d["close"], period=14),
        "macd": lambda d: feat.macd(d),
        "bollinger": lambda d: feat.bollinger_bands(d, window=20, n_std=2),
        "ema_cross": lambda d: feat.ema_crossover(d, short=12, long=26),
        "sharpe": lambda d: feat.rolling_sharpe(d, window=63),
        "vol": lambda d: feat.rolling_volatility(d, window=63),
        "atr": lambda d: feat.atr(d, period=14),
    }
    
    # Add EMA features (for crossover signals)
    ema_12 = feat.ema(df["close"], span=12)
    ema_26 = feat.ema(df["close"], span=26)
    ema_50 = feat.ema(df["close"], span=50)
    df["ema_12"] = ema_12
    df["ema_26"] = ema_26
    df["ema_50"] = ema_50
    
    # Generate requested features
    for feature_name in feature_list:
        if feature_name in feature_map:
            try:
                result = feature_map[feature_name](df)
                if isinstance(result, pd.Series):
                    df[result.name] = result.values
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        df[col] = result[col].values
            except Exception as e:
                logger.warning(f"Failed to generate feature {feature_name}: {e}")
    
    # Add regime features
    vix_df = fetch_vix_data(storage)
    df = add_regime_features(df, vix_df)
    
    # Create target: next-day return
    df["next_day_return"] = df["close"].pct_change().shift(-1)
    
    # Add lagged returns (common features)
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    
    # Add volume features
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()
    df["volume_change"] = df["volume"].pct_change(1)
    
    # Drop rows with NaN (from feature calculations and target)
    df = df.dropna()
    
    # Remove the last row (target is NaN)
    df = df[df["next_day_return"].notna()]
    
    logger.info(f"Prepared ML data for {ticker}: {len(df)} rows, {len(df.columns)} features")
    
    return df

