from __future__ import annotations
from pathlib import Path
from typing import Dict, Union
import pandas as pd
import logging
from .backtest_engine import run_vectorized_backtest
from .metrics import max_drawdown

# Set up logging
logger = logging.getLogger(__name__)

def son_backtest_stub(
        file_path: Union[str, Path],
        price_col: str = 'adj_close',
        date_col: str = 'datetime',
        cost_bps: float = 0.0,
) -> Dict[str, float]:
    """Run a buy-and-hold backtest on price data from a CSV file.
    
    Args:
        file_path: Path to CSV file containing price data
        price_col: Name of the price column (default: 'adj_close')
        date_col: Name of the date column (default: 'datetime')
        cost_bps: Transaction costs in basis points (default: 0.0)
        
    Returns:
        Dictionary containing backtest statistics
        
    Raises:
        ValueError: If required columns are missing or data is invalid
        FileNotFoundError: If the CSV file doesn't exist
    """
    fp = Path(file_path)
    
    if not fp.exists():
        raise FileNotFoundError(f"CSV file not found: {fp}")
    
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {fp}: {e}")
    
    # Clean the data - remove any malformed rows
    df = df.dropna()
    
    # Validate required columns
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in {fp}. Available columns: {list(df.columns)}")
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in {fp}. Available columns: {list(df.columns)}")
    
    # Convert date column to datetime and sort
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
    except Exception as e:
        raise ValueError(f"Error processing date column '{date_col}': {e}")
    
    # Extract and validate price data
    try:
        prices = df[price_col].astype(float)
        if prices.empty:
            raise ValueError("No valid price data found")
        if prices.isna().any():
            logger.warning("Found NaN values in price data, dropping them")
            prices = prices.dropna()
    except Exception as e:
        raise ValueError(f"Error processing price column '{price_col}': {e}")
    
    # Create buy-and-hold signal (always 100% invested)
    signal = pd.Series(1.0, index=prices.index)
    
    logger.info(f"Running backtest on {len(prices)} observations from {prices.index[0]} to {prices.index[-1]}")
    
    # Run the backtest
    res = run_vectorized_backtest(prices=prices, signal=signal, cost_bps=cost_bps)
    
    # Extract statistics
    stats = dict(res.stats)
    
    # Calculate max drawdown on equity curve (not raw prices)
    try:
        stats["max_drawdown"] = max_drawdown(res.equity_curve)
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        stats["max_drawdown"] = 0.0
    
    # Add additional metrics
    stats["final_equity"] = float(res.equity_curve.iloc[-1]) if not res.equity_curve.empty else 1.0
    stats["n_obs"] = int(res.returns.shape[0])
    stats["total_return"] = float(stats["final_equity"] - 1.0)
    
    logger.info(f"Backtest completed: {stats['total_return']:.2%} total return, {stats['sharpe']:.2f} Sharpe ratio")
    
    return stats

"""The main testing run file is in scripts"""






