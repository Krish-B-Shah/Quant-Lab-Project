from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
from .backtest_engine import run_vectorized_backtest
from .metrics import max_drawdown

def son_backtest_stub(
        file_path: str | Path,
        price_col: str = 'Adj Close',
        cost_bps: float = 0.0,
) -> Dict[str, float]:
    """What the function above does:
    - It loads the CSV file (with Date + Price column. These are 2 important factors I'm assuming we'll use
    """


    fp = Path(file_path)
    df = pd.read_csv(fp)

    """Now, we sort by Date and Index"""

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")

    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in {fp}")

    """Extract and ensure the price is a float"""
    prices = df[price_col].astype(float)

    """Buy and hold signal"""
    signal = pd.Series(1.0, index=prices.index)

    res = run_vectorized_backtest(prices=prices, signal=signal, cost_bps=cost_bps)

    stats = dict(res.stats)

    # Max drawdown should be computed on the equity curve, not raw prices
    try:
        stats["max_drawdown"] = max_drawdown(res.equity_curve)
    except Exception:
        stats["max_drawdown"] = max_drawdown(prices)

    stats["final_equity"] = float(res.equity_curve.iloc[-1]) if not res.equity_curve.empty else 1.0

    stats["n_obs"] = int(res.returns.shape[0])

    return stats

"""The main testing run file is in scripts"""






