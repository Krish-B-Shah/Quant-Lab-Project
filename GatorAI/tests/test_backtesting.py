import sys
from pathlib import Path

# Add src to path for testing
root = Path(__file__).resolve().parents[1]
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from backtesting.backtest_engine import run_vectorized_backtest
import pandas as pd


def test_backtest_runs():
	prices = pd.Series([100, 101, 102, 103], index=pd.date_range("2020-01-01", periods=4))
	signal = pd.Series([0, 1, 1, 0], index=prices.index)
	res = run_vectorized_backtest(prices, signal)
	assert hasattr(res, "equity_curve")
	assert len(res.equity_curve) == 4
