from GatorAI.src.backtesting.backtest_engine import run_vectorized_backtest
import pandas as pd


def test_backtest_runs():
	prices = pd.Series([100, 101, 102, 103], index=pd.date_range("2020-01-01", periods=4))
	signal = pd.Series([0, 1, 1, 0], index=prices.index)
	res = run_vectorized_backtest(prices, signal)
	assert hasattr(res, "equity_curve")
	assert len(res.equity_curve) == 4
