from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
	if equity.empty:
		return 0.0
	roll_max = equity.cummax()
	drawdown = equity / roll_max - 1.0
	return float(drawdown.min())


def annualized_sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
	"""Compute annualized Sharpe ratio from period returns.

	Sharpe = (mean_return * periods_per_year) / (std_return * sqrt(periods_per_year))

	Returns 0.0 if insufficient data or std is zero/nan.
	"""
	if returns is None or len(returns.dropna()) <= 1:
		return 0.0
	std = returns.std()
	mean = returns.mean()
	if std is None or std == 0 or np.isnan(std):
		return 0.0
	return float((mean * periods_per_year) / (std * np.sqrt(periods_per_year)))
