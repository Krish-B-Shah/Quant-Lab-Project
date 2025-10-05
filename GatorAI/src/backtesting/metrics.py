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
	std = returns.std()
	if std == 0 or np.isnan(std):
		return 0.0
	return float((returns.mean() * periods_per_year) / (std * np.sqrt(periods_per_year)))
