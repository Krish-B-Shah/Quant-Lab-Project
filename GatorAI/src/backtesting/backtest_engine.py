from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
	returns: pd.Series
	equity_curve: pd.Series
	stats: Dict[str, float]


def run_vectorized_backtest(
	prices: pd.Series,
	signal: pd.Series,
	cost_bps: float = 0.0,
) -> BacktestResult:
	aligned = pd.concat([prices, signal], axis=1).dropna()
	aligned.columns = ["price", "signal"]
	ret = aligned["price"].pct_change().fillna(0.0)
	pos = aligned["signal"].shift(1).fillna(0.0)
	gross = pos * ret
	cost = (pos.diff().abs().fillna(0.0)) * (cost_bps / 10000.0)
	net = gross - cost
	equity = (1.0 + net).cumprod()
	# Annualization settings (assumes daily data by default)
	periods_per_year = 252.0

	# Use number of return observations (after alignment) as the period count
	n_periods = int(net.dropna().shape[0])

	# CAGR: (final_equity) ** (periods_per_year / n_periods) - 1
	if n_periods <= 1 or equity.empty:
		cagr = 0.0
	else:
		try:
			cagr = float(equity.iloc[-1] ** (periods_per_year / float(n_periods)) - 1.0)
		except Exception:
			cagr = 0.0

	# Volatility: annualized std of net returns
	try:
		vol = float(net.std() * np.sqrt(periods_per_year)) if n_periods > 1 else 0.0
	except Exception:
		vol = 0.0

	# Sharpe: annualized mean / annualized std = (mean * periods_per_year) / (std * sqrt(periods_per_year))
	try:
		std = net.std()
		if std is None or std == 0 or np.isnan(std) or n_periods <= 1:
			sharpe = 0.0
		else:
			sharpe = float((net.mean() * periods_per_year) / (std * np.sqrt(periods_per_year)))
	except Exception:
		sharpe = 0.0

	stats = {"cagr": cagr, "vol": vol, "sharpe": sharpe}
	return BacktestResult(returns=net, equity_curve=equity, stats=stats)
