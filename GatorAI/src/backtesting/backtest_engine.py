from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd
from typing import Iterable, Union

from .strategy import BaseStrategy


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


def _get_rebalance_dates(index: pd.DatetimeIndex, freq: Union[str, Iterable]) -> pd.DatetimeIndex:
	"""Return a DatetimeIndex of rebalance dates based on freq.

	freq may be 'daily','weekly','monthly','quarterly' or an iterable of dates.
	"""
	if isinstance(freq, (list, tuple, set, pd.DatetimeIndex)):
		return pd.DatetimeIndex(sorted(pd.to_datetime(list(freq))))

	freq = str(freq).lower()
	if freq == "daily":
		return index
	if freq == "weekly":
		return index.to_series().resample("W").first().dropna().index
	if freq == "monthly":
		return index.to_series().resample("M").first().dropna().index
	if freq in ("quarterly", "quarter"):
		return index.to_series().resample("Q").first().dropna().index

	# fallback: try to use pandas offset alias directly
	try:
		return index.to_series().resample(freq).first().dropna().index
	except Exception:
		return index


def run_backtest_strategy(
	prices: pd.DataFrame,
	strategy: BaseStrategy,
	rebalance: Union[str, Iterable] = "monthly",
	cost_bps: float = 0.0,
	slippage: float = 0.0,
	periods_per_year: float = 252.0,
) -> BacktestResult:
	"""Run a multi-asset backtest driven by a Strategy instance.

	prices: DataFrame indexed by datetime with tickers as columns.
	strategy: BaseStrategy instance implementing generate_signals and allocate.
	rebalance: frequency string ('daily','weekly','monthly','quarterly') or iterable of dates.
	cost_bps: trading cost in basis points applied to turnover.
	slippage: additional slippage per unit turnover (fraction of notional).
	"""
	if not isinstance(prices, pd.DataFrame):
		raise ValueError("prices must be a DataFrame with tickers as columns")

	# Generate signals and allocation from strategy
	signals = strategy.generate_signals(prices)
	weights = strategy.allocate(signals, prices)

	# Align weights to price index and apply rebalancing frequency
	weights = weights.reindex(prices.index).copy()
	reb_dates = _get_rebalance_dates(prices.index, rebalance)

	# Keep weights only on rebalance dates, then forward-fill positions
	mask = weights.index.isin(reb_dates)
	weights = weights.copy()
	weights.loc[~mask, :] = np.nan
	weights = weights.ffill().fillna(0.0)

	# Period returns (matrix)
	returns = prices.pct_change().fillna(0.0)

	# Positions are applied with one-period lag (trade at next bar)
	pos = weights.shift(1).fillna(0.0)

	# Portfolio gross returns
	port_ret = (pos * returns).sum(axis=1)

	# Turnover and costs
	turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
	cost = turnover * (cost_bps / 10000.0)
	slip = turnover * float(slippage)

	net = port_ret - cost - slip
	equity = (1.0 + net).cumprod()

	# Stats (similar logic as single-series backtest)
	n_periods = int(net.dropna().shape[0])
	if n_periods <= 1 or equity.empty:
		cagr = 0.0
	else:
		try:
			cagr = float(equity.iloc[-1] ** (periods_per_year / float(n_periods)) - 1.0)
		except Exception:
			cagr = 0.0

	try:
		vol = float(net.std() * np.sqrt(periods_per_year)) if n_periods > 1 else 0.0
	except Exception:
		vol = 0.0

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
