from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import pandas as pd
from typing import Iterable, Union

from .strategy import BaseStrategy


@dataclass
class TradeLog:
	"""Individual trade record for detailed analysis."""
	timestamp: pd.Timestamp
	ticker: str
	signal: float
	prediction: Optional[float] = None  # ML prediction if available
	confidence: Optional[float] = None  # ML confidence if available
	actual_return: Optional[float] = None  # Realized return (filled post-trade)
	position_size: float = 0.0


@dataclass
class MLMetrics:
	"""ML-specific performance metrics."""
	prediction_mae: Optional[float] = None  # Mean Absolute Error of predictions
	prediction_mse: Optional[float] = None  # Mean Squared Error of predictions  
	hit_rate: Optional[float] = None  # Directional accuracy
	confidence_calibration: Optional[float] = None  # Confidence vs accuracy correlation
	trades_above_threshold: int = 0  # Number of trades above confidence threshold
	total_predictions: int = 0  # Total number of predictions made


@dataclass
class BacktestResult:
	returns: pd.Series
	equity_curve: pd.Series
	stats: Dict[str, float]
	trade_log: List[TradeLog] = None  # Detailed trade-by-trade log
	ml_metrics: MLMetrics = None  # ML-specific metrics
	monthly_stats: pd.DataFrame = None  # Monthly performance breakdown
	rolling_sharpe: pd.Series = None  # Rolling Sharpe ratio


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
	log_trades: bool = True,
	calculate_ml_metrics: bool = True,
) -> BacktestResult:
	"""Run a multi-asset backtest driven by a Strategy instance with enhanced logging.

	prices: DataFrame indexed by datetime with tickers as columns.
	strategy: BaseStrategy instance implementing generate_signals and allocate.
	rebalance: frequency string ('daily','weekly','monthly','quarterly') or iterable of dates.
	cost_bps: trading cost in basis points applied to turnover.
	slippage: additional slippage per unit turnover (fraction of notional).
	log_trades: whether to create detailed trade log for analysis.
	calculate_ml_metrics: whether to compute ML-specific metrics (if strategy supports it).
	"""
	if not isinstance(prices, pd.DataFrame):
		raise ValueError("prices must be a DataFrame with tickers as columns")

	# Generate signals and allocation from strategy
	signals = strategy.generate_signals(prices)
	weights = strategy.allocate(signals, prices)

	# Initialize trade logging
	trade_log = [] if log_trades else None
	
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

	# Log trades if requested
	if log_trades:
		for date in pos.index:
			if date in signals.index:
				for ticker in pos.columns:
					if pos.loc[date, ticker] != 0:  # Only log non-zero positions
						trade = TradeLog(
							timestamp=date,
							ticker=ticker,
							signal=signals.loc[date, ticker] if date in signals.index else 0.0,
							position_size=pos.loc[date, ticker]
						)
						
						# Add ML predictions if strategy supports it
						if hasattr(strategy, 'get_prediction_log'):
							try:
								pred_log = strategy.get_prediction_log()
								if not pred_log.empty:
									matching_preds = pred_log[pred_log['timestamp'] == date]
									if not matching_preds.empty:
										latest_pred = matching_preds.iloc[-1]
										trade.prediction = latest_pred['prediction']
										trade.confidence = latest_pred['confidence']
							except:
								pass  # Ignore errors in ML prediction logging
						
						trade_log.append(trade)

	# Portfolio gross returns
	port_ret = (pos * returns).sum(axis=1)

	# Turnover and costs
	turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
	cost = turnover * (cost_bps / 10000.0)
	slip = turnover * float(slippage)

	net = port_ret - cost - slip
	equity = (1.0 + net).cumprod()

	# Fill in actual returns for trade log
	if trade_log:
		for trade in trade_log:
			if trade.timestamp in returns.index:
				next_date = returns.index[returns.index.get_loc(trade.timestamp) + 1:] 
				if len(next_date) > 0:
					trade.actual_return = returns.loc[next_date[0], trade.ticker]

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

	# Calculate max drawdown
	try:
		rolling_max = equity.expanding().max()
		drawdowns = equity / rolling_max - 1
		max_drawdown = float(drawdowns.min())
	except Exception:
		max_drawdown = 0.0

	stats = {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_drawdown": max_drawdown}

	# Calculate ML metrics if requested and available
	ml_metrics = None
	if calculate_ml_metrics and hasattr(strategy, 'get_prediction_log'):
		ml_metrics = _calculate_ml_metrics(strategy, trade_log)

	# Calculate additional analytics
	monthly_stats = _calculate_monthly_stats(net, equity) if len(net) > 30 else None
	rolling_sharpe = _calculate_rolling_sharpe(net, window=252//4) if len(net) > 60 else None

	return BacktestResult(
		returns=net, 
		equity_curve=equity, 
		stats=stats,
		trade_log=trade_log,
		ml_metrics=ml_metrics,
		monthly_stats=monthly_stats,
		rolling_sharpe=rolling_sharpe
	)


def _calculate_ml_metrics(strategy, trade_log: Optional[List[TradeLog]]) -> MLMetrics:
	"""Calculate ML-specific performance metrics."""
	try:
		pred_log_df = strategy.get_prediction_log()
		if pred_log_df.empty:
			return MLMetrics()
		
		# Basic prediction accuracy metrics
		predictions = []
		actuals = []
		confidences = []
		
		if trade_log:
			for trade in trade_log:
				if trade.prediction is not None and trade.actual_return is not None:
					predictions.append(trade.prediction)
					actuals.append(trade.actual_return)
					if trade.confidence is not None:
						confidences.append(trade.confidence)
		
		ml_metrics = MLMetrics(total_predictions=len(pred_log_df))
		
		if len(predictions) > 0:
			predictions = np.array(predictions)
			actuals = np.array(actuals)
			
			# Calculate MAE and MSE
			ml_metrics.prediction_mae = float(np.mean(np.abs(predictions - actuals)))
			ml_metrics.prediction_mse = float(np.mean((predictions - actuals) ** 2))
			
			# Calculate hit rate (directional accuracy)
			pred_signs = np.sign(predictions)
			actual_signs = np.sign(actuals)
			ml_metrics.hit_rate = float(np.mean(pred_signs == actual_signs))
			
			# Calculate confidence calibration if available
			if len(confidences) > 0:
				confidences = np.array(confidences)
				high_conf_mask = confidences > 0.7
				if np.sum(high_conf_mask) > 0:
					high_conf_accuracy = np.mean((pred_signs == actual_signs)[high_conf_mask])
					ml_metrics.confidence_calibration = float(high_conf_accuracy)
		
		# Count trades above confidence threshold
		if hasattr(strategy, '_ml_predictor') and strategy._ml_predictor is not None:
			min_conf = getattr(strategy._ml_predictor, 'min_confidence', 0.6)
			above_threshold = pred_log_df['confidence'] >= min_conf
			ml_metrics.trades_above_threshold = int(above_threshold.sum())
		
		return ml_metrics
		
	except Exception as e:
		print(f"Warning: Could not calculate ML metrics: {e}")
		return MLMetrics()


def _calculate_monthly_stats(returns: pd.Series, equity: pd.Series) -> pd.DataFrame:
	"""Calculate monthly performance breakdown."""
	try:
		monthly_rets = returns.resample('M').sum()
		monthly_equity = equity.resample('M').last()
		monthly_vol = returns.resample('M').std() * np.sqrt(21)  # Assume 21 trading days per month
		
		monthly_stats = pd.DataFrame({
			'return': monthly_rets,
			'equity': monthly_equity,
			'volatility': monthly_vol
		})
		
		# Calculate monthly Sharpe
		monthly_stats['sharpe'] = monthly_stats['return'] / monthly_stats['volatility']
		monthly_stats['sharpe'] = monthly_stats['sharpe'].replace([np.inf, -np.inf], 0)
		
		return monthly_stats
		
	except Exception as e:
		print(f"Warning: Could not calculate monthly stats: {e}")
		return pd.DataFrame()


def _calculate_rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
	"""Calculate rolling Sharpe ratio."""
	try:
		rolling_mean = returns.rolling(window).mean()
		rolling_std = returns.rolling(window).std()
		rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)  # Annualized
		return rolling_sharpe.replace([np.inf, -np.inf], 0)
	except Exception as e:
		print(f"Warning: Could not calculate rolling Sharpe: {e}")
		return pd.Series(dtype=float)
