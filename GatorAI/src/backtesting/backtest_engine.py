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
	stats = {
		"cagr": float((equity.iloc[-1]) ** (252.0 / max(len(equity), 1)) - 1.0) if len(equity) > 0 else 0.0,
		"vol": float(net.std() * np.sqrt(252.0)) if len(net) > 1 else 0.0,
		"sharpe": float((net.mean() * 252.0) / (net.std() * np.sqrt(252.0))) if net.std() > 0 else 0.0,
	}
	return BacktestResult(returns=net, equity_curve=equity, stats=stats)
