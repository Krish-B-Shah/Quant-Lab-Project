"""Strategies: abstractions and small example implementations.

This module defines a minimal research-oriented strategy contract used by
the lightweight backtest/demo tools in this repository.

Contract summary
----------------
- generate_signals(price_df[, features]) -> pd.DataFrame
    - index: pd.DatetimeIndex matching the price_df index
    - columns: tickers (same order/names as price_df.columns)
    - values: numeric scores (scale is arbitrary). Implementations may
        return raw scores, z-scores, past returns, etc.

- allocate(signals, price_df) -> pd.DataFrame
    - returns weights with same shape as signals
    - rows should sum to 1.0 (or 0.0 for cash/no-exposure rows)
    - long-only strategies should return non-negative weights

Notes and examples
------------------
The code in this file is intentionally small and explicitly written for
readability. Example usage (demo):

        from backtesting.strategy import MomentumStrategy, StrategyConfig
        strat = MomentumStrategy(config=StrategyConfig(params={'lookback': 20}))
        prices = load_prices_from_csv('data/processed/SPY.csv')  # pandas DataFrame
        sig = strat.generate_signals(prices)
        w = strat.allocate(sig, prices)

Edge cases
----------
- If the lookback/window parameter is larger than the available rows, most
    implementations will produce many NaNs/zeros; callers should guard and/or
    choose smaller lookbacks for very short test series.
- All implementations return filled numeric DataFrames (NaNs are converted
    to zeros where appropriate) for convenience in demo/test code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import numpy as np


@dataclass
class StrategyConfig:
    params: Dict = None


class BaseStrategy(ABC):
    """Base strategy abstraction.

    Subclasses must implement two methods:
      - generate_signals(price_df, features=None) -> pd.DataFrame
      - allocate(signals, price_df) -> pd.DataFrame

    Implementations in this file follow a simple pattern: signals are
    calculated from price history and then normalized into long-only
    weights (rows sum to 1). Callers can override or extend allocation
    behavior as needed.

    The class stores a lightweight `StrategyConfig` to keep parameters
    together and readable in demos.
    """

    name: str
    config: StrategyConfig

    def __init__(self, name: str = "base", config: Optional[StrategyConfig] = None):
        self.name = name
        self.config = config or StrategyConfig(params={})

    @abstractmethod
    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return a DataFrame of raw signals/scores.

        - price_df: DataFrame of prices (index=dates, cols=tickers)
        - Return must be same shape (index & columns) as price_df.
        """
        raise NotImplementedError()

    @abstractmethod
    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw signals into weights.

        Typical behavior is to clip to long-only and row-normalize so each
        date's weights sum to 1. Returning zeros for rows with no exposure
        is allowed.
        """
        raise NotImplementedError()


class EqualWeightStrategy(BaseStrategy):
    """Simple equal-weight strategy: allocate equal weight across all tickers in the universe.

    Parameters (via config.params):
      - universe: Optional[list] restrict tickers
      - long_only: bool (default True)
    """

    def __init__(self, name: str = "equal_weight", config: Optional[StrategyConfig] = None):
        super().__init__(name=name, config=config)

    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        # Simple constant signal of 1 for all tickers and dates in price_df
        sig = pd.DataFrame(1.0, index=price_df.index, columns=price_df.columns)
        return sig

    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        # Ignore signals content and produce equal weights across columns
        cols = price_df.columns.tolist()
        if len(cols) == 0:
            return pd.DataFrame(index=price_df.index, columns=cols)
        w = 1.0 / len(cols)
        weights = pd.DataFrame(w, index=price_df.index, columns=cols)
        return weights


class MomentumStrategy(BaseStrategy):
    """Momentum strategy: signal = past return over lookback window.

    Params (config.params):
      - lookback: int (periods for past return), default 90
      - top_k: Optional[int] to only take top-k longs (long_only mode)
      - long_only: bool (default True)
    """

    def __init__(self, name: str = "momentum", config: Optional[StrategyConfig] = None):
        super().__init__(name=name, config=config)

    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        params = self.config.params or {}
        lookback = int(params.get("lookback", 90))
        # past return over lookback
        sig = price_df.pct_change(periods=lookback)
        return sig.fillna(0.0)

    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params or {}
        top_k = params.get("top_k", None)
        long_only = bool(params.get("long_only", True))

        s = signals.copy()
        # if top_k specified, zero out others per row
        if top_k is not None:
            def topk_row(r):
                if r.isna().all():
                    return r * 0.0
                idx = r.nlargest(top_k).index
                out = pd.Series(0.0, index=r.index)
                out.loc[idx] = r.loc[idx]
                return out

            s = s.apply(topk_row, axis=1)

        if long_only:
            s = s.clip(lower=0.0)

        # normalize rows to sum to 1
        row_sum = s.sum(axis=1).replace(0.0, np.nan)
        weights = s.div(row_sum, axis=0).fillna(0.0)
        return weights


class VolatilityWeightedStrategy(BaseStrategy):
    """Weight assets inversely proportional to historical volatility.

    Params (config.params):
      - vol_window: int (rolling window for volatility), default 21
      - long_only: bool (default True)
    """

    def __init__(self, name: str = "vol_weighted", config: Optional[StrategyConfig] = None):
        super().__init__(name=name, config=config)

    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        params = self.config.params or {}
        vol_window = int(params.get("vol_window", 21))
        rets = price_df.pct_change()
        vol = rets.rolling(vol_window).std()
        eps = 1e-8
        sig = 1.0 / (vol + eps)
        return sig.fillna(0.0)

    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params or {}
        long_only = bool(params.get("long_only", True))

        s = signals.copy()
        if long_only:
            s = s.clip(lower=0.0)

        row_sum = s.sum(axis=1).replace(0.0, np.nan)
        weights = s.div(row_sum, axis=0).fillna(0.0)
        return weights


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion: signal = - zscore of recent returns (buy when underperformed).

    Params (config.params):
      - lookback: int for rolling mean/std of returns, default 20
      - long_only: bool (default True)
      - top_k: Optional[int] to only take extreme names
    """

    def __init__(self, name: str = "mean_reversion", config: Optional[StrategyConfig] = None):
        super().__init__(name=name, config=config)

    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        params = self.config.params or {}
        lookback = int(params.get("lookback", 20))
        rets = price_df.pct_change()
        mu = rets.rolling(lookback).mean()
        sigma = rets.rolling(lookback).std()
        eps = 1e-8
        z = (rets - mu) / (sigma + eps)
        sig = -z  # mean-revert: negative z-score -> buy
        return sig.fillna(0.0)

    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params or {}
        long_only = bool(params.get("long_only", True))
        top_k = params.get("top_k", None)

        s = signals.copy()
        # if top_k specified, pick top (most negative z -> largest positive signals after negation)
        if top_k is not None:
            def topk_row(r):
                if r.isna().all():
                    return r * 0.0
                idx = r.nlargest(top_k).index
                out = pd.Series(0.0, index=r.index)
                out.loc[idx] = r.loc[idx]
                return out

            s = s.apply(topk_row, axis=1)

        if long_only:
            s = s.clip(lower=0.0)

        row_sum = s.sum(axis=1).replace(0.0, np.nan)
        weights = s.div(row_sum, axis=0).fillna(0.0)
        return weights
