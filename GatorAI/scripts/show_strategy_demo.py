#!/usr/bin/env python3
"""Small demo and human-friendly runner for strategies.

This script is a lightweight utility intended for rapid inspection of
signals, allocations and a short backtest summary for any of the
strategies implemented in `GatorAI/src/backtesting/strategy.py`.

Purpose
-------
- Give teammates a one-file example to drop in a CSV and immediately see
    signals, weights and a tiny equity summary.
- Produce reproducible output for unit/integration tests during local
    development.

Usage examples
--------------
Run against the committed SPY sample CSV and show the last calendar month:

        python3 GatorAI/scripts/show_strategy_demo.py --csv GatorAI/data/processed/SPY_sontest.csv --month

Show last 20 rows instead of the month:

        python3 GatorAI/scripts/show_strategy_demo.py --csv GatorAI/data/processed/SPY_sontest.csv --rows 20

Notes for maintainers
---------------------
- The script inserts `GatorAI/src` onto `sys.path` at runtime so the
    repository does not need to be installed into the environment. This
    keeps the demo runnable in CI and when paired with the project's
    virtualenv. The editor's static analysis may still flag imports; if
    that is distracting consider `pip install -e .` in the venv.
- The `--month` flag currently selects a calendar 30-day window
    (end − 30 days) and then shows only trading rows that exist in the
    CSV. That means the first printed date in the slice is the first
    available trading day ≥ start. If you'd prefer 30 trading days,
    switch to `pd.tseries.offsets.BDay(30)` (optional todo).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

# ensure repo src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtesting.strategy import (
    EqualWeightStrategy,
    MomentumStrategy,
    VolatilityWeightedStrategy,
    MeanReversionStrategy,
    StrategyConfig,
)


def make_price_df(n_days=10, tickers=("SPY", "QQQ", "IWM"), seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    # Simulate log returns and convert to prices
    rets = rng.normal(loc=0.0004, scale=0.01, size=(n_days, len(tickers)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    return df


def load_prices_from_csv(csv_path: str, ticker: str | None = None) -> pd.DataFrame:
    """Load a CSV and return a price DataFrame (dates index, columns=tickers).

    Supports two common shapes:
      - single-asset OHLC CSV with columns like 'Adj Close' or 'Close' -> returns a one-column DF named by `ticker` or filename
      - multi-column CSV with prices per ticker (each column is a ticker) -> returns as-is
    """
    p = Path(csv_path)
    df = pd.read_csv(csv_path, parse_dates=[0])
    # try to use the first column as Date
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df.set_index("Date", inplace=True)

    # If there is a column named 'Adj Close' or 'Close', treat as single-asset OHLC
    for col in ("Adj Close", "Close"):
        if col in df.columns:
            name = ticker or p.stem.split("_")[0]
            out = df[[col]].rename(columns={col: name})
            return out

    # otherwise assume each column is a ticker and take numeric columns
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric price columns found in {csv_path}")
    return numeric

    # Notes: the function supports two common CSV shapes:
    #  - single-asset OHLC files (with 'Adj Close' or 'Close') -> returns a
    #    single-column DataFrame named by the supplied `ticker` or the file
    #    stem. This is handy for testing single-ticker backtests.
    #  - multi-column numeric CSVs where each column is a ticker's price.
    #
    # The returned DataFrame should have a DatetimeIndex and numeric columns.


def summarize_portfolio(prices: pd.DataFrame, weights: pd.DataFrame):
    """Compute a tiny executed backtest summary.

    This helper performs a lag-1 execution (weights are applied to next-day
    returns), computes cumulative equity and a few simple annualized
    statistics. It is intentionally small — use the `backtest_engine` for
    production/backtesting-grade calculations.
    """

    # Simple lag-1 fill (enter next day) execution: use previous day's weights
    daily_rets = prices.pct_change().fillna(0.0)
    executed_w = weights.shift(1).fillna(0.0)
    port_rets = (executed_w * daily_rets).sum(axis=1)
    equity = (1.0 + port_rets).cumprod()

    # basic stats
    n = len(port_rets.dropna())
    periods_per_year = 252
    mean = port_rets.mean()
    vol = port_rets.std()
    sharpe = (mean / vol) * np.sqrt(periods_per_year) if vol > 0 else np.nan
    cagr = equity.iloc[-1] ** (periods_per_year / max(n, 1)) - 1 if n > 0 else np.nan
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()

    return dict(final_equity=float(equity.iloc[-1]), cagr=float(cagr), vol=float(vol * np.sqrt(periods_per_year)), sharpe=float(sharpe), max_drawdown=float(max_dd))


def print_df(title: str, df: pd.DataFrame, max_rows: int = 10):
    print(f"\n--- {title} (shape={df.shape}) ---")
    # If max_rows <= 0 we mean "show all rows". df.tail(0) returns an empty
    # DataFrame, which was causing confusion when --month was used.
    if max_rows <= 0:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df)
    else:
        with pd.option_context("display.max_rows", max_rows, "display.max_columns", None):
            print(df.tail(max_rows))


def main():
    parser = argparse.ArgumentParser(description="Show strategy demo on synthetic data or a CSV file")
    parser.add_argument("--csv", help="Path to CSV file with prices (Date in first column). If omitted synthetic data is used.")
    parser.add_argument("--ticker", help="Ticker name to use when CSV contains a single-asset OHLC file (optional)")
    parser.add_argument("--days", type=int, default=60, help="If synthetic, number of days to simulate")
    parser.add_argument("--rows", "-r", type=int, default=10, help="Number of rows to display for each printed table; 0 = all rows")
    parser.add_argument("--month", action="store_true", help="Show the last calendar month (30 days) of data instead of tail rows")
    args = parser.parse_args()

    if args.csv:
        if not Path(args.csv).exists():
            print(f"CSV path not found: {args.csv}")
            return
        prices = load_prices_from_csv(args.csv, ticker=args.ticker)
        print(f"Loaded prices from {args.csv} -> shape={prices.shape}, columns={list(prices.columns)}")
    else:
        prices = make_price_df(n_days=args.days)

    show_rows = args.rows
    # Optionally show last calendar month (30 days)
    if args.month:
        end = prices.index.max()
        start = end - pd.Timedelta(days=30)
        view_prices = prices.loc[prices.index >= start]
        if view_prices.empty:
            print(f"Warning: no data in the last 30 days (start={start.date()}); falling back to full series")
            view_prices = prices
        header = f"month {start.date()} → {end.date()} ({len(view_prices)} rows)"
    else:
        view_prices = prices
        header = f"last {show_rows} rows" if show_rows != 0 else "all rows"

    print(f"\n=== Demo price data ({header}) ===")
    # if month view, show all rows in the month slice; otherwise respect --rows
    print_df("prices", view_prices, max_rows=(0 if args.month else show_rows))

    strategies = [
        (EqualWeightStrategy(), "Equal weight across universe"),
        (MomentumStrategy(config=StrategyConfig(params={"lookback": 5})), "Momentum (5-day)"),
        (VolatilityWeightedStrategy(config=StrategyConfig(params={"vol_window": 10})), "Vol-weighted (10-day)"),
        (MeanReversionStrategy(config=StrategyConfig(params={"lookback": 5})), "Mean-reversion (5-day)"),
    ]

    for strat, desc in strategies:
        print(dedent(f"\n=== Strategy: {strat.name} — {desc} ==="))
        # Emit a warning if the strategy uses a lookback/window longer than available rows
        cfg = getattr(strat, "config", None)
        params = (cfg.params or {}) if cfg is not None else {}
        lookback = params.get("lookback") or params.get("vol_window")
        if lookback is not None and lookback > len(prices):
            print(f"Warning: strategy {strat.name} lookback/window={lookback} > available rows={len(prices)}; results may be empty or truncated.")

        sig = strat.generate_signals(prices)
        w = strat.allocate(sig, prices)

        # Slice signals/weights to the same view (month or tail) for printing
        sig_view = sig.loc[view_prices.index.intersection(sig.index)]
        w_view = w.loc[view_prices.index.intersection(w.index)]

        print_df("signals", sig_view, max_rows=(0 if args.month else show_rows))
        print_df("weights", w_view, max_rows=(0 if args.month else show_rows))
        stats = summarize_portfolio(prices, w)
        print(dedent(f"""
        Summary:
          final equity: {stats['final_equity']:.4f}
          CAGR: {stats['cagr']:.2%}
          Annualized vol: {stats['vol']:.2%}
          Sharpe: {stats['sharpe']:.3f}
          Max drawdown: {stats['max_drawdown']:.2%}
        """))


if __name__ == "__main__":
    main()
