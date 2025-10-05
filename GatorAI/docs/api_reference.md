# API Reference (High-Level)

## Data
- `src/data/fetch_data.py`:
  - `fetch_ohlc(tickers, start=None, end=None, interval="1d") -> DataFrame`
- `src/data/clean_data.py`:
  - `standardize_ohlc_columns(df) -> DataFrame`
  - `drop_missing(df) -> DataFrame`

## Backtesting
- `src/backtesting/backtest_engine.py`:
  - `run_vectorized_backtest(prices, signal, cost_bps=0.0) -> BacktestResult`
- `src/backtesting/metrics.py`:
  - `max_drawdown(equity) -> float`
  - `annualized_sharpe(returns) -> float`

## Optimization
- `src/optimization/optimizer.py`:
  - `mean_variance_optimize(returns, risk_aversion=1.0, long_only=True, weights_sum_to_one=True) -> Series`
- `src/optimization/risk_models.py`:
  - `sample_covariance(returns) -> DataFrame`
  - `ewma_covariance(returns, lambda_=0.94) -> DataFrame`
- `src/optimization/ml_models.py`:
  - `ReturnForecaster(alpha=1.0).fit(X, y).predict(X)`
