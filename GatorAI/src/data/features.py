from __future__ import annotations

import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window=period).mean()
    down = -delta.clip(upper=0).rolling(window=period).mean()
    rs = up / down
    out = 100 - (100 / (1 + rs))
    out.name = series.name or f"rsi_{period}"
    return out


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = df["close"]
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger_bands(df: pd.DataFrame, window: int = 20, n_std: int = 2) -> pd.DataFrame:
    close = df["close"]
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower, "bb_mid": ma})


def ema_crossover(df: pd.DataFrame, short: int = 12, long: int = 26) -> pd.Series:
    short_ema = ema(df["close"], short)
    long_ema = ema(df["close"], long)
    return (short_ema > long_ema).astype(int).rename("ema_cross")


def rolling_sharpe(df: pd.DataFrame, window: int = 63) -> pd.Series:
    # assume daily returns, window default ~ 3 months
    returns = df["close"].pct_change()
    return (returns.rolling(window).mean() / returns.rolling(window).std()).rename("rolling_sharpe")


def rolling_volatility(df: pd.DataFrame, window: int = 63) -> pd.Series:
    returns = df["close"].pct_change()
    return returns.rolling(window).std().rename("rolling_vol")


def correlation_matrix(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    # expects two dataframes with 'datetime' and 'close'
    merged = df[["datetime", "close"]].merge(other[["datetime", "close"]], on="datetime", suffixes=("", "_other"))
    return merged[["datetime"]].assign(corr=merged["close"].rolling(63).corr(merged["close_other"]))
