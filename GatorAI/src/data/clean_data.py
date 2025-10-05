from __future__ import annotations

import pandas as pd


def standardize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
	cols = {
		"Open": "open",
		"High": "high",
		"Low": "low",
		"Close": "close",
		"Adj Close": "adj_close",
		"Volume": "volume",
	}
	return df.rename(columns=cols)


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
	return df.dropna().copy()
