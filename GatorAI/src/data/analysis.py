from __future__ import annotations

import pandas as pd
from typing import Iterable, List, Optional, Dict

from .storage.sqlite_adapter import SQLiteAdapter


def compare_sources(db: SQLiteAdapter, ticker: str, sources: Iterable[str]) -> Dict[str, object]:
    """Compare price data for a ticker across multiple sources.

    Returns a summary dict with per-source counts, latest timestamps, pairwise correlations and mean absolute differences.
    """
    dfs = {}
    for s in sources:
        df = db.read_price_data(ticker, source=s)
        dfs[s] = df.set_index("datetime")["close"].rename(s) if not df.empty else pd.Series(dtype=float)

    summary = {"ticker": ticker, "sources": {}, "pairwise": {}}
    for s, ser in dfs.items():
        if ser.empty:
            summary["sources"][s] = {"count": 0, "latest": None}
        else:
            summary["sources"][s] = {"count": int(ser.shape[0]), "latest": str(ser.index.max())}

    # align all series on intersection of datetimes
    if len(dfs) >= 2:
        keys = list(dfs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = dfs[keys[i]]
                b = dfs[keys[j]]
                if a.empty or b.empty:
                    summary["pairwise"][f"{keys[i]}__{keys[j]}"] = {"corr": None, "mean_abs_diff": None, "count": 0}
                    continue
                merged = pd.concat([a, b], axis=1, join="inner").dropna()
                if merged.empty:
                    summary["pairwise"][f"{keys[i]}__{keys[j]}"] = {"corr": None, "mean_abs_diff": None, "count": 0}
                    continue
                corr = float(merged.corr().iloc[0, 1])
                mad = float((merged.iloc[:, 0] - merged.iloc[:, 1]).abs().mean())
                summary["pairwise"][f"{keys[i]}__{keys[j]}"] = {"corr": corr, "mean_abs_diff": mad, "count": int(merged.shape[0])}

    return summary


def get_combined_price(db: SQLiteAdapter, ticker: str, preferred_sources: List[str]) -> pd.DataFrame:
    """Combine price data from multiple sources, preferring earlier sources in the list when filling gaps.

    Returns a DataFrame with columns: datetime, open, high, low, close, adj_close, volume, source
    """
    frames = []
    for s in preferred_sources:
        df = db.read_price_data(ticker, source=s)
        if df is None or df.empty:
            continue
        df = df.set_index("datetime")
        df = df[~df.index.duplicated(keep="first")]
        frames.append((s, df))

    if not frames:
        return pd.DataFrame()

    # build a union index
    idx = pd.Index(sorted({ts for _, f in frames for ts in f.index}))
    out = pd.DataFrame(index=idx)
    out["source"] = None
    for s, f in frames:
        # reindex to union index and fill missing values where out is not set
        tmp = f.reindex(idx)
        # fill only where out is NaN
        mask = out["source"].isna()
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            out.loc[mask, col] = tmp[col]
        out.loc[mask, "source"] = s

    out = out.reset_index().rename(columns={"index": "datetime"})
    return out
