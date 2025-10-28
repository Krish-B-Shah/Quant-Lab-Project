from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


class SQLiteAdapter:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "data_store.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT,
                datetime TIMESTAMP,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, datetime)
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS features (
                ticker TEXT,
                datetime TIMESTAMP,
                feature_key TEXT,
                feature_value REAL,
                PRIMARY KEY (ticker, datetime, feature_key)
            )"""
        )
        self.conn.commit()

    def get_latest_timestamp(self, ticker: str) -> Optional[pd.Timestamp]:
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(datetime) FROM prices WHERE ticker = ?", (ticker,))
        r = cur.fetchone()
        if r is None or r[0] is None:
            return None
        try:
            return pd.to_datetime(r[0])
        except Exception:
            # fallback: return None if parsing fails
            return None

    def upsert_price_data(self, ticker: str, df: pd.DataFrame) -> None:
        # df should contain datetime, open, high, low, close, adj_close, volume
        df = df[["datetime", "open", "high", "low", "close", "adj_close", "volume"]].copy()
        # prepare rows for INSERT OR REPLACE using column-wise iteration to avoid Series/duplicate-column issues
        rows = []
        for dt, o, h, l, c, ac, v in zip(df["datetime"], df["open"], df["high"], df["low"], df["close"], df["adj_close"], df["volume"]):
            # coerce and validate
            try:
                if pd.isna(dt) or pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c) or pd.isna(ac) or pd.isna(v):
                    continue
                # ensure python datetime
                if hasattr(dt, "to_pydatetime"):
                    dt_val = pd.Timestamp(dt).to_pydatetime()
                else:
                    dt_val = pd.to_datetime(dt).to_pydatetime()
                row = (
                    ticker,
                    dt_val,
                    float(o),
                    float(h),
                    float(l),
                    float(c),
                    float(ac),
                    int(v),
                )
                rows.append(row)
            except Exception:
                # skip rows that fail conversion
                continue

        if not rows:
            return

        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO prices (ticker, datetime, open, high, low, close, adj_close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def read_price_data(self, ticker: str) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM prices WHERE ticker = ? ORDER BY datetime", self.conn, params=(ticker,))
        # ensure datetime is parsed
        if not df.empty and "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def get_price_count(self, ticker: str) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM prices WHERE ticker = ?", (ticker,))
        r = cur.fetchone()
        return int(r[0]) if r and r[0] is not None else 0

    def get_total_price_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM prices")
        r = cur.fetchone()
        return int(r[0]) if r and r[0] is not None else 0

    def upsert_feature_data(self, ticker: str, feat_df: pd.DataFrame) -> None:
        # feat_df has datetime + feature columns; we'll write each feature as rows (normalized)
        rows = []
        cols = [c for c in feat_df.columns if c != "datetime"]
        for _, r in feat_df.iterrows():
            dt = r["datetime"]
            for c in cols:
                val = r[c]
                if pd.isna(val):
                    continue
                rows.append((ticker, pd.Timestamp(dt).to_pydatetime(), c, float(val)))
        cur = self.conn.cursor()
        cur.executemany("INSERT OR REPLACE INTO features (ticker, datetime, feature_key, feature_value) VALUES (?, ?, ?, ?)", rows)
        self.conn.commit()
