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
                source TEXT,
                PRIMARY KEY (ticker, datetime)
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS features (
                ticker TEXT,
                datetime TIMESTAMP,
                feature_key TEXT,
                feature_value REAL,
                source TEXT,
                PRIMARY KEY (ticker, datetime, feature_key)
            )"""
        )
        self.conn.commit()
        # ensure columns exist for older DBs
        # add source column if missing
        try:
            cur.execute("PRAGMA table_info(prices)")
            cols = [r[1] for r in cur.fetchall()]
            if "source" not in cols:
                cur.execute("ALTER TABLE prices ADD COLUMN source TEXT")
            cur.execute("PRAGMA table_info(features)")
            fcols = [r[1] for r in cur.fetchall()]
            if "source" not in fcols:
                cur.execute("ALTER TABLE features ADD COLUMN source TEXT")
            self.conn.commit()
        except Exception:
            # if ALTER TABLE fails for any reason, ignore — tables created above will include source
            pass

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
        # source is optional: can be included as df['source'] or passed via kwargs
        cols = ["datetime", "open", "high", "low", "close", "adj_close", "volume"]
        if "source" in df.columns:
            cols.append("source")
        df = df[cols].copy()
        # prepare rows for INSERT OR REPLACE using row iteration for clarity
        rows = []
        for _, r in df.iterrows():
            dt = r["datetime"]
            o = r["open"]
            h = r["high"]
            l = r["low"]
            c = r["close"]
            ac = r["adj_close"]
            v = r["volume"]
            src = r["source"] if "source" in r.index else None
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
                    src,
                )
                rows.append(row)
            except Exception:
                # skip rows that fail conversion
                continue

        if not rows:
            return

        cur = self.conn.cursor()
        # if source column exists in table, include it in insert
        cur.execute("PRAGMA table_info(prices)")
        cols = [r[1] for r in cur.fetchall()]
        if "source" in cols:
            cur.executemany(
                "INSERT OR REPLACE INTO prices (ticker, datetime, open, high, low, close, adj_close, volume, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        else:
            # fallback for older schema
            trimmed = [r[:-1] for r in rows]
            cur.executemany(
                "INSERT OR REPLACE INTO prices (ticker, datetime, open, high, low, close, adj_close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                trimmed,
            )
        self.conn.commit()

    def read_price_data(self, ticker: str, source: Optional[str] = None) -> pd.DataFrame:
        """Read price data for a ticker. If source is provided, filter by source.

        Returns a DataFrame ordered by datetime.
        """
        if source:
            df = pd.read_sql_query("SELECT * FROM prices WHERE ticker = ? AND source = ? ORDER BY datetime", self.conn, params=(ticker, source))
        else:
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
                # include optional source column if present on feat_df
                src = r.get("source") if "source" in feat_df.columns else None
                rows.append((ticker, pd.Timestamp(dt).to_pydatetime(), c, float(val), src))
        cur = self.conn.cursor()
        # adapt insert depending on schema
        cur.execute("PRAGMA table_info(features)")
        fcols = [r[1] for r in cur.fetchall()]
        if "source" in fcols:
            cur.executemany("INSERT OR REPLACE INTO features (ticker, datetime, feature_key, feature_value, source) VALUES (?, ?, ?, ?, ?)", rows)
        else:
            trimmed = [r[:-1] for r in rows]
            cur.executemany("INSERT OR REPLACE INTO features (ticker, datetime, feature_key, feature_value) VALUES (?, ?, ?, ?)", trimmed)
        self.conn.commit()
