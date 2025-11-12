from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import List
from pathlib import Path
import pandas as pd

from .manager import DataManager
from .storage.sqlite_adapter import SQLiteAdapter
from .fetcher import get_fetcher
from . import features as feat

logger = logging.getLogger("gatorai.data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser("gatorai-data")
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])  # e.g. SPY QQQ IWM
    p.add_argument("--features", default="macd,rsi,bollinger", help="comma separated features to compute")
    p.add_argument("--interval", default="1d")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--config", default=None, help="path to YAML config file defining tickers/features/interval")
    p.add_argument("--fetcher", default=None, help="which fetcher to use: yahoo|polygon|alpha_vantage")
    p.add_argument("--analyze", action="store_true", help="run cross-source analysis for provided tickers")
    p.add_argument("--sources", nargs="+", default=None, help="list of sources to compare (e.g. csv yahoo polygon)")
    p.add_argument("--dry-run", action="store_true", help="only print what would be ingested/generated without writing to DB")
    return p.parse_args(argv)


async def main(argv: List[str] | None = None):
    args = parse_args(argv)
    # if dry-run, use an in-memory DB so nothing is written to disk
    storage = SQLiteAdapter(db_path=":memory:") if getattr(args, "dry_run", False) else SQLiteAdapter()
    # fetcher selection happens after config parsing so YAML/CLI can override
    chosen_fetcher = None
    # point to the package's processed folder (GatorAI/data/processed)
    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    # If local processed CSVs exist for tickers, ingest them first to avoid unnecessary downloads
    def ingest_local_if_present(tickers: List[str]):
        for t in tickers:
            candidates = [processed_dir / f"{t}_1d.csv", processed_dir / f"{t}_processed.csv", processed_dir / f"{t}.csv"]
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            if not found:
                logger.debug("no local CSV found for %s", t)
                continue
            logger.info("Ingesting local CSV for %s from %s", t, found)
            try:
                pdf = pd.read_csv(found)
            except Exception:
                logger.exception("failed to read local CSV %s", found)
                continue
            # normalize columns: accept Date or datetime, Open/High/Low/Close/Adj Close/Volume
            cols = {c.lower(): c for c in pdf.columns}
            # map expected names
            mapping = {}
            if "date" in cols:
                mapping[cols["date"]] = "datetime"
            elif "datetime" in cols:
                mapping[cols["datetime"]] = "datetime"
            for key in ["open", "high", "low", "close", "adj close", "adj_close", "volume"]:
                if key in cols:
                    mapping[cols[key]] = key.replace(" ", "_")
            try:
                pdf = pdf.rename(columns=mapping)
                # ensure we have required columns
                expected = ["datetime", "open", "high", "low", "close", "adj_close", "volume"]
                if not all(c in pdf.columns for c in expected):
                    logger.warning("local CSV %s missing some expected columns, skipping", found)
                    continue
                pdf = pdf[expected].copy()
                # coerce datetime and numeric columns, drop bad rows
                pdf["datetime"] = pd.to_datetime(pdf["datetime"], errors="coerce")
                for col in ["open", "high", "low", "close", "adj_close"]:
                    pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
                pdf["volume"] = pd.to_numeric(pdf["volume"], errors="coerce")
                # drop rows missing required values
                before = len(pdf)
                pdf = pdf.dropna(subset=["datetime", "open", "high", "low", "close", "adj_close", "volume"])
                after = len(pdf)
                if after == 0:
                    logger.warning("after cleaning, no valid rows in %s for %s", found, t)
                    continue
                # mark provenance for CSV ingestion
                pdf["source"] = "csv"
                storage.upsert_price_data(t, pdf)
                logger.info("Ingested %d/%d rows for %s from %s", after, before, t, found)
            except Exception:
                logger.exception("failed to ingest data from %s for %s", found, t)
    try:
        # try ingesting existing local CSVs before downloading
        ingest_local_if_present(args.tickers)

        # feature_map entries should return a Series/DataFrame with named columns
        feature_map = {
            "rsi": lambda df: feat.rsi(df["close"].rename("close")),
            "macd": lambda df: feat.macd(df),
            "bollinger": lambda df: feat.bollinger_bands(df),
            "ema_cross": lambda df: feat.ema_crossover(df),
            "sharpe": lambda df: feat.rolling_sharpe(df),
            "vol": lambda df: feat.rolling_volatility(df),
            "atr": lambda df: feat.atr(df),
        }

    # load from config file if provided (overrides CLI tickers/features if specified)
        if getattr(args, "config", None):
            try:
                # yaml is optional; only try to import it if the user provided a config
                try:
                    import yaml as _yaml
                except Exception:
                    _yaml = None
                    logger.warning("PyYAML not installed; cannot read config %s", args.config)
                if _yaml:
                    cfg = _yaml.safe_load(Path(args.config).read_text())
                    cfg_tickers = cfg.get("tickers")
                    cfg_features = cfg.get("features")
                    cfg_interval = cfg.get("interval")
                    cfg_fetcher = cfg.get("fetcher")
                    cfg_api_key = cfg.get("api_key")
                    if cfg_tickers:
                        args.tickers = cfg_tickers
                    if cfg_features:
                        requested = cfg_features
                    else:
                        requested = [s.strip().lower() for s in args.features.split(",") if s.strip()]
                    if cfg_interval:
                        args.interval = cfg_interval
                    if cfg_fetcher:
                        chosen_fetcher = (cfg_fetcher, cfg_api_key)
                else:
                    requested = [s.strip().lower() for s in args.features.split(",") if s.strip()]
            except Exception:
                logger.exception("failed to read config %s", args.config)
                requested = [s.strip().lower() for s in args.features.split(",") if s.strip()]
        else:
            requested = [s.strip().lower() for s in args.features.split(",") if s.strip()]

        # if CLI provided a fetcher, it takes precedence over config
        if getattr(args, "fetcher", None):
            chosen_fetcher = (args.fetcher, None)

        # instantiate fetcher and DataManager now that config/args are known
        fetcher_name, fetcher_key = (None, None)
        if chosen_fetcher:
            fetcher_name, fetcher_key = chosen_fetcher
        elif getattr(args, "fetcher", None):
            fetcher_name = args.fetcher

        fetcher = get_fetcher(fetcher_name, api_key=fetcher_key)
        dm = DataManager(storage, fetcher=fetcher)

        funcs = [feature_map[f] for f in requested if f in feature_map]

        start = time.perf_counter()
        # If analyze flag set, perform cross-source analysis and exit
        if getattr(args, "analyze", False):
            from .analysis import compare_sources

            srcs = args.sources or ["csv", "YahooFetcher", "PolygonFetcher"]
            for t in args.tickers:
                try:
                    s = compare_sources(storage, t, srcs)
                    logger.info("Analysis for %s: %s", t, s)
                except Exception:
                    logger.exception("analysis failed for %s", t)
            return
        logger.info("Starting download for %s", args.tickers)
        combined = await dm.fetch(args.tickers, interval=args.interval, start=args.start, end=args.end, refresh=args.refresh)
        logger.info("Downloaded rows: %s", len(combined))
        # log DB counts to confirm data was written
        try:
            for t in args.tickers:
                cnt = storage.get_price_count(t)
                logger.info("DB rows for %s: %s", t, cnt)
            logger.info("Total DB price rows: %s", storage.get_total_price_count())
        except Exception:
            logger.exception("failed to query DB counts")
        logger.info("Generating features: %s", requested)
        dm.generate_and_store_features(args.tickers, funcs)
        logger.info("Done in %.2fs", time.perf_counter() - start)
    except Exception as exc:  # catch-all to provide stacktrace in logs
        logger.exception("Fatal error during run: %s", exc)
        raise


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
