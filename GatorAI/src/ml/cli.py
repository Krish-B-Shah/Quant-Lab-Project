"""
CLI for ML prediction pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List

from .predictor import MLPredictor
from ..data.storage.sqlite_adapter import SQLiteAdapter
from ..data.manager import DataManager
from ..data.fetcher import get_fetcher
from ..data import features as feat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ensure_data_available(tickers: List[str], storage: SQLiteAdapter):
    """Ensure price data and features are available for tickers."""
    logger.info("Ensuring data is available for ML training")
    
    # Check if we need to fetch VIX data
    vix_data = storage.read_price_data("^VIX")
    if vix_data.empty:
        logger.info("Fetching VIX data for regime indicators")
        fetcher = get_fetcher("yahoo")
        dm = DataManager(storage, fetcher=fetcher)
        # Fetch VIX data
        asyncio.run(dm.fetch(["^VIX"], interval="1d", refresh=False))
    
    # Check if we need to fetch ticker data
    for ticker in tickers:
        price_data = storage.read_price_data(ticker)
        if price_data.empty:
            logger.info(f"Fetching price data for {ticker}")
            fetcher = get_fetcher("yahoo")
            dm = DataManager(storage, fetcher=fetcher)
            asyncio.run(dm.fetch([ticker], interval="1d", refresh=False))
        
        # Generate features if needed
        feature_map = {
            "rsi": lambda df: feat.rsi(df["close"].rename("close")),
            "macd": lambda df: feat.macd(df),
            "bollinger": lambda df: feat.bollinger_bands(df),
            "ema_cross": lambda df: feat.ema_crossover(df),
            "sharpe": lambda df: feat.rolling_sharpe(df),
            "vol": lambda df: feat.rolling_volatility(df),
            "atr": lambda df: feat.atr(df),
        }
        
        feature_funcs = [feature_map[f] for f in feature_map.keys()]
        dm = DataManager(storage)
        dm.generate_and_store_features([ticker], feature_funcs)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML prediction pipeline for next-day returns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and validate models for SPY, QQQ, IWM
  python -m GatorAI.src.ml.cli --tickers SPY QQQ IWM --train
  
  # Generate predictions for backtester
  python -m GatorAI.src.ml.cli --tickers SPY QQQ IWM --predict
  
  # Train with custom model parameters
  python -m GatorAI.src.ml.cli --tickers SPY --train --model-types xgboost random_forest
        """
    )
    
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "IWM"],
        help="Stock tickers to process (default: SPY QQQ IWM)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train and validate models using walk-forward validation",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Generate predictions for backtester consumption",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["linear", "random_forest", "xgboost"],
        choices=["linear", "random_forest", "xgboost"],
        help="Model types to use (default: all three)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GatorAI/ml/outputs"),
        help="Output directory for results (default: GatorAI/ml/outputs)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch data before training (if not already available)",
    )
    
    args = parser.parse_args()
    
    # Initialize storage
    storage = SQLiteAdapter()
    
    # Ensure data is available
    if args.fetch_data:
        ensure_data_available(args.tickers, storage)
    
    # Initialize predictor
    predictor = MLPredictor(
        storage=storage,
        output_dir=args.output_dir,
        model_types=args.model_types,
    )
    
    # Train and validate
    if args.train:
        logger.info(f"Training models for {args.tickers}")
        for ticker in args.tickers:
            try:
                results = predictor.train_and_validate(
                    ticker=ticker,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                predictor.save_results(ticker, results)
                logger.info(f"Completed training for {ticker}")
            except Exception as e:
                logger.error(f"Failed to train models for {ticker}: {e}", exc_info=True)
    
    # Generate predictions
    if args.predict:
        logger.info(f"Generating predictions for {args.tickers}")
        try:
            # Use best model (xgboost by default)
            best_model = "xgboost" if "xgboost" in args.model_types else args.model_types[0]
            predictions_df = predictor.generate_predictions_file(
                tickers=args.tickers,
                model_type=best_model,
            )
            logger.info(f"Generated predictions for {len(predictions_df)} tickers")
            logger.info(f"Predictions saved to {predictor.output_dir}")
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}", exc_info=True)
    
    if not args.train and not args.predict:
        parser.print_help()


if __name__ == "__main__":
    main()


