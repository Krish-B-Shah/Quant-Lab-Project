"""
FastAPI server for GatorAI Quant Lab Platform.

Provides REST API endpoints for:
- Data fetching and retrieval
- ML predictions
- Backtesting
- Portfolio optimization
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys

# Add src to path for imports
root = Path(__file__).resolve().parents[2]
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data.manager import DataManager
from data.storage.sqlite_adapter import SQLiteAdapter
from data.features import rsi, macd, bollinger_bands, ema, rolling_volatility, rolling_sharpe
from backtesting.backtest_engine import run_backtest_strategy, BacktestResult
from backtesting.strategy import (
    BaseStrategy, EqualWeightStrategy, MomentumStrategy,
    VolatilityWeightedStrategy, MeanReversionStrategy, StrategyConfig
)
from backtesting.metrics import max_drawdown, annualized_sharpe
from optimization.optimizer import (
    mean_variance_optimize, black_litterman_optimize,
    risk_parity_optimize, cvar_optimize
)
from optimization.ml_models import ReturnForecaster

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="GatorAI Quant Lab API",
    description="REST API for quantitative trading platform",
    version="1.0.0"
)

# CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize storage and data manager
storage = SQLiteAdapter()
data_manager = DataManager(storage)

# In-memory cache for ML models (in production, use Redis or similar)
_model_cache: Dict[str, ReturnForecaster] = {}


# ==================== Pydantic Models ====================

class DataFetchRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols")
    interval: str = Field("1d", description="Data interval (1d, 1h, 1m)")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    refresh: bool = Field(False, description="Force refresh of all data")


class PredictRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols")
    features: Optional[List[str]] = Field(None, description="Features to use for prediction")
    retrain: bool = Field(False, description="Retrain the model")
    blend: float = Field(0.5, ge=0.0, le=1.0, description="Blend factor for ML predictions")


class BacktestRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols")
    strategy: str = Field(..., description="Strategy name")
    strategy_params: Dict = Field(default_factory=dict, description="Strategy parameters")
    rebalance: str = Field("monthly", description="Rebalancing frequency")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    cost_bps: float = Field(5.0, ge=0.0, description="Transaction costs in basis points")
    slippage: float = Field(0.001, ge=0.0, le=1.0, description="Slippage as fraction")
    use_ml_predictions: bool = Field(False, description="Use ML predictions in optimization")
    ml_blend: float = Field(0.5, ge=0.0, le=1.0, description="ML blend factor")


class OptimizeRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols")
    method: str = Field("mean_variance", description="Optimization method")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    risk_aversion: float = Field(1.0, ge=0.0, description="Risk aversion parameter")
    long_only: bool = Field(True, description="Long-only constraint")
    use_ml_predictions: bool = Field(False, description="Use ML predictions")
    ml_blend: float = Field(0.5, ge=0.0, le=1.0, description="ML blend factor")
    position_limit: Optional[float] = Field(None, description="Maximum position size")
    max_turnover: Optional[float] = Field(None, description="Maximum turnover")


# ==================== Helper Functions ====================

def _get_prices_df(tickers: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Load price data for tickers and return as DataFrame with tickers as columns."""
    prices_dict = {}
    for ticker in tickers:
        price_data = storage.read_price_data(ticker)
        if price_data.empty:
            raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")
        price_data = price_data.set_index("datetime")
        if start_date:
            price_data = price_data[price_data.index >= pd.to_datetime(start_date)]
        if end_date:
            price_data = price_data[price_data.index <= pd.to_datetime(end_date)]
        prices_dict[ticker] = price_data["adj_close"]
    
    prices_df = pd.DataFrame(prices_dict)
    if prices_df.empty:
        raise HTTPException(status_code=404, detail="No price data available for the specified date range")
    return prices_df


def _compute_features(ticker: str, price_series: pd.Series) -> pd.DataFrame:
    """Compute technical features for a ticker."""
    # Get full OHLC data from storage
    try:
        price_data = storage.read_price_data(ticker)
        if price_data.empty:
            # Fallback: create minimal DataFrame from series
            df = price_series.to_frame(name="close")
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            if "datetime" not in df.columns and df.index.name != "datetime":
                df = df.reset_index()
                if "datetime" not in df.columns:
                    df["datetime"] = df.index
        else:
            # Use full OHLC data
            price_data = price_data.set_index("datetime")
            df = price_data[["open", "high", "low", "close", "adj_close"]].copy()
            df["close"] = df["adj_close"]  # Use adjusted close for calculations
            df = df.reset_index()
    except Exception as e:
        logger.warning(f"Failed to load full data for {ticker}, using simplified: {e}")
        # Fallback
        df = price_series.to_frame(name="close").reset_index()
        if "datetime" not in df.columns:
            df["datetime"] = df.index
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
    
    # Ensure datetime column
    if "datetime" not in df.columns:
        df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    
    # Compute features
    features_df = pd.DataFrame(index=df.index)
    
    try:
        features_df["rsi"] = rsi(df["close"], period=14)
    except Exception as e:
        logger.warning(f"Failed to compute RSI for {ticker}: {e}")
    
    try:
        macd_df = macd(df)
        features_df = pd.concat([features_df, macd_df], axis=1)
    except Exception as e:
        logger.warning(f"Failed to compute MACD for {ticker}: {e}")
    
    try:
        bb_df = bollinger_bands(df)
        features_df = pd.concat([features_df, bb_df], axis=1)
    except Exception as e:
        logger.warning(f"Failed to compute Bollinger Bands for {ticker}: {e}")
    
    try:
        features_df["rolling_vol"] = rolling_volatility(df)
    except Exception as e:
        logger.warning(f"Failed to compute rolling volatility for {ticker}: {e}")
    
    try:
        features_df["rolling_sharpe"] = rolling_sharpe(df)
    except Exception as e:
        logger.warning(f"Failed to compute rolling Sharpe for {ticker}: {e}")
    
    return features_df.ffill().fillna(0)


def _train_ml_model(tickers: List[str], features_list: Optional[List[str]] = None) -> ReturnForecaster:
    """Train ML model on historical data."""
    try:
        # Get price data
        prices_df = _get_prices_df(tickers)
        
        # Compute features for all tickers
        all_features = []
        all_returns = []
        
        for ticker in tickers:
            price_series = prices_df[ticker]
            features_df = _compute_features(ticker, price_series.to_frame())
            
            # Prepare features
            feature_cols = features_list or features_df.columns.tolist()
            available_features = [f for f in feature_cols if f in features_df.columns]
            
            if not available_features:
                logger.warning(f"No features available for {ticker}")
                continue
            
            X = features_df[available_features].fillna(0)
            
            # Target: forward returns
            y = price_series.pct_change().shift(-1).fillna(0)
            
            # Align indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) < 50:  # Need sufficient data
                logger.warning(f"Insufficient data for {ticker}: {len(X)} samples")
                continue
            
            all_features.append(X)
            all_returns.append(y)
        
        if not all_features:
            raise ValueError("No features computed for any ticker")
        
        # Combine features across tickers
        X_combined = pd.concat(all_features, axis=1).fillna(0)
        y_combined = pd.concat(all_returns, axis=0).fillna(0)
        
        # Align indices
        common_idx = X_combined.index.intersection(y_combined.index)
        X_combined = X_combined.loc[common_idx]
        y_combined = y_combined.loc[common_idx]
        
        if len(X_combined) < 50:
            raise ValueError(f"Insufficient combined data: {len(X_combined)} samples")
        
        # Train model
        model = ReturnForecaster(alpha=1.0)
        model.fit(X_combined, y_combined)
        
        return model
        
    except Exception as e:
        logger.error(f"Error training ML model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to train ML model: {str(e)}")


def _get_ml_predictions(tickers: List[str], model: Optional[ReturnForecaster] = None, 
                        features_list: Optional[List[str]] = None) -> pd.Series:
    """Get ML predictions for tickers."""
    try:
        if model is None:
            model_key = "_".join(sorted(tickers))
            if model_key not in _model_cache:
                _model_cache[model_key] = _train_ml_model(tickers, features_list)
            model = _model_cache[model_key]
        
        # Get latest price data
        prices_df = _get_prices_df(tickers)
        
        # Compute features for latest data
        predictions = []
        ticker_names = []
        
        for ticker in tickers:
            price_series = prices_df[ticker]
            features_df = _compute_features(ticker, price_series)
            
            # Get latest features
            if features_df.empty:
                continue
            
            feature_cols = features_list or features_df.columns.tolist()
            available_features = [f for f in feature_cols if f in features_df.columns]
            
            if not available_features:
                continue
            
            X_latest = features_df[available_features].iloc[-1:].fillna(0)
            
            # Predict
            pred = model.predict(X_latest)
            if len(pred) > 0:
                predictions.append(pred.iloc[0])
                ticker_names.append(ticker)
        
        if not predictions:
            raise ValueError("No predictions generated")
        
        return pd.Series(predictions, index=ticker_names)
        
    except Exception as e:
        logger.error(f"Error generating ML predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate predictions: {str(e)}")


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "success",
        "message": "GatorAI Quant Lab API",
        "version": "1.0.0",
        "endpoints": {
            "data": "/data/prices, /data/features, /data/fetch",
            "ml": "/predict",
            "backtesting": "/backtest",
            "optimization": "/optimize"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/data/fetch")
async def fetch_data(request: DataFetchRequest, background_tasks: BackgroundTasks):
    """Fetch market data for tickers."""
    try:
        async def _fetch():
            try:
                await data_manager.fetch(
                    request.tickers,
                    interval=request.interval,
                    start=request.start_date,
                    end=request.end_date,
                    refresh=request.refresh
                )
                # Generate features after fetching
                # Note: DataManager expects functions that take a DataFrame and return Series or DataFrame
                # The price_df from storage has columns: datetime, open, high, low, close, adj_close, volume
                def rsi_feature(df):
                    return rsi(df["adj_close"] if "adj_close" in df.columns else df["close"], period=14)
                
                def macd_feature(df):
                    # Create a copy with 'close' column for macd function
                    df_copy = df.copy()
                    if "adj_close" in df_copy.columns:
                        df_copy["close"] = df_copy["adj_close"]
                    return macd(df_copy)
                
                def bb_feature(df):
                    df_copy = df.copy()
                    if "adj_close" in df_copy.columns:
                        df_copy["close"] = df_copy["adj_close"]
                    return bollinger_bands(df_copy)
                
                def vol_feature(df):
                    df_copy = df.copy()
                    if "adj_close" in df_copy.columns:
                        df_copy["close"] = df_copy["adj_close"]
                    return rolling_volatility(df_copy)
                
                def sharpe_feature(df):
                    df_copy = df.copy()
                    if "adj_close" in df_copy.columns:
                        df_copy["close"] = df_copy["adj_close"]
                    return rolling_sharpe(df_copy)
                
                feature_funcs = [rsi_feature, macd_feature, bb_feature, vol_feature, sharpe_feature]
                data_manager.generate_and_store_features(request.tickers, feature_funcs)
                logger.info(f"Successfully fetched and processed data for {request.tickers}")
            except Exception as e:
                logger.error(f"Error in background fetch task: {e}", exc_info=True)
        
        # Run in background
        background_tasks.add_task(_fetch)
        
        return {
            "status": "success",
            "message": "Data fetch initiated",
            "tickers": request.tickers,
            "interval": request.interval
        }
    except Exception as e:
        logger.error(f"Error initiating data fetch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/prices")
async def get_prices(
    tickers: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get price data for tickers."""
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        prices_df = _get_prices_df(ticker_list, start_date, end_date)
        
        # Convert to dict for JSON serialization
        data = {
            "datetime": prices_df.index.strftime("%Y-%m-%d").tolist(),
            "prices": {}
        }
        for ticker in ticker_list:
            if ticker in prices_df.columns:
                data["prices"][ticker] = prices_df[ticker].tolist()
        
        return {
            "status": "success",
            "data": data,
            "tickers": ticker_list
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/features")
async def get_features(
    tickers: str,
    features: Optional[str] = None
):
    """Get computed features for tickers."""
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        feature_list = [f.strip() for f in features.split(",")] if features else None
        
        result = {}
        for ticker in ticker_list:
            price_data = storage.read_price_data(ticker)
            if price_data.empty:
                continue
            
            price_series = price_data.set_index("datetime")["adj_close"]
            features_df = _compute_features(ticker, price_series)
            
            if feature_list:
                available_features = [f for f in feature_list if f in features_df.columns]
                features_df = features_df[available_features]
            
            result[ticker] = {
                "datetime": features_df.index.strftime("%Y-%m-%d").tolist(),
                "features": features_df.to_dict(orient="list")
            }
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        logger.error(f"Error getting features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictRequest):
    """Generate ML predictions for tickers."""
    try:
        # Train or retrieve model
        model_key = "_".join(sorted(request.tickers))
        if request.retrain or model_key not in _model_cache:
            model = _train_ml_model(request.tickers, request.features)
            _model_cache[model_key] = model
        else:
            model = _model_cache[model_key]
        
        # Get predictions
        predictions = _get_ml_predictions(request.tickers, model, request.features)
        
        return {
            "status": "success",
            "predictions": predictions.to_dict(),
            "tickers": request.tickers,
            "blend": request.blend
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
async def backtest(request: BacktestRequest):
    """Run a backtest with specified strategy."""
    try:
        # Get price data
        prices_df = _get_prices_df(request.tickers, request.start_date, request.end_date)
        
        # Create strategy
        strategy_map = {
            "equal_weight": EqualWeightStrategy,
            "momentum": MomentumStrategy,
            "volatility_weighted": VolatilityWeightedStrategy,
            "mean_reversion": MeanReversionStrategy
        }
        
        if request.strategy not in strategy_map:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
        
        strategy_class = strategy_map[request.strategy]
        config = StrategyConfig(params=request.strategy_params)
        strategy = strategy_class(config=config)
        
        # Get ML predictions if requested
        ml_predictions = None
        if request.use_ml_predictions:
            try:
                ml_predictions = _get_ml_predictions(request.tickers)
            except Exception as e:
                logger.warning(f"Failed to get ML predictions: {e}")
        
        # Run backtest
        result = run_backtest_strategy(
            prices_df,
            strategy,
            rebalance=request.rebalance,
            cost_bps=request.cost_bps,
            slippage=request.slippage
        )
        
        # Compute additional metrics
        max_dd = max_drawdown(result.equity_curve)
        sharpe = annualized_sharpe(result.returns)
        
        # Prepare response
        equity_curve_dict = {
            date.strftime("%Y-%m-%d"): float(value)
            for date, value in result.equity_curve.items()
        }
        
        returns_dict = {
            date.strftime("%Y-%m-%d"): float(value)
            for date, value in result.returns.items()
        }
        
        return {
            "status": "success",
            "backtest_id": f"bt_{datetime.now().timestamp()}",
            "results": {
                "equity_curve": equity_curve_dict,
                "returns": returns_dict,
                "stats": {
                    "cagr": result.stats.get("cagr", 0.0),
                    "volatility": result.stats.get("vol", 0.0),
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "total_return": float(result.equity_curve.iloc[-1] - 1.0) if len(result.equity_curve) > 0 else 0.0
                },
                "strategy": request.strategy,
                "parameters": request.strategy_params
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize(request: OptimizeRequest):
    """Optimize portfolio weights."""
    try:
        # Get price data and compute returns
        prices_df = _get_prices_df(request.tickers, request.start_date, request.end_date)
        returns_df = prices_df.pct_change().dropna()
        
        if returns_df.empty:
            raise HTTPException(status_code=400, detail="Insufficient data for optimization")
        
        # Get ML predictions if requested
        predicted_returns = None
        if request.use_ml_predictions:
            try:
                predictions = _get_ml_predictions(request.tickers)
                predicted_returns = predictions.reindex(returns_df.columns).fillna(returns_df.mean())
            except Exception as e:
                logger.warning(f"Failed to get ML predictions: {e}")
        
        # Run optimization
        method_map = {
            "mean_variance": mean_variance_optimize,
            "black_litterman": black_litterman_optimize,
            "risk_parity": risk_parity_optimize,
            "cvar": cvar_optimize
        }
        
        if request.method not in method_map:
            raise HTTPException(status_code=400, detail=f"Unknown optimization method: {request.method}")
        
        opt_func = method_map[request.method]
        
        # Prepare optimization parameters
        opt_kwargs = {
            "returns": returns_df,
            "predicted_returns": predicted_returns,
            "blend": request.ml_blend if request.use_ml_predictions else 0.0,
            "long_only": request.long_only,
            "position_limit": request.position_limit,
            "max_turnover": request.max_turnover
        }
        
        if request.method == "mean_variance":
            opt_kwargs["risk_aversion"] = request.risk_aversion
        
        weights = opt_func(**opt_kwargs)
        
        # Compute portfolio metrics
        portfolio_return = (weights * returns_df.mean()).sum() * 252
        portfolio_vol = np.sqrt((weights @ returns_df.cov() @ weights) * 252)
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0
        
        return {
            "status": "success",
            "optimization_id": f"opt_{datetime.now().timestamp()}",
            "weights": weights.to_dict(),
            "expected_return": float(portfolio_return),
            "expected_volatility": float(portfolio_vol),
            "sharpe_ratio": float(sharpe_ratio),
            "method": request.method
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

