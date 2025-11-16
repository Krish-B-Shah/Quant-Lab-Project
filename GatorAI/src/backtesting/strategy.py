"""Strategies: abstractions and small example implementations.

This module defines a minimal research-oriented strategy contract used by
the lightweight backtest/demo tools in this repository.

Contract summary
----------------
- generate_signals(price_df[, features]) -> pd.DataFrame
    - index: pd.DatetimeIndex matching the price_df index
    - columns: tickers (same order/names as price_df.columns)
    - values: numeric scores (scale is arbitrary). Implementations may
        return raw scores, z-scores, past returns, etc.

- allocate(signals, price_df) -> pd.DataFrame
    - returns weights with same shape as signals
    - rows should sum to 1.0 (or 0.0 for cash/no-exposure rows)
    - long-only strategies should return non-negative weights

Notes and examples
------------------
The code in this file is intentionally small and explicitly written for
readability. Example usage (demo):

        from backtesting.strategy import MomentumStrategy, StrategyConfig
        strat = MomentumStrategy(config=StrategyConfig(params={'lookback': 20}))
        prices = load_prices_from_csv('data/processed/SPY.csv')  # pandas DataFrame
        sig = strat.generate_signals(prices)
        w = strat.allocate(sig, prices)

Edge cases
----------
- If the lookback/window parameter is larger than the available rows, most
    implementations will produce many NaNs/zeros; callers should guard and/or
    choose smaller lookbacks for very short test series.
- All implementations return filled numeric DataFrames (NaNs are converted
    to zeros where appropriate) for convenience in demo/test code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import numpy as np


@dataclass
class StrategyConfig:
    params: Dict = None


class BaseStrategy(ABC):
    """Base strategy abstraction.

    Subclasses must implement two methods:
      - generate_signals(price_df, features=None) -> pd.DataFrame
      - allocate(signals, price_df) -> pd.DataFrame

    Implementations in this file follow a simple pattern: signals are
    calculated from price history and then normalized into long-only
    weights (rows sum to 1). Callers can override or extend allocation
    behavior as needed.

    The class stores a lightweight `StrategyConfig` to keep parameters
    together and readable in demos.
    """

    name: str
    config: StrategyConfig

    def __init__(self, name: str = "base", config: Optional[StrategyConfig] = None):
        self.name = name
        self.config = config or StrategyConfig(params={})

    @abstractmethod
    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return a DataFrame of raw signals/scores.

        - price_df: DataFrame of prices (index=dates, cols=tickers)
        - Return must be same shape (index & columns) as price_df.
        """
        raise NotImplementedError()

    @abstractmethod
    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw signals into weights.

        Typical behavior is to clip to long-only and row-normalize so each
        date's weights sum to 1. Returning zeros for rows with no exposure
        is allowed.
        """
        raise NotImplementedError()


class EqualWeightStrategy(BaseStrategy):
    """Simple equal-weight strategy: allocate equal weight across all tickers in the universe.

    Parameters (via config.params):
      - universe: Optional[list] restrict tickers
      - long_only: bool (default True)
    """

    def __init__(self, name: str = "equal_weight", config: Optional[StrategyConfig] = None):
        super().__init__(name=name, config=config)

    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        # Simple constant signal of 1 for all tickers and dates in price_df
        sig = pd.DataFrame(1.0, index=price_df.index, columns=price_df.columns)
        return sig

    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        # Ignore signals content and produce equal weights across columns
        cols = price_df.columns.tolist()
        if len(cols) == 0:
            return pd.DataFrame(index=price_df.index, columns=cols)
        w = 1.0 / len(cols)
        weights = pd.DataFrame(w, index=price_df.index, columns=cols)
        return weights


class MomentumStrategy(BaseStrategy):
    """Momentum strategy: signal = past return over lookback window.

    Params (config.params):
      - lookback: int (periods for past return), default 90
      - top_k: Optional[int] to only take top-k longs (long_only mode)
      - long_only: bool (default True)
    """

    def __init__(self, name: str = "momentum", config: Optional[StrategyConfig] = None):
        super().__init__(name=name, config=config)

    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        params = self.config.params or {}
        lookback = int(params.get("lookback", 90))
        # past return over lookback
        sig = price_df.pct_change(periods=lookback)
        return sig.fillna(0.0)

    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params or {}
        top_k = params.get("top_k", None)
        long_only = bool(params.get("long_only", True))

        s = signals.copy()
        # if top_k specified, zero out others per row
        if top_k is not None:
            def topk_row(r):
                if r.isna().all():
                    return r * 0.0
                idx = r.nlargest(top_k).index
                out = pd.Series(0.0, index=r.index)
                out.loc[idx] = r.loc[idx]
                return out

            s = s.apply(topk_row, axis=1)

        if long_only:
            s = s.clip(lower=0.0)

        # normalize rows to sum to 1
        row_sum = s.sum(axis=1).replace(0.0, np.nan)
        weights = s.div(row_sum, axis=0).fillna(0.0)
        return weights


class VolatilityWeightedStrategy(BaseStrategy):
    """Weight assets inversely proportional to historical volatility.

    Params (config.params):
      - vol_window: int (rolling window for volatility), default 21
      - long_only: bool (default True)
    """

    def __init__(self, name: str = "vol_weighted", config: Optional[StrategyConfig] = None):
        super().__init__(name=name, config=config)

    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        params = self.config.params or {}
        vol_window = int(params.get("vol_window", 21))
        rets = price_df.pct_change()
        vol = rets.rolling(vol_window).std()
        eps = 1e-8
        sig = 1.0 / (vol + eps)
        return sig.fillna(0.0)

    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params or {}
        long_only = bool(params.get("long_only", True))

        s = signals.copy()
        if long_only:
            s = s.clip(lower=0.0)

        row_sum = s.sum(axis=1).replace(0.0, np.nan)
        weights = s.div(row_sum, axis=0).fillna(0.0)
        return weights


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion: signal = - zscore of recent returns (buy when underperformed).

    Params (config.params):
      - lookback: int for rolling mean/std of returns, default 20
      - long_only: bool (default True)
      - top_k: Optional[int] to only take extreme names
    """

    def __init__(self, name: str = "mean_reversion", config: Optional[StrategyConfig] = None):
        super().__init__(name=name, config=config)

    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        params = self.config.params or {}
        lookback = int(params.get("lookback", 20))
        rets = price_df.pct_change()
        mu = rets.rolling(lookback).mean()
        sigma = rets.rolling(lookback).std()
        eps = 1e-8
        z = (rets - mu) / (sigma + eps)
        sig = -z  # mean-revert: negative z-score -> buy
        return sig.fillna(0.0)

    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params or {}
        long_only = bool(params.get("long_only", True))
        top_k = params.get("top_k", None)

        s = signals.copy()
        # if top_k specified, pick top (most negative z -> largest positive signals after negation)
        if top_k is not None:
            def topk_row(r):
                if r.isna().all():
                    return r * 0.0
                idx = r.nlargest(top_k).index
                out = pd.Series(0.0, index=r.index)
                out.loc[idx] = r.loc[idx]
                return out

            s = s.apply(topk_row, axis=1)

        if long_only:
            s = s.clip(lower=0.0)

        row_sum = s.sum(axis=1).replace(0.0, np.nan)
        weights = s.div(row_sum, axis=0).fillna(0.0)
        return weights


class MLReturnPredictionStrategy(BaseStrategy):
    """ML-based strategy that uses trained models to predict returns and size positions.
    
    This strategy integrates with the ML prediction pipeline to generate trading signals
    based on machine learning forecasts. Includes confidence-based position sizing and
    comprehensive logging of predictions vs actuals for performance analysis.
    
    Params (config.params):
      - model_path: str path to trained MLReturnPredictor model file
      - feature_engineer: FeatureEngineer instance for generating ML features
      - position_scaling: str method for position sizing ("confidence", "prediction", "fixed")
      - min_confidence: float threshold below which no trades are made (default 0.6)
      - max_position: float maximum position size per asset (default 1.0)
      - long_only: bool whether to allow short positions (default True)
    """
    
    def __init__(self, name: str = "ml_prediction", config: Optional[StrategyConfig] = None):
        super().__init__(name=name, config=config)
        self._ml_predictor = None
        self._feature_engineer = None
        self._prediction_log = []  # Store predictions for analysis
        
    def _initialize_ml_components(self):
        """Lazy initialization of ML predictor and feature engineer."""
        if self._ml_predictor is not None:
            return  # Already initialized
            
        params = self.config.params or {}
        
        # Try to get ML predictor directly from params first
        direct_predictor = params.get("ml_predictor")
        if direct_predictor is not None:
            self._ml_predictor = direct_predictor
        else:
            # Fallback to loading from model_path
            model_path = params.get("model_path")
            if model_path:
                try:
                    from .ml_models import MLReturnPredictor
                    self._ml_predictor = MLReturnPredictor.load_model(model_path)
                except Exception as e:
                    print(f"Warning: Could not load ML model from {model_path}: {e}")
                    self._ml_predictor = None
        
        # Initialize feature engineer
        feature_engineer = params.get("feature_engineer")
        if feature_engineer is not None:
            self._feature_engineer = feature_engineer
        else:
            # Create default feature engineer
            try:
                from .ml_features import FeatureEngineer
                lookback = params.get("lookback_window", 21)
                self._feature_engineer = FeatureEngineer(lookback_window=lookback)
            except Exception as e:
                print(f"Warning: Could not create feature engineer: {e}")
                self._feature_engineer = None
    
    def generate_signals(self, price_df: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate ML-based trading signals.
        
        Args:
            price_df: Price data for signal generation
            features: Pre-computed features (optional, will generate if None)
            
        Returns:
            DataFrame of ML predictions as signals (same shape as price_df)
        """
        self._initialize_ml_components()
        
        # Fallback to zero signals if ML components not available
        if self._ml_predictor is None:
            print("Warning: ML predictor not available, returning zero signals")
            return pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
        
        # Use pre-computed features if available
        params = self.config.params or {}
        pre_computed_features = params.get('pre_computed_features')
        
        if pre_computed_features is not None:
            # Use pre-computed features
            aligned_features = pre_computed_features.reindex(price_df.index).fillna(0.0)
        elif features is not None:
            # Use provided features
            aligned_features = features.reindex(price_df.index).fillna(0.0)
        else:
            # Try to generate features (this will fail with single Close price column)
            if self._feature_engineer is None:
                print("Warning: No feature engineer available, returning zero signals")
                return pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
                
            try:
                generated_features = self._feature_engineer.create_technical_features(price_df)
                aligned_features = generated_features.reindex(price_df.index).fillna(0.0)
            except Exception as e:
                print(f"Warning: Feature generation failed: {e}")
                return pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
        
        # Make ML predictions
        try:
            predictions_df = self._ml_predictor.predict_with_confidence(aligned_features)
        except Exception as e:
            print(f"Warning: ML prediction failed: {e}")
            return pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
        
        # Store predictions for later analysis
        for idx, row in predictions_df.iterrows():
            self._prediction_log.append({
                'timestamp': idx,
                'prediction': row['prediction'],
                'confidence': row['confidence'],
                'model': self._ml_predictor.model_type
            })
        
        # Convert predictions to signals for each ticker
        signals = pd.DataFrame(index=price_df.index, columns=price_df.columns)
        
        for col in price_df.columns:
            # For now, assume same prediction applies to all tickers
            # In practice, you'd want separate models per ticker or cross-sectional modeling
            signals[col] = predictions_df['prediction'].reindex(price_df.index).fillna(0.0)
            
        return signals
    
    def allocate(self, signals: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """Convert ML signals into position weights with confidence-based sizing.
        
        Args:
            signals: ML prediction signals from generate_signals()
            price_df: Price data for context
            
        Returns:
            DataFrame of position weights (same shape as signals)
        """
        params = self.config.params or {}
        position_scaling = params.get("position_scaling", "confidence")
        min_confidence = params.get("min_confidence", 0.6)
        max_position = params.get("max_position", 1.0)
        long_only = bool(params.get("long_only", True))
        
        # Initialize weights
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        
        # Get confidence data from ML predictor if available
        if hasattr(self, '_ml_predictor') and self._ml_predictor is not None:
            try:
                # Re-generate features and predictions to get confidence
                if self._feature_engineer is not None:
                    features = self._feature_engineer.create_technical_features(price_df)
                    aligned_features = features.reindex(price_df.index).fillna(0.0)
                    predictions_df = self._ml_predictor.predict_with_confidence(aligned_features)
                    
                    for col in signals.columns:
                        for idx in signals.index:
                            if idx not in predictions_df.index:
                                continue
                                
                            pred_row = predictions_df.loc[idx]
                            prediction = pred_row['prediction']
                            confidence = pred_row['confidence']
                            
                            # Only trade if confidence exceeds threshold
                            if confidence < min_confidence:
                                weights.loc[idx, col] = 0.0
                                continue
                            
                            # Position sizing based on method
                            if position_scaling == "confidence":
                                # Scale position by confidence level
                                base_position = np.sign(prediction) * min(max_position, confidence)
                            elif position_scaling == "prediction":
                                # Scale by prediction magnitude and confidence
                                base_position = np.tanh(prediction * 10) * confidence * max_position
                            else:  # "fixed"
                                # Fixed position size based on prediction direction
                                base_position = np.sign(prediction) * max_position
                            
                            # Apply long-only constraint
                            if long_only:
                                base_position = max(0.0, base_position)
                                
                            weights.loc[idx, col] = base_position
                            
            except Exception as e:
                print(f"Warning: Advanced position sizing failed: {e}")
                # Fallback to simple signal-based allocation
                weights = signals.copy()
        else:
            # Simple fallback: use signals directly
            weights = signals.copy()
        
        # Normalize weights per row (optional - depends on strategy preference)
        normalize_weights = params.get("normalize_weights", True)
        if normalize_weights:
            row_sums = weights.abs().sum(axis=1).replace(0.0, np.nan)
            weights = weights.div(row_sums, axis=0).fillna(0.0)
        
        return weights
    
    def get_prediction_log(self) -> pd.DataFrame:
        """Return log of ML predictions for analysis.
        
        Returns:
            DataFrame with columns: timestamp, prediction, confidence, model
        """
        if not self._prediction_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self._prediction_log)
