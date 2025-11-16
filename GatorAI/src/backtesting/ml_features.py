"""Feature engineering pipeline for ML backtesting.

This module creates technical indicators and features from price data for use in
ML-based trading strategies.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path


class FeatureEngineer:
    """Generate technical indicators and features from price data."""
    
    def __init__(self, lookback_window: int = 21):
        """
        Args:
            lookback_window: Number of periods to use for rolling calculations (default 21 ~= 1 month)
        """
        self.lookback_window = lookback_window
        
    def create_technical_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features from price data.
        
        Args:
            prices: DataFrame with OHLC data or just Close prices
            
        Returns:
            DataFrame with technical features aligned to price index
        """
        # Use Close/Adj Close if available, otherwise first numeric column
        if 'Adj Close' in prices.columns:
            close = prices['Adj Close']
        elif 'Close' in prices.columns:
            close = prices['Close']
        else:
            # Take first numeric column
            close = prices.select_dtypes(include=[np.number]).iloc[:, 0]
            
        features = pd.DataFrame(index=prices.index)
        
        # Returns
        features['returns_1d'] = close.pct_change()
        features['returns_5d'] = close.pct_change(5)
        features['returns_21d'] = close.pct_change(self.lookback_window)
        
        # Moving averages
        features['sma_5'] = close.rolling(5).mean()
        features['sma_21'] = close.rolling(self.lookback_window).mean()
        features['sma_50'] = close.rolling(50).mean()
        
        # Price relative to moving averages
        features['price_to_sma_5'] = close / features['sma_5'] - 1
        features['price_to_sma_21'] = close / features['sma_21'] - 1
        features['price_to_sma_50'] = close / features['sma_50'] - 1
        
        # Volatility features
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_21d'] = features['returns_1d'].rolling(self.lookback_window).std()
        
        # RSI (Relative Strength Index)
        features['rsi_14'] = self._calculate_rsi(close, 14)
        
        # Momentum indicators
        features['momentum_5'] = close / close.shift(5) - 1
        features['momentum_21'] = close / close.shift(self.lookback_window) - 1
        
        # Volume features if available
        if 'Volume' in prices.columns:
            volume = prices['Volume']
            features['volume_ma_21'] = volume.rolling(self.lookback_window).mean()
            features['volume_ratio'] = volume / features['volume_ma_21']
            
        # Bollinger Bands
        bb_middle = features['sma_21']
        bb_std = features['returns_1d'].rolling(self.lookback_window).std()
        features['bb_upper'] = bb_middle + (2 * bb_std * close)
        features['bb_lower'] = bb_middle - (2 * bb_std * close)
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_targets(self, prices: pd.DataFrame, target_horizon: int = 1) -> pd.Series:
        """Create target variable (future returns) for ML training.
        
        Args:
            prices: Price DataFrame
            target_horizon: Number of periods ahead to predict (default 1 = next day)
            
        Returns:
            Series of target returns aligned with feature dates
        """
        # Use same price column logic as features
        if 'Adj Close' in prices.columns:
            close = prices['Adj Close']
        elif 'Close' in prices.columns:
            close = prices['Close']
        else:
            close = prices.select_dtypes(include=[np.number]).iloc[:, 0]
            
        # Future returns (shift backwards so target[t] = return[t+horizon])
        targets = close.pct_change(target_horizon).shift(-target_horizon)
        targets.name = f'target_return_{target_horizon}d'
        
        return targets
    
    def prepare_ml_dataset(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Create complete feature matrix and targets for ML training.
        
        Returns:
            Tuple of (features_df, targets_series) with aligned indices and no NaNs
        """
        features = self.create_technical_features(prices)
        targets = self.create_targets(prices)
        
        # Align and remove NaN rows
        combined = pd.concat([features, targets], axis=1).dropna()
        
        feature_cols = features.columns
        X = combined[feature_cols]
        y = combined[targets.name]
        
        return X, y
    
    def split_time_series(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        train_pct: float = 0.7, 
        val_pct: float = 0.15
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split time series data chronologically.
        
        Args:
            X: Feature matrix
            y: Target series  
            train_pct: Percentage for training (default 0.7)
            val_pct: Percentage for validation (default 0.15)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end] 
        X_test = X.iloc[val_end:]
        
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_prepare_features(csv_path: str, lookback_window: int = 21) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience function to load CSV and create ML dataset.
    
    Args:
        csv_path: Path to price CSV file
        lookback_window: Lookback window for features
        
    Returns:
        Tuple of (features, targets) ready for ML training
    """
    # Load price data
    prices = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    
    # Create feature engineer and prepare dataset
    engineer = FeatureEngineer(lookback_window=lookback_window)
    X, y = engineer.prepare_ml_dataset(prices)
    
    return X, y


def create_sample_data_for_testing() -> pd.DataFrame:
    """Create synthetic SPY-like data for testing when real data unavailable."""
    
    # Create date range (1 year of trading days)
    dates = pd.bdate_range("2024-01-01", periods=252)
    
    # Simulate SPY-like price data
    np.random.seed(42)
    initial_price = 400.0
    returns = np.random.normal(0.0008, 0.015, size=len(dates))  # ~0.08% daily return, 1.5% vol
    returns = np.cumsum(returns)  # Random walk
    prices = initial_price * np.exp(returns)
    
    # Create OHLC data
    price_data = pd.DataFrame({
        "Open": prices * (1 + np.random.normal(0, 0.001, len(prices))),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.003, len(prices)))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.003, len(prices)))),
        "Close": prices,
        "Adj Close": prices,
        "Volume": np.random.lognormal(15, 0.3, len(prices)).astype(int)
    }, index=dates)
    
    return price_data


if __name__ == "__main__":
    # Demo usage
    from pathlib import Path
    
    # Load SPY data
    repo_root = Path(__file__).parents[2]
    spy_path = repo_root / "data" / "processed" / "SPY_sontest.csv"
    
    if spy_path.exists():
        print(f"Loading data from {spy_path}")
        X, y = load_and_prepare_features(str(spy_path))
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target series length: {len(y)}")
        print(f"\nFeature columns:")
        for col in X.columns:
            print(f"  {col}")
            
        print(f"\nFirst few target values:")
        print(y.head())
        
        # Show data split
        engineer = FeatureEngineer()
        X_train, X_val, X_test, y_train, y_val, y_test = engineer.split_time_series(X, y)
        
        print(f"\nData splits:")
        print(f"  Train: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})")
        print(f"  Val:   {len(X_val)} samples ({X_val.index[0] if len(X_val) > 0 else 'N/A'} to {X_val.index[-1] if len(X_val) > 0 else 'N/A'})")
        print(f"  Test:  {len(X_test)} samples ({X_test.index[0] if len(X_test) > 0 else 'N/A'} to {X_test.index[-1] if len(X_test) > 0 else 'N/A'})")
    else:
        print(f"SPY data file not found at {spy_path}")