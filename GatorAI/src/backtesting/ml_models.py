"""ML model training and prediction module for backtesting.

This module provides ML models for predicting returns with confidence estimates.
Integrates with the feature engineering pipeline and backtesting framework.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass 
class MLPrediction:
    """Container for ML prediction with confidence."""
    prediction: float
    confidence: float  # 0-1 scale, higher = more confident
    timestamp: pd.Timestamp
    model_name: str


@dataclass
class MLModelMetrics:
    """Container for ML model performance metrics."""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error  
    hit_rate: float  # Directional accuracy (0-1)
    confidence_calibration: float  # How well confidence matches actual accuracy
    

class MLReturnPredictor:
    """ML model for predicting next-day returns with confidence estimates."""
    
    def __init__(
        self, 
        model_type: str = "ridge",
        confidence_method: str = "ensemble",
        min_confidence: float = 0.6,
        **model_kwargs
    ):
        """
        Args:
            model_type: "ridge", "random_forest", or "ensemble"
            confidence_method: "ensemble" or "residual_based"  
            min_confidence: Minimum confidence threshold for trading (0.0-1.0)
            **model_kwargs: Parameters passed to underlying model
        """
        self.model_type = model_type
        self.confidence_method = confidence_method
        self.min_confidence = min_confidence
        self.model_kwargs = model_kwargs
        
        # Initialize models
        self.models = {}
        self.is_fitted = False
        self.feature_names = None
        self.training_metrics = None
        
    def _create_model(self, model_type: str) -> Any:
        """Create sklearn model instance."""
        if model_type == "ridge":
            return Ridge(alpha=self.model_kwargs.get("alpha", 1.0), random_state=42)
        elif model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.model_kwargs.get("n_estimators", 100),
                max_depth=self.model_kwargs.get("max_depth", 10),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "MLReturnPredictor":
        """Train the ML model(s) on historical data.
        
        Args:
            X_train: Training features
            y_train: Training targets (returns)
            X_val: Optional validation features for metrics
            y_val: Optional validation targets
            
        Returns:
            Self for method chaining
        """
        self.feature_names = list(X_train.columns)
        
        if self.model_type == "ensemble":
            # Train multiple models for ensemble confidence
            model_types = ["ridge", "random_forest"]
            for mtype in model_types:
                model = self._create_model(mtype)
                model.fit(X_train.values, y_train.values)
                self.models[mtype] = model
        else:
            # Single model
            model = self._create_model(self.model_type)
            model.fit(X_train.values, y_train.values)
            self.models[self.model_type] = model
        
        self.is_fitted = True
        
        # Calculate training metrics if validation data provided
        if X_val is not None and y_val is not None:
            self.training_metrics = self._calculate_metrics(X_val, y_val)
            
        return self
    
    def predict_with_confidence(
        self, 
        X: pd.DataFrame,
        return_raw_predictions: bool = False
    ) -> pd.DataFrame:
        """Make predictions with confidence estimates.
        
        Args:
            X: Feature matrix for prediction
            return_raw_predictions: If True, include raw model outputs
            
        Returns:
            DataFrame with columns: prediction, confidence, timestamp
            If return_raw_predictions=True, also includes individual model predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        if self.model_type == "ensemble":
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(X.values)
                predictions[f"pred_{name}"] = pred
                
            # Ensemble prediction (simple average)
            pred_values = np.array(list(predictions.values()))
            ensemble_pred = np.mean(pred_values, axis=0)
            
            # Confidence based on agreement between models
            pred_std = np.std(pred_values, axis=0)
            # Higher std = lower confidence; normalize to 0-1 scale
            max_std = np.percentile(pred_std, 95)  # Use 95th percentile as max
            confidence = 1.0 - np.clip(pred_std / max_std, 0, 1)
            
        else:
            # Single model prediction
            model = list(self.models.values())[0]
            ensemble_pred = model.predict(X.values)
            
            # Simple confidence based on prediction magnitude (heuristic)
            # Closer to zero = less confident
            confidence = np.tanh(np.abs(ensemble_pred) * 10)  # Scale and bound to 0-1
            predictions = {f"pred_{self.model_type}": ensemble_pred}
        
        # Create result DataFrame
        result = pd.DataFrame({
            "prediction": ensemble_pred,
            "confidence": confidence,
            "timestamp": X.index
        }, index=X.index)
        
        # Add raw predictions if requested
        if return_raw_predictions:
            for name, values in predictions.items():
                result[name] = values
                
        return result
    
    def _calculate_metrics(self, X_test: pd.DataFrame, y_test: pd.Series) -> MLModelMetrics:
        """Calculate performance metrics on test data."""
        predictions_df = self.predict_with_confidence(X_test)
        y_pred = predictions_df["prediction"].values
        y_true = y_test.values
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        # Hit rate (directional accuracy)
        direction_correct = np.sign(y_pred) == np.sign(y_true)
        hit_rate = np.mean(direction_correct)
        
        # Confidence calibration (simplified)
        confidence = predictions_df["confidence"].values
        high_conf_mask = confidence > 0.7
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(direction_correct[high_conf_mask])
            confidence_calibration = high_conf_accuracy
        else:
            confidence_calibration = 0.5  # Random baseline
            
        return MLModelMetrics(
            mae=mae,
            mse=mse,
            hit_rate=hit_rate,
            confidence_calibration=confidence_calibration
        )
    
    def should_trade(self, prediction_row: pd.Series) -> bool:
        """Determine if we should trade based on confidence threshold.
        
        Args:
            prediction_row: Row from predict_with_confidence() output
            
        Returns:
            True if confidence >= min_confidence threshold
        """
        return prediction_row["confidence"] >= self.min_confidence
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
            
        model_data = {
            "models": self.models,
            "model_type": self.model_type,
            "confidence_method": self.confidence_method,
            "min_confidence": self.min_confidence,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "is_fitted": self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        
    @classmethod 
    def load_model(cls, filepath: str) -> "MLReturnPredictor":
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        predictor = cls(
            model_type=model_data["model_type"],
            confidence_method=model_data["confidence_method"],
            min_confidence=model_data["min_confidence"]
        )
        
        predictor.models = model_data["models"]
        predictor.feature_names = model_data["feature_names"]
        predictor.training_metrics = model_data["training_metrics"]  
        predictor.is_fitted = model_data["is_fitted"]
        
        return predictor


def create_sample_data_for_testing() -> Tuple[pd.DataFrame, pd.Series]:
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
    """Demo the ML training pipeline."""
    from ml_features import FeatureEngineer
    
    print("=== ML Return Predictor Demo ===")
    
    # Create or load data
    print("Creating sample data for demo...")
    price_data = create_sample_data_for_testing()
    print(f"Generated price data: {price_data.shape}")
    
    # Engineer features
    engineer = FeatureEngineer(lookback_window=21)
    X, y = engineer.prepare_ml_dataset(price_data)
    print(f"Feature matrix: {X.shape}, Target series: {len(y)}")
    
    # Split data  
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.split_time_series(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train models
    print("\nTraining Ridge model...")
    ridge_model = MLReturnPredictor(model_type="ridge", min_confidence=0.6)
    ridge_model.fit(X_train, y_train, X_val, y_val)
    
    print("\nTraining Ensemble model...")
    ensemble_model = MLReturnPredictor(model_type="ensemble", min_confidence=0.6)
    ensemble_model.fit(X_train, y_train, X_val, y_val)
    
    # Make predictions
    print("\n=== Predictions ===")
    ridge_preds = ridge_model.predict_with_confidence(X_test)
    ensemble_preds = ensemble_model.predict_with_confidence(X_test)
    
    print(f"Ridge predictions shape: {ridge_preds.shape}")
    print(f"Ensemble predictions shape: {ensemble_preds.shape}")
    
    # Show sample predictions
    print("\nSample Ridge predictions:")
    print(ridge_preds.head())
    
    print("\nSample Ensemble predictions:")
    print(ensemble_preds.head())
    
    # Calculate test metrics
    ridge_metrics = ridge_model._calculate_metrics(X_test, y_test)
    ensemble_metrics = ensemble_model._calculate_metrics(X_test, y_test)
    
    print("\n=== Model Performance ===")
    print(f"Ridge - MAE: {ridge_metrics.mae:.4f}, Hit Rate: {ridge_metrics.hit_rate:.3f}")
    print(f"Ensemble - MAE: {ensemble_metrics.mae:.4f}, Hit Rate: {ensemble_metrics.hit_rate:.3f}")
    
    # Test confidence thresholding
    ridge_trades = ridge_preds.apply(lambda row: ridge_model.should_trade(row), axis=1)
    ensemble_trades = ensemble_preds.apply(lambda row: ensemble_model.should_trade(row), axis=1)
    
    print(f"\nTrades above confidence threshold:")
    print(f"Ridge: {ridge_trades.sum()}/{len(ridge_trades)} = {ridge_trades.mean():.1%}")
    print(f"Ensemble: {ensemble_trades.sum()}/{len(ensemble_trades)} = {ensemble_trades.mean():.1%}")