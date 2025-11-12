"""
Baseline ML models for next-day return prediction.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class MLModel:
    """Wrapper for ML models with prediction and confidence intervals."""
    
    def __init__(self, model_type: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize ML model.
        
        Args:
            model_type: Type of model ('linear', 'random_forest', 'xgboost')
            model_params: Model-specific parameters
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self._create_model()
    
    def _create_model(self):
        """Create the model instance."""
        if self.model_type == "linear":
            self.model = LinearRegression(**self.model_params)
        elif self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            self.model = RandomForestRegressor(**default_params)
        elif self.model_type == "xgboost":
            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            self.model = xgb.XGBRegressor(**default_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
        # Store residual std for linear models
        if self.model_type == "linear":
            train_pred = self.model.predict(X)
            residuals = train_pred - y
            self._residual_std = np.std(residuals)
        return self
    
    def predict(self, X, return_std: bool = False):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            return_std: If True, return standard deviation for confidence intervals
        
        Returns:
            Predictions (and optionally std)
        """
        predictions = self.model.predict(X)
        
        if return_std:
            # Calculate prediction intervals
            if self.model_type == "linear":
                # For linear regression, use stored residual standard error
                if hasattr(self, '_residual_std'):
                    std = np.full(len(predictions), self._residual_std)
                else:
                    # Fallback: use prediction std
                    std = np.std(predictions) * np.ones_like(predictions)
            elif self.model_type == "random_forest":
                # Use tree predictions for uncertainty
                tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                std = np.std(tree_predictions, axis=0)
            elif self.model_type == "xgboost":
                # XGBoost doesn't provide uncertainty directly
                # Use a conservative estimate based on typical daily return volatility
                # For daily returns, typical std is around 0.01-0.02 (1-2%)
                # Use a scaled version based on prediction magnitude
                base_std = 0.015  # 1.5% baseline volatility
                pred_magnitude = np.abs(predictions)
                # Scale std based on prediction magnitude (larger predictions = higher uncertainty)
                std = base_std * (1 + pred_magnitude * 0.5)
            else:
                std = np.std(predictions) * np.ones_like(predictions)
            
            return predictions, std
        
        return predictions
    
    def get_feature_importance(self) -> Optional[Dict[int, float]]:
        """Get feature importance scores.
        
        Returns:
            Dictionary mapping feature index to importance score
        """
        if self.model_type == "linear":
            # For linear regression, use absolute coefficients
            if hasattr(self.model, "coef_"):
                return {i: float(abs(coef)) for i, coef in enumerate(self.model.coef_)}
        elif self.model_type == "random_forest":
            if hasattr(self.model, "feature_importances_"):
                return {i: float(imp) for i, imp in enumerate(self.model.feature_importances_)}
        elif self.model_type == "xgboost":
            if hasattr(self.model, "feature_importances_"):
                return {i: float(imp) for i, imp in enumerate(self.model.feature_importances_)}
        return None


def create_model(model_type: str, model_params: Optional[Dict[str, Any]] = None) -> MLModel:
    """
    Create an ML model instance.
    
    Args:
        model_type: Type of model ('linear', 'random_forest', 'xgboost')
        model_params: Model-specific parameters
    
    Returns:
        MLModel instance
    """
    return MLModel(model_type, model_params)

