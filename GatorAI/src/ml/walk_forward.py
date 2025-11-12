"""
Walk-forward validation for time series ML models.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .models import MLModel

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation for time series models."""
    
    def __init__(
        self,
        train_window_days: int = 252,  # 1 year
        test_window_days: int = 21,    # 1 month
        step_days: int = 21,           # Step forward 1 month at a time
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_window_days: Number of days for training window
            test_window_days: Number of days for testing window
            step_days: Number of days to step forward each iteration
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
    
    def split_data(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into train/test windows using walk-forward validation.
        
        Args:
            df: DataFrame with datetime index
            datetime_col: Name of datetime column
        
        Returns:
            List of (train_df, test_df) tuples
        """
        df = df.sort_values(datetime_col).reset_index(drop=True)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        splits = []
        min_date = df[datetime_col].min()
        max_date = df[datetime_col].max()
        
        # Start from train_window_days after min_date
        current_date = min_date + pd.Timedelta(days=self.train_window_days)
        
        while current_date + pd.Timedelta(days=self.test_window_days) <= max_date:
            train_end = current_date
            test_start = current_date
            test_end = test_start + pd.Timedelta(days=self.test_window_days)
            
            train_df = df[df[datetime_col] < train_end].copy()
            test_df = df[(df[datetime_col] >= test_start) & (df[datetime_col] < test_end)].copy()
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
            
            # Step forward
            current_date += pd.Timedelta(days=self.step_days)
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def validate(
        self,
        model: MLModel,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "next_day_return",
        datetime_col: str = "datetime",
        return_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Args:
            model: MLModel instance
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            datetime_col: Name of datetime column
            return_predictions: If True, return predictions for each split
        
        Returns:
            Dictionary with validation results
        """
        splits = self.split_data(df, datetime_col)
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        all_metrics = []
        
        for i, (train_df, test_df) in enumerate(splits):
            # Prepare data
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values
            test_dates = test_df[datetime_col].values
            
            # Check for NaN values
            train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            test_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
            
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
            test_dates = test_dates[test_mask]
            
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(f"Split {i}: Skipping due to empty train/test set")
                continue
            
            # Train model (create a fresh instance for each split)
            from .models import create_model
            model_copy = create_model(model.model_type, model.model_params)
            model_copy.fit(X_train, y_train)
            
            # Make predictions with confidence intervals
            predictions, std = model_copy.predict(X_test, return_std=True)
            
            # Calculate metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(mse)
            
            # Calculate R-squared
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate directional accuracy
            direction_actual = (y_test > 0).astype(int)
            direction_pred = (predictions > 0).astype(int)
            directional_accuracy = np.mean(direction_actual == direction_pred)
            
            metrics = {
                "split": i,
                "train_start": train_df[datetime_col].min(),
                "train_end": train_df[datetime_col].max(),
                "test_start": test_df[datetime_col].min(),
                "test_end": test_df[datetime_col].max(),
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "directional_accuracy": directional_accuracy,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
            
            all_metrics.append(metrics)
            
            if return_predictions:
                for j in range(len(predictions)):
                    all_predictions.append(predictions[j])
                    all_actuals.append(y_test[j])
                    all_dates.append(test_dates[j])
        
        # Aggregate metrics
        avg_metrics = {
            "mean_mse": np.mean([m["mse"] for m in all_metrics]),
            "mean_mae": np.mean([m["mae"] for m in all_metrics]),
            "mean_rmse": np.mean([m["rmse"] for m in all_metrics]),
            "mean_r2": np.mean([m["r2"] for m in all_metrics]),
            "mean_directional_accuracy": np.mean([m["directional_accuracy"] for m in all_metrics]),
            "n_splits": len(all_metrics),
        }
        
        results = {
            "metrics": all_metrics,
            "avg_metrics": avg_metrics,
        }
        
        if return_predictions:
            results["predictions"] = np.array(all_predictions)
            results["actuals"] = np.array(all_actuals)
            results["dates"] = np.array(all_dates)
        
        return results

