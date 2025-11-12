"""
Main ML predictor class that coordinates data preparation, model training, and prediction.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

from .data_preparation import prepare_ml_data
from .models import create_model, MLModel
from .walk_forward import WalkForwardValidator
from ..data.storage.sqlite_adapter import SQLiteAdapter

logger = logging.getLogger(__name__)


class MLPredictor:
    """Main class for ML-based next-day return prediction."""
    
    def __init__(
        self,
        storage: SQLiteAdapter,
        output_dir: Optional[Path] = None,
        model_types: Optional[List[str]] = None,
    ):
        """
        Initialize ML predictor.
        
        Args:
            storage: SQLiteAdapter instance for data access
            output_dir: Directory for saving outputs
            model_types: List of model types to use (default: ['linear', 'random_forest', 'xgboost'])
        """
        self.storage = storage
        self.output_dir = Path(output_dir) if output_dir else Path("GatorAI/ml/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_types is None:
            model_types = ["linear", "random_forest", "xgboost"]
        self.model_types = model_types
        
        self.validator = WalkForwardValidator(
            train_window_days=252,
            test_window_days=21,
            step_days=21,
        )
    
    def prepare_features(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prepare features for ML model.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
        
        Returns:
            DataFrame with features and target
        """
        return prepare_ml_data(
            ticker=ticker,
            storage=self.storage,
            start_date=start_date,
            end_date=end_date,
        )
    
    def train_and_validate(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Train and validate models using walk-forward validation.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            model_params: Model-specific parameters
        
        Returns:
            Dictionary with validation results for each model
        """
        logger.info(f"Training and validating models for {ticker}")
        
        # Prepare data
        df = self.prepare_features(ticker, start_date, end_date)
        
        # Identify feature columns (exclude target and metadata)
        exclude_cols = ["datetime", "next_day_return", "ticker"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if model_params is None:
            model_params = {}
        
        results = {}
        
        for model_type in self.model_types:
            logger.info(f"Training {model_type} model for {ticker}")
            
            # Create model
            params = model_params.get(model_type, {})
            model = create_model(model_type, params)
            
            # Perform walk-forward validation
            validation_results = self.validator.validate(
                model=model,
                df=df,
                feature_cols=feature_cols,
                target_col="next_day_return",
                datetime_col="datetime",
                return_predictions=True,
            )
            
            # Train final model on all available data
            X_all = df[feature_cols].values
            y_all = df["next_day_return"].values
            
            # Remove NaN rows
            mask = ~(np.isnan(X_all).any(axis=1) | np.isnan(y_all))
            X_all = X_all[mask]
            y_all = y_all[mask]
            
            final_model = create_model(model_type, params)
            final_model.fit(X_all, y_all)
            
            # Get feature importance
            feature_importance = final_model.get_feature_importance()
            if feature_importance:
                # Map to feature names (feature_importance is dict of {index: importance})
                importance_dict = {
                    feature_cols[idx]: float(importance)
                    for idx, importance in feature_importance.items()
                    if idx < len(feature_cols)
                }
                # Sort by importance
                importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                importance_dict = {}
            
            results[model_type] = {
                "validation": validation_results,
                "feature_importance": importance_dict,
                "model": final_model,
                "feature_cols": feature_cols,
            }
            
            logger.info(
                f"{model_type} - RMSE: {validation_results['avg_metrics']['mean_rmse']:.4f}, "
                f"R2: {validation_results['avg_metrics']['mean_r2']:.4f}, "
                f"Directional Accuracy: {validation_results['avg_metrics']['mean_directional_accuracy']:.2%}"
            )
        
        return results
    
    def predict_next_day(
        self,
        ticker: str,
        model_type: str = "xgboost",
        model: Optional[MLModel] = None,
        feature_cols: Optional[List[str]] = None,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predict next-day return for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            model_type: Type of model to use
            model: Pre-trained model (if None, will train)
            feature_cols: List of feature columns (if None, will prepare data)
            date: Date for prediction (if None, uses latest available data)
        
        Returns:
            Dictionary with prediction and confidence intervals
        """
        # Prepare data
        df = self.prepare_features(ticker, end_date=date)
        
        if df.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Get latest row
        latest_row = df.iloc[-1:].copy()
        
        # Identify feature columns
        if feature_cols is None:
            exclude_cols = ["datetime", "next_day_return", "ticker"]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = latest_row[feature_cols].values
        
        # Check for NaN
        if np.isnan(X).any():
            logger.warning("NaN values in features, filling with median")
            X = pd.DataFrame(X, columns=feature_cols).fillna(df[feature_cols].median()).values
        
        # Make prediction
        if model is None:
            # Train model on all available data
            X_all = df[feature_cols].values
            y_all = df["next_day_return"].values
            mask = ~(np.isnan(X_all).any(axis=1) | np.isnan(y_all))
            X_all = X_all[mask]
            y_all = y_all[mask]
            
            model = create_model(model_type)
            model.fit(X_all, y_all)
        
        prediction, std = model.predict(X, return_std=True)
        
        # Calculate confidence intervals (95%)
        lower_bound = prediction[0] - 1.96 * std[0]
        upper_bound = prediction[0] + 1.96 * std[0]
        
        return {
            "ticker": ticker,
            "date": latest_row["datetime"].iloc[0],
            "prediction": float(prediction[0]),
            "std": float(std[0]),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "confidence_level": 0.95,
        }
    
    def save_results(
        self,
        ticker: str,
        results: Dict[str, Any],
        suffix: str = "",
    ):
        """
        Save prediction results to files.
        
        Args:
            ticker: Stock ticker symbol
            results: Results dictionary from train_and_validate
            suffix: Optional suffix for filenames
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{ticker}_{timestamp}{suffix}"
        
        # Save predictions for each model
        for model_type, model_results in results.items():
            validation = model_results["validation"]
            
            # Save predictions CSV
            if "predictions" in validation:
                predictions_df = pd.DataFrame({
                    "date": validation["dates"],
                    "prediction": validation["predictions"],
                    "actual": validation["actuals"],
                })
                predictions_file = self.output_dir / f"{base_name}_{model_type}_predictions.csv"
                predictions_df.to_csv(predictions_file, index=False)
                logger.info(f"Saved predictions to {predictions_file}")
            
            # Save metrics
            metrics_file = self.output_dir / f"{base_name}_{model_type}_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(validation["avg_metrics"], f, indent=2, default=str)
            logger.info(f"Saved metrics to {metrics_file}")
            
            # Save feature importance
            importance_file = self.output_dir / f"{base_name}_{model_type}_feature_importance.json"
            with open(importance_file, "w") as f:
                json.dump(model_results["feature_importance"], f, indent=2)
            logger.info(f"Saved feature importance to {importance_file}")
        
        # Save summary
        summary = {
            "ticker": ticker,
            "timestamp": timestamp,
            "models": list(results.keys()),
            "metrics": {
                model_type: results[model_type]["validation"]["avg_metrics"]
                for model_type in results.keys()
            },
        }
        summary_file = self.output_dir / f"{base_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary to {summary_file}")
    
    def generate_predictions_file(
        self,
        tickers: List[str],
        model_type: str = "xgboost",
        output_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions file for backtester consumption.
        
        Args:
            tickers: List of ticker symbols
            model_type: Type of model to use
            output_file: Output file path (if None, uses default)
        
        Returns:
            DataFrame with predictions
        """
        all_predictions = []
        
        for ticker in tickers:
            try:
                # Train model
                results = self.train_and_validate(ticker, model_params={model_type: {}})
                model = results[model_type]["model"]
                feature_cols = results[model_type]["feature_cols"]
                
                # Generate prediction
                prediction = self.predict_next_day(
                    ticker=ticker,
                    model_type=model_type,
                    model=model,
                    feature_cols=feature_cols,
                )
                
                all_predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Failed to generate prediction for {ticker}: {e}")
                continue
        
        predictions_df = pd.DataFrame(all_predictions)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"predictions_{timestamp}.csv"
        
        predictions_df.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")
        
        return predictions_df

