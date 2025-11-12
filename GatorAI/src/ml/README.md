# ML Prediction Module

This module implements machine learning models for next-day return prediction for SPY, QQQ, and IWM.

## Features

- **Baseline Models**: Linear Regression, Random Forest, XGBoost
- **Walk-Forward Validation**: Robust time-series validation to prevent overfitting
- **Confidence Intervals**: Risk-aware predictions with uncertainty estimates
- **Feature Importance**: Transparency into what drives predictions
- **Regime Indicators**: VIX buckets and day-of-week effects
- **Technical Indicators**: RSI, MACD, EMAs, ATR, Bollinger Bands, rolling volatility

## Usage

### Training and Validation

```bash
# Train models for SPY, QQQ, IWM
python -m GatorAI.src.ml.cli --tickers SPY QQQ IWM --train --fetch-data

# Train with specific models
python -m GatorAI.src.ml.cli --tickers SPY --train --model-types xgboost random_forest
```

### Generate Predictions

```bash
# Generate predictions for backtester
python -m GatorAI.src.ml.cli --tickers SPY QQQ IWM --predict --fetch-data
```

### Data Preparation

The module automatically:
1. Fetches price data for tickers (if not available)
2. Fetches VIX data for regime indicators
3. Generates technical indicators (RSI, MACD, ATR, etc.)
4. Adds regime features (VIX buckets, day-of-week)
5. Creates next-day return target

## Output Structure

All outputs are saved to `GatorAI/ml/outputs/`:

- `{ticker}_{timestamp}_{model}_predictions.csv`: Predictions vs actuals
- `{ticker}_{timestamp}_{model}_metrics.json`: Validation metrics
- `{ticker}_{timestamp}_{model}_feature_importance.json`: Feature importance scores
- `{ticker}_{timestamp}_summary.json`: Summary of all models
- `predictions_{timestamp}.csv`: Predictions file for backtester consumption

## Model Performance Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: R-squared (coefficient of determination)
- **Directional Accuracy**: Percentage of correct direction predictions

## Feature Importance

Feature importance is calculated for each model:
- **Linear Regression**: Absolute coefficients
- **Random Forest**: Mean decrease in impurity
- **XGBoost**: Gain-based importance

## Walk-Forward Validation

The walk-forward validator:
- Uses 252 days (1 year) for training
- Tests on 21 days (1 month) windows
- Steps forward 21 days at a time
- Prevents look-ahead bias and overfitting

## Dependencies

- scikit-learn
- xgboost
- pandas
- numpy

## Integration with Backtester

The predictions file format:
```csv
ticker,date,prediction,std,lower_bound,upper_bound,confidence_level
SPY,2025-01-15,0.0012,0.005,0.001,0.002,0.95
```

This format is designed to be consumed by the backtester and optimizer modules.


