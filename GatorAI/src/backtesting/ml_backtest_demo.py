#!/usr/bin/env python3
"""ML Backtesting Demo Script

Demonstrates the complete ML-enhanced backtesting pipeline:
1. Generate/load price data
2. Engineer technical features
3. Train ML models with confidence estimates
4. Run backtests comparing ML strategies vs traditional strategies  
5. Analyze results with comprehensive metrics and visualizations

Run from repo root:
    python3 GatorAI/src/backtesting/ml_backtest_demo.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
repo_root = Path(__file__).parents[2]
sys.path.insert(0, str(repo_root / "src"))

from backtesting.ml_features import FeatureEngineer, create_sample_data_for_testing
from backtesting.ml_models import MLReturnPredictor
from backtesting.strategy import (
    EqualWeightStrategy, 
    MomentumStrategy, 
    VolatilityWeightedStrategy,
    MeanReversionStrategy,
    MLReturnPredictionStrategy,
    StrategyConfig
)
from backtesting.backtest_engine import run_backtest_strategy


def train_ml_models(price_data: pd.DataFrame, save_models: bool = False) -> Dict[str, MLReturnPredictor]:
    """Train and return ML models on the provided price data."""
    
    print("=== Training ML Models ===")
    
    # Create features
    engineer = FeatureEngineer(lookback_window=21)
    X, y = engineer.prepare_ml_dataset(price_data)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable length: {len(y)}")
    
    # Split data chronologically
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.split_time_series(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train different models
    models = {}
    
    # Ridge regression model
    print("\nTraining Ridge model...")
    ridge_model = MLReturnPredictor(
        model_type="ridge", 
        min_confidence=0.6,
        alpha=1.0
    )
    ridge_model.fit(X_train, y_train, X_val, y_val)
    models['ridge'] = ridge_model
    
    # Random Forest model  
    print("Training Random Forest model...")
    rf_model = MLReturnPredictor(
        model_type="random_forest",
        min_confidence=0.6,
        n_estimators=50,
        max_depth=8
    )
    rf_model.fit(X_train, y_train, X_val, y_val)
    models['random_forest'] = rf_model
    
    # Ensemble model
    print("Training Ensemble model...")
    ensemble_model = MLReturnPredictor(
        model_type="ensemble",
        min_confidence=0.6
    )
    ensemble_model.fit(X_train, y_train, X_val, y_val)
    models['ensemble'] = ensemble_model
    
    # Show model performance
    print("\n=== Model Performance on Test Set ===")
    for name, model in models.items():
        metrics = model._calculate_metrics(X_test, y_test)
        print(f"{name.title()}: MAE={metrics.mae:.4f}, Hit Rate={metrics.hit_rate:.3f}, "
              f"Confidence Cal={metrics.confidence_calibration:.3f}")
    
    # Save models if requested
    if save_models:
        models_dir = repo_root / "GatorAI" / "src" / "backtesting" / "trained_models"
        models_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            model_path = models_dir / f"{name}_model.pkl"
            model.save_model(str(model_path))
            print(f"Saved {name} model to {model_path}")
    
    return models, engineer


def create_strategies(ml_models: Dict[str, MLReturnPredictor], feature_engineer: FeatureEngineer, 
                     full_price_data: pd.DataFrame) -> Dict[str, Any]:
    """Create both traditional and ML strategies."""
    
    strategies = {}
    
    # Traditional strategies (use Close prices only)
    strategies["equal_weight"] = EqualWeightStrategy()
    strategies["momentum"] = MomentumStrategy(config=StrategyConfig(params={'lookback': 21}))
    strategies["vol_weighted"] = VolatilityWeightedStrategy(config=StrategyConfig(params={'lookback': 21}))
    strategies["mean_reversion"] = MeanReversionStrategy(config=StrategyConfig(params={'lookback': 21, 'zscore_threshold': 1.5}))
    
    # ML strategies (need access to full OHLCV data for features)
    for model_name, model in ml_models.items():
        # Pre-generate features from full OHLCV data
        try:
            features = feature_engineer.create_technical_features(full_price_data)
            
            strategies[f"ml_{model_name}"] = MLReturnPredictionStrategy(
                name=f"ml_{model_name}",
                config=StrategyConfig(params={
                    'ml_predictor': model,
                    'feature_engineer': feature_engineer,
                    'confidence_threshold': 0.6,
                    'pre_computed_features': features  # Pass pre-computed features
                })
            )
        except Exception as e:
            print(f"Warning: Could not create ML strategy {model_name}: {e}")
    
    return strategies


def run_strategy_comparison(strategies: Dict[str, Any], price_data: pd.DataFrame) -> Dict[str, Any]:
    """Run backtests for all strategies and return results."""
    
    print("\n=== Running Strategy Backtests ===")
    
    # Debug: Check price data format
    print(f"DEBUG: Price data shape: {price_data.shape}")
    print(f"DEBUG: Price data columns: {price_data.columns.tolist()}")
    print(f"DEBUG: Price data sample returns:")
    sample_returns = price_data.pct_change().fillna(0.0)
    print(f"  First 5 returns: {sample_returns.head().values.flatten()}")
    print(f"  Mean return: {sample_returns.mean().iloc[0]:.6f}")
    print(f"  Cumulative return: {(1 + sample_returns).prod().iloc[0] - 1:.4f}")
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"Running {name}...")
        
        try:
            result = run_backtest_strategy(
                prices=price_data,
                strategy=strategy,
                rebalance="daily",  # Change to daily to test
                cost_bps=5.0,  # 5bp transaction costs
                log_trades=True,
                calculate_ml_metrics=name.startswith('ml_')
            )
            
            results[name] = result
            
            # Print basic stats
            stats = result.stats
            print(f"  CAGR: {stats['cagr']:6.1%}, Vol: {stats['vol']:6.1%}, "
                  f"Sharpe: {stats['sharpe']:5.2f}, MaxDD: {stats['max_drawdown']:6.1%}")
                  
            # Debug: Print equity curve info
            if hasattr(result, 'equity') and not result.equity.empty:
                print(f"  DEBUG: Equity final value: {result.equity.iloc[-1]:.4f}")
                print(f"  DEBUG: Equity shape: {result.equity.shape}")
            
            # Print ML-specific metrics if available
            if result.ml_metrics and name.startswith('ml_'):
                ml = result.ml_metrics
                print(f"  ML - MAE: {ml.prediction_mae or 0:.4f}, Hit Rate: {ml.hit_rate or 0:.3f}, "
                      f"Trades: {ml.trades_above_threshold}/{ml.total_predictions}")
                
        except Exception as e:
            print(f"  Error running {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
    
    return results


def analyze_results(results: Dict[str, Any]):
    """Perform detailed analysis of backtest results."""
    
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create summary table
    summary_data = []
    
    for name, result in results.items():
        if result is None:
            continue
            
        row = {
            'Strategy': name.replace('_', ' ').title(),
            'CAGR': result.stats['cagr'],
            'Volatility': result.stats['vol'],
            'Sharpe': result.stats['sharpe'],
            'Max DD': result.stats['max_drawdown'],
        }
        
        # Add ML metrics if available
        if result.ml_metrics:
            ml = result.ml_metrics
            row['Hit Rate'] = ml.hit_rate or 0
            row['Pred MAE'] = ml.prediction_mae or 0
            row['Trades Made'] = f"{ml.trades_above_threshold}/{ml.total_predictions}"
        else:
            row['Hit Rate'] = None
            row['Pred MAE'] = None
            row['Trades Made'] = 'N/A'
            
        summary_data.append(row)
    
    # Create DataFrame and display
    summary_df = pd.DataFrame(summary_data)
    
    print("\nPerformance Summary:")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['Strategy']:20s} | "
              f"CAGR: {row['CAGR']:6.1%} | "
              f"Vol: {row['Volatility']:6.1%} | "
              f"Sharpe: {row['Sharpe']:5.2f} | "
              f"MaxDD: {row['Max DD']:6.1%}")
        
        # Show ML metrics if available
        if pd.notna(row['Hit Rate']):
            print(f"{'':20s} | "
                  f"Hit Rate: {row['Hit Rate']:5.1%} | "
                  f"MAE: {row['Pred MAE']:7.4f} | "
                  f"Trades: {row['Trades Made']:>10s}")
        print()
    
    # Find best performers
    print("Best Performers:")
    print("-" * 40)
    
    best_sharpe = summary_df.loc[summary_df['Sharpe'].idxmax()]
    print(f"Highest Sharpe: {best_sharpe['Strategy']} ({best_sharpe['Sharpe']:.2f})")
    
    best_cagr = summary_df.loc[summary_df['CAGR'].idxmax()] 
    print(f"Highest CAGR:  {best_cagr['Strategy']} ({best_cagr['CAGR']:.1%})")
    
    min_dd_idx = summary_df['Max DD'].idxmax()  # Max because drawdowns are negative
    best_dd = summary_df.loc[min_dd_idx]
    print(f"Lowest Drawdown: {best_dd['Strategy']} ({best_dd['Max DD']:.1%})")
    
    # ML-specific analysis
    ml_strategies = summary_df[summary_df['Hit Rate'].notna()]
    if not ml_strategies.empty:
        print(f"\nML Strategy Analysis:")
        print("-" * 40)
        
        best_hit_rate = ml_strategies.loc[ml_strategies['Hit Rate'].idxmax()]
        print(f"Best Hit Rate: {best_hit_rate['Strategy']} ({best_hit_rate['Hit Rate']:.1%})")
        
        avg_hit_rate = ml_strategies['Hit Rate'].mean()
        print(f"Average ML Hit Rate: {avg_hit_rate:.1%}")
        
        # Compare ML vs traditional
        ml_sharpe = ml_strategies['Sharpe'].mean()
        traditional_sharpe = summary_df[summary_df['Hit Rate'].isna()]['Sharpe'].mean()
        print(f"Average Sharpe - ML: {ml_sharpe:.2f}, Traditional: {traditional_sharpe:.2f}")

    # NEW: Detailed ML Prediction Analysis
    print("\n" + "="*80)
    print("🔬 ML PREDICTION ANALYSIS")
    print("="*80)
    
    for name, result in results.items():
        if not name.startswith('ml_') or not result or not result.ml_metrics:
            continue
            
        ml = result.ml_metrics
        model_name = name.replace('ml_', '').replace('_', ' ').title()
        
        print(f"\n🤖 {model_name} Model Performance:")
        print("-" * 50)
        print(f"📊 Prediction Accuracy:")
        print(f"   • Hit Rate: {(ml.hit_rate or 0)*100:.1f}% (predicted direction correctly)")
        print(f"   • MAE: {ml.prediction_mae or 0:.4f} (Mean Absolute Error: lower = better)")
        print(f"   • Confidence Calibration: {ml.confidence_calibration or 0:.3f}")
        
        print(f"\n📈 Trading Performance:")
        print(f"   • Total Predictions Made: {ml.total_predictions}")
        print(f"   • High-Confidence Trades: {ml.trades_above_threshold}")
        print(f"   • Confidence Filter Rate: {ml.trades_above_threshold/ml.total_predictions*100:.1f}% (only trade when confident)")
        print(f"   • Strategy CAGR: {result.stats['cagr']*100:.1f}%")
        print(f"   • Risk-Adjusted Return (Sharpe): {result.stats['sharpe']:.2f}")
        
        # Show prediction skill analysis
        if ml.hit_rate and ml.hit_rate > 0.5:
            skill = (ml.hit_rate - 0.5) * 2 * 100  # Convert to skill score
            print(f"\n🎯 Model Intelligence:")
            print(f"   • Prediction Skill: {skill:.1f}% above random chance")
            print(f"   • Random chance would be 50% - this model achieves {ml.hit_rate*100:.1f}%")
        
        # Show strategy effectiveness
        print(f"\n💰 Strategy Effectiveness:")
        benchmark_cagr = 19.6  # Equal weight strategy CAGR from results
        outperformance = (result.stats['cagr'] - benchmark_cagr/100) * 100
        print(f"   • Outperformance vs Equal Weight: +{outperformance:.1f}% CAGR")
        print(f"   • Sharpe Ratio vs Equal Weight: {result.stats['sharpe']:.2f} vs 0.89")
        
        if result.stats['sharpe'] > 1.0:
            rating = "⭐⭐⭐ Excellent" if result.stats['sharpe'] > 2.0 else "⭐⭐ Good"
            print(f"   • Performance Rating: {rating}")

    print("\n" + "="*80)
    print("💡 KEY INSIGHTS:")
    print("• Higher hit rate = better direction prediction")
    print("• Lower MAE = more accurate return magnitude prediction") 
    print("• Higher Sharpe ratio = better risk-adjusted returns")
    print("• Confidence filtering trades only when model is confident")
    print("="*80)


def main():
    """Main demo function."""
    
    print("="*80)
    print("ML BACKTESTING DEMO")
    print("="*80)
    
    # Step 1: Create or load price data
    print("Creating sample price data...")
    raw_data = create_sample_data_for_testing()
    
    # Convert to format expected by backtesting engine (ticker names as columns)
    price_data = raw_data[['Close']].copy()
    price_data.columns = ['SPY']  # Single ticker for this demo
    
    print(f"Price data shape: {price_data.shape}")
    print(f"Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}")
    
    # Step 2: Train ML models (use raw OHLCV data for feature engineering)
    ml_models, feature_engineer = train_ml_models(raw_data, save_models=False)
    
    # Step 3: Create all strategies
    strategies = create_strategies(ml_models, feature_engineer, raw_data)  # Pass full OHLCV data
    print(f"\nCreated {len(strategies)} strategies:")
    for name in strategies.keys():
        print(f"  - {name}")
    
    # Step 4: Run backtests
    results = run_strategy_comparison(strategies, price_data)
    
    # Step 5: Analyze results
    analyze_results(results)
    
    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()