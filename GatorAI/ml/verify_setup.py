#!/usr/bin/env python3
"""
Verification script to check if ML module is working properly.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported."""
    print("🔍 Checking imports...")
    try:
        # Add src to path for imports
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from ml import MLPredictor, prepare_ml_data, create_model, WalkForwardValidator
        print("  ✅ ML module imports successful")
        
        from data.storage.sqlite_adapter import SQLiteAdapter
        print("  ✅ Storage adapter imports successful")
        
        import xgboost
        import sklearn
        import pandas as pd
        import numpy as np
        print(f"  ✅ XGBoost {xgboost.__version__} installed")
        print(f"  ✅ Scikit-learn {sklearn.__version__} installed")
        print(f"  ✅ Pandas {pd.__version__} installed")
        print(f"  ✅ NumPy {np.__version__} installed")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def check_data_availability():
    """Check if data is available in the database."""
    print("\n🔍 Checking data availability...")
    try:
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from data.storage.sqlite_adapter import SQLiteAdapter
        storage = SQLiteAdapter()
        
        tickers = ["SPY", "QQQ", "IWM"]
        all_available = True
        
        for ticker in tickers:
            data = storage.read_price_data(ticker)
            if len(data) > 0:
                print(f"  ✅ {ticker}: {len(data)} rows ({data['datetime'].min()} to {data['datetime'].max()})")
            else:
                print(f"  ❌ {ticker}: No data found")
                all_available = False
        
        # Check VIX (optional)
        vix_data = storage.read_price_data("^VIX")
        if len(vix_data) > 0:
            print(f"  ✅ VIX: {len(vix_data)} rows")
        else:
            print(f"  ⚠️  VIX: No data (optional, will use defaults)")
        
        return all_available
    except Exception as e:
        print(f"  ❌ Error checking data: {e}")
        return False

def check_data_preparation():
    """Check if data preparation works."""
    print("\n🔍 Checking data preparation...")
    try:
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from ml.data_preparation import prepare_ml_data
        from data.storage.sqlite_adapter import SQLiteAdapter
        
        storage = SQLiteAdapter()
        df = prepare_ml_data("SPY", storage, start_date="2020-01-01")
        
        if len(df) > 0:
            feature_count = len([col for col in df.columns if col not in ["datetime", "next_day_return"]])
            print(f"  ✅ Data preparation successful")
            print(f"     - Rows: {len(df)}")
            print(f"     - Features: {feature_count}")
            print(f"     - Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"     - Target column present: {'next_day_return' in df.columns}")
            return True
        else:
            print("  ❌ No data prepared")
            return False
    except Exception as e:
        print(f"  ❌ Error in data preparation: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_models():
    """Check if models can be created and trained."""
    print("\n🔍 Checking model creation...")
    try:
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from ml.models import create_model
        import numpy as np
        
        # Test each model type
        model_types = ["linear", "random_forest", "xgboost"]
        all_work = True
        
        for model_type in model_types:
            try:
                model = create_model(model_type)
                # Create dummy data
                X = np.random.randn(100, 10)
                y = np.random.randn(100)
                
                # Train
                model.fit(X, y)
                
                # Predict
                predictions = model.predict(X[:5])
                
                # Check feature importance
                importance = model.get_feature_importance()
                
                print(f"  ✅ {model_type}: Created, trained, and predicted successfully")
                if importance:
                    print(f"     - Feature importance available: {len(importance)} features")
            except Exception as e:
                print(f"  ❌ {model_type}: Error - {e}")
                all_work = False
        
        return all_work
    except Exception as e:
        print(f"  ❌ Error checking models: {e}")
        return False

def check_walk_forward():
    """Check if walk-forward validation works."""
    print("\n🔍 Checking walk-forward validation...")
    try:
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from ml.walk_forward import WalkForwardValidator
        from ml.models import create_model
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        df = pd.DataFrame({
            "datetime": dates,
            "feature1": np.random.randn(500),
            "feature2": np.random.randn(500),
            "next_day_return": np.random.randn(500) * 0.01,
        })
        
        validator = WalkForwardValidator(train_window_days=100, test_window_days=20, step_days=20)
        splits = validator.split_data(df)
        
        if len(splits) > 0:
            print(f"  ✅ Walk-forward validation works")
            print(f"     - Created {len(splits)} splits")
            
            # Test validation
            model = create_model("linear")
            results = validator.validate(
                model=model,
                df=df,
                feature_cols=["feature1", "feature2"],
                target_col="next_day_return",
                return_predictions=False,
            )
            
            if "avg_metrics" in results:
                print(f"     - Validation metrics calculated")
                print(f"     - RMSE: {results['avg_metrics']['mean_rmse']:.4f}")
                return True
        else:
            print("  ❌ No splits created")
            return False
    except Exception as e:
        print(f"  ❌ Error in walk-forward validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_output_directory():
    """Check if output directory exists and is writable."""
    print("\n🔍 Checking output directory...")
    try:
        output_dir = Path("GatorAI/ml/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to write a test file
        test_file = output_dir / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()
        
        print(f"  ✅ Output directory exists and is writable: {output_dir}")
        return True
    except Exception as e:
        print(f"  ❌ Error with output directory: {e}")
        return False

def check_end_to_end():
    """Check end-to-end workflow."""
    print("\n🔍 Checking end-to-end workflow...")
    try:
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from ml.predictor import MLPredictor
        from data.storage.sqlite_adapter import SQLiteAdapter
        
        storage = SQLiteAdapter()
        predictor = MLPredictor(storage, model_types=["linear"])
        
        # Prepare data
        df = predictor.prepare_features("SPY", start_date="2023-01-01")
        
        if len(df) > 0:
            print(f"  ✅ End-to-end workflow works")
            print(f"     - Data preparation: {len(df)} rows")
            
            # Test prediction
            try:
                prediction = predictor.predict_next_day("SPY", model_type="linear")
                print(f"     - Prediction generated: {prediction['prediction']:.4f}")
                print(f"     - Confidence interval: [{prediction['lower_bound']:.4f}, {prediction['upper_bound']:.4f}]")
                return True
            except Exception as e:
                print(f"     ⚠️  Prediction test failed: {e}")
                return False
        else:
            print("  ❌ No data available for end-to-end test")
            return False
    except Exception as e:
        print(f"  ❌ Error in end-to-end workflow: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("ML Module Verification")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Data Availability", check_data_availability),
        ("Data Preparation", check_data_preparation),
        ("Models", check_models),
        ("Walk-Forward Validation", check_walk_forward),
        ("Output Directory", check_output_directory),
        ("End-to-End Workflow", check_end_to_end),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! ML module is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

