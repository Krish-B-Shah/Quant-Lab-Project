# ✅ How to Check if Everything is Working

## Quick Verification (30 seconds)

Run this command to verify everything:

```bash
cd /Users/sidrad/Quant-Lab-Project
bash GatorAI/ml/check_setup.sh
```

This will check:
- ✅ All packages installed
- ✅ Data available
- ✅ CLI works
- ✅ Output directory ready
- ✅ Quick test run

## Manual Verification Steps

### 1. Check Packages (5 seconds)
```bash
python -c "import xgboost; import sklearn; print('✅ Packages OK')"
```

### 2. Check Data (5 seconds)
```bash
python -c "
from data.storage.sqlite_adapter import SQLiteAdapter
import sys
from pathlib import Path
sys.path.insert(0, 'GatorAI/src')
from data.storage.sqlite_adapter import SQLiteAdapter
storage = SQLiteAdapter()
print(f'SPY: {len(storage.read_price_data(\"SPY\"))} rows')
print(f'QQQ: {len(storage.read_price_data(\"QQQ\"))} rows')
print(f'IWM: {len(storage.read_price_data(\"IWM\"))} rows')
"
```

### 3. Test CLI (5 seconds)
```bash
python -m GatorAI.src.ml.cli --help
```

### 4. Quick Training Test (1-2 minutes)
```bash
python -m GatorAI.src.ml.cli --tickers SPY --train --model-types linear --start-date 2023-01-01
```

**Expected output:**
- Should see "Training models for ['SPY']"
- Should see "Prepared ML data for SPY: X rows, 34 features"
- Should see "Created X walk-forward splits"
- Should see "RMSE: X, R2: X, Directional Accuracy: X%"
- Should see "Saved predictions to..." messages
- Should see "Completed training for SPY"

### 5. Check Output Files (5 seconds)
```bash
ls -lh GatorAI/ml/outputs/ | tail -10
```

You should see files like:
- `SPY_*_predictions.csv`
- `SPY_*_metrics.json`
- `SPY_*_feature_importance.json`
- `SPY_*_summary.json`

## Full End-to-End Test (5-10 minutes)

### Test All Models for One Ticker
```bash
python -m GatorAI.src.ml.cli --tickers SPY --train --model-types linear random_forest xgboost --start-date 2020-01-01
```

**What to look for:**
- ✅ All 3 models train successfully
- ✅ Each model shows RMSE, R², and Directional Accuracy
- ✅ Files are saved to `GatorAI/ml/outputs/`
- ✅ No errors in the output

### Generate Predictions
```bash
python -m GatorAI.src.ml.cli --tickers SPY QQQ IWM --predict --model-types xgboost
```

**What to look for:**
- ✅ Models train for each ticker
- ✅ Predictions file created: `predictions_*.csv`
- ✅ File contains: ticker, date, prediction, std, lower_bound, upper_bound

## What Success Looks Like

### ✅ All Good If You See:
1. **Packages**: No import errors
2. **Data**: SPY/QQQ/IWM have 5000+ rows each
3. **CLI**: Help message shows correctly
4. **Training**: Models train without errors
5. **Outputs**: Files created in `GatorAI/ml/outputs/`
6. **Metrics**: RMSE, R², Directional Accuracy shown
7. **Predictions**: CSV file with predictions created

### ❌ Problems to Watch For:

**"No module named 'xgboost'"**
- Fix: `pip install xgboost`

**"No data found for SPY"**
- Fix: `python -m GatorAI.src.data.cli --tickers SPY QQQ IWM --features rsi,macd,atr --refresh`

**"Import error"**
- Fix: `cd GatorAI && pip install -e .`

**"No splits created"**
- Fix: Check date range - need at least 1 year of data

## Quick Status Check

Run this one-liner to see everything at once:

```bash
echo "=== ML Module Status ===" && \
python -c "import xgboost, sklearn; print('✅ Packages')" && \
python -c "import sys; sys.path.insert(0, 'GatorAI/src'); from data.storage.sqlite_adapter import SQLiteAdapter; s = SQLiteAdapter(); print(f'✅ Data: SPY={len(s.read_price_data(\"SPY\"))}, QQQ={len(s.read_price_data(\"QQQ\"))}, IWM={len(s.read_price_data(\"IWM\"))}')" && \
ls -1 GatorAI/ml/outputs/*.csv 2>/dev/null | wc -l | xargs echo "✅ Output files:" && \
echo "✅ Ready to use!"
```

## Expected Performance

For SPY (2020-2025):
- **RMSE**: ~0.13-0.14 (1.3-1.4% daily return error)
- **R²**: Negative values are normal (models perform worse than naive baseline)
- **Directional Accuracy**: ~55-58% (slightly better than random)

**Note**: Negative R² is expected for next-day return prediction - it's a very difficult problem!

## Next Steps After Verification

1. ✅ **Review outputs**: Check `GatorAI/ml/outputs/` for results
2. ✅ **Analyze feature importance**: See which features matter most
3. ✅ **Tune models**: Adjust parameters in `models.py`
4. ✅ **Integrate with backtester**: Use predictions file
5. ✅ **Experiment**: Try different features or time periods

## Summary

**Everything is working if:**
- ✅ Packages install without errors
- ✅ Data is available (5000+ rows per ticker)
- ✅ CLI runs without errors
- ✅ Training completes successfully
- ✅ Output files are created
- ✅ Predictions can be generated

**You're all set!** 🎉


