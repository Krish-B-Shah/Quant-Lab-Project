#!/bin/bash
# Simple verification script for ML module setup

echo "============================================================"
echo "ML Module Setup Verification"
echo "============================================================"
echo ""

# Check 1: Python and packages
echo "1. Checking Python packages..."
python -c "import xgboost; import sklearn; import pandas; import numpy; print('  ✅ All required packages installed')" 2>/dev/null || echo "  ❌ Missing packages"

# Check 2: Data availability
echo ""
echo "2. Checking data availability..."
python -c "
from data.storage.sqlite_adapter import SQLiteAdapter
import sys
from pathlib import Path
sys.path.insert(0, str(Path('GatorAI/src')))
from data.storage.sqlite_adapter import SQLiteAdapter
storage = SQLiteAdapter()
spy = storage.read_price_data('SPY')
qqq = storage.read_price_data('QQQ')
iwm = storage.read_price_data('IWM')
print(f'  ✅ SPY: {len(spy)} rows')
print(f'  ✅ QQQ: {len(qqq)} rows')
print(f'  ✅ IWM: {len(iwm)} rows')
" 2>/dev/null || echo "  ❌ Data check failed"

# Check 3: CLI works
echo ""
echo "3. Checking CLI..."
python -m GatorAI.src.ml.cli --help > /dev/null 2>&1 && echo "  ✅ ML CLI works" || echo "  ❌ ML CLI not working"

# Check 4: Output directory
echo ""
echo "4. Checking output directory..."
if [ -d "GatorAI/ml/outputs" ] && [ -w "GatorAI/ml/outputs" ]; then
    echo "  ✅ Output directory exists and is writable"
    file_count=$(ls -1 GatorAI/ml/outputs/*.csv GatorAI/ml/outputs/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✅ Found $file_count output files"
else
    echo "  ❌ Output directory issue"
fi

# Check 5: Quick test run
echo ""
echo "5. Running quick test (this may take a minute)..."
python -m GatorAI.src.ml.cli --tickers SPY --train --model-types linear --start-date 2023-01-01 2>&1 | grep -E "(INFO|ERROR|Completed)" | tail -3

echo ""
echo "============================================================"
echo "Verification complete!"
echo "============================================================"
echo ""
echo "To run full verification:"
echo "  python -m GatorAI.src.ml.cli --tickers SPY --train"
echo ""
echo "To generate predictions:"
echo "  python -m GatorAI.src.ml.cli --tickers SPY QQQ IWM --predict"


