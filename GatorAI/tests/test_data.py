from pathlib import Path
import pandas as pd
import sys

# Add src to path for testing
root = Path(__file__).resolve().parents[1]
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from backtesting.backtest_runner import run_csv_backtest


def test_placeholder_data_dirs():
	# This is a placeholder test to ensure test discovery works
	assert isinstance(Path(".").resolve(), Path)


def test_csv_reading():
	"""Test that CSV files can be read correctly."""
	# Check if processed data exists
	data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	spy_file = data_dir / "SPY_1d.csv"
	
	if spy_file.exists():
		# Test CSV reading
		df = pd.read_csv(spy_file)
		assert not df.empty, "CSV file is empty"
		assert "adj_close" in df.columns, "adj_close column missing"
		assert "datetime" in df.columns, "datetime column missing"
		
		# Test backtest integration
		stats = run_csv_backtest(spy_file)
		assert "cagr" in stats, "CAGR metric missing"
		assert "sharpe" in stats, "Sharpe ratio missing"
		assert "max_drawdown" in stats, "Max drawdown missing"
	else:
		# Skip test if data doesn't exist
		import pytest
		pytest.skip("No processed data found - run fetch_data.py first")
