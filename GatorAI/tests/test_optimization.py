import sys
from pathlib import Path

# Add src to path for testing
root = Path(__file__).resolve().parents[1]
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from optimization.optimizer import mean_variance_optimize, placeholder_weights
import pandas as pd


def test_optimizer_outputs_weights():
	ret = pd.DataFrame({"A": [0.01, -0.01, 0.0], "B": [0.0, 0.02, -0.01]})
	w = mean_variance_optimize(ret)
	assert abs(w.sum() - 1.0) < 1e-6
	assert set(w.index) == {"A", "B"}


def test_placeholder_weights():
	"""Test that dummy optimizer produces correct weights."""
	weights = placeholder_weights()
	
	# Check weights sum to 100%
	assert abs(weights.sum() - 1.0) < 1e-6, f"Weights sum to {weights.sum()}, not 1.0"
	
	# Check correct tickers
	expected_tickers = {"SPY", "QQQ", "IWM"}
	assert set(weights.index) == expected_tickers, f"Expected {expected_tickers}, got {set(weights.index)}"
	
	# Check specific allocations
	assert abs(weights["SPY"] - 0.5) < 1e-6, f"SPY weight is {weights['SPY']}, not 0.5"
	assert abs(weights["QQQ"] - 0.3) < 1e-6, f"QQQ weight is {weights['QQQ']}, not 0.3"
	assert abs(weights["IWM"] - 0.2) < 1e-6, f"IWM weight is {weights['IWM']}, not 0.2"
