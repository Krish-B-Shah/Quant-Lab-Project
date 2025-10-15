from pathlib import Path
import sys

# Ensure the package `backtesting` (located in GatorAI/src/backtesting) is importable
# when running this script directly. We add GatorAI/src to sys.path.
root = Path(__file__).resolve().parents[1]
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from backtesting.backtest_runner import run_csv_backtest

"""The testing file is in the data/processed directory"""
def main():
    # Pick one of your processed CSVs (resolve relative to GatorAI/ root)
    data_dir = Path(__file__).resolve().parents[1] / "data"
    csv_path = data_dir / "processed" / "SPY_1d.csv"

    # If the expected CSV is missing, provide a clear message and exit
    if not csv_path.exists():
        print(f"ERROR: expected CSV not found at {csv_path}\nPlease generate sample data with `python GatorAI/scripts/generate_sample_data.py` or provide a processed CSV at that path.")
        return

    # Run the backtest
    stats = run_csv_backtest(csv_path, price_col="adj_close", cost_bps=1.0)

    # Print results
    print("=== Backtest Results ===")
    for k, v in stats.items():
        print(f"{k:15s}: {v}")

if __name__ == "__main__":
    main()
