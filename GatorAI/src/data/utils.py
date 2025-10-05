from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXAMPLES_DIR = DATA_DIR / "examples"

DEFAULT_TICKERS: List[str] = ["SPY", "QQQ", "IWM"]


def ensure_data_directories() -> None:
	for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, EXAMPLES_DIR):
		d.mkdir(parents=True, exist_ok=True)
