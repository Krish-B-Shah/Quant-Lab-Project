#!/usr/bin/env python3
"""
Script to start the Streamlit dashboard with API integration.
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    dashboard_path = Path(__file__).resolve().parents[2] / "src" / "dashboard" / "dashboard_api.py"
    
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])

