#!/usr/bin/env python3
"""
Script to start the FastAPI server for GatorAI Quant Lab.

Usage:
    python scripts/start_api.py
    
Or from the GatorAI directory:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn
import sys
import os
from pathlib import Path

# Get the project root directory (GatorAI/)
project_root = Path(__file__).resolve().parents[2]

# Add src to Python path
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Change working directory to project root
os.chdir(project_root)

if __name__ == "__main__":
    # Run uvicorn with the app module path
    # The app is located at src/api/main.py, so we use "api.main:app"
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

