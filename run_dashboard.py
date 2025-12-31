#!/usr/bin/env python
"""
Launcher script for the Predictive Maintenance Dashboard.

Usage:
    python run_dashboard.py
    
Or run directly with Streamlit:
    streamlit run src/dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App not found at {app_path}")
        sys.exit(1)
    
    print(f"Starting dashboard from {app_path}...")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
