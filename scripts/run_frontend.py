import subprocess
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
frontend_dir = root_dir / "frontend"

if __name__ == "__main__":
    cmd = [sys.executable, "-P", "-m", "streamlit", "run", "streamlit/main.py"]
    # Safe-path mode avoids importing frontend/streamlit as the framework itself.
    sys.exit(subprocess.run(cmd, cwd=frontend_dir).returncode)
