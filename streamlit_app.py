"""Entry point for Streamlit Community Cloud."""
import runpy, sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

runpy.run_module("ai_exposure_api.app", run_name="__main__")
