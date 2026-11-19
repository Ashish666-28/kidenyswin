"""Allow the same package launch command as the reference project."""
import runpy
import sys
from pathlib import Path

directory = Path(__file__).resolve().parent
sys.path.insert(0, str(directory))
runpy.run_path(str(directory / "app.py"), run_name="__main__")
