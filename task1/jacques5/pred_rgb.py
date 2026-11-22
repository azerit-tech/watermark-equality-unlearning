"""t1/jacques3/pred_informed.py with t1/jacques5/informed_rgb.py in place of jacques3/informed.py (same arguments, same outputs)."""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jacques3")); sys.path.insert(0, str(HERE))
import informed_rgb  # noqa: E402
sys.modules["informed"] = informed_rgb
runpy.run_path(str(HERE.parent / "jacques3" / "pred_informed.py"), run_name="__main__")
