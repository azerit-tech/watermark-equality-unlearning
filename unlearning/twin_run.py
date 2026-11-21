"""Run one of the task scripts (scrub.py, targets.py, unlearn.py, ...) unchanged on the twin architecture.

Those scripts build the model through common.build_model(), hard-coded to the real ViT-B/16. This wrapper
replaces that one function by the twin regime's architecture, then runs the script as if called directly:

  python twin_run.py --regime tiny scrub.py --model runs/twin/tiny/orig.pt --forget runs/twin/F.npz --out runs/twin/tiny/m/scrub
  python twin_run.py --regime tiny targets.py --model runs/twin/tiny/orig.pt --forget runs/twin/F.npz --out ... --mean-loss 1.0
  python twin_run.py --regime tiny unlearn.py --model runs/twin/tiny/orig.pt --forget runs/twin/F.npz --targets ... --out ...

`--twin` is a shortcut that appends `--model <regime>/orig.pt --forget runs/twin/F.npz`.
Nothing in the other scripts is edited; they see only the twin-original and the twin-forget images.
"""
import runpy
import sys

import common
from twin_common import F_NPZ, TWIN_DIR, build, regime_arch

argv = sys.argv[1:]
assert argv and argv[0] == "--regime", __doc__
regime, script, rest = argv[1], argv[2], argv[3:]
arch = regime_arch(TWIN_DIR / regime)
common.build_model = lambda: build(arch)
if "--twin" in rest:
    rest.remove("--twin")
    rest += ["--model", str(TWIN_DIR / regime / "orig.pt"), "--forget", str(F_NPZ)]
sys.argv = [script] + rest
runpy.run_path(script, run_name="__main__")
