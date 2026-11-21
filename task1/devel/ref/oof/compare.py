"""Paired comparison of two prediction files on their common train images (guard binarisation), bootstrap CI,
plus agreement of the two test files. Run from TASK: python t1/oof/compare.py A_dir B_dir   (dirs with *_probs128.npz)"""
import sys, os, glob
from pathlib import Path
import numpy as np
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/common"), str(TASK / "t1/train"), str(TASK / "t1/synth"), str(TASK / "t1/oof")]
from evaluate import evaluate  # noqa
from regime_guard import guard, load_regimes  # noqa
from analyze_oof import ci  # noqa

def ld(p):
    z = np.load(p); return dict(zip([str(n) for n in z["names"]], z["probs"].astype(np.float32)))
A, B = sys.argv[1], sys.argv[2]
fa = [f for f in glob.glob(A + "/*_probs128.npz") if "test" not in f][0]; fb = [f for f in glob.glob(B + "/*_probs128.npz") if "test" not in f][0]
pa, pb = ld(fa), ld(fb); names = sorted(set(pa) & set(pb)); reg = load_regimes("train")
sa = evaluate({n: guard(pa[n], reg[n])[0] for n in names}, names, return_per_image=True)["per_image"]
sb = evaluate({n: guard(pb[n], reg[n])[0] for n in names}, names, return_per_image=True)["per_image"]
print(f"n={len(names)}  A={ci([sa[n] for n in names])}  B={ci([sb[n] for n in names])}  A-B={ci([sa[n] - sb[n] for n in names])}")
ta, tb = ld(A + "/test_probs128.npz"), ld(B + "/test_probs128.npz"); treg = load_regimes("test"); tn = sorted(ta)
d = np.mean([np.abs(ta[n] - tb[n]).mean() for n in tn])
agree = np.mean([(guard(ta[n], treg[n])[0] == guard(tb[n], treg[n])[0]).mean() for n in tn])
print(f"test: mean|dp|={d:.3f}  cell agreement of guarded masks={agree:.3f}")
