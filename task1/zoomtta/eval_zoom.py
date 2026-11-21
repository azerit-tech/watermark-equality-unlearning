"""Fold-0 evaluation (no tuning): each view alone and simple mean with the current map, threshold 0.5 + regime guard,
by edit type and minority-region size (<2 %, 2-10 %, >10 % of the 128 grid), paired bootstrap vs current.
Current: TM = t1/out/tiles2/trustmark/fold0_probs128.npz (=s320), PS = t1/out/tiles3/pixelseal/fold0_probs128.npz (mean t256,t320).
python ~/zoomtta/eval_zoom.py"""
import sys, json
from pathlib import Path
import numpy as np
TASK = Path("/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task")
for p in (TASK / "t1" / "train", TASK / "t1" / "common", TASK / "t1" / "tmdense", TASK):
    sys.path.insert(0, str(p))
from regime_guard import guard, load_regimes  # noqa
from evaluate import evaluate, load_train  # noqa
ti = {d["image_name"].rsplit(".", 1)[0] if "." in d["image_name"] else d["image_name"]: d for d in json.loads((TASK / "t1/common/train_index.json").read_text())} if isinstance(json.loads((TASK / "t1/common/train_index.json").read_text()), list) else json.loads((TASK / "t1/common/train_index.json").read_text())
_, gt = load_train(); reg = load_regimes("train"); Z = Path.home() / "zoomtta" / "out"
def ld(p, names=None):
    z = np.load(p); d = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
    return d
CUR = {"trustmark": TASK / "t1/out/tiles2/trustmark/fold0_probs128.npz", "pixelseal": TASK / "t1/out/tiles3/pixelseal/fold0_probs128.npz"}
V192 = {"trustmark": TASK / "t1/out/tiles2/trustmark/fold0_s192_probs128.npz", "pixelseal": TASK / "t1/out/tiles3/pixelseal/s2_22min/fold0_t192_probs128.npz"}
BINS = [(0, .02, "<2%"), (.02, .10, "2-10%"), (.10, .51, ">10%")]
rng = np.random.default_rng(0); res = {}
for m in ("trustmark", "pixelseal"):
    cur = ld(CUR[m]); names = list(cur); views = {"cur": cur, "s192": ld(V192[m])}
    f128 = Z / m / "fold0_s128_probs128.npz"
    if f128.exists(): views["s128"] = ld(f128)
    for v in ("s192", "s128"):
        if v in views: views[f"cur+{v}"] = {n: (cur[n] + views[v][n]) / 2 for n in names}
    if "s128" in views: views["cur+s192+s128"] = {n: (cur[n] + views["s192"][n] + views["s128"][n]) / 3 for n in names}
    mf = {n: min(gt[n].mean(), 1 - gt[n].mean()) for n in names}
    per = {}; res[m] = {}
    print(f"== {m} ({len(names)} images; bins " + ", ".join(f"{l}:{sum(lo <= mf[n] < hi for n in names)}" for lo, hi, l in BINS) + ")")
    for k, d in views.items():
        r = evaluate({n: guard(d[n], reg[n])[0] for n in names}, names, return_per_image=True)["per_image"]; per[k] = r
        row = {"all": np.mean([r[n] for n in names])}
        for et in ("hole", "transplant", "outpainting"):
            row[et] = np.mean([r[n] for n in names if ti[n]["edit_type"] == et])
        for lo, hi, l in BINS:
            row[l] = np.mean([r[n] for n in names if lo <= mf[n] < hi])
        if k != "cur":
            dd = np.array([r[n] - per["cur"][n] for n in names]); idx = rng.integers(0, len(names), (5000, len(names)))
            bm = dd[idx].mean(1); row["d"] = dd.mean(); row["ci"] = (np.percentile(bm, 2.5), np.percentile(bm, 97.5)); row["P>0"] = (bm > 0).mean()
            for lo, hi, l in BINS:
                row["d" + l] = np.mean([r[n] - per["cur"][n] for n in names if lo <= mf[n] < hi])
        res[m][k] = {a: (np.round(b, 4).tolist() if not isinstance(b, float) else round(b, 4)) for a, b in row.items()}
        print(f"{k:14s} " + " ".join(f"{a}={np.round(b, 4).tolist()}" for a, b in row.items()), flush=True)
(Path.home() / "zoomtta" / "out" / "eval_fold0.json").write_text(json.dumps(res, indent=1, default=float))
