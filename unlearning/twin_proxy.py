"""Which local retain proxy predicts the TRUE retain score? Across all runs of a twin regime.

  python twin_proxy.py --regime pb10q

True values come from twin_score.py (methods.json: KL(orig || sub) at T=2 and accuracy ratio on the unseen
control set T). Proxies are what each run measured on itself without any unseen data (metrics.jsonl, last row):
  ret_kl    composites of the training recipe      ret2_kl   composites of another recipe
  noise_kl  smooth noise images                    held_kl   forget images held out of the unlearning (--holdout runs only)
  wdist     relative weight distance to the original
For SCRUB runs (ProxyEvaluator) only the composite proxy `retain_kl` exists, per epoch (base_scrub.json).
Prints rank and log-log correlations with the true retain KL, and the typical ratio true / proxy.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from twin_common import ALTER_RECIPES, TWIN_DIR

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--regime", required=True)
args = p.parse_args()
R = TWIN_DIR / args.regime
rows = []
scores = json.loads((R / "methods.json").read_text()) if (R / "methods.json").exists() else {}
for path, s in scores.items():
    run = Path(path).parent
    mfile = run / "metrics.jsonl"
    if not mfile.exists():
        continue
    last = [json.loads(l) for l in mfile.read_text().splitlines() if l.strip()][-1]
    a = s[ALTER_RECIPES[0]]
    rows.append({"name": run.name, "true_kl": a["retain_kl"], "true_ratio": a["retain_acc_ratio"], "true_retain": a["retain"],
                 **{k: last.get(k, float("nan")) for k in ("ret_kl", "ret2_kl", "noise_kl", "held_kl", "heldalt_kl", "wdist", "ret_agree", "held_agree")}})
scrub = json.loads((R / "base_scrub.json").read_text()) if (R / "base_scrub.json").exists() else {}
for path, s in scrub.items():
    run, ep = Path(path).parent, int(Path(path).stem[-2:])
    hist = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    h = next((x for x in hist if x["epoch"] == ep), None)
    if h:
        a = s[ALTER_RECIPES[0]]
        rows.append({"name": f"{run.name}_ep{ep}", "true_kl": a["retain_kl"], "true_ratio": a["retain_acc_ratio"], "true_retain": a["retain"],
                     "ret_kl": h["retain_kl"], "ret_agree": h["retain_agree"]})


def ranks(v):
    return torch.tensor(v).argsort().argsort().double()


def corr(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12)


print(f"{len(rows)} runs. True retain KL ranges {min(r['true_kl'] for r in rows):.4f} .. {max(r['true_kl'] for r in rows):.4f}")
print(f"{'proxy':<12} {'n':>3} {'spearman':>9} {'log-log r':>9} {'median true/proxy':>18} {'min':>7} {'max':>7}")
for k in ("ret_kl", "ret2_kl", "noise_kl", "held_kl", "heldalt_kl", "wdist"):
    sel = [r for r in rows if k in r and isinstance(r[k], float) and math.isfinite(r[k]) and r[k] > 0 and r["true_kl"] > 0]
    if len(sel) < 3:
        print(f"{k:<12} {len(sel):>3}  (too few runs)")
        continue
    t, q = [r["true_kl"] for r in sel], [r[k] for r in sel]
    ratio = torch.tensor([a / b for a, b in zip(t, q)])
    print(f"{k:<12} {len(sel):>3} {corr(ranks(t), ranks(q)):9.3f} {corr([math.log(x) for x in t], [math.log(x) for x in q]):9.3f} "
          f"{ratio.median().item():18.2f} {ratio.min().item():7.2f} {ratio.max().item():7.2f}")
print("\nper run (sorted by the true retain KL):")
print(f"{'run':<40} {'true KL':>8} {'acc ratio':>9} {'retain':>7} | {'ret_kl':>7} {'ret2_kl':>7} {'noise':>7} {'held':>7} {'wdist':>7}")
for r in sorted(rows, key=lambda r: r["true_kl"]):
    g = lambda k: f"{r[k]:7.4f}" if k in r and isinstance(r[k], float) and math.isfinite(r[k]) else "      -"
    print(f"{r['name'][:40]:<40} {r['true_kl']:8.4f} {r['true_ratio']:9.3f} {r['true_retain']:7.3f} | {g('ret_kl')} {g('ret2_kl')} {g('noise_kl')} {g('held_kl')} {g('wdist')}")
