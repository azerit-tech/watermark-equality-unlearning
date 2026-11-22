"""Does the pseudo-gold bench rank models like the leaderboard? Spearman rank correlation on the models already submitted.

Reads submissions.jsonl (one JSON line per attempt; successful ones have http 200 and reply.score), keeps each file
once (its successful score), scores every file still on disk with pgb_score.Bench, then prints the Spearman
correlation between the public score and: the bench's final score, its variants and its components, plus simple
one-number baselines (altered-copy CE, clean CE). The bench is useful only if `final` beats those baselines.

GO rule (plan): Spearman(final, public) >= --go (0.7) -> the bench may pick models; else it picks nothing.

  python pgb_rank.py --submissions submissions.jsonl --orig ../model.pt --forget ../forget_images.npz \
      --retain runs/pg/bank_t2i.npz --refs runs/pg/models/pg_2.pt runs/pg/models/pg_3.pt --csv runs/pgb/rank.csv
  (sisters as reference instead of / next to the pseudo-golds: --sis runs/sisters/sis_probs.pt)
Extra models (not submitted) can be scored in the same run with --extra; they are listed but not in the correlation.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from pgb_score import add_bench_args, bench_from_args, fmt


def rankdata(x):
    """Ranks 1..n, ties get their average rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def read_submissions(path):
    """[(resolved file, public score, tag)] for successful submissions, one per file (last success wins)."""
    base = Path(path).resolve().parent
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        score = (d.get("reply") or {}).get("score")
        if d.get("http") != 200 or score is None:
            continue
        f = Path(d["file"])
        f = f if f.is_absolute() else base / f
        note = d.get("note", "")
        tag = note[1:note.index("]")] if note.startswith("[") and "]" in note else f"{f.parent.name}/{f.stem}"
        out[str(f.resolve())] = (Path(f), float(score), tag)
    return list(out.values())


KEYS = ("final", "final_gated", "final_held", "p30_mean", "forget", "mean_std", "pred", "retain", "ret_kl",
        "ret_agree", "fam_crop", "fam_flip", "fam_blur", "fam_noise", "fam_color", "fam_jpeg")
BASELINES = ("alt_ce_mean", "clean_ce_mean", "clean_acc")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bench_args(p)
    p.add_argument("--submissions", type=Path, required=True)
    p.add_argument("--extra", type=Path, nargs="*", default=[], help="other models to score (not in the correlation)")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None, help="summary (correlations) as JSON")
    p.add_argument("--go", type=float, default=0.7)
    a = p.parse_args()
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    subs = read_submissions(a.submissions)
    missing = [t for f, _, t in subs if not f.exists()]
    subs = [s for s in subs if s[0].exists()]
    print(f"[rank] {len(subs)} submitted models on disk" + (f", missing: {missing}" if missing else ""), flush=True)
    bench = bench_from_args(a)
    rows = []
    for f, pub, tag in subs + [(e, float("nan"), "extra:" + e.stem) for e in a.extra]:
        r = {"tag": tag, "model": str(f), "public": pub, **bench.score(f)}
        rows.append(r)
        print(f"public {pub:.4f} | " + fmt(tag, r), flush=True)
    sub_rows = [r for r in rows if not np.isnan(r["public"])]
    pub = [r["public"] for r in sub_rows]
    nz = [i for i, r in enumerate(sub_rows) if r["public"] > 0]
    corr = {}
    print("\nSpearman with the public score (all | public > 0):")
    sis_keys = tuple("sis_" + k for k in ("final", "final_held", "forget", "mean_std", "pred") if "sis_" + k in sub_rows[0])
    for k in KEYS + sis_keys + BASELINES:
        v = [r[k] for r in sub_rows]
        corr[k] = (spearman(v, pub), spearman([v[i] for i in nz], [pub[i] for i in nz]))
        print(f"  {k:14s} {corr[k][0]:+.3f} | {corr[k][1]:+.3f}" + ("   (baseline)" if k in BASELINES else ""))
    rho = corr["final"][0]
    verdict = "GO" if rho >= a.go else "NO-GO"
    print(f"\nBENCH {verdict}: Spearman(final, public) = {rho:.3f} over {len(sub_rows)} models (threshold {a.go}); "
          f"public>0 only: {corr['final'][1]:.3f}; refs = {sorted(bench.refs)}")
    order = sorted(rows, key=lambda r: -r["final"])
    print("bench order: " + ", ".join(f"{r['tag']}={r['final']:.4f}(pub {r['public']:.4f})" for r in order))
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        cols = ["tag", "public", "model"] + [k for k in rows[0] if k not in ("tag", "public", "model")]
        with a.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"verdict": verdict, "rho_final": rho, "n": len(sub_rows), "refs": sorted(bench.refs),
                                      "corr": corr}, indent=1))


if __name__ == "__main__":
    main()
