"""Summary tables of a run (run from TASK): python t1/train/report.py t1/out/train/v0"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
METH = ["pixelseal", "trustmark", "stegastamp"]
TYPES = ["outpainting", "transplant", "hole"]


def recs(name):
    p = R / name / "log.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


print("## Curves (it, min, val real fold0, synth val all types)")
for name in METH + ["real_only"]:
    rs = recs(name)
    print(name, " | ".join(f"{r['it']}/{r['time_min']:.0f}m:{r['val']:.3f}" for r in rs))

print("\n## Best checkpoint: real fold 0 by method x edit_type, synthetic val, gap (synth - real)")
print("| model | method | type | real fold0 (n) | synth val (n) | gap |")
for name in METH + ["real_only"]:
    rs = recs(name)
    if not rs:
        continue
    b = max(rs, key=lambda r: r["val"])
    for m in (METH if name == "real_only" else [name]):
        for t in TYPES:
            g = b["groups"].get(f"{m}/{t}"); s = b.get("synth_val", {}).get(f"{m}/{t}")
            gap = f"{s[0] - g[0]:+.3f}" if g and s else "-"
            print(f"| {name} (it {b['it']}) | {m} | {t} | {g[0]:.3f} ({g[1]}) | {s[0] if s else '-'} ({s[1] if s else '-'}) | {gap} |")
        g = b["groups"].get(f"method/{m}")
        print(f"| {name} | {m} | ALL | {g[0]:.3f} ({g[1]}) | | |")

print("\n## Fidelity (bit accuracy vs message; hf_ratio median)")
keys = ["bit_full", "bit_grey_out", "bit_crop_lt40", "bit_crop_40_80", "bit_crop_ge80", "hf_ratio", "minority_frac"]
print("| method | src | type | n | " + " | ".join(keys) + " |")
for m in METH:
    p = R / f"fidelity_{m}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    for src in ("real", "synth"):
        for t in TYPES:
            v = d[src].get(t)
            if not v:
                continue
            cells = []
            for k in keys:
                x = v.get(k)
                n = v.get(k + "_n")
                cells.append("-" if x is None else (f"{x:.3f}" + (f" ({n})" if n is not None else "")))
            print(f"| {m} | {src} | {t} | {v['n']} | " + " | ".join(cells) + " |")
    for kind, g in d["gen_stats"].items():
        print(m, "gen", kind, {k: (v["n"], round(v["mad"], 3), round(v["frac_unchanged"], 3)) for k, v in g.items()})
