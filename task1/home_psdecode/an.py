import json, sys, numpy as np
R = json.load(open(sys.argv[1])); thr = float(sys.argv[2])
op = {k: v for k, v in R.items() if v["kind"] == "outpainting"}
tp = {k: v for k, v in R.items() if v["kind"] != "outpainting"}
keys = [k for k in next(iter(op.values())) if isinstance(next(iter(op.values()))[k], float)]
print(f"n_outpaint={len(op)} n_tp_items={len(tp)} thr={thr}")
rows = []
for k in keys:
    a = np.array([v[k] for v in op.values()]); t = np.array([v.get(k, 0) for v in tp.values()]) if tp else np.zeros(1)
    tb = [v.get(k, 0) for v in tp.values() if v["kind"] == "tp_bbox"]; tbest = [v.get(k, 0) for v in tp.values() if v["kind"] == "tp_best"]
    rows.append((int((a >= thr).sum()), k, float(np.median(a)), float(np.mean(a)), max(tb or [0]), max(tbest or [0]), int((t >= thr).sum())))
for r in sorted(rows, key=lambda r: (-r[0], -r[3])):
    print(f"{r[1]:28s} ok={r[0]:2d}/{len(op)} med={r[2]:.3f} mean={r[3]:.3f} | tp max bbox={r[4]:.3f} best={r[5]:.3f} FP={r[6]}")
