import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l); g = r["groups"]
    print(r["it"], r["time_min"], r["val"], {k[5:]: g[k][0] for k in g if k.startswith("edit")}, "synth_tr", r.get("synth_val", {}).get("all/transplant"))
