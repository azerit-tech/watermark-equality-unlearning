import json, numpy as np, sys
S = json.load(open(sys.argv[1]))
for lo, hi in [(0, 20), (20, 40), (40, 80), (80, 999)]:
    v = [s for s in S.values() if lo <= s["minside"] < hi]
    f = lambda key: (np.median([s[key] for s in v]), sum(s[key] >= .9 for s in v))
    print("minside [%d,%d) n=%d" % (lo, hi, len(v)), "sim_exact med %.3f ok %d | sim_m3 med %.3f ok %d | real_m3 med %.3f ok %d" % (*f("sim_exact"), *f("sim_m3"), *f("real_m3")))
print("total ok sim_exact", sum(s["sim_exact"] >= .9 for s in S.values()), "sim_m3", sum(s["sim_m3"] >= .9 for s in S.values()), "real_m3", sum(s["real_m3"] >= .9 for s in S.values()))
