"""Expected score of a candidate partition under simulated truths (real metric, intra-family MCC)."""
import csv, random, statistics, sys
sys.path.insert(0, "/Users/jacquesmougeot/Desktop/Azerit-hack")
from model_equality.scoring.metrics import score
def run(path, lures, sizes=(5, 6, 7), wrongs=(0, 1), n=400, seed=0):
    pred = {r[0]: (r[1], r[2]) for r in csv.reader(open(path)) if r[0] != "endpoint"}
    pf = {e: v[0] for e, v in pred.items()}; pm = {e: v[1] for e, v in pred.items()}
    fams = {}
    for e, (f, m) in pred.items(): fams.setdefault(f, []).append(e)
    rng = random.Random(seed); out = {}
    for s in sizes:
        for wrong in wrongs:
            sc = []
            for _ in range(n):
                tm = {}
                badfam = rng.choice(sorted(fams))
                for f, eps in fams.items():
                    L = [e for e in eps if e in lures]; U = [e for e in eps if e not in lures]
                    rng.shuffle(L)
                    bad = L[:wrong] if f == badfam else []
                    good = [e for e in L if e not in bad]
                    extra = rng.sample(U, max(0, s - len(good)))
                    for e in eps: tm[e] = f + "_base" if (e in good or e in extra) else f + "_" + e
                sc.append(score(pf, pm, pf, tm)["score"])
            out[(s, wrong)] = (statistics.mean(sc), min(sc), max(sc))
    return out
if __name__ == "__main__":
    lures = set(sys.argv[2].split(","))
    for k, v in run(sys.argv[1], lures).items():
        print(f"base group size {k[0]}, wrong lure labels {k[1]}: mean {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
