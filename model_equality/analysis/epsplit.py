"""Which endpoints are public?  Joint posterior of (truth, public endpoint set) given the rounded public scores,
under the hypothesis "a pair is public when both its endpoints belong to a public set of m endpoints".

  python3.12 epsplit.py [--m 18] [--n 4000000] [--seed 1] [--split endpoint|pair] NAME=score NAME=score ...

Truth prior (written in the journal, 23 h 05): the 11 labelled lures other than ep-02 are right; ep-02 right 0.88;
ep-07 right 0.85, ep-14 0.90, ep-27 0.90; ep-32 and ep-13 are bare bases with their white-box priors (0.68, 0.55);
the bare base of qwen_A is one of ep-01, ep-12, ep-17, ep-25 (0.235 each) or none of them (0.06);
the eleventh kind is a lure with probability 0.15 per family (ep-26, ep-19, and in qwen_A a second of the four).
Score = 0.1 + 0.9 * max(0, MCC) on the public intra-family pairs (formula of scoring/metrics.py), rounded to 0.05.
Output: P(scores), posterior of every uncertain truth item, and P(public) for each endpoint.
"""
import csv, itertools, sys
import numpy as np
ROOT = "/Users/jacquesmougeot/Desktop/Azerit-hack/model_equality/"
C = ROOT + "analysis/candidates/"
fam = {r[0]: r[1] for r in csv.reader(open(ROOT + "analysis/families_ep.csv")) if r[0] != "endpoint"}
EPS = sorted(fam); IDX = {e: i for i, e in enumerate(EPS)}
PAIRS = [(a, b) for a, b in itertools.combinations(EPS, 2) if fam[a] == fam[b]]
PA = np.array([IDX[a] for a, b in PAIRS]); PB = np.array([IDX[b] for a, b in PAIRS])
SURE = ["ep-10", "ep-16", "ep-29", "ep-31", "ep-04", "ep-23", "ep-33", "ep-06", "ep-08", "ep-20", "ep-30"]
PRI = {"ep-02": .88, "ep-07": .85, "ep-14": .90, "ep-27": .90, "ep-32": .68, "ep-13": .55}
QA4 = ["ep-01", "ep-12", "ep-17", "ep-25"]

def arg(a, k, d=None):
    if k in a:
        i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return v
    return d

def pred_vec(name):
    m = {r[0]: (r[1], r[2]) for r in csv.reader(open(C + name + ".csv")) if r[0] != "endpoint"}
    return np.array([m[a] == m[b] for a, b in PAIRS])

RMODE = 'nearest'

def score(P, T, pub, step=0.05):
    tp = (pub & P & T).sum(1).astype(np.float64); fp = (pub & P & ~T).sum(1).astype(np.float64)
    fn = (pub & ~P & T).sum(1).astype(np.float64); tn = (pub & ~P & ~T).sum(1).astype(np.float64)
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.sqrt(np.where(den > 0, den, 1)), 0.0)
    raw = (0.1 + 0.9 * np.maximum(mcc, 0)) / step
    q = np.round(raw) if RMODE == 'nearest' else (np.floor(raw + 1e-9) if RMODE == 'floor' else np.ceil(raw - 1e-9))
    return np.round(q * step, 4)

def main():
    a = sys.argv[1:]
    m = int(arg(a, "--m", "18")); n = int(arg(a, "--n", "4000000")); seed = int(arg(a, "--seed", "1")); split = arg(a, "--split", "endpoint")
    global RMODE; RMODE = arg(a, "--round", "nearest")
    obs = [(x.split("=")[0], float(x.split("=")[1])) for x in a]
    Pobs = [(pred_vec(nm), v) for nm, v in obs]
    rng = np.random.default_rng(seed); chunk = 200000
    kept = 0; pubsum = np.zeros(len(EPS)); truthsum = np.zeros(len(EPS)); bare = {e: 0 for e in QA4 + ["none"]}; mcount = {}
    partial = [0] * len(obs); lasthist = {}
    for _ in range(n // chunk):
        B = np.zeros((chunk, len(EPS)), bool)
        for e in SURE: B[:, IDX[e]] = True
        for e, p in PRI.items(): B[:, IDX[e]] = rng.random(chunk) < p
        which = rng.choice(5, size=chunk, p=[.235, .235, .235, .235, .06])
        for j, e in enumerate(QA4): B[:, IDX[e]] = which == j
        B[:, IDX["ep-26"]] = rng.random(chunk) < .15
        B[:, IDX["ep-19"]] = rng.random(chunk) < .15
        x = rng.random(chunk) < .15; second = rng.integers(0, 4, chunk)
        for j, e in enumerate(QA4): B[:, IDX[e]] |= x & (second == j)
        T = B[:, PA] & B[:, PB]
        if split == "pair":
            pub = rng.random((chunk, len(PAIRS))) < 0.30; R = None
        else:
            R = rng.random((chunk, len(EPS))).argsort(1).argsort(1) < m
            pub = R[:, PA] & R[:, PB]
        ok = np.ones(chunk, bool)
        for j, (P, v) in enumerate(Pobs):
            sc = score(P, T[ok], pub[ok])
            if j == len(Pobs) - 1:
                for x in sc: lasthist[float(x)] = lasthist.get(float(x), 0) + 1
            ok[ok] = sc == v
            partial[j] += int(ok.sum())
            if not ok.any(): break
        if not ok.any(): continue
        kept += int(ok.sum()); truthsum += B[ok].sum(0)
        if R is not None: pubsum += R[ok].sum(0)
        w = which[ok]
        for j, e in enumerate(QA4 + ["none"]): bare[e] += int((w == j).sum())
    print(f"[{RMODE}] split {split}{' m=' + str(m) if split == 'endpoint' else ''}: {kept} of {n} draws reproduce the {len(obs)} scores -> P(scores) = {kept / n:.3e}")
    print("   cumulative: " + ", ".join(f"{obs[j][1]}: {partial[j] / n:.1e}" for j in range(len(obs))))
    tot = sum(lasthist.values())
    print("   last file, simulated score among draws compatible with the earlier scores: " + ", ".join(f"{k:.2f}:{v / max(tot, 1):.2f}" for k, v in sorted(lasthist.items())) + f"  (n={tot})")
    if not kept: return
    print("   truth, P(in the base group): " + ", ".join(f"{e} {truthsum[IDX[e]] / kept:.2f}" for e in list(PRI) + QA4 + ["ep-26", "ep-19"]))
    print("   first bare draw of qwen_A: " + ", ".join(f"{e} {v / kept:.2f}" for e, v in bare.items()))
    if split == "endpoint":
        for f in ("qwen_A", "qwen_B", "llama3"):
            print(f"   P(public) {f}: " + ", ".join(f"{e[3:]}:{pubsum[IDX[e]] / kept:.2f}" for e in EPS if fam[e] == f))

if __name__ == "__main__":
    main()
