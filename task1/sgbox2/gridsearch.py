"""Map-independent coarse grid of rectangles + local 4-edge refinement, CPU decoder, fixed decode budget per image.
Usage (from TASK): python t1/sgbox2/gridsearch.py <split> <out.json> name1 name2 ...   conf = acc of crop R+3 (same as stegaedge)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
BUDGET, TOPK, A = int(os.environ.get("BUDGET", 1225)), 6, 16


def ok(r):
    h, w = r[1] - r[0], r[3] - r[2]
    return r[0] >= 0 and r[2] >= 0 and r[1] <= 512 and r[3] <= 512 and h >= 8 and w >= 8 and max(h / w, w / h) <= A


def grid(sizes=(64, 128, 256), div=2):
    pos = {}
    for s in sizes:
        p = list(range(0, 512 - s + 1, max(4, s // div)))
        if p[-1] != 512 - s: p.append(512 - s)
        pos[s] = p
    return [(y, y + h, x, x + w) for h in sizes for w in sizes if max(h / w, w / h) <= A for y in pos[h] for x in pos[w]]


def refine(D, x, msg, r, obj, stop):
    for st in (8, 4, 2, 1):
        for _ in range(8):
            if D.n >= stop: return r, obj
            c = []
            for e in range(4):
                for d in (-st, st):
                    q = list(r); q[e] += d; c.append(tuple(q))
            for (p, q_) in ((0, 1), (2, 3)):
                for d in (-st, st):
                    q = list(r); q[p] += d; q[q_] += d; c.append(tuple(q))
            for d in (-st, st):
                c.append((r[0] - d, r[1] + d, r[2] - d, r[3] + d))
            c = [q for q in c if ok(q)]
            o, _, _ = D.score(x, c, msg); j = int(np.argmax(o))
            if o[j] <= obj: break
            r, obj = c[j], o[j]
    return r, obj


if __name__ == "__main__":
    split, fout = sys.argv[1], Path(sys.argv[2]); M = meta(split); D = Dec()
    G = grid(); print("grid", len(G), "budget", BUDGET, flush=True)
    res = json.loads(fout.read_text()) if fout.exists() else {}
    for nm in sys.argv[3:]:
        if nm in res: continue
        t0 = time.time(); n0 = D.n; x = load(split, nm); msg = msgof(M[nm])
        o, acc, _ = D.score(x, G, msg); order = np.argsort(-o); starts = []
        for j in order:
            if all(iou(G[j], s) < 0.3 for s, _ in starts): starts.append((G[j], o[j]))
            if len(starts) >= TOPK: break
        best = None; per = (BUDGET - (D.n - n0)) // TOPK
        for k, (s, o0) in enumerate(starts):
            r, ob = refine(D, x, msg, s, o0, D.n + per)
            if best is None or ob > best[1]: best = (r, ob)
            if ob >= 1.0: break
        _, ac, mar = D.score(x, [best[0]], msg)
        res[nm] = {"rect": list(map(int, best[0])), "conf": float(ac[0]), "margin": float(mar[0]), "grid_max_acc": float(acc.max()),
                   "n_decodes": D.n - n0, "sec": time.time() - t0}
        print(nm, res[nm], flush=True); fout.write_text(json.dumps(res))
    print("DONE s/dec %.3f" % (D.t / max(D.n, 1)))
