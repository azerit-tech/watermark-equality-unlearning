"""sgseed2: same seeds/objective/refine as ~/sgseed/seedrefine.py but high budget: K=10 starts (seeds topped up with
+-8px jitter variants, scored once, dedup), PER=400 decodes per start (refine runs to convergence, max 448).
Usage (from TASK): python ~/sgseed2/seed2.py <val|test> <out.json> <wid> <nw> name1 name2 ...   (THREADS env)."""
import sys, os
TH = int(os.environ.get("THREADS", 2)); os.environ["OMP_NUM_THREADS"] = str(TH); os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.environ["TASK"], "t1/sgbox2"))
from common import *  # noqa
import common
from gridsearch import refine, ok  # noqa
from scipy.ndimage import label
torch.set_num_threads(TH)
T = TASK / "t1/out"; H = Path.home()
PER, K, THR, MINA = int(os.environ.get("PER", 400)), int(os.environ.get("K", 10)), 0.5, 16


class Dec2(common.Dec):
    def __init__(self, bs=16):
        so = ort.SessionOptions(); so.intra_op_num_threads = TH; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(TASK / "models/stegastamp/stega_stamp_logits.onnx"), so, providers=["CPUExecutionProvider"])
        self.bs = bs; self.n = 0; self.t = 0.0


def maps(tag):
    if tag == "test" or tag == "fold0":
        fs = [T / f"stegatta/stegastamp/{tag}_v{i}_probs128.npz" for i in range(4)]
        fs += [T / f"stegashift/maps/{tag}_v{i}_probs128.npz" for i in range(1, 8)]
        fs += sorted((T / "stegaens/models").glob(f"*/{tag}_probs128.npz"))
        if tag == "test": fs.append(T / "jacques/sg_all_s1/test_probs128.npz")
    else:  # out-of-fold train folds: sgoof maps
        fs = sorted((H / "sgoof/stegastamp").glob(f"{tag}_*probs128.npz"))
    out = []
    for f in fs:
        if f.exists():
            z = np.load(f); out.append(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
    return out


def boxes_of(p):
    lab, n = label(p >= THR, structure=np.ones((3, 3)))
    if n == 0: return []
    ar = np.bincount(lab.ravel())[1:]; order = np.argsort(-ar); bs = []
    for j in order[:2]:
        if ar[j] < MINA or ar[j] < 0.1 * ar[order[0]]: continue
        ys, xs = np.nonzero(lab == j + 1)
        bs.append((4 * int(ys.min()), 4 * int(ys.max() + 1), 4 * int(xs.min()), 4 * int(xs.max() + 1)))
    return bs


def far(b, S): return all(max(abs(b[i] - s[i]) for i in range(4)) > 4 for s in S)


def starts_of(nm, M):
    S = []
    for m in M:
        if nm in m:
            for b in boxes_of(m[nm]):
                if far(b, S): S.append(b)
    return [s for s in S if s[1] - s[0] >= 8 and s[3] - s[2] >= 8]


def jitter(s):
    y0, y1, x0, x1 = s; d = 8
    c = [(y0 - d, y1 + d, x0 - d, x1 + d), (y0 + d, y1 - d, x0 + d, x1 - d), (y0 - d, y1 - d, x0, x1), (y0 + d, y1 + d, x0, x1),
         (y0, y1, x0 - d, x1 - d), (y0, y1, x0 + d, x1 + d)]
    return [tuple(min(512, max(0, v)) for v in q) for q in c]


if __name__ == "__main__":
    mode, fout, wid, nw = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); names = sys.argv[5:][wid::nw]
    D = Dec2(); split = "train" if mode == "val" else "test"; Me = meta(split)
    sp = json.loads((TASK / "t1/common/splits_5fold.json").read_text())["folds"]
    fold_of = {}
    if mode == "val":
        for k, f in enumerate(sp):
            for n in (f if isinstance(f, list) else f.get("val", f.get("names", []))): fold_of[n] = k
    Mc = {}
    res = json.loads(fout.read_text()) if fout.exists() else {}
    for nm in names:
        if nm in res: continue
        tag = "test" if mode == "test" else f"fold{fold_of[nm]}"
        if tag not in Mc: Mc[tag] = maps(tag)
        t0 = time.time(); n0 = D.n; x = load(split, nm); msg = msgof(Me[nm])
        S = starts_of(nm, Mc[tag]); P = list(S)
        for s in S[:10]:
            for q in jitter(s):
                if ok(q) and far(q, P): P.append(q)
        r_ = {"n_seeds": len(S), "n_pool": len(P)}
        if P:
            o, acc, _ = D.score(x, P, msg); order = np.argsort(-o)
            sel = [P[j] for j in order[:K]]; ob0 = [o[j] for j in order[:K]]; per = []
            best = None
            for s, o0 in zip(sel, ob0):
                n1 = D.n; r, ob = refine(D, x, msg, s, o0, D.n + PER)
                _, ac, _ = D.score(x, [r], msg); per.append(round(float(ac[0]), 3))
                if best is None or ac[0] > best[1]: best = (r, float(ac[0]), s)
                if mode == "test" and ac[0] >= 0.9: break
            r_.update(rect=list(map(int, best[0])), conf=best[1], start=list(best[2]), pool_acc_max=float(acc.max()), per_start=per)
        r_.update(n_decodes=D.n - n0, sec=round(time.time() - t0, 1)); res[nm] = r_
        print(mode, nm, r_, flush=True); fout.write_text(json.dumps(res))
    print("DONE s/dec %.3f" % (D.t / max(D.n, 1)), flush=True)
