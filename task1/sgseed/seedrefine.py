"""Extra starts for the StegaStamp rectangle search: bounding boxes of the main components of several probability maps
(TTA flips, shifts, ensemble members, jacques/sg_all_s1 if DONE), deduplicated, scored once, top-K refined locally with the
sgbox2 refine (same objective, CPU decoder), budget PER decodes per start. Threshold 0.9 on conf (acc of crop R+3), fixed.
Run from TASK:  python ~/sgseed/seedrefine.py   (fold 0 first; test only if no transplant reaches 0.9)
Outputs ~/sgseed/seeds_{fold0,test}.json and ~/sgseed/stegastamp_{train,test}_boxes.json (stegaedge format)."""
import sys, os
sys.path.insert(0, os.path.join(os.environ["TASK"], "t1/sgbox2"))
from common import *  # noqa  (CPU decoder, iou, load, meta, msgof)
from gridsearch import refine  # noqa  same local refinement as sgbox2
from scipy.ndimage import label

OUT = Path.home() / "sgseed"; T = TASK / "t1/out"
PER, K, THR, MINA = int(os.environ.get("PER", 60)), int(os.environ.get("K", 3)), 0.5, 16


def maps(tag):
    fs = [T / f"stegatta/stegastamp/{tag}_v{i}_probs128.npz" for i in range(4)]
    fs += [T / f"stegashift/maps/{tag}_v{i}_probs128.npz" for i in range(1, 8)]
    fs += sorted((T / "stegaens/models").glob(f"*/{tag}_probs128.npz"))
    if (T / "jacques/sg_all_s1/DONE").exists() and (T / f"jacques/sg_all_s1/{tag}_probs128.npz").exists():
        fs.append(T / f"jacques/sg_all_s1/{tag}_probs128.npz")
    out = []
    for f in fs:
        if f.exists():
            z = np.load(f); out.append(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
    print(tag, "maps", len(out), [str(f.relative_to(T)) for f in fs if f.exists()], flush=True)
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


def starts_of(nm, M):
    S = []
    for m in M:
        if nm not in m: continue
        for b in boxes_of(m[nm]):
            if all(max(abs(b[i] - s[i]) for i in range(4)) > 4 for s in S): S.append(b)
    return S


def run(split, tag, names, M, D, fout):
    res = json.loads(fout.read_text()) if fout.exists() else {}; Me = meta(split)
    for nm in names:
        if nm in res: continue
        t0 = time.time(); n0 = D.n; x = load(split, nm); msg = msgof(Me[nm])
        S = [s for s in starts_of(nm, M) if s[1] - s[0] >= 8 and s[3] - s[2] >= 8]
        r_ = {"n_starts": len(S)}
        if S:
            o, acc, _ = D.score(x, S, msg); order = np.argsort(-o); best = None
            for j in order[:K]:
                r, ob = refine(D, x, msg, S[j], o[j], D.n + PER)
                _, ac, _ = D.score(x, [r], msg)
                if best is None or ac[0] > best[1]: best = (r, float(ac[0]), S[j])
                if ac[0] >= 0.9: break
            r_.update(rect=list(map(int, best[0])), conf=best[1], start=list(best[2]), start_acc_max=float(acc.max()))
        r_.update(n_decodes=D.n - n0, sec=time.time() - t0); res[nm] = r_
        print(tag, nm, r_, flush=True); fout.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    D = Dec(); idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    fa = {s: dict(zip(*(lambda z: (z["names"].tolist(), z["acc"].tolist()))(np.load(T / f"signal/fullacc_stegastamp_{s}.npz")))) for s in ("train", "test")}
    old = {s: json.loads((T / f"stegaedge/stegastamp_{s}_boxes.json").read_text()) for s in ("train", "test")}
    oc = lambda s, n: old[s]["boxes"].get(n, {}).get("conf", 0)
    # ---- fold 0 validation: all outpaintings + transplants of fold 0 (old rect >= 0.9 outpaintings skipped: nothing to gain)
    M0 = maps("fold0"); f0 = sorted(set().union(*[set(m) for m in M0]))
    val = [n for n in f0 if idx[n]["edit_type"] == "transplant"] + [n for n in f0 if idx[n]["edit_type"] == "outpainting" and oc("train", n) < 0.9]
    r0 = run("train", "fold0", val, M0, D, OUT / "seeds_fold0.json")
    fp = [n for n in val if idx[n]["edit_type"] == "transplant" and r0[n].get("conf", 0) >= 0.9]
    rec = [n for n in val if idx[n]["edit_type"] == "outpainting" and r0[n].get("conf", 0) >= 0.9]
    print("FOLD0 outpaint total", sum(idx[n]["edit_type"] == "outpainting" for n in f0), "old>=0.9", sum(idx[n]["edit_type"] == "outpainting" and oc("train", n) >= 0.9 for n in f0),
          "searched", sum(idx[n]["edit_type"] == "outpainting" for n in val), "recovered", [(n, r0[n]["conf"], round(iou(r0[n]["rect"], idx[n]["bbox512"]), 3)) for n in rec],
          "| transplants", sum(idx[n]["edit_type"] == "transplant" for n in val), "FP>=0.9", fp,
          "max transplant conf", max([r0[n].get("conf", 0) for n in val if idx[n]["edit_type"] == "transplant"] or [0]), flush=True)
    d = old["train"]
    for n in rec: d["boxes"][n] = {"rect": r0[n]["rect"], "conf": r0[n]["conf"], "src": "sgseed"}
    (OUT / "stegastamp_train_boxes.json").write_text(json.dumps(d, indent=0))
    if fp:
        print("STOP: transplant false positive"); (OUT / "DONE").write_text("stop fp"); sys.exit(0)
    # ---- test: minority-regime StegaStamp images without rect >= 0.9
    M1 = maps("test"); Mt = meta("test")
    todo = sorted(n for n in Mt if fa["test"][n] < 0.755 and oc("test", n) < 0.9)
    print("test todo", len(todo), flush=True)
    r1 = run("test", "test", todo, M1, D, OUT / "seeds_test.json")
    d = old["test"]; new = []
    for n, v in r1.items():
        if v.get("conf", 0) >= 0.9 and oc("test", n) < 0.9:
            d["boxes"][n] = {"rect": v["rect"], "conf": v["conf"], "src": "sgseed"}; new.append(n)
    (OUT / "stegastamp_test_boxes.json").write_text(json.dumps(d, indent=0))
    print("TEST new rects", len(new), new, "| total conf>=0.9", sum(v["conf"] >= 0.9 for v in d["boxes"].values()), "s/dec %.3f" % (D.t / max(D.n, 1)))
    (OUT / "DONE").write_text("ok")
