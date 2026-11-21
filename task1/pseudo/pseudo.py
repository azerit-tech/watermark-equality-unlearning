"""Self-training with pseudo-labels on the TEST images, on top of the frozen-feature decoder of t1/frozen/dec.py.
Run from TASK (CPU):  FCACHE=cache9 OMP_NUM_THREADS=8 python t1/pseudo/pseudo.py --iters 2
Iteration k: pseudo-label test from model k-1 (cells with p>thr or p<1-thr, others ignored; image dropped if its
binarized mask contradicts the decoded regime), retrain from scratch on syn + real folds 1-4 + pseudo test,
evaluate on real fold 0 (never pseudo-labelled). Writes t1/out/pseudo/trustmark/it<k>/{fold0,test}_probs128.npz, res.json"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK / "t1/frozen"), str(TASK / "t1/train"), str(TASK / "t1/common")]
from dec import Dec, Multi, load, batch, loss_fn, predict, C  # noqa
from evaluate import evaluate  # noqa
from train import binarize  # noqa
from regime_guard import guard, load_regimes  # noqa


def ram(d):
    return {k: np.ascontiguousarray(v) for k, v in d.items()}


def evaluate_f0(P0, names, typ, reg):
    res = {}
    for tag, pred in (("raw", {n: binarize(p) for n, p in zip(names, P0)}), ("guard", {n: guard(p, reg[n])[0] for n, p in zip(names, P0)})):
        r = evaluate(pred, names, return_per_image=True); g = defaultdict(list)
        for n in names: g[typ[n]].append(r["per_image"][n])
        res[tag] = {"all": round(r["all"][0], 4), **{k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(g.items())}}
    return res


def pseudo(Pt, tnames, reg, thr):
    lab = (Pt > 0.5).astype(np.uint8)
    conf = (Pt > thr) | (Pt < 1 - thr)
    ign = (~conf).astype(np.uint8); drop = 0
    for i, n in enumerate(tnames):
        if (binarize(Pt[i]).mean() >= 0.5) != reg[n]:
            ign[i] = 1; drop += 1
    kept = 1 - ign.mean()
    return lab, ign, {"dropped_images": drop, "kept_cell_frac": round(float(kept), 4), "pos_frac_kept": round(float((lab * (1 - ign)).sum() / max(1, (1 - ign).sum())), 4)}


def train(S, tr_s, R, tr_r, T, stages, a):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    m = Dec(stages, a.rgb)
    f0, _, _ = batch(S, rng.choice(tr_s, 64, replace=False), stages)
    for s in stages:
        getattr(m, f"mu_{s}").copy_(f0[s].mean((0, 2, 3), keepdim=True)); getattr(m, f"sd_{s}").copy_(f0[s].std((0, 2, 3), keepdim=True) + 1e-3)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    nr = max(1, int(round(a.bs * a.p_real))); nt = int(round(a.bs * a.p_test)); ns = a.bs - nr - nt
    t0 = time.time(); run = 0
    for st in range(a.steps):
        parts = [batch(S, rng.choice(tr_s, ns, replace=False), stages), batch(R, rng.choice(tr_r, nr, replace=True), stages)]
        if nt: parts.append(batch(T, rng.choice(len(T["lab"]), nt, replace=False), stages))
        f = {k: torch.cat([p[0][k] for p in parts]) for k in parts[0][0]}
        y = torch.cat([p[1] for p in parts]); w = torch.cat([p[2] for p in parts])
        loss = loss_fn(m(f), y, w); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        run = 0.95 * run + 0.05 * float(loss) if st else float(loss)
        if st % 100 == 0 or st == a.steps - 1: print(f"  step {st} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    return m, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="t1/out/frozen/trustmark/D_l234_rgb_8k/dec.pt")
    ap.add_argument("--iters", type=int, default=2); ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--p_real", type=float, default=0.2); ap.add_argument("--p_test", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--syn_tags", default="syn,syn2")
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--out", default="t1/out/pseudo/trustmark")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    out = TASK / a.out; out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(TASK / a.init, map_location="cpu"); stages = ck["args"]["stages"].split(","); a.rgb = ck["args"]["rgb"]
    S = Multi([load(t) for t in a.syn_tags.split(",")]); tr_s = np.arange(len(S))
    R = ram(load("real")); T = ram(load("test"))
    rm = json.load(open(C / "real_meta.json")); tnames = json.load(open(C / "test_meta.json"))["names"]
    tr_r = np.array([i for i, f in enumerate(rm["fold"]) if f != 0]); va_r = np.array([i for i, f in enumerate(rm["fold"]) if f == 0])
    names0 = [rm["names"][i] for i in va_r]; typ = dict(zip(rm["names"], rm["type"]))
    reg_tr, reg_te = load_regimes("train"), load_regimes("test")
    m = Dec(stages, a.rgb); m.load_state_dict(ck["model"])
    Pt = predict(m, T, stages, len(tnames)); Pr = predict(m, R, stages, len(rm["names"]))
    hist = {"it0_init": evaluate_f0(Pr[va_r], names0, typ, reg_tr)}
    print("it0", json.dumps(hist["it0_init"]), flush=True)
    for it in range(1, a.iters + 1):
        lab, ign, st = pseudo(Pt, tnames, reg_te, a.thr); T["lab"], T["ign"] = lab, ign
        print(f"it{it} pseudo {st}", flush=True)
        m, ts = train(S, tr_s, R, tr_r, T, stages, a)
        Pr = predict(m, R, stages, len(rm["names"])); Pt = predict(m, T, stages, len(tnames))
        res = evaluate_f0(Pr[va_r], names0, typ, reg_tr); res["pseudo"] = st; res["train_s"] = round(ts)
        pt = {rm["names"][i]: binarize(Pr[i]) for i in tr_r}; res["real_train_folds_raw"] = round(evaluate(pt, list(pt))["all"][0], 4)
        hist[f"it{it}"] = res; print(f"it{it}", json.dumps(res), flush=True)
        d = out / f"it{it}"; d.mkdir(exist_ok=True)
        torch.save({"model": m.state_dict(), "args": {**vars(a), "stages": ",".join(stages)}}, d / "dec.pt")
        np.savez(d / "fold0_probs128.npz", names=np.array(names0), probs=Pr[va_r].astype(np.float16))
        np.savez(d / "test_probs128.npz", names=np.array(tnames), probs=Pt.astype(np.float16))
        json.dump(hist, open(out / "res.json", "w"), indent=1)
    (out / "done.txt").write_text("ok\n")


if __name__ == "__main__":
    main()
