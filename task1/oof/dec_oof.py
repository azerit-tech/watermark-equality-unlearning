"""Out-of-fold training of TrustMark frozen-stage decoder, variant D recipe (l2+l3+l4+rgb, 1200 steps, bs 32,
p_real 0.2, lr 2e-3, seed 0). Run from TASK (CPU, one fold at a time):
  OMP_NUM_THREADS=8 python t1/oof/dec_oof.py --fold k
Train = real TrustMark images of folds != k (cache9) + leak-free synthetic samples with src_fold != k (t1/out/oof/cache).
Writes t1/out/oof/trustmark/fold{k}/{dec.pt, oof_probs128.npz (fold k), test_probs128.npz, res.json}"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/frozen"), str(TASK / "t1/synth"), str(TASK / "t1/train"), str(TASK / "t1/common")]
from dec import Dec, batch, loss_fn, predict  # noqa  (t1/frozen/dec.py, read only)

C9 = TASK / "t1/out/frozen/cache9"; CO = TASK / "t1/out/oof/cache"


def load(d, tag):
    return {k: np.load(d / f"{tag}_{k}.npy", mmap_mode="r") for k in ("l2", "l3", "l4", "rgb", "lab", "ign")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--p_real", type=float, default=0.2); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(8); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    stages = ["l2", "l3", "l4"]; out = TASK / "t1/out/oof/trustmark" / f"fold{a.fold}"; out.mkdir(parents=True, exist_ok=True)
    S = load(CO, "syn"); R = load(C9, "real"); T = load(C9, "test")
    rm = json.load(open(C9 / "real_meta.json")); tm = json.load(open(C9 / "test_meta.json")); sm = json.load(open(CO / "syn_meta.json"))
    src = np.array(sm["src_fold"]); assert len(src) == len(S["lab"])
    tr_s = np.where(src != a.fold)[0]
    tr_r = np.array([i for i, f in enumerate(rm["fold"]) if f != a.fold]); va_r = np.array([i for i, f in enumerate(rm["fold"]) if f == a.fold])
    print(f"fold {a.fold}: syn={len(tr_s)} real_tr={len(tr_r)} real_val={len(va_r)}", flush=True)
    m = Dec(stages, 1)
    f0, _, _ = batch(S, rng.choice(tr_s, 64, replace=False), stages)
    for s in stages:
        getattr(m, f"mu_{s}").copy_(f0[s].mean((0, 2, 3), keepdim=True)); getattr(m, f"sd_{s}").copy_(f0[s].std((0, 2, 3), keepdim=True) + 1e-3)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    nr = max(1, int(round(a.bs * a.p_real))); t0 = time.time(); run = 0
    for st in range(a.steps):
        fs, ys, ws = batch(S, rng.choice(tr_s, a.bs - nr, replace=False), stages)
        fr, yr, wr = batch(R, rng.choice(tr_r, nr, replace=True), stages)
        f = {k: torch.cat([fs[k], fr[k]]) for k in fs}; y = torch.cat([ys, yr]); w = torch.cat([ws, wr])
        loss = loss_fn(m(f), y, w); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        run = 0.95 * run + 0.05 * float(loss) if st else float(loss)
        if st % 100 == 0 or st == a.steps - 1: print(f"step {st} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save({"model": m.state_dict(), "args": vars(a)}, out / "dec.pt")
    Pr = predict(m, R, stages, len(rm["names"]))
    np.savez(out / "oof_probs128.npz", names=np.array([rm["names"][i] for i in va_r]), probs=Pr[va_r].astype(np.float16))
    Pt = predict(m, T, stages, len(tm["names"]))
    np.savez(out / "test_probs128.npz", names=np.array(tm["names"]), probs=Pt.astype(np.float16))
    json.dump({"train_s": time.time() - t0, "nsyn": int(len(tr_s)), "nreal": int(len(tr_r))}, open(out / "res.json", "w"))
    (out / "done").write_text("ok\n"); print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
