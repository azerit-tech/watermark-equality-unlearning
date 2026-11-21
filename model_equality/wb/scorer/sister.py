"""sister: matched filter from a sibling LoRA, joint MLE of the strength s and the temperature T.

Idea (research/VEILLE.md, idea 1). In a family the same adapter is served at several strengths. Distil it
from a STRONG sibling (loradir.py train, H0 half of the sibling only), then for every other endpoint fit
    l(s, T) = sum over response tokens of log p_{B + s.dW^, T}(x_t | context)
on a grid, profile T out, and test s = 0 with the profiled likelihood ratio (invariant to T):
    L = max_{s,T} l - max_T l(0, T),   z_LLR = sign(s^) sqrt(2 L),   z_boot = s^ / sd_bootstrap(s^),
    z = min(z_LLR, z_boot); flagged iff z >= 4 (rules_r5_sister.txt).
The signal is first order in s (likelihood ratio along the right direction), not s^2.

Usage (cluster, login node, CUDA_VISIBLE_DEVICES=1):
  python sister.py split --responses SNAP --endpoint ref-07 --half 0 --out H0.jsonl
  python loradir.py train --responses H0.jsonl --strong ref-07:1.0 --max-rows 0 --out ADIR [...]
  python sister.py score --model M --adapter ADIR --responses SNAP [LOCAL.jsonl] --out ODIR [--endpoints ...]
  python sister.py test ODIR --sister ref-07 [--half 1] [--cats nondiag|all] [--parity 45]
Nothing in mescore or loradir is modified; their helpers are imported.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from loradir import FLAG_ALT, half_of, cat, safe  # noqa: E402

S_GRID = [round(-0.6 + 0.1 * i, 2) for i in range(33)]          # -0.6 .. 2.6
T_GRID = [round(0.50 + 0.01 * i, 2) for i in range(81)]         # 0.50 .. 1.30
LURES = ["ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08", "LOCAL"]


# ------------------------------------------------------------------ split
def cmd_split(a):
    from mescore.data import _norm
    n = 0
    with open(a.out, "w") as o:
        for p in a.responses:
            for line in open(p):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r = _norm(raw)
                if r is None or r["endpoint"] != a.endpoint or half_of(r["id"]) != a.half:
                    continue
                o.write(json.dumps(raw) + "\n")
                n += 1
    print(f"split: {a.endpoint} half {a.half}: {n} rows -> {a.out}")


# ------------------------------------------------------------------ score (GPU)
def score_rows(model, tok, scaler, hyp, rows, sgrid, tgrid, batch_tokens, device):
    import torch
    from loradir import make_items, batches, resp_logits
    items = make_items(tok, hyp, rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    R, S, M = len(rows), len(sgrid), len(tgrid)
    LL = np.zeros((R, S, M), np.float64)
    ntok = np.zeros(R, np.int64)
    nalt = np.zeros(R, np.int64)
    pieces = {}

    def piece(i):
        s = pieces.get(i)
        if s is None:
            s = pieces[i] = tok.decode([int(i)], skip_special_tokens=False)
        return s

    Tt = torch.tensor(tgrid, device=device, dtype=torch.float32)
    lengths = [len(it["p"]) + len(it["ids"]) for it in items]
    with torch.inference_mode():
        for bi in batches(lengths, batch_tokens):
            bi = [i for i in bi if len(items[i]["ids"]) > 0]
            if not bi:
                continue
            seqs = [(items[i]["p"], items[i]["ids"]) for i in bi]
            y = torch.tensor([t for _, ids in seqs for t in ids], device=device)
            rowidx = torch.tensor([j for j, (_, ids) in enumerate(seqs) for _ in ids], device=device)
            scaler.set(0.0)
            LB = resp_logits(model, pad, seqs, device)
            lo = LB.gather(1, y[:, None])[:, 0]
            rank = ((LB > lo[:, None]).sum(-1) + 1).cpu().numpy()
            top1 = LB.argmax(-1).cpu().numpy()
            keep = np.ones(len(rank), np.float32)
            k = 0
            for j, i in enumerate(bi):
                ids = items[i]["ids"]
                n = len(ids)
                for t in np.nonzero(rank[k:k + n] > 1)[0]:
                    s1 = piece(top1[k + t])
                    if s1 and "�" not in s1 and "".join(piece(q) for q in ids[t:t + 8]).startswith(s1):
                        keep[k + t] = 0.0
                ntok[i] = int(keep[k:k + n].sum())
                nalt[i] = n - ntok[i]
                k += n
            keep_t = torch.tensor(keep, device=device)
            out = torch.zeros((len(bi), S, M), device=device, dtype=torch.float64)
            for si, s in enumerate(sgrid):
                if s == 0.0:
                    L = LB
                else:
                    scaler.set(s)
                    L = resp_logits(model, pad, seqs, device)
                ly = L.gather(1, y[:, None])[:, 0]
                for m in range(M):
                    lp = (ly / Tt[m] - torch.logsumexp(L / Tt[m], -1)) * keep_t
                    out[:, si, m].index_add_(0, rowidx, lp.double())
                del L
            scaler.set(0.0)
            LL[bi] = out.cpu().numpy()
            del LB, out
    meta = [{"id": r["id"], "endpoint": r["endpoint"], "prompt_id": r.get("prompt_id"), "half": half_of(r["id"]),
             "cat": cat(r.get("prompt_id")), "finish_eff": it["eff"], "n_tok": int(len(it["ids"]))}
            for r, it in zip(rows, items)]
    return LL, ntok, nalt, meta


def cmd_score(a):
    import torch
    from loradir import _limit_mem, build, Scaler, make_hyp
    from mescore.data import load_responses
    _limit_mem(a.mem_gb)
    model, tok = build(a.model, a.dtype, a.device, adapter=a.adapter)
    model.eval()
    scaler = Scaler(model)
    hyp = make_hyp(a.model, a.system_prompt, a.chat_template)
    groups = load_responses(a.responses, set(a.endpoints.split(",")) if a.endpoints else None)
    os.makedirs(a.out, exist_ok=True)
    for ep, rows in groups.items():
        name = "LOCAL" if ep.startswith("local:") else ep
        p = os.path.join(a.out, safe(name) + ".npz")
        if os.path.exists(p[:-4] + ".json") and not a.force:
            continue
        t0 = time.time()
        LL, ntok, nalt, meta = score_rows(model, tok, scaler, hyp, rows, S_GRID, T_GRID, a.batch_tokens, a.device)
        np.savez(p + ".tmp.npz", LL=LL, ntok=ntok, nalt=nalt)
        os.replace(p + ".tmp.npz", p)
        json.dump({"endpoint": ep, "model": a.model, "adapter": a.adapter, "dtype": a.dtype, "s_grid": S_GRID,
                   "t_grid": T_GRID, "rows": meta, "seconds": round(time.time() - t0, 1), "torch": torch.__version__},
                  open(p[:-4] + ".json", "w"))
        print(f"score: {name}: rows={len(rows)} tok={int(ntok.sum())} alt={int(nalt.sum())} "
              f"{time.time() - t0:.0f}s -> {p}", flush=True)


# ------------------------------------------------------------------ test (numpy only)
def para(y):
    """Parabolic refinement of the max along the last axis (uniform grid). Returns (value, fractional index)."""
    K = y.shape[-1]
    i = np.clip(np.argmax(y, -1), 1, K - 2)
    y0 = np.take_along_axis(y, (i - 1)[..., None], -1)[..., 0]
    y1 = np.take_along_axis(y, i[..., None], -1)[..., 0]
    y2 = np.take_along_axis(y, (i + 1)[..., None], -1)[..., 0]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den < 0, 0.5 * (y0 - y2) / den, 0.0)
    d = np.clip(d, -1.0, 1.0)
    v = np.maximum(y1 - 0.25 * (y0 - y2) * d, np.max(y, -1))
    return v, i + d


def profile(LLsum, sgrid, tgrid):
    """LLsum [..., S, M] -> dict of s_hat, T_hat, T0_hat, Lambda (arrays of shape [...])."""
    ds, dt = sgrid[1] - sgrid[0], tgrid[1] - tgrid[0]
    vT, iT = para(LLsum)                     # [..., S]
    v, iS = para(vT)                         # [...]
    s0 = sgrid.index(0.0)
    s_hat = sgrid[0] + iS * ds
    near = np.clip(np.rint(iS).astype(int), 0, len(sgrid) - 1)
    T_hat = tgrid[0] + np.take_along_axis(iT, near[..., None], -1)[..., 0] * dt
    lam = np.maximum(v - vT[..., s0], 0.0)
    return {"s_hat": s_hat, "T_hat": T_hat, "T0_hat": tgrid[0] + iT[..., s0] * dt, "Lambda": lam,
            "edge_s": (iS <= 0.5) | (iS >= len(sgrid) - 1.5)}


class EP:
    def __init__(self, path):
        z = np.load(path)
        self.meta = json.load(open(path[:-4] + ".json"))
        self.LL, self.ntok = z["LL"], z["ntok"]
        self.sgrid, self.tgrid = self.meta["s_grid"], self.meta["t_grid"]
        rows = self.meta["rows"]
        self.half = np.array([r["half"] for r in rows])
        self.cat = np.array([r["cat"] for r in rows])


def decide(e: EP, sel, n_boot, rng):
    idx = np.nonzero(sel)[0]
    R = len(idx)
    if R < 5:
        return {"rows": R}
    X = e.LL[idx]
    P = profile(X.sum(0), e.sgrid, e.tgrid)
    s_hat, lam = float(P["s_hat"]), float(P["Lambda"])
    W = rng.multinomial(R, np.full(R, 1.0 / R), size=n_boot).astype(np.float64)
    Xb = np.einsum("br,rk->bk", W, X.reshape(R, -1)).reshape((n_boot,) + X.shape[1:])
    sb = profile(Xb, e.sgrid, e.tgrid)["s_hat"]
    sd = float(sb.std(ddof=1))
    z_llr = math.copysign(math.sqrt(2 * lam), s_hat)
    z_boot = s_hat / sd if sd > 0 else float("nan")
    return {"rows": R, "tok": int(e.ntok[idx].sum()), "s_hat": round(s_hat, 3), "sd_boot": round(sd, 3),
            "T_hat": round(float(P["T_hat"]), 3), "T0_hat": round(float(P["T0_hat"]), 3),
            "Lambda": round(lam, 2), "z_LLR": round(z_llr, 2), "z_boot": round(z_boot, 2),
            "z": round(min(z_llr, z_boot), 2), "edge_s": bool(P["edge_s"]), "flag": bool(min(z_llr, z_boot) >= 4)}


def profile_ctx(X, sgrid, tgrid):
    """X [..., C, S, M] (contexts) -> s_hat, Lambda profiled over (T, context) under both hypotheses."""
    vT, iT = para(X)                          # [..., C, S]
    v, iS = para(vT)                          # [..., C]
    s0 = sgrid.index(0.0)
    best = np.argmax(v, -1)
    s_hat = sgrid[0] + np.take_along_axis(iS, best[..., None], -1)[..., 0] * (sgrid[1] - sgrid[0])
    lam = np.maximum(v.max(-1) - vT[..., s0].max(-1), 0.0)
    return {"s_hat": s_hat, "Lambda": lam, "ctx": best, "ctx0": np.argmax(vT[..., s0], -1)}


def decide_ctx(es, names, sel, n_boot, rng):
    """Descriptive: same statistic, the hidden context profiled over the scored contexts (rows aligned by id)."""
    idx = np.nonzero(sel)[0]
    R = len(idx)
    if R < 5:
        return {"rows": R}
    X = np.stack([e.LL[idx] for e in es], 1)            # [R,C,S,M]
    P = profile_ctx(X.sum(0), es[0].sgrid, es[0].tgrid)
    s_hat, lam = float(P["s_hat"]), float(P["Lambda"])
    W = rng.multinomial(R, np.full(R, 1.0 / R), size=n_boot).astype(np.float64)
    Xb = np.einsum("br,rk->bk", W, X.reshape(R, -1)).reshape((n_boot,) + X.shape[1:])
    sd = float(profile_ctx(Xb, es[0].sgrid, es[0].tgrid)["s_hat"].std(ddof=1))
    z_llr = math.copysign(math.sqrt(2 * lam), s_hat)
    z_boot = s_hat / sd if sd > 0 else float("nan")
    return {"rows": R, "s_hat": round(s_hat, 3), "sd_boot": round(sd, 3), "Lambda": round(lam, 2),
            "z_LLR": round(z_llr, 2), "z_boot": round(z_boot, 2), "z": round(min(z_llr, z_boot), 2),
            "ctx_hat": names[int(P["ctx"])], "ctx_hat_s0": names[int(P["ctx0"])]}


def cmd_test(a):
    rng = np.random.default_rng(0)
    eps = sorted(f[:-4] for f in os.listdir(a.dir) if f.endswith(".npz") and os.path.exists(os.path.join(a.dir, f[:-4] + ".json")))
    E = {ep: EP(os.path.join(a.dir, ep + ".npz")) for ep in eps}
    res = {"dir": a.dir, "sister": a.sister, "half": a.half, "cats": a.cats, "endpoints": {}}
    for ep, e in E.items():
        catsel = (e.cat != "diag") if a.cats == "nondiag" else np.ones(len(e.cat), bool)
        if ep == a.sister and a.half == 0:
            continue                                   # H0 of the sister = training rows
        sel = catsel & (e.half == a.half)
        r = {"primary": decide(e, sel, a.boot, rng)}
        if a.secondary:
            r["all_cats"] = decide(e, e.half == a.half, a.boot, rng)
            if ep != a.sister:
                r["other_half"] = decide(e, catsel & (e.half != a.half), a.boot, rng)
        if a.ctx:
            names, es = ["default"], [e]
            for spec in a.ctx:
                nm, d = spec.split("=", 1)
                p = os.path.join(d, ep + ".npz")
                if os.path.exists(p) and os.path.exists(p[:-4] + ".json"):
                    ec = EP(p)
                    assert [r["id"] for r in ec.meta["rows"]] == [r["id"] for r in e.meta["rows"]], (ep, nm)
                    names.append(nm); es.append(ec)
            r["ctx_profiled"] = decide_ctx(es, names, sel, a.boot, rng)
            r["ctx_profiled"]["contexts"] = names
        if a.parity:
            fl, zs = [], []
            base = np.nonzero(e.half == a.half)[0]
            for _ in range(a.draws):
                pick = np.zeros(len(e.cat), bool)
                pick[rng.choice(base, size=min(a.parity, len(base)), replace=False)] = True
                d = decide(e, pick & catsel, 200, rng)
                zs.append(d.get("z", float("nan")))
                fl.append(d.get("flag", False))
            zs = np.array(zs)
            r["parity"] = {"n_resp": a.parity, "draws": a.draws, "median_z": round(float(np.nanmedian(zs)), 2),
                           "q05_z": round(float(np.nanquantile(zs, 0.05)), 2), "q95_z": round(float(np.nanquantile(zs, 0.95)), 2),
                           "P_flag": round(float(np.mean(fl)), 3), "P_z_ge3": round(float(np.mean(zs >= 3)), 3)}
        res["endpoints"][ep] = r
    if "LOCAL" in res["endpoints"]:
        L = res["endpoints"]["LOCAL"]["primary"]
        for ep, r in res["endpoints"].items():
            p = r["primary"]
            if ep != "LOCAL" and "s_hat" in p and "s_hat" in L:
                r["z_rel_LOCAL"] = round((p["s_hat"] - L["s_hat"]) / math.hypot(p["sd_boot"], L["sd_boot"]), 2)
    print(json.dumps(res, indent=1))
    print(f"\n{'endpoint':10s} {'rows':>4s} {'s_hat':>6s} {'sd':>6s} {'T_hat':>6s} {'T0':>6s} {'zLLR':>6s} {'zboot':>6s} {'z':>6s} flag",
          file=sys.stderr)
    for ep, r in res["endpoints"].items():
        p = r["primary"]
        if "s_hat" in p:
            print(f"{ep:10s} {p['rows']:4d} {p['s_hat']:6.3f} {p['sd_boot']:6.3f} {p['T_hat']:6.2f} {p['T0_hat']:6.2f} "
                  f"{p['z_LLR']:6.2f} {p['z_boot']:6.2f} {p['z']:6.2f} {'FLAG' if p['flag'] else ''}"
                  f"{' EDGE' if p['edge_s'] else ''}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--responses", nargs="+", required=True); s.add_argument("--endpoint", required=True)
    s.add_argument("--half", type=int, default=0); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_split)
    c = sub.add_parser("score")
    c.add_argument("--model", required=True); c.add_argument("--adapter", required=True)
    c.add_argument("--responses", nargs="+", required=True); c.add_argument("--out", required=True)
    c.add_argument("--endpoints"); c.add_argument("--batch-tokens", type=int, default=3072)
    c.add_argument("--dtype", default="float32"); c.add_argument("--device", default="cuda")
    c.add_argument("--system-prompt", default=None); c.add_argument("--chat-template", default=None)
    c.add_argument("--force", action="store_true"); c.add_argument("--mem-gb", type=float, default=18.0)
    c.set_defaults(fn=cmd_score)
    t = sub.add_parser("test")
    t.add_argument("dir"); t.add_argument("--sister", required=True); t.add_argument("--half", type=int, default=1)
    t.add_argument("--cats", default="nondiag", choices=["nondiag", "all"]); t.add_argument("--boot", type=int, default=2000)
    t.add_argument("--secondary", action="store_true"); t.add_argument("--parity", type=int, default=0)
    t.add_argument("--draws", type=int, default=1000)
    t.add_argument("--ctx", action="append", help="NAME=DIR: same endpoint scored under another context (descriptive)")
    t.set_defaults(fn=cmd_test)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
