"""Update surgery: edit the weight change of a finished unlearning run instead of training again.

  Delta = theta_ckpt - theta_orig.  A variant is  theta = theta_orig + sum_terms c * scale_group(name) * mask(name) * Delta.

Masks (elementwise weights on a Delta):
  topk     keep the top `frac` of |Delta| (scope global or per tensor)
  fisher   diagonal Fisher of the ORIGINAL model's T=2 output (labels sampled from the model) on forget images
           (clean + alter) F_f and on proxies F_p (composites, ret2 composites, smooth noise, retain_synth bank);
           r = (F_f / mean F_f) / (F_p / mean F_p + eps), weight = min(1, (r / alpha) ** power)  (power 0 = hard)
  ig       integrated-gradients attribution of each element of Delta along theta_orig -> theta_ckpt:
           A_f = Delta * mean_t grad(forget CE vs pseudo-label)   (sums to the CE increase of the checkpoint)
           A_p = Delta * mean_t grad(proxy KL(orig || model) T=2)  (sums to the proxy drift of the checkpoint)
           keep where A_f - lam * A_p > thr   (elements that forget more than they drift)
  agree    keep where sign(Delta) == sign(Delta_other)
  proj     remove the per-tensor component of Delta along another direction (e.g. ep6 - ep2)
Groups: embed, bXX.{attn,mlp,norm,ls}, norm, head; a term's "groups" maps a regex (fullmatch on the group name)
to a scale (first match wins, default "default" or 1).
"calib": a target clean CE: the variant's global multiplier is bisected so the forget clean CE mean hits it.

  python surgery_run.py --delta ep2=runs/cal_lr1e-5/ckpt_ep02.pt --specs runs/surgery/specs_a.json \
         --imp ep2 --bench screen --log runs/surgery/a.jsonl
Specs: a JSON list of {"name", "terms": [{"d", "c", "groups", "mask"}], "calib", "save"}; or a builtin name
("groups:ep2" = every group alone and removed). Saved variants go to runs/surgery/<name>.pt (check_submission).
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment import alter, composite, composite_alt, smooth_noise, strong, strong_alt
from common import (FORGET_IMAGES_FILE, MODEL_FILE, build_model, check_submission, get_device, kl_T,
                    load_forget_images, preprocess, server_logits)

OUT = Path("runs/surgery")
BENCH = {  # n_forget, n_ret, n_ret2, n_noise, n_bank
    "screen": (768, 768, 384, 256, 256),
    "full": (3500, 2048, 1024, 512, 256),
}


def group_of(name: str) -> str:
    if name.startswith(("cls_token", "pos_embed", "patch_embed")):
        return "embed"
    if name.startswith("blocks."):
        _, b, part = name.split(".")[:3]
        kind = {"norm1": "norm", "norm2": "norm", "attn": "attn", "mlp": "mlp", "ls1": "ls", "ls2": "ls"}[part]
        return f"b{int(b):02d}.{kind}"
    return "norm" if name.startswith("norm") else "head"


GROUPS = ["embed"] + [f"b{b:02d}.{k}" for b in range(12) for k in ("attn", "mlp", "norm", "ls")] + ["norm", "head"]


# ------------------------------------------------------------------------------------------------- bench
class Bench:
    """Fixed evaluation inputs (stored fp16 on GPU) and the original model's logits on them."""

    def __init__(self, teacher, images, sizes, bank_paths, target_path, seed=4321, bs=128, perm_seed=0, bank_offset=0):
        n_f, n_r, n_r2, n_n, n_b = sizes
        dev = images.device
        self.bs = bs
        g = torch.Generator(device=dev).manual_seed(seed)
        perm = torch.randperm(len(images), generator=torch.Generator().manual_seed(perm_seed))[:n_f].to(dev)
        self.f_idx = perm
        inp = {"clean": [], "alt": [], "ret": [], "ret2": [], "noise": [], "bank": []}
        for i in range(0, n_f, bs):
            x = preprocess(images[perm[i:i + bs]])
            inp["clean"].append(x.half())
            inp["alt"].append(alter(x, g).half())
        for kind, n in (("ret", n_r), ("ret2", n_r2)):
            for _ in range(max(1, n // bs)):
                j = torch.randint(0, len(images), (bs,), generator=g, device=dev)
                x = strong(preprocess(images[j]), g) if kind == "ret" else strong_alt(preprocess(images[j]), g)
                inp[kind].append((composite(x, g) if kind == "ret" else composite_alt(x, g)).half())
        for _ in range(max(1, n_n // 64)):
            inp["noise"].append(smooth_noise(64, g, dev).half())
        bank = [np.load(p)["images"] for p in bank_paths if Path(p).exists()]
        if bank and n_b:
            bank = torch.from_numpy(np.concatenate(bank)[bank_offset:bank_offset + n_b]).to(dev)
            for i in range(0, len(bank), bs):
                inp["bank"].append(preprocess(bank[i:i + bs]).half())
        self.inp = {k: v for k, v in inp.items() if v}
        self.ref = {k: self.logits(teacher, k) for k in self.inp}
        self.labels = self.ref["clean"].argmax(1)
        self.target = None
        if target_path and Path(target_path).exists():
            probs = torch.load(target_path, map_location="cpu", weights_only=False)["probs"].float()
            self.target = probs.clamp_min(1e-8).log().to(dev)[perm]
        print("bench:", {k: sum(len(x) for x in v) for k, v in self.inp.items()}, flush=True)

    def logits(self, model, kind):
        return torch.cat([server_logits(model, x.float()) for x in self.inp[kind]])

    @torch.no_grad()
    def forget_ce(self, model, kind="clean"):
        """Forget CE mean on clean, alt, or "mean" (average of both)."""
        if kind == "mean":
            return 0.5 * (self.forget_ce(model, "clean") + self.forget_ce(model, "alt"))
        return F.cross_entropy(self.logits(model, kind), self.labels).item()

    @torch.no_grad()
    def __call__(self, model) -> dict:
        out = {}
        for kind in ("clean", "alt"):
            z = self.logits(model, kind)
            ce = F.cross_entropy(z, self.labels, reduction="none")
            out[f"{kind}_ce"], out[f"{kind}_std"] = ce.mean().item(), ce.std().item()
            out[f"{kind}_acc"] = (z.argmax(1) == self.labels).float().mean().item()
            out[f"{kind}_kl"] = kl_T(self.ref[kind], z).mean().item()
            if self.target is not None:
                out[f"{kind}_tkl"] = kl_T(self.target, z).mean().item()
        for kind in ("ret", "ret2", "noise", "bank"):
            if kind in self.inp:
                z = self.logits(model, kind)
                out[f"{kind}_kl"] = kl_T(self.ref[kind], z).mean().item()
                out[f"{kind}_agree"] = (z.argmax(1) == self.ref[kind].argmax(1)).float().mean().item()
        return out


HDR = (f"{'name':<34} {'mult':>5} | {'cl ce':>6} {'std':>5} {'acc':>5} {'tkl':>5} | {'al ce':>6} {'std':>5} {'acc':>5} | "
       f"{'ret kl':>6} {'agr':>5} | {'ret2':>6} {'agr':>5} | {'noise':>6} {'agr':>5} | {'bank':>6} {'agr':>5} | {'frac':>5}")


def fmt(name, m):
    g = lambda k: m.get(k, float("nan"))
    return (f"{name:<34} {g('mult'):5.2f} | {g('clean_ce'):6.3f} {g('clean_std'):5.2f} {g('clean_acc'):5.3f} "
            f"{g('clean_tkl'):5.3f} | {g('alt_ce'):6.3f} {g('alt_std'):5.2f} {g('alt_acc'):5.3f} | "
            f"{g('ret_kl'):6.3f} {g('ret_agree'):5.3f} | {g('ret2_kl'):6.3f} {g('ret2_agree'):5.3f} | "
            f"{g('noise_kl'):6.3f} {g('noise_agree'):5.3f} | {g('bank_kl'):6.3f} {g('bank_agree'):5.3f} | "
            f"{g('keep_frac'):5.3f}")


# ------------------------------------------------------------------------------------------------- importances
def _set(model, sd0, delta=None, t=0.0):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(sd0[n])
            if delta is not None and t:
                p.add_(delta[n], alpha=t)


def _grad_pass(model, batches, loss_fn, scale=1024.0):
    """Sum over batches of grad(loss_fn) (each loss already divided by the total count). fp16 + loss scaling."""
    params = dict(model.named_parameters())
    acc = {n: torch.zeros_like(p) for n, p in params.items()}
    for x, aux in batches:
        s = scale
        while True:
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = model(x.float()).float()
            (loss_fn(z, aux) * s).backward()
            if all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None):
                break
            s /= 16
            if s < 1e-3:
                raise RuntimeError("non-finite gradients")
        for n, p in params.items():
            if p.grad is not None:
                acc[n].add_(p.grad, alpha=1 / s)
    model.zero_grad(set_to_none=True)
    return acc


def _forget_batches(bench, n, bs):
    xs = torch.cat([torch.cat(bench.inp["clean"]), torch.cat(bench.inp["alt"])])
    ys = torch.cat([bench.labels, bench.labels])
    sel = torch.randperm(len(xs), generator=torch.Generator().manual_seed(1))[:n].to(xs.device)
    return [(xs[sel[i:i + bs]], ys[sel[i:i + bs]]) for i in range(0, len(sel), bs)]


def _proxy_batches(bench, n, bs):
    xs, refs = [], []
    for k in ("ret", "ret2", "noise", "bank"):
        if k in bench.inp:
            xs.append(torch.cat(bench.inp[k]))
            refs.append(bench.ref[k])
    xs, refs = torch.cat(xs), torch.cat(refs)
    sel = torch.randperm(len(xs), generator=torch.Generator().manual_seed(2))[:n].to(xs.device)
    return [(xs[sel[i:i + bs]], refs[sel[i:i + bs]]) for i in range(0, len(sel), bs)]


def compute_ig(model, sd0, delta, bench, steps=10, n=512, bs=32):
    fb, pb = _forget_batches(bench, n, bs), _proxy_batches(bench, n, bs)
    nf, npx = sum(len(x) for x, _ in fb), sum(len(x) for x, _ in pb)
    Af = {k: torch.zeros_like(v) for k, v in delta.items()}
    Ap = {k: torch.zeros_like(v) for k, v in delta.items()}
    for i in range(steps):
        t = (i + 0.5) / steps
        _set(model, sd0, delta, t)
        gf = _grad_pass(model, fb, lambda z, y: F.cross_entropy(z, y, reduction="sum") / nf)
        gp = _grad_pass(model, pb, lambda z, r: kl_T(r, z).sum() / npx)
        for k in Af:
            Af[k].addcmul_(gf[k], delta[k], value=1 / steps)
            Ap[k].addcmul_(gp[k], delta[k], value=1 / steps)
        del gf, gp
        print(f"  ig step {i + 1}/{steps} t={t:.2f}", flush=True)
    _set(model, sd0)
    sf, sp = sum(v.sum().item() for v in Af.values()), sum(v.sum().item() for v in Ap.values())
    print(f"  ig totals: forget CE {sf:.4f}  proxy KL {sp:.4f}", flush=True)
    return Af, Ap


def compute_fisher(model, sd0, bench, n=768, bs=8, T=2.0):
    _set(model, sd0)
    gen = torch.Generator(device=bench.labels.device).manual_seed(3)
    def loss(z, _):
        with torch.no_grad():
            y = torch.multinomial(F.softmax(z / T, 1), 1, generator=gen).squeeze(1)
        return F.cross_entropy(z / T, y, reduction="sum")
    out = []
    for batches in (_forget_batches(bench, n, bs), _proxy_batches(bench, n, bs)):
        acc = None
        for x, aux in batches:
            g = _grad_pass(model, [(x, aux)], loss)
            if acc is None:
                acc = {k: v.pow(2) for k, v in g.items()}
            else:
                for k in acc:
                    acc[k].add_(g[k].pow(2))
        out.append(acc)
        print("  fisher pass done", flush=True)
    return out


# ------------------------------------------------------------------------------------------------- variants
class Surgeon:
    def __init__(self, model, sd0, deltas, imps, bench):
        self.model, self.sd0, self.deltas, self.imps, self.bench = model, sd0, deltas, imps, bench
        self._thr = {}

    def delta(self, d):
        if d in self.deltas:
            return self.deltas[d]
        if "-" in d:  # difference of two named deltas, e.g. "ep6-ep2"
            a, b = d.split("-")
            return {k: self.deltas[a][k] - self.deltas[b][k] for k in self.sd0}
        raise KeyError(d)

    def _global_thr(self, dname, frac):
        key = (dname, frac)
        if key not in self._thr:
            D = self.delta(dname)
            flat = torch.cat([v.abs().flatten() for v in D.values()])
            idx = torch.randint(0, len(flat), (8_000_000,), device=flat.device, generator=torch.Generator(device=flat.device).manual_seed(0))
            self._thr[key] = torch.quantile(flat[idx].float(), 1 - frac).item()
        return self._thr[key]

    def mask(self, k, d, m, dname):
        t = m["type"]
        if t == "topk":
            if m.get("scope", "global") == "global":
                return (d.abs() >= self._global_thr(dname, m["frac"])).float()
            kk = max(1, int(round(d.numel() * m["frac"])))
            thr = d.abs().flatten().kthvalue(d.numel() - kk + 1).values if kk < d.numel() else -1
            return (d.abs() >= thr).float()
        if t == "fisher":
            Ff, Fp = self.imps[m.get("imp", "fisher")]
            mf, mp = self._fmeans(m.get("imp", "fisher"))
            r = (Ff[k] / mf) / (Fp[k] / mp + m.get("eps", 1e-2))
            p = m.get("power", 1.0)
            return (r >= m["alpha"]).float() if p == 0 else (r / m["alpha"]).pow(p).clamp(max=1.0)
        if t == "ig":
            Af, Ap = self.imps[m.get("imp", "ig")]
            s = Af[k] - m.get("lam", 1.0) * Ap[k]
            agg = m.get("agg", "elem")
            if agg != "elem" and s.dim() >= 2:  # one decision per output row (or input column) of a weight
                dims = tuple(range(1, s.dim())) if agg == "row" else tuple(i for i in range(s.dim()) if i != 1)
                s = s.sum(dim=dims, keepdim=True).expand_as(s)
            return (s > m.get("thr", 0.0)).float()
        if t == "agree":
            return (torch.sign(d) == torch.sign(self.delta(m["with"])[k])).float()
        raise ValueError(t)

    def _fmeans(self, name):
        key = ("fm", name)
        if key not in self._thr:
            Ff, Fp = self.imps[name]
            nf = sum(v.numel() for v in Ff.values())
            self._thr[key] = (sum(v.sum().item() for v in Ff.values()) / nf, sum(v.sum().item() for v in Fp.values()) / nf)
        return self._thr[key]

    @torch.no_grad()
    def build(self, spec, mult=1.0):
        """Writes the variant into self.model's parameters. Returns the kept share of Delta entries (first term)."""
        _set(self.model, self.sd0)
        params = dict(self.model.named_parameters())
        kept = tot = 0
        for ti, term in enumerate(spec["terms"]):
            D = self.delta(term["d"])
            Dproj = self.delta(term["proj"]) if "proj" in term else None
            c = term.get("c", 1.0) * (mult if term.get("calib", True) else 1.0)
            groups = [(re.compile(r), s) for r, s in term.get("groups", {}).items() if r != "default"]
            default = term.get("groups", {}).get("default", 1.0)
            for k, p in params.items():
                gname = group_of(k)
                s = next((s for r, s in groups if r.fullmatch(gname)), default)
                if s == 0:
                    continue
                d = D[k]
                if Dproj is not None:
                    v = Dproj[k]
                    d = d - (d * v).sum() / (v * v).sum().clamp_min(1e-30) * v
                for m in ([term["mask"]] if isinstance(term.get("mask"), dict) else term.get("mask", [])):
                    w = self.mask(k, d, m, term["d"])
                    d = d * w
                    if ti == 0:
                        kept += (w > 0).sum().item()
                        tot += w.numel()
                p.add_(d, alpha=c * s)
        return kept / tot if tot else 1.0

    def calibrate(self, spec, target, lo=0.3, hi=8.0, iters=8):
        """Geometric bisection of the global multiplier so that the forget CE mean (bench subset, on
        spec["calib_kind"]: clean | alt | mean) = target. spec "lo" / "hi" / "iters" narrow the search."""
        lo, hi, iters = spec.get("lo", lo), spec.get("hi", hi), spec.get("iters", iters)
        kind = spec.get("calib_kind", "clean")
        f = lambda m: (self.build(spec, m), self.bench.forget_ce(self.model, kind))[1]
        flo, fhi = f(lo), f(hi)
        if not (flo <= target <= fhi):
            print(f"  calib: target {target} outside [{flo:.3f}, {fhi:.3f}] for mult [{lo}, {hi}]", flush=True)
            return lo if target < flo else hi
        for _ in range(iters):
            mid = math.sqrt(lo * hi)
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def run(self, spec, log):
        t0 = time.time()
        mult = self.calibrate(spec, spec["calib"]) if spec.get("calib") else spec.get("mult", 1.0)
        frac = self.build(spec, mult)
        m = self.bench(self.model)
        m.update(name=spec["name"], mult=mult, keep_frac=frac, spec=spec, secs=time.time() - t0)
        print(fmt(spec["name"], m), flush=True)
        with open(log, "a") as fh:
            fh.write(json.dumps(m) + "\n")
        if spec.get("save"):
            path = OUT / f"{spec['name']}.pt"
            torch.save({k: v.detach().cpu().float().contiguous() for k, v in self.model.state_dict().items()}, path)
            check_submission(path)
            print(f"  saved + checked {path}", flush=True)
        return m


def ratio_groups(imps, power, cap=3.0, ref=1.5):
    """Group scales s_g = min(cap, (r_g / ref) ** power), r_g = IG forget-CE / proxy-KL attribution of group g."""
    Af, Ap = imps["ig"]
    out = {}
    for g in GROUPS:
        ks = [k for k in Af if group_of(k) == g]
        a, b = sum(Af[k].sum().item() for k in ks), sum(Ap[k].sum().item() for k in ks)
        r = a / b if b > 1e-9 else (cap * ref if a > 0 else 0.0)
        out[re.escape(g)] = round(min(cap, max(r, 0.0) / ref) ** power if power else float(r >= ref), 4)
    return out


def builtin(name, deltas, imps=None):
    kind, d = name.split(":")[:2]
    if kind == "ratio":  # ratio:<delta>:<calib>:<calib_kind>:<ref>:<powers, e.g. 1-2>
        f = name.split(":") + [""] * 6
        cal, ck, ref = float(f[2] or 0.5), f[3] or "clean", float(f[4] or 1.5)
        powers = [float(x) for x in (f[5] or "0-1-2-4").split("-")]
        return [{"name": f"{d}_ratio_r{ref}_p{p:g}_{ck}{cal}", "calib": cal, "calib_kind": ck,
                 "terms": [{"d": d, "groups": ratio_groups(imps, p, ref=ref)}]} for p in powers]
    if kind == "groups":
        specs = []
        for g in GROUPS:
            gg = re.escape(g)
            specs.append({"name": f"{d}:only:{g}", "terms": [{"d": d, "groups": {gg: 1.0, "default": 0.0}}]})
            specs.append({"name": f"{d}:drop:{g}", "terms": [{"d": d, "groups": {gg: 0.0}}]})
        return specs
    if kind == "blocks":  # coarser: block ranges and component types
        specs = []
        for lab, rx in [("embed", "embed"), ("b0-3", r"b0[0-3]\..*"), ("b4-7", r"b0[4-7]\..*"),
                        ("b8-11", r"b(0[89]|1[01])\..*"), ("normhead", "norm|head"), ("attn", r"b\d\d\.attn"),
                        ("mlp", r"b\d\d\.mlp"), ("norms", r"b\d\d\.norm|norm"), ("ls", r"b\d\d\.ls")]:
            specs.append({"name": f"{d}:only:{lab}", "terms": [{"d": d, "groups": {rx: 1.0, "default": 0.0}}]})
            specs.append({"name": f"{d}:drop:{lab}", "terms": [{"d": d, "groups": {rx: 0.0}}]})
        return specs
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delta", action="append", default=[], help="name=checkpoint path (Delta = ckpt - orig)")
    p.add_argument("--specs", nargs="+", default=[], help="JSON spec files or builtin names (groups:<d>, blocks:<d>)")
    p.add_argument("--imp", default="", help="delta name for IG and Fisher importances (cached under runs/surgery/cache)")
    p.add_argument("--fisher", action="store_true")
    p.add_argument("--ig-steps", type=int, default=10)
    p.add_argument("--imp-seed", type=int, default=777, help="seed of the inputs used for IG / Fisher")
    p.add_argument("--ig-n", type=int, default=512)
    p.add_argument("--bench", default="screen", choices=list(BENCH))
    p.add_argument("--bank", nargs="*", default=["runs/retain_synth/bank_t_shuf.npz", "runs/retain_synth/bank_t_inv.npz"])
    p.add_argument("--target", default="runs/theory/theory_C_rec.pt")
    p.add_argument("--log", type=Path, default=OUT / "log.jsonl")
    p.add_argument("--baseline", action="store_true", help="also evaluate each delta at c=1")
    p.add_argument("--only", default="", help="comma-separated spec names to run (default all)")
    args = p.parse_args()

    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cache").mkdir(exist_ok=True)
    dev = get_device()
    sd0 = {k: v.to(dev).float() for k, v in torch.load(MODEL_FILE, map_location="cpu", weights_only=True).items()}
    deltas = {}
    for item in args.delta:
        name, path = item.split("=", 1)
        sd1 = torch.load(path, map_location="cpu", weights_only=True)
        deltas[name] = {k: sd1[k].to(dev).float() - sd0[k] for k in sd0}
        del sd1
    model = build_model().to(dev).eval()
    _set(model, sd0)
    images = load_forget_images(FORGET_IMAGES_FILE).to(dev)
    bench = Bench(model, images, BENCH[args.bench], args.bank, args.target)
    # importances are estimated on OTHER inputs (other seed, other forget subset, other bank images) than the
    # evaluation, otherwise elementwise selections overfit the evaluation proxies
    need = ((args.imp and not (OUT / "cache" / f"ig_{args.imp}_s{args.imp_seed}.pt").exists())
            or (args.fisher and not (OUT / "cache" / f"fisher_orig_s{args.imp_seed}.pt").exists()))
    ibench = Bench(model, images, BENCH["screen"], args.bank, None, seed=args.imp_seed, perm_seed=args.imp_seed,
                   bank_offset=256) if need else None
    del images
    imps = {}
    if args.imp:
        d = args.imp
        cpath = OUT / "cache" / f"ig_{d}_s{args.imp_seed}.pt"
        if cpath.exists():
            Af, Ap = torch.load(cpath, map_location=dev, weights_only=True)
        else:
            Af, Ap = compute_ig(model, sd0, deltas[d], ibench, args.ig_steps, args.ig_n)
            torch.save((Af, Ap), cpath)
        imps["ig"] = (Af, Ap)
        print("IG per group (forget CE attribution, proxy KL attribution, ratio):", flush=True)
        for g in GROUPS:
            ks = [k for k in sd0 if group_of(k) == g]
            a, b = sum(Af[k].sum().item() for k in ks), sum(Ap[k].sum().item() for k in ks)
            print(f"  {g:<10} {a:+.4f} {b:+.4f} {a / b if abs(b) > 1e-9 else float('nan'):+7.2f}", flush=True)
    if args.fisher:
        cpath = OUT / "cache" / f"fisher_orig_s{args.imp_seed}.pt"
        if cpath.exists():
            imps["fisher"] = torch.load(cpath, map_location=dev, weights_only=True)
        else:
            imps["fisher"] = compute_fisher(model, sd0, ibench)
            torch.save(imps["fisher"], cpath)
    surgeon = Surgeon(model, sd0, deltas, imps, bench)
    print(HDR, flush=True)
    m = bench(model)
    m.update(name="orig", mult=0.0, keep_frac=0.0)
    print(fmt("orig", m), flush=True)
    if args.baseline:
        for d in deltas:
            surgeon.run({"name": f"{d}:raw", "terms": [{"d": d}]}, args.log)
    for s in args.specs:
        specs = json.load(open(s)) if s.endswith(".json") else builtin(s, deltas, imps)
        for spec in specs:
            if args.only and spec["name"] not in args.only.split(","):
                continue
            try:
                surgeon.run(spec, args.log)
            except Exception as exc:
                print(f"{spec['name']}: FAILED {exc!r}", flush=True)


if __name__ == "__main__":
    main()
