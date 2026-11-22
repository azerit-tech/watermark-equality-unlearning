"""Level / drift measurement and the empirical law  score ~ 0.047 * level / (1 + KL_ext).

level  = CE mean + CE std on the forget images (label = the original's argmax), averaged over the clean images and
         one altered copy each (pgb_score families, crop/flip-heavy mix, fixed seed)
KL_ext = mean KL(p_orig || p_model) at T=2 on the held-out natural photos (runs/ext/bank_ext.npz, last 10%)
also   KL / agreement with the original on the held-out in-domain synthetic images (bank_t2i split == 1)
law    = 0.047 * level / (1 + KL_ext)   (fitted on E18 = 0.1647 and sM = 0.0673 public); law_t2i uses KL_t2i.
No gold reference is used (the sisters bench is circular for models trained on the sisters' targets).

  python hyb_law.py --orig ../model.pt --forget ../forget_images.npz --t2i runs/pg/bank_t2i.npz \
      --ext runs/ext/bank_ext.npz --cands runs/oracle/E18.pt runs/pgb/repair/e18/ckpt_s*.pt --out runs/pgb/repair/law.jsonl
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pgb_score import (ALT_VERSION, FAMILIES, MIX_WEIGHTS, _file_sig, _sig, load_model, load_u8, logits_u8,
                       make_altered, retain_split)

T = 2.0


def kl_agree(zo, z):
    lo, lz = F.log_softmax(zo / T, 1), F.log_softmax(z / T, 1)
    return (lo.exp() * (lo - lz)).sum(1).mean().item(), (zo.argmax(1) == z.argmax(1)).float().mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orig", type=Path, required=True)
    p.add_argument("--forget", type=Path, required=True)
    p.add_argument("--t2i", type=Path, required=True, help="in-domain synthetic bank (held-out part used)")
    p.add_argument("--ext", type=Path, required=True, help="natural photos bank (held-out = last 10%)")
    p.add_argument("--cands", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=Path("runs/pgb/cache_law"))
    p.add_argument("--coef", type=float, default=0.047)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-ret", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--queue-min", type=float, default=None, help="write a submit_queue note when law >= this")
    p.add_argument("--queue", type=Path, default=Path("runs/submit_queue"))
    p.add_argument("--tag", default="law")
    p.add_argument("--recipe", default="")
    a = p.parse_args()
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    dev = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda")
    f = load_u8(a.forget)[:a.limit]
    n = len(f)
    fam_of = np.random.default_rng((a.seed, 99)).choice(len(FAMILIES), n, p=MIX_WEIGHTS)
    alt = np.empty_like(f)
    for k, fam in enumerate(FAMILIES):
        m = np.nonzero(fam_of == k)[0]
        if len(m):
            alt[m] = make_altered(f[m], fam, a.seed)
    views = {"clean": f, "alt": alt}
    for name, path in (("t2i", a.t2i), ("ext", a.ext)):
        _, ho, _ = retain_split(path, 0.1)
        ho = ho[:a.n_ret]
        with np.load(path, allow_pickle=False) as d:
            views[name] = np.ascontiguousarray(d["images"][ho])
    sig = _sig(_file_sig(a.forget), _file_sig(a.t2i), _file_sig(a.ext), n, a.n_ret, a.seed, ALT_VERSION)
    a.cache.mkdir(parents=True, exist_ok=True)
    print(f"[law] views: {n} forget (clean + altered), {len(views['t2i'])} t2i held-out, {len(views['ext'])} ext held-out", flush=True)

    def logits(path):
        c = a.cache / f"{path.stem}_{_sig(_file_sig(path), sig)}.pt"
        if c.exists():
            return torch.load(c, weights_only=True)
        m = load_model(path, dev)
        out = {k: logits_u8(m, v, dev) for k, v in views.items()}
        tmp = c.with_name(c.name + f".{os.getpid()}.tmp")
        torch.save(out, tmp)
        os.replace(tmp, c)
        return out

    zo = logits(a.orig)
    y = zo["clean"].argmax(1)
    for cand in a.cands:
        z = logits(cand)
        r = {"model": str(cand)}
        for g in ("clean", "alt"):
            ce = F.cross_entropy(z[g], y, reduction="none")
            r.update({f"{g}_ce": ce.mean().item(), f"{g}_std": ce.std().item(), f"{g}_acc": (z[g].argmax(1) == y).float().mean().item()})
        r["level"] = (r["clean_ce"] + r["clean_std"] + r["alt_ce"] + r["alt_std"]) / 2
        r["kl_t2i"], r["agree_t2i"] = kl_agree(zo["t2i"], z["t2i"])
        r["kl_ext"], r["agree_ext"] = kl_agree(zo["ext"], z["ext"])
        r["law"] = a.coef * r["level"] / (1 + r["kl_ext"])
        r["law_t2i"] = a.coef * r["level"] / (1 + r["kl_t2i"])
        print(f"{cand.parent.name + '/' + cand.stem:32s} law {r['law']:.4f} (t2i {r['law_t2i']:.4f}) | level {r['level']:.2f} "
              f"clean {r['clean_ce']:.2f}/{r['clean_std']:.2f}/{r['clean_acc']:.2f} alt {r['alt_ce']:.2f}/{r['alt_std']:.2f} | "
              f"KL_ext {r['kl_ext']:.3f} agree {r['agree_ext']:.3f} | KL_t2i {r['kl_t2i']:.3f} agree {r['agree_t2i']:.3f}", flush=True)
        if a.out:
            with a.out.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
        if a.queue_min is not None and r["law"] >= a.queue_min and "oracle" not in str(cand):
            a.queue.mkdir(parents=True, exist_ok=True)
            note = a.queue / f"{a.tag}_{cand.parent.name}_{cand.stem}.txt"
            note.write_text(f"{cand.resolve()} | {a.recipe} | law 0.047*level/(1+KL_ext) = {r['law']:.4f} (t2i {r['law_t2i']:.4f}); "
                            f"level {r['level']:.2f} (clean CE {r['clean_ce']:.2f}/{r['clean_std']:.2f} acc {r['clean_acc']:.2f}, "
                            f"alt {r['alt_ce']:.2f}/{r['alt_std']:.2f}); KL_ext {r['kl_ext']:.3f} agree {r['agree_ext']:.3f}; "
                            f"KL_t2i {r['kl_t2i']:.3f} agree {r['agree_t2i']:.3f} | holdout 0\n")
            print("[law] queued", note, flush=True)


if __name__ == "__main__":
    main()
