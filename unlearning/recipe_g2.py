"""G2 recipe: twin validation (score of the target as if it were the submission) + real-model target files.

  python recipe_g2.py twin            # table on the twin
  python recipe_g2.py real            # writes runs/recipe/real_G2.pt, real_G2hi.pt (+ prints realised stats)
"""
import sys

import torch
import torch.nn.functional as F

from recipe_fit import FAMS, build, ce64, decompose, describe, fmt, load_twin, score

torch.set_num_threads(4)
BASE = {"sig": "ent", "within": False, "beta": 0.25, "tau": 1.0, "floor": 1e-3}


def twin():
    tw = load_twin()
    y = tw["y"]
    print("twin gold on F (clean): ", {k: round(v, 3) for k, v in score(tw, tw["O"]["F"]).items() if k in ("gm", "gs")})
    th = tw["D"]["theory"]
    k0 = next(iter(th))
    print("theory entry type:", type(th[k0]), getattr(th[k0], "shape", None))
    rows = []
    for name in ("th_C_rec", "th_A_mild", "th_E_strong", "th_x1.0", "th_x0.5", "th_x1.0_oracle"):
        if name in th:
            t = th[name]
            t = t["logits"] if isinstance(t, dict) else t
            t = t.float()
            if (t >= 0).all():
                t = t.clamp_min(1e-8).log()
            t = F.log_softmax(t, 1)
            rows.append((name, score(tw, t)))
    rows.append(("orig itself", score(tw, tw["O"]["F"])))
    gl = torch.stack([g["F"] for g in tw["G"]])
    rows.append(("gold avg logits (upper ref)", score(tw, gl.mean(0))))
    rows.append(("margin L3 (ln shape, ent)", score(tw, build(tw, "margin", {"mean": 0.5, "std": 1.5, "err": 0.12}))))
    for r in rows:
        print(fmt(*r))
    print("\n--- G2 = lognormal(mean, std) by difficulty, beta tilt, tau 1, floor 1e-3 ---")
    best = None
    for sig in ("ce", "ent"):
        for beta in (0.0, 0.25):
            for m in (0.2, 0.3, 0.4, 0.53, 0.65):
                for ratio in (1.5, 1.8, 2.2):
                    P = dict(BASE, sig=sig, beta=beta, mean=m, std=m * ratio)
                    s = score(tw, build(tw, "ln", P))
                    print(fmt(f"ln {sig} b{beta} m{m} sd{m * ratio:.2f}", s), flush=True)
                    if best is None or s["forget"] > best[1]["forget"]:
                        best = (P, s)
    print("\nBEST", best[0], fmt("", best[1]))
    P = best[0]
    for name, Q in (("best", P), ("same, alt via recipe", P)):
        t = build(tw, "ln", Q)
        d = decompose(tw, t)
        right = (F.softmax(torch.stack([g["F"] for g in tw["G"]]) / 2, -1).mean(0).argmax(1) == y)
        print(f"{name}: KL {d['kl'].mean():.3f} = binary {d['bin'].mean():.3f} + rest {d['rest'].mean():.3f} | gold-right imgs "
              f"KL {d['kl'][right].mean():.3f} (bin {d['bin'][right].mean():.3f}) | gold-wrong KL {d['kl'][~right].mean():.3f} "
              f"(bin {d['bin'][~right].mean():.3f}, rest {d['rest'][~right].mean():.3f})")
        break
    from recipe_fit import evaluate
    print(fmt("best, alt target = recipe(alt logits)", evaluate(tw, "ln", P, alt_mode="recipe")))
    for k in range(3):
        gi = tuple(j for j in range(3) if j != k)
        print(fmt(f"best vs golds {gi}", score(tw, build(tw, "ln", P), gold_idx=gi)))


def real():
    d = torch.load("runs/diag/diag.pt", map_location="cpu", weights_only=False)
    z, y = d["clean_logits"].float(), d["pseudo_labels"].long()
    cc = torch.bincount(y, minlength=100).float()
    tw = {"y": y, "ccount": cc, "O": {"F": z}}
    Lc = ce64(z, y)
    print(f"real clean CE: mean {Lc.mean():.2e} median {Lc.median():.2e} min {Lc.min():.2e} | distinct values {len(Lc.unique())}/{len(Lc)}")
    from recipe_fit import signal
    for s in ("ce", "ent", "margin"):
        v = signal(z, y, s)
        print(f"  signal {s}: quantiles {[f'{q:.3g}' for q in torch.quantile(v, torch.tensor([0., .1, .5, .9, 1.], dtype=v.dtype)).tolist()]} ties {len(v) - len(v.unique())}")
    files = {"G2": dict(BASE, sig="ent", mean=0.5, std=1.4), "G2hi": dict(BASE, sig="ent", mean=0.65, std=1.7)}
    if len(sys.argv) > 2:
        import json
        files = {sys.argv[2]: dict(BASE, **json.loads(sys.argv[3]))}
    for name, P in files.items():
        t = FAMS["ln"](z, y, cc, P)
        p = t.double().exp()
        p = p / p.sum(1, keepdim=True)
        st = describe(p.clamp_min(1e-8).log().float(), y)  # what targets.py sees after its 1e-8 clamp
        info = (f"real_{name}: recipe_fit 'ln' family {P}: T=1 loss L_i = lognormal(mean, std) quantiles by difficulty = "
                f"normal score of clean {P['sig']} rank + beta*z(log class forget-count); other classes softmax(clean logits/tau) "
                f"+ floor. Realised CE {st['ce']:.3f} std {st['std']:.3f} acc {st['acc']:.3f} T2-entropy {st['ent2']:.2f}")
        torch.save({"probs": p.float(), "info": info, "pseudo_labels": y, "params": P}, f"runs/recipe/real_{name}.pt")
        print(info)


if __name__ == "__main__":
    {"twin": twin, "real": real}[sys.argv[1]]()
