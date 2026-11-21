"""loradir: directional (matched-filter) test for a WEAK LoRA of a known base model.

Idea. For B + s.dW with small s, the non-directional statistics (excess surprise, residual
energy) are second order in s, while the likelihood ratio along the right direction is first
order: z ~ rho * s * sqrt(N * I). The direction is not known, but the same adapter is served at
larger strengths by sibling endpoints. So:

  train   fit a surrogate adapter dW^ on the responses of the STRONG siblings by maximum
          likelihood with tied scales (endpoint e is modelled as B + s_e.dW^, sampled at T),
          e.g. ref-04 <-> 0.7 and ref-07 <-> 1.5. The tested endpoints are never used.
  score   teacher-force any endpoint under B and under B + s.dW^ (s in --scales) and, per token
          and per temperature T, store for every direction g_j(.) (full-vocabulary vector)
              lora   g = log p_{B+s.dW^,T}(.|c) - log p_{B,T}(.|c)
              ctx    g = log p_{B,T}(.|other system prompt, c) - log p_{B,T}(.|c)   (optional)
              T      g = log p_{B,T}(.|c)   (its centred value is minus the excess surprise;
                                             it is the score of the inverse temperature)
          the observed value g_j[x_t], its conditional mean E_p[g_j] and the conditional
          covariance Cov_p(g_i, g_j) under p = p_{B,T}(.|c_t), all exact (full vocabulary).
  test    numpy only. For a target direction a and nuisance directions b (T, optionally ctx):
              D = sum_t (g_a[x_t] - E g_a) - gamma' sum_t (g_b[x_t] - E g_b),
              gamma = (sum Cov_bb)^-1 sum Cov_ba,   Var = sum Cov_aa - gamma' sum Cov_ba
          Under H0 (base weights, this context, some temperature near T) every centred term is
          a martingale increment: D has mean 0 and variance Var whatever the sampled
          trajectory, the prompt mix or the lengths, so z_mart = D / sqrt(Var) needs no
          reference sample. z_emp is the same sum with the empirical row-to-row variance.
          T is the endpoint's own base-likelihood maximiser on the grid (nuisance fitted under
          H0), and the projection on the T direction makes D first-order insensitive to it.
  info    rank probes by the efficient Fisher information per response along the direction
          (white box, computed on base samples), for probe design.

Lures (measured on the bench, rules_r4_loradir.txt). The T projection is exact (T = 1.0 gives z = 0.4)
but a hidden system prompt projects on the LoRA direction as much as a weak LoRA does. Remedy:
  score --profile-ctx --ctx NAME=sys:TEXT ...   each endpoint is scored around the library context
          (default included) that maximises its base likelihood, the others stay as directions;
  test --nuis T,ctx                             T and every context direction are projected out;
  test --cats stream,free                       free-text and random-stream probes only (the lures act
          on the continuation, format and identity probes).
A lure adds a bias proportional to the information, not noise: compare the dose D / Var (strength in
units of the target direction; 0.3 x dose estimated the true strength on the bench), not z alone.
test --dose0 d also reports the test of dose > d.

Usage (cluster, card 0):
  python loradir.py train --model M --responses LEDGER --strong ref-04:0.7,ref-07:1.5 --out ADIR
  python loradir.py score --model M --adapter ADIR --responses LEDGER [...] --out ODIR [--endpoints a,b]
  python loradir.py test  ODIR --target s0.3 [--ref ref-01,LOCAL] [--parity 100]
  python loradir.py info  ODIR --ep LOCAL --target s0.3 [--nuis-ctx] [--lures SIM_a,SIM_b]
Nothing here modifies mescore; its data module is reused for context reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FLAG_EOS, FLAG_ALT, FLAG_TRUNC = 1, 2, 4
TEMPS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1]
ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def half_of(rid: str) -> int:           # same split as analysis_r1.half_of
    return sha(rid) % 2


def cat(pid) -> str:
    """Probe category. 'diag' = identity / system-prompt / persona probes (D*, E01_identity*)."""
    pid = pid or ""
    p = pid.split("_")[0]
    if p[:2] == "SL" or (p[:1] == "S" and p[1:].isdigit()):
        return "stream"
    if p[:1] == "F" and p[1:].isdigit():
        return "form"
    if (p[:1] == "D" and p[1:].isdigit()) or pid.startswith("E01_identity"):
        return "diag"
    if p[:1] == "C" and p[1:].isdigit():
        return "cont"
    return "free"


# ------------------------------------------------------------------ GPU side
def _limit_mem(gb: float):
    import torch
    if gb and torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2 ** 30 / tot), 0)


class Scaler:
    """Multiplies the scaling of every LoRA layer by s (s = 0 gives the base model exactly)."""

    def __init__(self, model):
        self.mods = []
        for m in model.modules():
            sc = getattr(m, "scaling", None)
            if isinstance(sc, dict) and sc:
                self.mods.append((m, dict(sc)))
        if not self.mods:
            raise RuntimeError("no LoRA layer found")

    def set(self, s: float):
        for m, base in self.mods:
            for k, v in base.items():
                m.scaling[k] = v * s


def build(model_id: str, dtype: str, device: str = "cuda", adapter: str | None = None, new_lora: dict | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    elif new_lora:
        from peft import LoraConfig, get_peft_model
        tg = ALL_TARGETS if new_lora["targets"] == "all" else ["q_proj", "v_proj"]
        model = get_peft_model(model, LoraConfig(r=new_lora["r"], lora_alpha=new_lora["r"],
                                                 lora_dropout=new_lora.get("dropout", 0.0), target_modules=tg,
                                                 task_type="CAUSAL_LM"))
    return model, tok


def make_hyp(model_id, system_prompt=None, chat_template=None, template_kwargs=None):
    from mescore.hyp import Hyp
    if isinstance(template_kwargs, str):
        template_kwargs = json.loads(template_kwargs)
    return Hyp(name="loradir", model=model_id, system_prompt=system_prompt, chat_template=chat_template,
               template_kwargs=template_kwargs or {})


def make_items(tok, hyp, rows):
    from mescore.data import prompt_ids, response_ids
    out = []
    for r in rows:
        ids, flags, eff = response_ids(tok, hyp, r)
        out.append({"p": prompt_ids(tok, hyp, r), "ids": ids, "flags": flags, "eff": eff})
    return out


def batches(lengths, budget):
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    cur, cur_max = [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and m * (len(cur) + 1) > budget:
            yield cur
            cur, m = [], lengths[i]
        cur.append(i)
        cur_max = m
    if cur:
        yield cur


def resp_logits(model, pad, seqs, device):
    """seqs: list of (prompt_ids, response_ids). float32 logits [sum n_i, V] of the response tokens."""
    import torch
    L = max(len(p) + len(x) for p, x in seqs)
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, (p, ids) in enumerate(seqs):
        s = p + ids
        x[j, :len(s)] = torch.tensor(s)
        am[j, :len(s)] = 1
    out = model(input_ids=x.to(device), attention_mask=am.to(device), use_cache=False).logits
    return torch.cat([out[j, len(p) - 1:len(p) - 1 + len(ids)] for j, (p, ids) in enumerate(seqs)]).float()


# ------------------------------------------------------------------ train
def cmd_train(a):
    import torch
    import torch.nn.functional as F
    from mescore.data import load_responses
    _limit_mem(a.mem_gb)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    strong = [(x.split(":")[0], float(x.split(":")[1])) for x in a.strong.split(",")]
    groups = load_responses(a.responses, {e for e, _ in strong})
    model, tok = build(a.model, a.dtype, a.device, new_lora={"r": a.rank, "targets": a.targets, "dropout": a.dropout})
    scaler = Scaler(model)
    hyp = make_hyp(a.model, a.system_prompt, a.chat_template, a.template_kwargs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    data, split = {}, {}
    for ep, s in strong:
        rows = sorted(groups[ep], key=lambda r: sha("train|" + r["id"]))
        if a.max_rows:
            rows = rows[:a.max_rows]
        nv = int(round(a.val_frac * len(rows)))
        it = [dict(i, id=r["id"]) for i, r in zip(make_items(tok, hyp, rows), rows)]
        it = [i for i in it if len(i["ids"]) > 0]
        val_ids = {r["id"] for r in rows[:nv]}
        data[ep] = {"s": s, "val": [i for i in it if i["id"] in val_ids], "tr": [i for i in it if i["id"] not in val_ids]}
        split[ep] = {"val": sorted(val_ids), "train": sorted(i["id"] for i in data[ep]["tr"])}
        print(f"train: {ep} scale {s}: {len(data[ep]['tr'])} train rows, {len(data[ep]['val'])} val rows, "
              f"{sum(len(i['ids']) for i in data[ep]['tr'])} train tokens", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)

    def mk_batches(items, shuffle):
        if shuffle:
            # random buckets of similar length: shuffle, then cut by token budget
            idx = list(rng.permutation(len(items)))
            out, cur = [], []
            for i in idx:
                L = max([len(items[k]["p"]) + len(items[k]["ids"]) for k in cur + [i]])
                if cur and L * (len(cur) + 1) > a.batch_tokens:
                    out.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                out.append(cur)
            return out
        return list(batches([len(i["p"]) + len(i["ids"]) for i in items], a.batch_tokens))

    n_steps = a.epochs * sum(len(mk_batches(d["tr"], True)) for d in data.values())
    warm = max(1, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, k / n_steps)))))
    base_cache = {}

    def gain(ep, part):
        """mean per-token log-likelihood gain of B + s.dW^ over B at T, on rows of `part`."""
        d = data[ep]
        tot, n = 0.0, 0
        model.eval()
        with torch.no_grad():
            for bi, b in enumerate(mk_batches(d[part], False)):
                seqs = [(d[part][i]["p"], d[part][i]["ids"]) for i in b]
                y = torch.tensor([t for _, ids in seqs for t in ids], device=a.device)
                key = (ep, part, bi)
                if key not in base_cache:
                    scaler.set(0.0)
                    base_cache[key] = torch.log_softmax(resp_logits(model, pad, seqs, a.device) / a.T, -1).gather(1, y[:, None])[:, 0].sum().item()
                scaler.set(d["s"])
                lp = torch.log_softmax(resp_logits(model, pad, seqs, a.device) / a.T, -1).gather(1, y[:, None])[:, 0].sum().item()
                tot += lp - base_cache[key]
                n += len(y)
        model.train()
        return tot / max(n, 1)

    hist, best, best_state = [], -1e9, None
    t0 = time.time()
    model.train()
    step = 0
    for epoch in range(a.epochs):
        plan = [(ep, b) for ep, d in data.items() for b in mk_batches(d["tr"], True)]
        plan = [plan[i] for i in rng.permutation(len(plan))]
        run = 0.0
        for ep, b in plan:
            d = data[ep]
            seqs = [(d["tr"][i]["p"], d["tr"][i]["ids"]) for i in b]
            y = torch.tensor([t for _, ids in seqs for t in ids], device=a.device)
            scaler.set(d["s"])
            loss = F.cross_entropy(resp_logits(model, pad, seqs, a.device) / a.T, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += float(loss)
            step += 1
        rec = {"epoch": epoch + 1, "train_loss": round(run / len(plan), 4)}
        for ep in data:
            rec[f"val_gain_{ep}"] = round(gain(ep, "val"), 5)
        rec["val_gain"] = round(float(np.mean([rec[f"val_gain_{ep}"] for ep in data])), 5)
        hist.append(rec)
        print(f"train: {json.dumps(rec)} ({time.time() - t0:.0f}s)", flush=True)
        if rec["val_gain"] > best:
            best = rec["val_gain"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
            best_epoch = epoch + 1
    model.load_state_dict(best_state, strict=False)
    scaler.set(1.0)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    json.dump({"args": {k: v for k, v in vars(a).items() if k != "fn"}, "strong": strong, "history": hist, "best_epoch": best_epoch, "best_val_gain": best,
               "split": split, "seconds": round(time.time() - t0, 1)}, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
    print(f"train: best epoch {best_epoch} val_gain {best:.5f} -> {a.out}", flush=True)


# ------------------------------------------------------------------ score
def token_stats(LB, Ld, y, temps, n_lora, chunk=int(os.environ.get("LORADIR_CHUNK", "192"))):
    """LB [n,V] base logits, Ld list of direction logits [n,V], y [n] observed tokens.
    Directions: Ld (log-ratio to base at T) then the T direction (base log-prob at T).
    Returns obs [n,M,J], mean [n,M,J], cov [n,M,J(J+1)/2] (upper triangle, row order), kl1 [n,M,n_lora]."""
    import torch
    n, J, M = LB.shape[0], len(Ld) + 1, len(temps)
    iu = torch.triu_indices(J, J)
    obs = torch.zeros((n, M, J)); mean = torch.zeros((n, M, J))
    cov = torch.zeros((n, M, iu.shape[1])); kl1 = torch.zeros((n, M, n_lora))
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(n, c0 + chunk))
        yy = y[sl]
        for m, T in enumerate(temps):
            lB = torch.log_softmax(LB[sl] / T, -1)
            P = lB.exp()
            G = torch.stack([torch.log_softmax(L[sl] / T, -1) - lB for L in Ld] + [lB], 1)      # [c,J,V]
            obs[sl, m] = G.gather(2, yy[:, None, None].expand(-1, J, 1))[..., 0].cpu()
            mu = (G * P[:, None, :]).sum(-1)
            for j in range(n_lora):                                                             # KL(p_s || p_B)
                kl1[sl, m, j] = ((G[:, j] + lB).exp() * G[:, j]).sum(-1).cpu()
            G = G - mu[:, :, None]
            C = torch.bmm(G * P[:, None, :], G.transpose(1, 2))
            mean[sl, m] = mu.cpu()
            cov[sl, m] = C[:, iu[0], iu[1]].cpu()
            del G, C, P, lB
    return obs.numpy(), mean.numpy(), cov.numpy(), kl1.numpy()


def score_rows(model, tok, scaler, hyp, ctx_hyps, rows, scales, temps, batch_tokens, device):
    import torch
    items = make_items(tok, hyp, rows)
    ctx_items = [make_items(tok, h, rows) for _, h in ctx_hyps]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    res = [None] * len(rows)
    pieces = {}

    def piece(i):
        s = pieces.get(i)
        if s is None:
            s = pieces[i] = tok.decode([int(i)], skip_special_tokens=False)
        return s

    longest = [max([len(it["p"])] + [len(c[i]["p"]) for c in ctx_items]) + len(it["ids"]) for i, it in enumerate(items)]
    with torch.inference_mode():
        for bi in batches(longest, batch_tokens):
            bi = [i for i in bi if len(items[i]["ids"]) > 0]
            if not bi:
                continue
            seqs = [(items[i]["p"], items[i]["ids"]) for i in bi]
            y = torch.tensor([t for _, ids in seqs for t in ids], device=device)
            scaler.set(0.0)
            LB = resp_logits(model, pad, seqs, device)
            Ld = []
            for s in scales:
                scaler.set(s)
                Ld.append(resp_logits(model, pad, seqs, device))
            scaler.set(0.0)
            for c in ctx_items:
                for i in bi:
                    assert c[i]["ids"] == items[i]["ids"]
                Ld.append(resp_logits(model, pad, [(c[i]["p"], c[i]["ids"]) for i in bi], device))
            obs, mean, cov, kl1 = token_stats(LB, Ld, y, temps, len(scales))
            lo = LB.gather(1, y[:, None])[:, 0]
            rank = ((LB > lo[:, None]).sum(-1) + 1).cpu().numpy().astype(np.int32)
            top1 = LB.argmax(-1).cpu().numpy()
            del LB, Ld
            k = 0
            for i in bi:
                ids = items[i]["ids"]
                n = len(ids)
                fl = np.asarray(items[i]["flags"], np.uint8).copy()
                for t in np.nonzero(rank[k:k + n] > 1)[0]:
                    s1 = piece(top1[k + t])
                    if s1 and "�" not in s1 and "".join(piece(q) for q in ids[t:t + 8]).startswith(s1):
                        fl[t] |= FLAG_ALT
                res[i] = {"tok": np.asarray(ids, np.int32), "flags": fl, "rank": rank[k:k + n],
                          "obs": obs[k:k + n], "mean": mean[k:k + n], "cov": cov[k:k + n], "kl1": kl1[k:k + n]}
                k += n
    J, M = len(scales) + len(ctx_hyps) + 1, len(temps)
    empty = {"tok": np.zeros(0, np.int32), "flags": np.zeros(0, np.uint8), "rank": np.zeros(0, np.int32),
             "obs": np.zeros((0, M, J), np.float32), "mean": np.zeros((0, M, J), np.float32),
             "cov": np.zeros((0, M, J * (J + 1) // 2), np.float32), "kl1": np.zeros((0, M, len(scales)), np.float32)}
    res = [r if r is not None else empty for r in res]
    arr = {k: np.concatenate([r[k] for r in res]) for k in empty}
    arr["offsets"] = np.concatenate([[0], np.cumsum([len(r["tok"]) for r in res])]).astype(np.int64)
    rows_meta = [{"id": r["id"], "endpoint": r["endpoint"], "prompt_id": r.get("prompt_id"), "key": r.get("key"),
                  "finish_eff": it["eff"], "n_tok": len(it["ids"])} for r, it in zip(rows, items)]
    return arr, rows_meta


def base_scan(model, tok, scaler, hyp, rows, temps, batch_tokens, device, cats):
    """Base log-likelihood on the temperature grid, summed over the rows whose probe category is in cats."""
    import torch
    items = make_items(tok, hyp, rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    keep = [i for i, r in enumerate(rows) if cat(r.get("prompt_id")) in cats and len(items[i]["ids"]) > 0]
    tot, ntok = np.zeros(len(temps)), 0
    scaler.set(0.0)
    with torch.inference_mode():
        for bi in batches([len(items[i]["p"]) + len(items[i]["ids"]) for i in keep], batch_tokens):
            seqs = [(items[keep[i]]["p"], items[keep[i]]["ids"]) for i in bi]
            y = torch.tensor([t for _, ids in seqs for t in ids], device=device)
            LB = resp_logits(model, pad, seqs, device)
            lo = LB.gather(1, y[:, None])[:, 0]
            for m, T in enumerate(temps):
                tot[m] += float((lo / T - torch.logsumexp(LB / T, -1)).sum())
            ntok += len(y)
            del LB
    return tot, ntok


def safe(ep: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.+=" else "_" for c in ep)


def cmd_score(a):
    import torch
    from mescore.data import load_responses
    _limit_mem(a.mem_gb)
    scales = [float(s) for s in a.scales.split(",")]
    temps = [float(t) for t in a.temps.split(",")] if a.temps else TEMPS
    model, tok = build(a.model, a.dtype, a.device, adapter=a.adapter)
    model.eval()
    scaler = Scaler(model)
    hyp = make_hyp(a.model, a.system_prompt, a.chat_template, a.template_kwargs)
    ctx_hyps = []
    for spec in a.ctx or []:
        name, val = spec.split("=", 1)
        if val.startswith("sys:"):
            ctx_hyps.append((name, make_hyp(a.model, val[4:], a.chat_template, a.template_kwargs)))
        elif val.startswith("tpl:"):
            ctx_hyps.append((name, make_hyp(a.model, None, "file:" + val[4:])))
        elif val.startswith("kw:"):          # same template, other template arguments (e.g. a date string)
            ctx_hyps.append((name, make_hyp(a.model, a.system_prompt, a.chat_template, val[3:])))
        else:
            raise ValueError(spec)
    contexts = [("default", hyp)] + ctx_hyps
    groups = load_responses(a.responses, set(a.endpoints.split(",")) if a.endpoints else None)
    os.makedirs(a.out, exist_ok=True)
    for ep, rows in groups.items():
        p = os.path.join(a.out, safe(a.rename or ep) + ".npz")
        if os.path.exists(p[:-4] + ".json") and not a.force:
            continue
        t0 = time.time()
        if a.max_rows and len(rows) > a.max_rows:        # pseudo-random subset, same for every run
            rows = sorted(rows, key=lambda r: sha("score|" + r["id"]))[:a.max_rows]
        # context profile: the primary context is the one of the library (default included) that maximises the
        # base likelihood of this endpoint over the temperature grid (nuisance fitted under H0, like T); the
        # other contexts stay as nuisance directions around it
        prim, hyp_e, others, ctx_scan = "default", hyp, ctx_hyps, None
        if a.profile_ctx and ctx_hyps:
            ctx_scan = {}
            for name, h in contexts:
                tot, _ = base_scan(model, tok, scaler, h, rows, temps, a.batch_tokens, a.device, a.pcats.split(","))
                ctx_scan[name] = round(float(tot.max()), 3)
            prim = max(ctx_scan, key=ctx_scan.get)
            hyp_e = dict(contexts)[prim]
            others = [(n, h) for n, h in contexts if n != prim]
        dirs = [f"s{s:g}" for s in scales] + [f"ctx:{n}" for n, _ in others] + ["T"]
        heavy, scan = temps, None
        if len(temps) > 1 and not a.all_temps:
            # the full statistics are heavy: keep them at 0.7 and at the endpoint's own T_mle (grid, base
            # likelihood on the rows of --tcats); the T projection absorbs the remaining mismatch
            tot, ntok = base_scan(model, tok, scaler, hyp_e, rows, temps, a.batch_tokens, a.device, a.tcats.split(","))
            t_hat = temps[int(np.argmax(tot))]
            heavy = sorted({0.7, t_hat})
            scan = {"temps": temps, "loglik": [round(float(x), 3) for x in tot], "tokens": ntok, "T_mle": t_hat, "cats": a.tcats}
        arr, rows_meta = score_rows(model, tok, scaler, hyp_e, others, rows, scales, heavy, a.batch_tokens, a.device)
        np.savez(p + ".tmp.npz", **arr)
        os.replace(p + ".tmp.npz", p)
        meta = {"endpoint": ep, "model": a.model, "adapter": a.adapter, "dtype": a.dtype, "temps": heavy, "T_scan": scan, "dirs": dirs,
                "primary_ctx": prim, "ctx_scan": ctx_scan, "scales": scales, "ctx": a.ctx or [], "rows": rows_meta, "seconds": round(time.time() - t0, 1),
                "torch": torch.__version__}
        json.dump(meta, open(p[:-4] + ".json", "w"))
        print(f"score: {ep}: rows={len(rows)} tok={int(arr['offsets'][-1])} ctx={prim} {time.time() - t0:.0f}s -> {p}", flush=True)


# ------------------------------------------------------------------ test (numpy only)
def _rowsum(x, off):
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x.astype(np.float64), axis=0)])
    return c[off[1:]] - c[off[:-1]]


class EP:
    """Row-level sufficient statistics of one scored endpoint (tokens flagged ALT_SEG dropped)."""

    def __init__(self, path):
        z = np.load(path)
        self.meta = json.load(open(path[:-4] + ".json"))
        self.temps, self.dirs = self.meta["temps"], self.meta["dirs"]
        off = z["offsets"]
        keep = ((z["flags"] & FLAG_ALT) == 0).astype(np.float64)
        J = len(self.dirs)
        self.n = _rowsum(keep, off)
        self.A = _rowsum((z["obs"].astype(np.float64) - z["mean"]) * keep[:, None, None], off)     # [R,M,J]
        self.O = _rowsum(z["obs"].astype(np.float64) * keep[:, None, None], off)                     # [R,M,J]
        cu = _rowsum(z["cov"].astype(np.float64) * keep[:, None, None], off)
        iu = np.triu_indices(J)
        self.C = np.zeros(cu.shape[:2] + (J, J))
        self.C[:, :, iu[0], iu[1]] = cu
        self.C[:, :, iu[1], iu[0]] = cu
        nl = z["kl1"].shape[2]
        self.K = _rowsum((z["kl1"].astype(np.float64) - z["mean"][:, :, :nl]) * keep[:, None, None], off)  # Jeffreys
        self.R = len(off) - 1
        rows = self.meta["rows"]
        self.pid = np.array([r["prompt_id"] or "" for r in rows])
        self.cat = np.array([cat(r["prompt_id"]) for r in rows])
        self.half = np.array([half_of(r["id"]) for r in rows])
        self.ids = [r["id"] for r in rows]
        self.rank1 = float((z["rank"] == 1).mean()) if len(z["rank"]) else float("nan")

    def j(self, name):
        return self.dirs.index(name)

    def tidx(self, T):
        return min(range(len(self.temps)), key=lambda i: abs(self.temps[i] - T))

    def t_mle(self, sel):
        return int(np.argmax(self.O[sel][:, :, self.j("T")].sum(0)))


DOSE0 = 0.0          # tolerance of the minimum-effect test (test --dose0)


def ratio(d, n):
    if len(d) < 3 or n.sum() == 0:
        return float("nan"), float("nan")
    mu = d.sum() / n.sum()
    r = d - mu * n
    return mu, math.sqrt((r ** 2).sum() * len(d) / (len(d) - 1)) / n.sum()


def stat(e: EP, sel, target: str, nuis: list, m: int | None = None, sign: float = 1.0) -> dict:
    """Efficient (nuisance-projected) martingale statistic on the rows `sel` (bool mask or indices)."""
    sel = np.asarray(sel)
    if sel.dtype == bool:
        sel = np.nonzero(sel)[0]
    if len(sel) < 3:
        return {"rows": int(len(sel)), "z_mart": float("nan"), "z_emp": float("nan")}
    if m is None:
        m = e.t_mle(sel)
    t, nb = e.j(target), [e.j(x) for x in nuis]
    A, C = e.A[sel][:, m], e.C[sel][:, m]
    Ct = C.sum(0)
    if nb:
        g = np.linalg.solve(Ct[np.ix_(nb, nb)], Ct[nb, t])
        D = A[:, t] - A[:, nb] @ g
        V = C[:, t, t] - 2 * C[:, t][:, nb] @ g + np.einsum("i,rij,j->r", g, C[:, nb][:, :, nb], g)
    else:
        g = np.zeros(0)
        D, V = A[:, t].copy(), C[:, t, t].copy()
    D = sign * D
    n = e.n[sel]
    sd = D.std(ddof=1)
    mu, se = ratio(D, n)
    out = {"rows": int(len(sel)), "tok": int(n.sum()), "T": e.temps[m], "D": float(D.sum()), "var": float(V.sum()),
           "z_mart": float(D.sum() / math.sqrt(V.sum())) if V.sum() > 0 else float("nan"),
           "z_emp": float(D.mean() / (sd / math.sqrt(len(D)))) if sd > 0 else float("nan"),
           "dose": float(D.sum() / V.sum()) if V.sum() > 0 else float("nan"),
           "dose_se": float(1 / math.sqrt(V.sum())) if V.sum() > 0 else float("nan"),
           "dose_se_emp": float(sd * math.sqrt(len(D)) / V.sum()) if V.sum() > 0 else float("nan"),
           "mu": mu, "se": se, "gamma": [float(x) for x in g]}
    if DOSE0 and V.sum() > 0:
        # minimum-effect test: a lure shifts the statistic by a bias proportional to the information, so the
        # null is "dose <= DOSE0" (tolerance measured on simulated lures), not "dose = 0"
        E = D - DOSE0 * V
        sde = E.std(ddof=1)
        out["z_mart_d0"] = float(E.sum() / math.sqrt(V.sum()))
        out["z_emp_d0"] = float(E.mean() / (sde / math.sqrt(len(E)))) if sde > 0 else float("nan")
    out["_D"], out["_V"], out["_n"], out["_m"] = D, V, n, m
    return out


def pub(d: dict, nd=3) -> dict:
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("_")}


def z2(a, b):
    return (a["mu"] - b["mu"]) / math.sqrt(a["se"] ** 2 + b["se"] ** 2)


def profile_s(e: EP, sel):
    """Profile likelihood over the scale grid (T profiled on the grid): s_hat and the gains."""
    jT = e.j("T")
    O = e.O[sel].sum(0)                                   # [M,J]
    base = O[:, jT].max()
    out = {"0": 0.0}
    for d in e.dirs:
        if d.startswith("s"):
            out[d[1:]] = float((O[:, jT] + O[:, e.j(d)]).max() - base)
    return max(out, key=out.get), {k: round(v, 2) for k, v in out.items()}


def parity(e: EP, base_sel, target, nuis, n_rows, n_draw, rng, m=None, sign=1.0):
    """z over random subsets of n_rows responses of the endpoint (all categories), then restricted to base_sel."""
    zm, ze, zd = [], [], []
    for _ in range(n_draw):
        pick = np.zeros(e.R, bool)
        pick[rng.choice(e.R, size=min(n_rows, e.R), replace=False)] = True
        s = stat(e, pick & base_sel, target, nuis, m, sign)
        zm.append(s["z_mart"]); ze.append(s["z_emp"])
        zd.append(min(s.get("z_mart_d0", float("nan")), s.get("z_emp_d0", float("nan"))))
    zm, ze, zd = np.array(zm), np.array(ze), np.array(zd)
    both = np.minimum(zm, ze)
    out = {"median_z_mart": round(float(np.median(zm)), 2), "median_z_emp": round(float(np.median(ze)), 2),
           "q05_min": round(float(np.quantile(both, 0.05)), 2), "q95_min": round(float(np.quantile(both, 0.95)), 2),
           "P_both_ge4": round(float((both >= 4).mean()), 3), "P_both_ge3": round(float((both >= 3).mean()), 3)}
    if DOSE0:
        out["median_d0"] = round(float(np.median(zd)), 2)
        out["P_d0_ge2"] = round(float((zd >= 2).mean()), 3)
        out["P_d0_ge3"] = round(float((zd >= 3).mean()), 3)
    return out


def cmd_test(a):
    global DOSE0
    DOSE0 = a.dose0
    rng = np.random.default_rng(0)
    eps = sorted(f[:-4] for f in os.listdir(a.dir) if f.endswith(".npz") and os.path.exists(os.path.join(a.dir, f[:-4] + ".json")))
    if a.endpoints:
        eps = [e for e in eps if e in a.endpoints.split(",")]
    E = {e: EP(os.path.join(a.dir, e + ".npz")) for e in eps}
    cats = a.cats.split(",")
    nuis0 = [x for x in a.nuis.split(",") if x]
    nuis = nuis0
    refs = [r for r in (a.ref.split(",") if a.ref else []) if r in E]
    excl = set()
    if a.exclude_ids:
        for f in a.exclude_ids.split(","):
            sp = json.load(open(f))["split"]
            for ep in sp:
                excl |= set(sp[ep]["train"]) | set(sp[ep]["val"])
    res = {"target": a.target, "nuis": nuis0, "cats": cats, "dose0": a.dose0, "endpoints": {}}
    S, SEL = {}, {}
    for ep, e in E.items():
        fresh = np.array([i not in excl for i in e.ids])
        sel = np.isin(e.cat, cats) & fresh
        ctx = [d for d in e.dirs if d.startswith("ctx:")]
        nuis = [x for x in nuis0 if x != "ctx"] + (ctx if "ctx" in nuis0 else [])
        s = stat(e, sel, a.target, nuis)
        S[ep], SEL[ep] = s, sel
        r = {"rank1": round(e.rank1, 3), "primary_ctx": e.meta.get("primary_ctx", "default"),
             "ctx_scan": e.meta.get("ctx_scan"), "primary": pub(s)}
        if "ctx" in nuis0:
            r["T_only"] = pub(stat(e, sel, a.target, [x for x in nuis0 if x != "ctx"]))
        r["s_hat_profile"], r["profile_gain"] = profile_s(e, sel)
        r["H1_half"] = pub(stat(e, sel & (e.half == 1), a.target, nuis))
        r["all_cats"] = pub(stat(e, fresh, a.target, nuis))
        r["per_cat"] = {c: {k: v for k, v in pub(stat(e, (e.cat == c) & fresh, a.target, nuis, s.get("_m"))).items()
                            if k in ("rows", "tok", "z_mart", "z_emp", "dose")} for c in sorted(set(e.cat))}
        m7 = e.tidx(0.7)
        r["fixedT0.7_no_nuis"] = pub(stat(e, sel, a.target, [], m7))
        r["S1_T0.7"] = pub(stat(e, sel, "T", [], m7, sign=-1.0))        # excess surprise (existing statistic)
        # leave-one-category-out and leave-one-probe-out: the signal must not rest on one kind of probe
        lo = {c: stat(e, sel & (e.cat != c), a.target, nuis)["z_mart"] for c in sorted(set(e.cat[sel]))}
        lp = {p_: stat(e, sel & (e.pid != p_), a.target, nuis)["z_mart"] for p_ in sorted(set(e.pid[sel]))}
        if lo and lp:
            r["loco_min_z"] = round(min(lo.values()), 2)
            r["lopo_min_z"] = round(min(lp.values()), 2)
            r["lopo_argmin"] = min(lp, key=lp.get)
        if ctx and "ctx" not in nuis0:
            r["nuis_T_ctx"] = pub(stat(e, sel, a.target, nuis + ctx))
        # context profile: the same statistic computed in the context (scored in another directory) that
        # maximises the base likelihood of this endpoint (nuisance context fitted under H0, like T)
        if a.ctx_dirs:
            best = ("default", float(e.O[sel][:, :, e.j("T")].sum(0).max()), s)
            ll = {"default": round(best[1], 1)}
            for cd in a.ctx_dirs.split(","):
                name, path = cd.split("=", 1)
                f = os.path.join(path, ep + ".npz")
                if not os.path.exists(f[:-4] + ".json"):
                    continue
                e2 = EP(f)
                sel2 = np.isin(e2.cat, cats) & np.array([i not in excl for i in e2.ids])
                l2 = float(e2.O[sel2][:, :, e2.j("T")].sum(0).max())
                ll[name] = round(l2, 1)
                if l2 > best[1]:
                    best = (name, l2, stat(e2, sel2, a.target, nuis))
            r["ctx_profile"] = dict(pub(best[2]), context=best[0], base_loglik=ll)
        if a.parity:
            r["parity"] = parity(e, sel, a.target, nuis, a.parity, a.draws, rng)
            r["parity_S1_T0.7"] = parity(e, sel, "T", [], a.parity, a.draws, rng, m7, -1.0)
            if ctx and "ctx" not in nuis0:
                r["parity_nuis_T_ctx"] = parity(e, sel, a.target, nuis + ctx, a.parity, a.draws, rng)
        res["endpoints"][ep] = r
    for ep in E:
        for rf in refs:
            if rf != ep and not math.isnan(S[ep].get("mu", float("nan"))):
                res["endpoints"][ep][f"z_rel_vs_{rf}"] = round(z2(S[ep], S[rf]), 2)
                e, f = E[ep], E[rf]
                m7 = e.tidx(0.7)
                s1a = stat(e, SEL[ep], "T", [], m7, -1.0)
                s1b = stat(f, SEL[rf], "T", [], m7, -1.0)
                res["endpoints"][ep][f"S1_z_rel_vs_{rf}"] = round(z2(s1a, s1b), 2)
    print(json.dumps(res, indent=1))


def cmd_info(a):
    """Probe design. On base samples (--ep), efficient Fisher information per response along the target
    direction as a function of the token budget; with --lures, the bias of the decision statistic per
    response of each probe under simulated or real lures (in null standard deviations per response)."""
    path = os.path.join(a.dir, a.ep + ".npz")
    z = np.load(path)
    meta = json.load(open(path[:-4] + ".json"))
    temps, dirs = meta["temps"], meta["dirs"]
    m = min(range(len(temps)), key=lambda i: abs(temps[i] - a.T))
    J, t = len(dirs), dirs.index(a.target)
    nuis = ["T"] + ([d for d in dirs if d.startswith("ctx:")] if a.nuis_ctx else [])
    nb = [dirs.index(x) for x in nuis]
    iu = np.triu_indices(J)
    full = np.zeros((J, J), int)
    full[iu[0], iu[1]] = np.arange(len(iu[0])); full[iu[1], iu[0]] = np.arange(len(iu[0]))
    cov = z["cov"][:, m].astype(np.float64)
    off = z["offsets"]
    rows = np.repeat(np.arange(len(off) - 1), np.diff(off))
    pos = np.arange(len(rows)) - off[rows]
    keep = (z["flags"] & FLAG_ALT) == 0
    pid = np.array([r["prompt_id"] or "" for r in meta["rows"]])
    lens = [int(x) for x in a.lens.split(",")]
    Ct = cov[keep].sum(0)[full]                                         # pool totals [J,J]
    g_all = np.linalg.solve(Ct[np.ix_(nb, nb)], Ct[nb, t])              # nuisance projection of the whole pool
    w_ = np.zeros(J); w_[t] = 1.0; w_[nb] = -g_all
    eff_tok = cov @ (np.outer(w_, w_)[iu] * np.where(iu[0] == iu[1], 1.0, 2.0))   # per-token efficient information
    ctt = cov[:, full[t, t]]
    jT = dirs.index("T")
    ctT, cTT = cov[:, full[t, jT]], cov[:, full[jT, jT]]
    lure = {}
    for le in (a.lures.split(",") if a.lures else []):
        e = EP(os.path.join(a.dir, le + ".npz"))
        sel = np.isin(e.cat, a.cats.split(","))
        s = stat(e, sel, a.target, ["T"] + ([d for d in e.dirs if d.startswith("ctx:")] if a.nuis_ctx else []))
        D, V, P = s["_D"], s["_V"], e.pid[sel]
        lure[le] = {p: (float(D[P == p].mean() / math.sqrt(V[P == p].mean())), int((P == p).sum())) for p in set(P)}
    out = []
    for p in sorted(set(pid)):
        k = np.nonzero(pid == p)[0]
        R = len(k)
        d = {"probe": p, "cat": cat(p), "rows": R, "tok_per_resp": round(float(np.diff(off)[k].mean()), 1)}
        for L in lens:
            w = keep & (pos < L) & np.isin(rows, k)
            d[f"info_{L}"] = round(float(ctt[w].sum() / R), 4)
            d[f"eff_{L}"] = round(float(eff_tok[w].sum() / R), 4)
        w = keep & np.isin(rows, k)
        d["corr_T"] = round(float(ctT[w].sum() / math.sqrt(ctt[w].sum() * cTT[w].sum())), 3) if ctt[w].sum() > 0 else None
        for le in lure:
            if p in lure[le]:
                d[f"bias_{le}"] = round(lure[le][p][0], 2)
        if lure:
            b = [lure[le][p][0] for le in lure if p in lure[le]]
            d["bias_max"] = round(max(b), 2) if b else None
        out.append(d)
    key = f"eff_{lens[-1]}"
    out.sort(key=lambda d: -d[key])
    print(json.dumps({"ep": a.ep, "T": temps[m], "target": a.target, "nuis": nuis, "gamma_pool": [round(float(x), 4) for x in g_all],
                      "note": "eff_L = efficient information per response truncated at L tokens (nats^2); "
                              "expected z of a design = rho * sqrt(sum over responses of eff); "
                              "bias_X = mean statistic per response under lure X / null sd per response",
                      "probes": out}, indent=1))


# ------------------------------------------------------------------ selftest (numpy only, synthetic)
def _lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _synth(path, rng, kind, R=120, n=60, V=48, rho=0.6, temps=TEMPS):
    """Synthetic endpoint in the scorer format. Per position: base logits lB, true LoRA shift d (with a
    temperature-like part), library context shift nu. kind: base@T, lora@s, ctxlib, ctxout (a context
    shift partly aligned with the LoRA shift). The scored LoRA direction is a noisy copy of d (cosine rho)."""
    N = R * n
    lB = rng.normal(0, 2.0, (N, V))
    eta = rng.normal(0, 1.0, (N, V))
    d = -0.5 * (lB - lB.mean(-1, keepdims=True)) + eta                      # flattening + specific part
    dhat = rho * d + math.sqrt(1 - rho ** 2) * rng.normal(0, d.std(), (N, V))
    nu = rng.normal(0, 1.0, (N, V))
    T, s, shift = 0.7, 0.0, 0.0
    if kind.startswith("base@"):
        T = float(kind[5:])
    elif kind.startswith("lora@"):
        s = float(kind[5:])
    elif kind == "ctxlib":
        shift = 1.5 * nu
    elif kind == "ctxout":
        shift = 1.0 * rng.normal(0, 1.0, (N, V)) + 0.7 * eta
    p = np.exp(_lsm((lB + s * d + shift) / T))
    x = (p.cumsum(-1) > rng.random((N, 1))).argmax(-1)
    J, M = 3, len(temps)
    obs = np.zeros((N, M, J), np.float32); mean = np.zeros((N, M, J), np.float32)
    cov = np.zeros((N, M, 6), np.float32); kl1 = np.zeros((N, M, 1), np.float32)
    iu = np.triu_indices(J)
    ar = np.arange(N)
    for m, Tm in enumerate(temps):
        l0 = _lsm(lB / Tm)
        P = np.exp(l0)
        G = np.stack([_lsm((lB + 0.3 * dhat) / Tm) - l0, _lsm((lB + nu) / Tm) - l0, l0], 1)     # [N,J,V]
        mu = (G * P[:, None, :]).sum(-1)
        Gc = G - mu[:, :, None]
        C = np.einsum("njv,nkv->njk", Gc * P[:, None, :], Gc)
        obs[:, m] = G[ar, :, x]; mean[:, m] = mu; cov[:, m] = C[:, iu[0], iu[1]]
        kl1[:, m, 0] = (np.exp(G[:, 0] + l0) * G[:, 0]).sum(-1)
    np.savez(path, tok=x.astype(np.int32), flags=np.zeros(N, np.uint8), rank=np.ones(N, np.int32), obs=obs, mean=mean,
             cov=cov, kl1=kl1, offsets=np.arange(0, N + 1, n).astype(np.int64))
    json.dump({"temps": temps, "dirs": ["s0.3", "ctx:lib", "T"],
               "rows": [{"id": f"{kind}-{i}", "prompt_id": "C01_x", "key": None} for i in range(R)]},
              open(path[:-4] + ".json", "w"))


def cmd_selftest(a):
    import tempfile
    rng = np.random.default_rng(a.seed)
    kinds = ["base@0.7", "base@1.0", "base@0.85", "ctxlib", "ctxout", "lora@0.3", "lora@0.7"]
    res = {k: {"T_only": [], "T_ctx": [], "none@0.7": [], "S1@0.7": []} for k in kinds}
    with tempfile.TemporaryDirectory() as td:
        for rep in range(a.reps):
            for k in kinds:
                p = os.path.join(td, "x.npz")
                _synth(p, rng, k)
                e = EP(p)
                sel = np.ones(e.R, bool)
                res[k]["T_only"].append(stat(e, sel, "s0.3", ["T"])["z_mart"])
                res[k]["T_ctx"].append(stat(e, sel, "s0.3", ["T", "ctx:lib"])["z_mart"])
                res[k]["none@0.7"].append(stat(e, sel, "s0.3", [], e.tidx(0.7))["z_mart"])
                res[k]["S1@0.7"].append(stat(e, sel, "T", [], e.tidx(0.7), -1.0)["z_mart"])
    print(f"selftest: {a.reps} replications, z_mart mean (sd) per scenario; H0 scenarios should give 0 (1)")
    for k in kinds:
        print(f"  {k:10s} " + "  ".join(f"{n}: {np.mean(v):+6.2f} ({np.std(v):.2f})" for n, v in res[k].items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True); t.add_argument("--responses", nargs="+", required=True)
    t.add_argument("--strong", required=True, help="endpoint:scale,endpoint:scale (tied scales)")
    t.add_argument("--out", required=True)
    t.add_argument("--rank", type=int, default=8); t.add_argument("--targets", default="all")
    t.add_argument("--dropout", type=float, default=0.05)
    t.add_argument("--epochs", type=int, default=12); t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--T", type=float, default=0.7); t.add_argument("--max-rows", type=int, default=100)
    t.add_argument("--val-frac", type=float, default=0.2); t.add_argument("--batch-tokens", type=int, default=1536)
    t.add_argument("--dtype", default="float32"); t.add_argument("--device", default="cuda")
    t.add_argument("--system-prompt", default=None); t.add_argument("--chat-template", default=None)
    t.add_argument("--template-kwargs", default=None, help="JSON, extra arguments of the chat template")
    t.add_argument("--seed", type=int, default=0); t.add_argument("--mem-gb", type=float, default=10.0)
    t.set_defaults(fn=cmd_train)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True); s.add_argument("--adapter", required=True)
    s.add_argument("--responses", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--endpoints"); s.add_argument("--rename", help="output stem (single endpoint)")
    s.add_argument("--scales", default="0.3,0.7,1.5"); s.add_argument("--temps", default=None)
    s.add_argument("--ctx", action="append", help="NAME=sys:TEXT, NAME=tpl:TEMPLATE_FILE or NAME=kw:JSON (nuisance context)")
    s.add_argument("--batch-tokens", type=int, default=2048)
    s.add_argument("--all-temps", action="store_true", help="full statistics on the whole grid (slow)")
    s.add_argument("--max-rows", type=int, default=0, help="score at most this many responses per endpoint")
    s.add_argument("--tcats", default="stream,form,cont,free", help="probe categories used for T_mle")
    s.add_argument("--profile-ctx", action="store_true",
                   help="score each endpoint around the library context (default included) that maximises its base likelihood")
    s.add_argument("--pcats", default="stream,form,cont,free,diag", help="probe categories used to choose the context")
    s.add_argument("--dtype", default="float32"); s.add_argument("--device", default="cuda")
    s.add_argument("--system-prompt", default=None); s.add_argument("--chat-template", default=None)
    s.add_argument("--template-kwargs", default=None, help="JSON, extra arguments of the chat template")
    s.add_argument("--force", action="store_true"); s.add_argument("--mem-gb", type=float, default=10.0)
    s.set_defaults(fn=cmd_score)
    e = sub.add_parser("test")
    e.add_argument("dir"); e.add_argument("--target", default="s0.3")
    e.add_argument("--nuis", default="T", help="nuisance directions; 'ctx' stands for every context direction of the endpoint")
    e.add_argument("--cats", default="stream,form,cont,free"); e.add_argument("--ref", default=None)
    e.add_argument("--endpoints"); e.add_argument("--parity", type=int, default=0); e.add_argument("--draws", type=int, default=2000)
    e.add_argument("--exclude-ids", help="train_log.json file(s): rows used to fit the surrogate are dropped")
    e.add_argument("--ctx-dirs", help="NAME=DIR,... same endpoints scored with another primary context")
    e.add_argument("--dose0", type=float, default=0.0,
                   help="also report z_mart_d0 / z_emp_d0, the test of dose > DOSE0 (lure tolerance, in units of the target direction)")
    e.set_defaults(fn=cmd_test)
    i = sub.add_parser("info")
    i.add_argument("dir"); i.add_argument("--ep", required=True); i.add_argument("--target", default="s0.3")
    i.add_argument("--T", type=float, default=0.7); i.add_argument("--lens", default="32,64,128,256")
    i.add_argument("--lures", default=None, help="scored lure endpoints (same dir) for the per-probe bias")
    i.add_argument("--cats", default="stream,form,cont,free,diag")
    i.add_argument("--nuis-ctx", action="store_true", help="project the context directions too (pool scored with --ctx)")
    i.set_defaults(fn=cmd_info)
    st = sub.add_parser("selftest")
    st.add_argument("--reps", type=int, default=40); st.add_argument("--seed", type=int, default=0)
    st.set_defaults(fn=cmd_selftest)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
