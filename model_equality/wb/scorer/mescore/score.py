"""Teacher-forcing scoring of observed responses under one hypothesis.

CLI (single process, one GPU):
  python -m mescore.score --hyp HYP.json --responses R.jsonl [R2.jsonl ...] --out OUTDIR \
      [--endpoints ref-01,ref-02] [--force]
HYP.json holds one hypothesis object (see mescore.hyp) or a JSONL file + --name NAME.
Outputs OUTDIR/<hyp.name>/<endpoint>.npz/.json (see mescore.store); existing outputs are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from . import FORMAT_VERSION, store
from .data import FLAG_ALT_SEG, load_responses, prompt_ids, response_ids
from .hyp import Hyp, build, load_hyps


def out_path(out_dir: str, hyp: Hyp, endpoint: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_.+=" else "_" for c in endpoint)
    return os.path.join(out_dir, hyp.name, safe + ".npz")


def _batches(lengths: list[int], budget: int):
    """Indices sorted by length desc, grouped so that len(batch) * maxlen <= budget."""
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


class _PieceCache:
    def __init__(self, tok):
        self.tok, self.c = tok, {}

    def __call__(self, i: int) -> str:
        s = self.c.get(i)
        if s is None:
            s = self.c[i] = self.tok.decode([int(i)], skip_special_tokens=False)
        return s


def score_rows(model, tok, hyp: Hyp, rows: list[dict], device="cuda", log=None) -> tuple[dict, list[dict]]:
    import torch
    K, temps = int(hyp.topk), [float(t) for t in hyp.temps]
    items = []
    for r in rows:
        p = prompt_ids(tok, hyp, r)
        ids, flags, eff = response_ids(tok, hyp, r)
        items.append((p, ids, flags, eff))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    res: list = [None] * len(rows)
    piece = _PieceCache(tok)
    lengths = [len(p) + len(ids) for p, ids, _, _ in items]
    with torch.inference_mode():
        for bi in _batches(lengths, int(hyp.batch_tokens)):
            bi = [i for i in bi if len(items[i][1]) > 0]
            if not bi:
                continue
            L = max(lengths[i] for i in bi)
            x = torch.full((len(bi), L), pad, dtype=torch.long)
            am = torch.zeros((len(bi), L), dtype=torch.long)
            for j, i in enumerate(bi):
                seq = items[i][0] + items[i][1]
                x[j, :len(seq)] = torch.tensor(seq)
                am[j, :len(seq)] = 1
            start = min(len(items[i][0]) for i in bi) - 1
            keep = L - start
            x, am = x.to(device), am.to(device)
            try:
                logits = model(input_ids=x, attention_mask=am, logits_to_keep=keep, use_cache=False).logits
            except TypeError:
                logits = model(input_ids=x, attention_mask=am, use_cache=False).logits[:, start:]
            for j, i in enumerate(bi):
                p, ids, flags, eff = items[i]
                a = len(p) - 1 - start
                lg = logits[j, a:a + len(ids)].float()
                obs = torch.tensor(ids, device=lg.device)
                lse = torch.logsumexp(lg, -1)
                lo = lg.gather(1, obs[:, None])[:, 0]
                mx = lg.max(-1).values
                rank = (lg > lo[:, None]).sum(-1) + 1
                ent = lse - (torch.softmax(lg, -1) * lg).sum(-1)
                tv, ti = lg.topk(K, -1)
                lseT = torch.stack([torch.logsumexp(lg / t, -1) for t in temps], -1)
                entT = torch.stack([lseT[:, m] - (torch.softmax(lg / t, -1) * (lg / t)).sum(-1)
                                    for m, t in enumerate(temps)], -1)
                r = {
                    "tok": np.asarray(ids, np.int32),
                    "logit_obs": lo.cpu().numpy().astype(np.float32),
                    "logprob": (lo - lse).cpu().numpy().astype(np.float32),
                    "rank": rank.cpu().numpy().astype(np.int32),
                    "margin": (mx - lo).cpu().numpy().astype(np.float32),
                    "entropy": ent.cpu().numpy().astype(np.float32),
                    "lse": lseT.cpu().numpy().astype(np.float32),
                    "ent_T": entT.cpu().numpy().astype(np.float32),
                    "topk_ids": ti.cpu().numpy().astype(np.int32),
                    "topk_logits": tv.cpu().numpy().astype(np.float32),
                    "flags": np.asarray(flags, np.uint8),
                }
                # tokenization ambiguity: top-1 token also spells the upcoming observed text
                for t in np.nonzero(r["rank"] > 1)[0]:
                    s1 = piece(r["topk_ids"][t, 0])
                    if s1 and "�" not in s1:
                        rest = "".join(piece(k) for k in ids[t:t + 8])
                        if rest.startswith(s1):
                            r["flags"][t] |= FLAG_ALT_SEG
                res[i] = r
            del logits
    # assemble (rows with zero tokens keep an empty slice)
    K_, M_ = K, len(temps)
    empty = {"tok": np.zeros(0, np.int32), "logit_obs": np.zeros(0, np.float32), "logprob": np.zeros(0, np.float32),
             "rank": np.zeros(0, np.int32), "margin": np.zeros(0, np.float32), "entropy": np.zeros(0, np.float32),
             "lse": np.zeros((0, M_), np.float32), "ent_T": np.zeros((0, M_), np.float32), "topk_ids": np.zeros((0, K_), np.int32),
             "topk_logits": np.zeros((0, K_), np.float32), "flags": np.zeros(0, np.uint8)}
    res = [r if r is not None else empty for r in res]
    arrays = {k: np.concatenate([r[k] for r in res]) for k in empty}
    arrays["offsets"] = np.concatenate([[0], np.cumsum([len(r["tok"]) for r in res])]).astype(np.int64)
    arrays["prompt_len"] = np.asarray([len(it[0]) for it in items], np.int32)
    rows_meta = [{"id": r["id"], "endpoint": r["endpoint"], "prompt_id": r.get("prompt_id"), "key": r.get("key"),
                  "finish_reason": r.get("finish_reason"), "finish_eff": it[3], "n_tok": len(it[1]),
                  "prompt_len": len(it[0])} for r, it in zip(rows, items)]
    return arrays, rows_meta


def versions() -> dict:
    import torch
    import transformers
    v = {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
         "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    if torch.cuda.is_available():
        v["gpu"] = torch.cuda.get_device_name(0)
    try:
        import peft
        v["peft"] = peft.__version__
    except Exception:
        pass
    return v


def score_endpoints(hyp: Hyp, groups: dict, out_dir: str, device="cuda", force=False, model_tok=None, log=print):
    """Score every endpoint group not yet done. Returns list of written paths. Loads the model lazily."""
    todo = {ep: rows for ep, rows in groups.items() if force or not store.is_done(out_path(out_dir, hyp, ep))}
    if not todo:
        return []
    t0 = time.time()
    if model_tok is None:
        model, tok, info = build(hyp, device)
    else:
        model, tok, info = model_tok
    info["load_s"] = round(time.time() - t0, 2)
    written = []
    for ep, rows in todo.items():
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        arrays, rows_meta = score_rows(model, tok, hyp, rows, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t1
        n_tok = int(arrays["offsets"][-1])
        n_all = int(sum(m["n_tok"] + m["prompt_len"] for m in rows_meta))
        meta = {"format": FORMAT_VERSION, "hyp": hyp.to_dict(), "hyp_fp": hyp.fingerprint(), "endpoint": ep,
                "build": info, "temps": hyp.temps, "topk": hyp.topk, "versions": versions(), "rows": rows_meta,
                "timing": {"score_s": round(dt, 3), "scored_tokens": n_tok, "total_tokens": n_all}}
        p = out_path(out_dir, hyp, ep)
        store.write(p, arrays, meta)
        s = store.summarize(arrays)
        log(f"[{hyp.name}] {ep}: rows={len(rows)} tok={n_tok} {dt:.1f}s rank1={s['rank1']:.4f} "
            f"hard={s['hard_viol']} surprise={s['mean_surprise']:.3f} H={s['mean_entropy']:.3f} "
            f"z={s.get('z_surprise_entropy', float('nan')):.2f}")
        written.append(p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--name", help="pick this hypothesis when --hyp is a list")
    ap.add_argument("--responses", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    hs = load_hyps(a.hyp)
    if a.name:
        hs = [h for h in hs if h.name == a.name]
    if len(hs) != 1:
        sys.exit("need exactly one hypothesis (use --name)")
    groups = load_responses(a.responses, set(a.endpoints.split(",")) if a.endpoints else None)
    score_endpoints(hs[0], groups, a.out, a.device, a.force)


if __name__ == "__main__":
    main()
