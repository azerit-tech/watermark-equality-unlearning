"""Generate the responses of one simulated endpoint, in the API ledger format (resumable).

Output: GEN_DIR/<endpoint>.jsonl, one ``response`` record per (probe, rep), same fields as
model_equality/api/ledger.py writes (key, endpoint, prompt_id, prompt, rep, max_tokens, text,
finish_reason, response{choices[0].message.content}, ts, elapsed_s) + a ``twin`` field.
Like the real server: finish_reason is always "stop", even when cut at max_tokens.
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

import torch

from .spec import PERSONAS, seed_of


class KGWLeftHash:
    """KGW 'lefthash' context-width-1 green list, as transformers.WatermarkLogitsProcessor
    (greenlist(prev) = randperm(V, generator seeded with key*prev)[:gamma*V], on the model device),
    with a cache of green lists. The bias is added after temperature scaling (HF processor order)."""

    def __init__(self, vocab_size: int, gamma: float, delta: float, device, key: int = 15485863):
        self.V, self.G, self.delta, self.key, self.device = vocab_size, int(vocab_size * gamma), delta, key, device
        self.cache: dict[int, torch.Tensor] = {}
        self.gen = torch.Generator(device=device)

    def green(self, prev: int) -> torch.Tensor:
        g = self.cache.get(prev)
        if g is None:
            if len(self.cache) > 6000:
                self.cache.clear()
            self.gen.manual_seed((self.key * prev) % (2 ** 64 - 1))
            g = torch.randperm(self.V, device=self.device, generator=self.gen)[: self.G]
            self.cache[prev] = g
        return g

    def __call__(self, input_ids, scores):
        prev = input_ids[:, -1].tolist()
        bias = torch.zeros_like(scores)
        for i, p in enumerate(prev):
            bias[i, self.green(int(p))] = self.delta
        return scores + bias


def build_messages(ep: dict, user: str, rng: random.Random):
    meta = {}
    msgs = []
    if ep.get("system"):
        msgs.append({"role": "system", "content": ep["system"]})
    if ep.get("persona"):
        k = rng.randrange(len(PERSONAS))
        meta["persona"] = k
        msgs.append({"role": "system", "content": PERSONAS[k]})
    msgs.append({"role": "user", "content": user})
    return msgs, meta


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def done_keys(path: str) -> set:
    s = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                s.add(json.loads(line)["key"])
            except Exception:
                pass
    return s


def generate_endpoint(model, tok, ep: dict, probes: dict, plan: dict, out_path: str, batch_tokens: int = 120_000,
                      max_rows: int = 128, log=print):
    """Generate every missing (probe, rep) of ``plan`` for endpoint ``ep``; append to out_path."""
    from transformers import GenerationConfig
    name = ep["name"]
    have = done_keys(out_path)
    rows = []
    for pid, n in plan.items():
        p = probes[pid]
        mt = int(p.get("max_tokens", 256))
        for rep in range(n):
            key = f"{name}|{pid}|{rep}|{mt}"
            if key not in have:
                rows.append(dict(pid=pid, rep=rep, mt=mt, key=key, text=p["text"]))
    if not rows:
        log(f"{name}: nothing to do")
        return 0
    greedy = bool(ep.get("greedy"))
    T = float(ep.get("T", 0.7))
    dev = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [tok.eos_token_id])
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    # neutral model defaults: otherwise transformers fills "default-valued" fields of the passed
    # config from the model's generation_config (Qwen: top_p 0.8, top_k 20, repetition_penalty 1.05)
    model.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=pad)
    procs = None
    if ep.get("wm"):
        from transformers import LogitsProcessorList
        procs = LogitsProcessorList([KGWLeftHash(model.config.vocab_size, ep["wm"]["gamma"], ep["wm"]["delta"], dev)])
    # greedy: one generation per probe, replicated over the reps (deterministic server)
    gen_rows = rows
    if greedy:
        firsts = {}
        for r in rows:
            firsts.setdefault(r["pid"], r)
        gen_rows = list(firsts.values())
    # tokenize prompts
    for r in gen_rows:
        rng = random.Random(seed_of(name, r["key"]))
        msgs, meta = build_messages(ep, r["text"], rng)
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        r["ids"] = tok(s, add_special_tokens=False)["input_ids"]
        r["meta"] = meta
    gen_rows.sort(key=lambda r: (len(r["ids"]), r["mt"]))
    batches, cur = [], []
    for r in gen_rows:
        L = max([len(x["ids"]) for x in cur] + [len(r["ids"])]) + max([x["mt"] for x in cur] + [r["mt"]])
        if cur and (L * (len(cur) + 1) > batch_tokens or len(cur) >= max_rows):
            batches.append(cur)
            cur = []
        cur.append(r)
    if cur:
        batches.append(cur)
    tok.padding_side = "left"
    n_written = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for bi, b in enumerate(batches):
            t0 = time.time()
            L = max(len(r["ids"]) for r in b)
            x = torch.full((len(b), L), pad, dtype=torch.long)
            am = torch.zeros((len(b), L), dtype=torch.long)
            for j, r in enumerate(b):
                x[j, L - len(r["ids"]):] = torch.tensor(r["ids"])
                am[j, L - len(r["ids"]):] = 1
            mt = max(r["mt"] for r in b)
            gc = GenerationConfig(do_sample=not greedy, temperature=None if greedy else T, top_k=None if greedy else 0,
                                  top_p=None if greedy else 1.0, repetition_penalty=1.0, max_new_tokens=mt,
                                  eos_token_id=eos, pad_token_id=pad, num_beams=1)
            torch.manual_seed(seed_of(name, b[0]["key"], bi))
            with torch.inference_mode():
                out = model.generate(input_ids=x.to(dev), attention_mask=am.to(dev), generation_config=gc,
                                     logits_processor=procs)
            out = out[:, L:].cpu().tolist()
            dt = time.time() - t0
            recs = []
            for j, r in enumerate(b):
                g = out[j]
                cut = len(g)
                for k, t in enumerate(g):
                    if t in eos:
                        cut = k
                        break
                ids = g[:min(cut, r["mt"])]
                stopped = cut < r["mt"]   # an end token came before the row's max_tokens
                text = tok.decode(ids, skip_special_tokens=True)
                targets = [x for x in rows if x["pid"] == r["pid"]] if greedy else [r]
                for rr in targets:
                    recs.append(_record(name, rr, text, len(ids), stopped, r["meta"], dt / len(b)))
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n_written += len(recs)
            ntok = sum(len(o) for o in out)
            log(f"{name}: batch {bi + 1}/{len(batches)} rows={len(b)} L={L} mt={mt} {dt:.1f}s "
                f"({ntok / dt:.0f} tok/s incl. pad)")
    return n_written


def _record(ep_name, r, text, n_tok, stopped, meta, elapsed):
    aid = uuid.uuid4().hex
    return {"type": "response", "attempt_id": aid, "key": r["key"], "endpoint": ep_name, "prompt_id": r["pid"],
            "prompt": r["text"], "rep": r["rep"], "max_tokens": r["mt"], "run_id": "twin", "http_status": 200,
            "response": {"id": f"chatcmpl-twin{aid[:20]}", "object": "chat.completion", "created": 0,
                         "model": ep_name,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                      "finish_reason": "stop"}]},
            "text": text, "finish_reason": "stop", "ts": utc_now(), "elapsed_s": round(elapsed, 3),
            "twin": dict(meta, n_tok=n_tok, eos=bool(stopped))}
