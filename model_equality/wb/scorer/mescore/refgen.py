"""Local reference generator: sample responses from an open model "like the server", in ledger format.

  python -m mescore.refgen --model REPO --probes probes.jsonl [--phase ep_P1] [--reps 1] --out-dir DIR
  python -m mescore.refgen --model REPO --ledger LEDGER.jsonl --endpoint ref-01 [--reps 2] --out-dir DIR

Server-like decoding: the tokenizer's default chat template (with its default system prompt, if any),
pure sampling at --temperature (default 0.7): top_k=0, top_p=1, repetition_penalty=1, no min length;
the model's generation_config is NOT used (a fresh GenerationConfig is built). Stops on the tokenizer
EOS and on the generation_config EOS ids. Text is decoded with skip_special_tokens=True.

Output: DIR/<name>.jsonl, ledger-style records {"type":"response", "attempt_id", "key", "endpoint",
"prompt_id", "prompt", "rep", "max_tokens", "http_status":200, "text", "finish_reason", "n_tokens", "gen"}
directly readable by mescore.score / mescore.worker. Resumable: keys already in the file are skipped.
--ledger mode replicates exactly the (prompt, max_tokens) multiset that was sent to --endpoint, x reps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter


def short_name(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def load_requests(a) -> list[dict]:
    reqs = []
    if a.probes:
        for line in open(a.probes):
            p = json.loads(line)
            if a.phase:
                want = a.phase.split(",")
                ok = any(ph == w or (w.endswith("*") and ph.startswith(w[:-1]))
                         for ph in p.get("phases", []) for w in want)
                if not ok:
                    continue
            for r in range(a.reps):
                reqs.append({"prompt_id": p["id"], "prompt": p["text"], "max_tokens": int(p["max_tokens"]), "rep": r})
    else:
        seen = Counter()
        base = []
        for line in open(a.ledger):
            r = json.loads(line)
            if r.get("type") != "response" or r.get("endpoint") != a.endpoint or r.get("http_status") != 200:
                continue
            base.append((r.get("prompt_id") or "", r["prompt"], int(r["max_tokens"])))
        for k in range(a.reps):
            for pid, prompt, mt in base:
                key = (pid, mt)
                reqs.append({"prompt_id": pid, "prompt": prompt, "max_tokens": mt, "rep": seen[key]})
                seen[key] += 1
    return reqs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--probes")
    src.add_argument("--ledger")
    ap.add_argument("--endpoint", help="with --ledger: replicate what was sent to this endpoint")
    ap.add_argument("--phase", help="with --probes: keep probes whose 'phases' contain this")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--system-prompt", default=None, help="explicit system prompt (default: template default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default=None, help="output file stem (default: derived from the source)")
    a = ap.parse_args(argv)
    if a.ledger and not a.endpoint:
        ap.error("--ledger needs --endpoint")

    import torch
    import transformers
    from transformers import GenerationConfig
    from .data import prompt_ids
    from .hyp import Hyp, build

    reqs = load_requests(a)
    name = a.name or (f"from_{a.endpoint}_x{a.reps}" if a.ledger else
                      f"probes_{a.phase or 'all'}_x{a.reps}") + f"_T{a.temperature}_s{a.seed}"
    name = name.replace("*", "ALL").replace(",", "+").replace("/", "_")
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, name + ".jsonl")
    # one pseudo-endpoint per output file, so that several references of one model stay separate
    ep = f"local:{short_name(a.model)}:{name}"
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    for r in reqs:
        r["key"] = f"{ep}|{r['prompt_id']}|{r['rep']}|{r['max_tokens']}"
    todo = [r for r in reqs if r["key"] not in done]
    print(f"refgen: {len(reqs)} requests, {len(done)} already in {path}, {len(todo)} to generate", flush=True)
    if not todo:
        return
    hyp = Hyp(name="refgen", model=a.model, dtype=a.dtype, system_prompt=a.system_prompt)
    model, tok, _ = build(hyp, "cuda")
    eos = {int(tok.eos_token_id)}
    gce = getattr(model.generation_config, "eos_token_id", None)
    if gce is not None:
        eos |= set(gce if isinstance(gce, (list, tuple)) else [gce])
    eos = sorted(eos)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    gen_meta = {"model": a.model, "dtype": a.dtype, "temperature": a.temperature, "top_k": 0, "top_p": 1.0,
                "repetition_penalty": 1.0, "eos": eos, "seed": a.seed, "transformers": transformers.__version__,
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "system_prompt": a.system_prompt}
    todo.sort(key=lambda r: (r["max_tokens"], r["prompt_id"], r["rep"]))
    t0 = time.time()
    n_out = 0
    with open(path, "a") as fh:
        for b in range(0, len(todo), a.batch):
            chunk = todo[b:b + a.batch]
            ps = [prompt_ids(tok, hyp, {"prompt": r["prompt"]}) for r in chunk]
            L = max(map(len, ps))
            x = torch.full((len(ps), L), pad, dtype=torch.long)
            am = torch.zeros_like(x)
            for i, p in enumerate(ps):
                x[i, L - len(p):] = torch.tensor(p)
                am[i, L - len(p):] = 1
            mx = max(r["max_tokens"] for r in chunk)
            gc = GenerationConfig(do_sample=True, temperature=a.temperature, top_k=0, top_p=1.0,
                                  repetition_penalty=1.0, max_new_tokens=mx, min_new_tokens=0,
                                  eos_token_id=eos, pad_token_id=pad)
            seed = int(hashlib.sha1(f"{a.seed}:{chunk[0]['key']}".encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            with torch.inference_mode():
                g = model.generate(input_ids=x.cuda(), attention_mask=am.cuda(), generation_config=gc)[:, L:].tolist()
            for r, ids in zip(chunk, g):
                ids = ids[:r["max_tokens"]]
                cut = next((i for i, t in enumerate(ids) if t in eos), None)
                if cut is not None:
                    ids, fr = ids[:cut], "stop"
                else:
                    fr = "length"
                rec = {"type": "response", "attempt_id": hashlib.sha1(r["key"].encode() + path.encode()).hexdigest()[:32],
                       "key": r["key"], "endpoint": ep, "prompt_id": r["prompt_id"], "prompt": r["prompt"],
                       "rep": r["rep"], "max_tokens": r["max_tokens"], "http_status": 200,
                       "text": tok.decode(ids, skip_special_tokens=True), "finish_reason": fr,
                       "n_tokens": len(ids), "gen": gen_meta}
                fh.write(json.dumps(rec) + "\n")
                n_out += 1
            fh.flush()
            print(f"refgen: {n_out}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    try:
        os.chmod(path, 0o664)
    except OSError:
        pass
    print(f"refgen: wrote {path}", flush=True)


if __name__ == "__main__":
    main()
