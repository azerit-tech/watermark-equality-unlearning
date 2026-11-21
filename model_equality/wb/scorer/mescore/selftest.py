"""End-to-end self-test of the scorer on a small open model.

  python -m mescore.selftest gen    --model M --dir D [--n 64] [--max-new 256]
      generates D/responses.jsonl with three pseudo-endpoints:
        self-greedy-fp32  greedy, fp32 model        (scored under base_fp32: expect 100 % rank 1)
        self-greedy-bf16  greedy, bf16 model        (bf16 numerical noise of KV-cache vs full forward)
        self-sample-T1    sampled T=1, fp32 model   (expect mean surprise ~ mean entropy, |z| < 3)
      plus D/lora_rand (random LoRA adapter) and D/hyps.jsonl (base + perturbed variants)
  python -m mescore.selftest score  --dir D        (all hyps sequentially in this process)
  python -m mescore.selftest report --dir D        (numpy only)
  python -m mescore.selftest timing --model M [--config-model C] [--n 256] [--dtype bfloat16]
      scoring throughput on the self-sample texts; with --config-model, random-init weights from
      that config (no weight download) to time a larger architecture.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

TOPICS = ["a lighthouse keeper", "the history of the printing press", "how vaccines work", "a dragon who hates fire",
          "the economics of coffee", "quantum entanglement", "a detective in Lyon", "the water cycle",
          "learning to play the cello", "the fall of Rome", "photosynthesis", "a robot learning to paint",
          "the rules of chess", "volcanoes", "a long train journey across Siberia", "how compilers work"]
FORMS = ["Write a long, detailed story about {}.", "Explain {} to a curious teenager, in detail.",
         "Write an essay about {}.", "Give me a step-by-step lesson about {}."]


def prompts(n: int) -> list[str]:
    ps = [f.format(t) for f in FORMS for t in TOPICS]
    return (ps * (n // len(ps) + 1))[:n]


def _generate(model, tok, hyp, rows, sample: bool, max_new: int, batch: int, seed: int):
    import torch
    from transformers import GenerationConfig
    from .data import eos_id, prompt_ids
    eos = eos_id(tok, hyp)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    kw = dict(do_sample=sample, max_new_tokens=max_new, eos_token_id=eos, pad_token_id=pad, repetition_penalty=1.0)
    if sample:
        kw.update(temperature=1.0, top_p=1.0, top_k=0)
    gc = GenerationConfig(**kw)
    torch.manual_seed(seed)
    out, canon = [], 0
    for b in range(0, len(rows), batch):
        chunk = rows[b:b + batch]
        ps = [prompt_ids(tok, hyp, r) for r in chunk]
        L = max(map(len, ps))
        x = torch.full((len(ps), L), pad, dtype=torch.long)
        am = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x[i, L - len(p):] = torch.tensor(p)
            am[i, L - len(p):] = 1
        with torch.inference_mode():
            g = model.generate(input_ids=x.cuda(), attention_mask=am.cuda(), generation_config=gc)[:, L:].tolist()
        for r, ids in zip(chunk, g):
            if eos in ids:
                ids, fr = ids[:ids.index(eos)], "stop"
            else:
                ids = [t for t in ids if t != pad] if pad != eos else ids
                fr = "length"
            text = tok.decode(ids, skip_special_tokens=True)
            canon += tok(text, add_special_tokens=False)["input_ids"] == ids
            out.append(dict(r, text=text, finish_reason=fr, max_tokens=max_new, n_gen=len(ids)))
    return out, canon


def cmd_gen(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from .hyp import Hyp, build
    os.makedirs(a.dir, exist_ok=True)
    base = Hyp(name="base_fp32", model=a.model, dtype="float32")
    ps = prompts(a.n)
    rows_all, stats = [], {}
    for tag, dtype, sample in [("self-greedy-fp32", "float32", False), ("self-sample-T1", "float32", True),
                               ("self-greedy-bf16", "bfloat16", False)]:
        if tag == "self-greedy-fp32" or tag == "self-greedy-bf16":
            model, tok, _ = build(Hyp(name="g", model=a.model, dtype=dtype), "cuda")
        rows = [{"id": f"{tag}-{i}", "endpoint": tag, "prompt_id": f"p{i}", "prompt": p} for i, p in enumerate(ps)]
        t0 = time.time()
        out, canon = _generate(model, tok, base, rows, sample, a.max_new, a.batch, seed=1234)
        stats[tag] = {"gen_s": round(time.time() - t0, 1), "canonical_retok": f"{canon}/{len(out)}",
                      "stop": sum(r["finish_reason"] == "stop" for r in out),
                      "mean_len": sum(r["n_gen"] for r in out) / len(out)}
        print(tag, stats[tag], flush=True)
        rows_all += out
        if tag == "self-sample-T1":
            # random LoRA adapter on the fp32 model (B initialised non-zero)
            cfg = LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", init_lora_weights=False)
            pm = get_peft_model(model, cfg)
            pm.save_pretrained(os.path.join(a.dir, "lora_rand"))
            del pm
            torch.cuda.empty_cache()
    with open(os.path.join(a.dir, "responses.jsonl"), "w") as fh:
        for r in rows_all:
            fh.write(json.dumps(r) + "\n")
    json.dump(stats, open(os.path.join(a.dir, "gen_stats.json"), "w"), indent=1)
    lora = os.path.join(a.dir, "lora_rand")
    M = a.model
    hyps = [
        dict(name="base_fp32", model=M, dtype="float32"),
        dict(name="base_bf16", model=M, dtype="bfloat16"),
        dict(name="base_fp16", model=M, dtype="float16"),
        dict(name="noise_1e-3", model=M, dtype="float32", noise={"sigma_rel": 1e-3, "seed": 0}),
        dict(name="noise_1e-2", model=M, dtype="float32", noise={"sigma_rel": 1e-2, "seed": 0}),
        dict(name="fp8_tensor", model=M, dtype="float32", quant="fp8_e4m3/tensor"),
        dict(name="fp8_w8a8_channel_token", model=M, dtype="float32", quant="fp8_e4m3/channel",
             act_quant="fp8_e4m3/token"),
        dict(name="int4_g128", model=M, dtype="float32", quant="int4/group128"),
        dict(name="lora_rand_s0.3", model=M, dtype="float32", lora={"path": lora, "scale": 0.3}),
        dict(name="lora_rand_s1.0", model=M, dtype="float32", lora={"path": lora, "scale": 1.0}),
        dict(name="sysprompt_pirate", model=M, dtype="float32", system_prompt="You are a pirate. Talk like one."),
    ]
    with open(os.path.join(a.dir, "hyps.jsonl"), "w") as fh:
        for h in hyps:
            fh.write(json.dumps(h) + "\n")
    print("wrote", a.dir)


def cmd_score(a):
    from .data import load_responses
    from .hyp import load_hyps
    from .score import score_endpoints
    groups = load_responses([os.path.join(a.dir, "responses.jsonl")])
    for h in load_hyps(os.path.join(a.dir, "hyps.jsonl")):
        if a.only and h.name not in a.only.split(","):
            continue
        score_endpoints(h, groups, os.path.join(a.dir, "out"))
        import torch
        torch.cuda.empty_cache()


def cmd_report(a):
    from . import store
    out = os.path.join(a.dir, "out")
    files = sorted(glob.glob(os.path.join(out, "*", "*.npz")))
    print(f"{'hyp':26s} {'endpoint':18s} {'rank1':>8s} {'hard':>5s} {'surpr':>7s} {'H':>7s} {'z(s-H)':>7s} "
          f"{'altseg':>6s} {'LLR/tok vs base':>16s} {'z_LLR':>8s} {'flip':>6s} {'ktok/s':>7s}")
    for f in files:
        hyp, ep = f.split(os.sep)[-2], os.path.basename(f)[:-4]
        arr, meta = store.load(f)
        s = store.summarize(arr)
        bf = os.path.join(out, "base_fp32", ep + ".npz")
        c = {}
        if hyp != "base_fp32" and os.path.exists(bf):
            try:
                c = store.compare(arr, store.load(bf)[0])
            except ValueError:
                c = {}
        t = meta["timing"]
        print(f"{hyp:26s} {ep:18s} {s['rank1']:8.4f} {s['hard_viol']:5d} {s['mean_surprise']:7.3f} "
              f"{s['mean_entropy']:7.3f} {s.get('z_surprise_entropy', float('nan')):7.2f} {s['alt_seg']:6d} "
              f"{c.get('llr_per_token', float('nan')):16.4f} {c.get('z_llr', float('nan')):8.2f} "
              f"{c.get('argmax_flip_rate', float('nan')):6.3f} {t['scored_tokens'] / max(t['score_s'], 1e-9) / 1e3:7.1f}")


def cmd_timing(a):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from .data import load_responses
    from .hyp import Hyp, build
    from .score import score_rows
    rows = load_responses([os.path.join(a.dir, "responses.jsonl")])["self-sample-T1"]
    rows = (rows * (a.n // len(rows) + 1))[:a.n]
    h = Hyp(name="timing", model=a.model, dtype=a.dtype, batch_tokens=a.batch_tokens)
    if a.config_model:
        cfg = AutoConfig.from_pretrained(a.config_model)
        with torch.device("cuda"):
            model = AutoModelForCausalLM.from_config(cfg, dtype=getattr(torch, a.dtype)).eval()
        tok = AutoTokenizer.from_pretrained(a.model)
        label = f"{a.config_model} (random init)"
    else:
        model, tok, _ = build(h, "cuda")
        label = a.model
    score_rows(model, tok, h, rows[:8])  # warm-up
    torch.cuda.synchronize()
    t0 = time.time()
    arr, meta = score_rows(model, tok, h, rows)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = int(arr["offsets"][-1])
    per_k = dt / n * 256_000
    print(json.dumps({"model": label, "dtype": a.dtype, "rows": len(rows), "scored_tokens": n, "seconds": round(dt, 2),
                      "scored_tok_per_s": round(n / dt), "s_per_1000_resp_256tok": round(per_k, 1),
                      "gpu": torch.cuda.get_device_name(0),
                      "max_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1)}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gen", "score", "report", "timing"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dir", default="sim/selftest")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--only")
    ap.add_argument("--config-model")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    a = ap.parse_args(argv)
    {"gen": cmd_gen, "score": cmd_score, "report": cmd_report, "timing": cmd_timing}[a.cmd](a)


if __name__ == "__main__":
    main()
