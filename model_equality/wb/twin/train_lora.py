"""Short real LoRA fine-tune (peft), on an open instruction corpus, for realistic amplitude.

Corpora (downloaded on the login node into the HF cache, read offline):
  dolly      databricks/databricks-dolly-15k  (human-written instruction / response pairs)
  codealpaca sahil2801/CodeAlpaca-20k         (code instructions)
The prompt is formatted with the model's chat template (default system prompt), loss on the
response tokens only. Writes the adapter to OUT and OUT/train_log.json; OUT/adapter_config.json
+ OUT/DONE mark completion.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import time

import torch

HF = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def load_corpus(name: str) -> list[tuple[str, str]]:
    if name == "dolly":
        f = glob.glob(f"{HF}/hub/datasets--databricks--databricks-dolly-15k/snapshots/*/databricks-dolly-15k.jsonl")[0]
        out = []
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            u = d["instruction"] + (("\n\n" + d["context"]) if d.get("context") else "")
            if 20 < len(d["response"]) < 1500 and len(u) < 1500:
                out.append((u, d["response"]))
        return out
    if name == "codealpaca":
        f = glob.glob(f"{HF}/hub/datasets--sahil2801--CodeAlpaca-20k/snapshots/*/code_alpaca_20k.json")[0]
        out = []
        for d in json.load(open(f, encoding="utf-8")):
            u = d["instruction"] + (("\n\n" + d["input"]) if d.get("input") else "")
            if 10 < len(d["output"]) < 1500:
                out.append((u, d["output"]))
        return out
    raise ValueError(name)


def train(model_id: str, cfg: dict, out: str, dtype: str = "bfloat16", device: str = "cuda", max_len: int = 512,
          micro: int = 4, accum: int = 2, log=print):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    rnd = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    tg = (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if cfg["targets"] == "all"
          else ["q_proj", "v_proj"])
    lc = LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.05, target_modules=tg, task_type="CAUSAL_LM")
    model = get_peft_model(model, lc)
    data = load_corpus(cfg["data"])
    rnd.shuffle(data)
    eos = tok.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tok.get_vocab() else tok.eos_token_id

    def encode(u, a):
        p = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ai = tok(a, add_special_tokens=False)["input_ids"] + [eos]
        ids = (pi + ai)[:max_len]
        lab = ([-100] * len(pi) + ai)[:max_len]
        return ids, lab

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    steps, warm = cfg["steps"], max(1, cfg["steps"] // 15)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    k = 0
    hist = []
    model.train()
    for step in range(steps):
        tot = 0.0
        for _ in range(accum):
            batch = [encode(*data[(k + i) % len(data)]) for i in range(micro)]
            k += micro
            L = max(len(b[0]) for b in batch)
            x = torch.full((micro, L), pad, dtype=torch.long)
            y = torch.full((micro, L), -100, dtype=torch.long)
            am = torch.zeros((micro, L), dtype=torch.long)
            for j, (ids, lab) in enumerate(batch):
                x[j, :len(ids)] = torch.tensor(ids)
                y[j, :len(lab)] = torch.tensor(lab)
                am[j, :len(ids)] = 1
            loss = model(input_ids=x.to(device), attention_mask=am.to(device), labels=y.to(device)).loss / accum
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        hist.append(round(tot, 4))
        if step % 10 == 0 or step == steps - 1:
            log(f"lora {cfg['data']} step {step + 1}/{steps} loss {tot:.4f} ({time.time() - t0:.0f}s)")
    model.save_pretrained(out)
    json.dump({"model": model_id, "cfg": cfg, "loss": hist, "seconds": round(time.time() - t0, 1),
               "examples_seen": k}, open(os.path.join(out, "train_log.json"), "w"))
    open(os.path.join(out, "DONE"), "w").write("ok\n")
    del model, opt
    torch.cuda.empty_cache()
