"""Twin of the Model Equality bench: families, simulated endpoints, probe plan, truth.

A family = one open base model + simulated endpoints mirroring the organisers' bench:
plain, hidden system prompt, persona drawn per request, decoding watermark (2 strengths),
T=1.0, greedy, fp8 weight-only per-tensor, a second quantization, one LoRA at 3 strengths,
a second LoRA. Everything is sampled with temperature only (no top-k/top-p, no repetition
penalty), T=0.7 unless the endpoint says otherwise, model chat template with its default
system prompt, max_tokens from the probe (<= 256), one independent draw per request.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = ("You are Nova, the virtual assistant of Brightline Telecom. Be friendly and concise, "
                 "answer in the user's language, and never mention competitors.")

PERSONAS = [
    "You are a retired history teacher who loves telling anecdotes about the past.",
    "You are a cheerful fitness coach. Keep your answers upbeat and motivating.",
    "You are a meticulous software engineer who values precision above all.",
    "You are a poet at heart; you like vivid images and a touch of lyricism.",
    "You are a calm customer-support agent for an online bookstore.",
    "You are an enthusiastic amateur astronomer who relates things to the night sky.",
    "You are a pragmatic financial advisor. Be cautious and clear.",
    "You are a friendly chef who often draws comparisons with cooking.",
]

# ---------------------------------------------------------------- probe plan (per endpoint)
# Proportions of the plan for a 100-response budget, x5: diagnostic ~12, discrete battery ~70,
# free text (continuations, domain) ~18.
PLAN = {
    # diagnostic (60)
    "tp_story": 10, "E01_identity_long": 5, "E02_pi_digits": 10, "E03_story_zh": 10, "D03_leak_repeat": 10,
    "D01_identity": 5, "D06_persona": 5, "D08_fridge": 5,
    # discrete battery (350)
    "SL01_coin_long": 60, "SL06_abcd_long": 60, "F05_mcq": 50, "F20_tools_a": 50, "F02_yesno": 40,
    "S02_digits": 30, "SL02_digits_long": 30, "F01_ab_pref": 30,
    # free text (90)
    "C01_en": 6, "C02_fr": 6, "C06_en_science": 6, "C08_story": 6, "C09_en_long": 6,
    "L01_python": 6, "L03_medical": 6, "L05_poem": 6, "L07_math": 6, "L09_refusal": 6,
    "L13_alpaca": 6, "L16_emotional": 6, "L19_story_kids": 6, "L20_steps": 6, "L22_opinion": 6,
}

SMALL_PLAN_SCALE = None  # set by --plan-scale for login tests


def lora_dir(root, fam, name):
    return os.path.join(root, "lora", fam, name)


def family_specs(root: str) -> dict:
    """fam -> dict(model, dtype, endpoints: [dict(name, kind, model_label, ...)] , loras: {name: train cfg})."""
    def eps(prefix, fam, full=True):
        L = []
        n = [0]

        def add(kind, label, **kw):
            n[0] += 1
            L.append(dict(name=f"{prefix}-{n[0]:02d}", kind=kind, model_label=label, **kw))
        add("plain", "S")
        add("system prompt", "S", system=SYSTEM_PROMPT)
        add("persona", "S", persona=True)
        add("watermark (KGW lefthash gamma 0.25 delta 2.0)", "S", wm=dict(gamma=0.25, delta=2.0))
        if full:
            add("watermark (KGW lefthash gamma 0.5 delta 1.0)", "S", wm=dict(gamma=0.5, delta=1.0))
        add("sampling at temperature 1.0", "S", T=1.0)
        if full:
            add("greedy decoding", "S", greedy=True)
        add("quantized (fp8 e4m3 weight-only per-tensor)", "Q8T", quant="fp8_e4m3/tensor")
        add("quantized (int8 weight-only per-channel)", "Q8C", quant="int8/channel")
        add("LoRA A at strength 0.3", "LA03", lora="A", scale=0.3)
        add("LoRA A at strength 0.7", "LA07", lora="A", scale=0.7)
        add("LoRA A at strength 1.5", "LA15", lora="A", scale=1.5)
        if full:
            add("LoRA B at strength 1.0", "LB10", lora="B", scale=1.0)
            # LoRA A was much stronger than the bench's at low strength (excess surprise at T=0.7:
            # 0.25 nat/token at 0.3 vs 0.03 on ref-09): calibrated copy, A scaled by 0.25
            for s_ in (0.3, 0.7, 1.5):
                add(f"LoRA A calibrated (A x 0.25) at strength {s_}", f"LC{int(s_ * 10):02d}", lora="A",
                    scale=round(0.25 * s_, 4))
        return L

    return {
        "Q": dict(model="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16", prefix="tw", endpoints=eps("tw", "Q"),
                  loras={"A": dict(data="dolly", r=16, alpha=32, targets="all", steps=150, lr=2e-4, seed=1),
                         "B": dict(data="codealpaca", r=8, alpha=16, targets="qv", steps=150, lr=3e-4, seed=2)}),
        "O": dict(model="allenai/OLMo-2-0425-1B-Instruct", dtype="float32", prefix="to", gen_batch_tokens=40_000, endpoints=eps("to", "O", full=False),
                  loras={"A": dict(data="dolly", r=16, alpha=32, targets="all", steps=60, lr=3e-4, seed=3)}),
        # login-node test family (small model already cached)
        "q": dict(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float32", prefix="tq", endpoints=eps("tq", "q"),
                  loras={"A": dict(data="dolly", r=16, alpha=32, targets="all", steps=30, lr=2e-4, seed=1),
                         "B": dict(data="codealpaca", r=8, alpha=16, targets="qv", steps=30, lr=3e-4, seed=2)}),
    }


def write_truth(path: str, fams: dict, which: list):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["endpoint", "family", "model", "kind"])
        for f in which:
            for e in fams[f]["endpoints"]:
                w.writerow([e["name"], f, f"{f}_{e['model_label']}", e["kind"]])


def load_probes(path: str) -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d
    return out


def plan(scale: float = 1.0) -> dict:
    return {p: max(1, int(round(n * scale))) for p, n in PLAN.items()}


def seed_of(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)
