"""Synthetic endpoints with known truth, producing *text* responses so that the
whole pipeline (parsers included) is exercised.

A "model" is a set of logit tables, one per probe context:
  * SK  (K block, first_word): 6 prompts x 6 options, one word per response;
  * SS  (short stream, digits, comma): order-1 Markov chain on 10 digits,
        100 items per response;
  * SLL (long stream, letters a-h, comma): order-1 Markov chain, 100 items;
  * SF  (long form, 24 lines, options A/B/C): independent lines.
Deployments (``Deployment``): base logits plus
  * ``quant`` / ``lora``: additive perturbation eps * N(0,1) of every logit
    (``direction`` seed fixes the perturbation; strength scales it);
  * ``T``: temperature (logits / T); ``greedy``: argmax (deterministic);
  * ``wm_delta``: green-list watermark, + delta on a pseudo-random half of the
    options of each block, independent of the context (hash context constant);
  * ``sp``: system-prompt-like perturbation of the *short* probes only
    (sigma sp), long probes untouched (dilution).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset, Probe, Response

DIGITS = [str(i) for i in range(10)]
LETTERS = list("abcdefgh")
KOPTS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
FOPTS = ["A", "B", "C"]
N_K, N_FORM, STREAM_LEN = 6, 24, 100


def synth_probes() -> dict:
    P = {}
    for j in range(N_K):
        pid = f"SK{j + 1:02d}"
        P[pid] = Probe(pid, "K_single", "SK", {"type": "first_word", "options": KOPTS}, f"k prompt {j}", "SK", 4)
    P["SS"] = Probe("SS", "S_stream", "SS", {"type": "stream", "alphabet": DIGITS, "sep": "comma"}, "ss", "SS")
    P["SLL"] = Probe("SLL", "SL_stream_diluted", "SLL", {"type": "stream", "alphabet": LETTERS, "sep": "comma"},
                     "sll", "SLL")
    P["SF"] = Probe("SF", "F_form", "SF", {"type": "form", "n": N_FORM, "options": FOPTS}, "sf", "SF")
    P["SC"] = Probe("SC", "C_continuation", "SC", {"type": "free"}, "sc", None)
    return P


@dataclass
class Deployment:
    name: str
    base_seed: int = 0          # the checkpoint (family / base model)
    quant: float = 0.0
    lora: float = 0.0
    lora_dir: int = 1           # which LoRA adapter (direction seed)
    T: float = 1.0
    greedy: bool = False
    wm_delta: float = 0.0
    wm_key: int = 7
    sp: float = 0.0
    sp_key: int = 11
    quant_key: int = 3


def _base_tables(seed: int) -> dict:
    r = np.random.default_rng(1000 + seed)
    return {
        "SK": r.normal(0, 1.0, (N_K, len(KOPTS))),
        "SS": r.normal(0, 0.8, (11, 10)),          # row 10 = START
        "SLL": r.normal(0, 0.8, (9, len(LETTERS))),
        "SF": r.normal(0, 1.0, (N_FORM, len(FOPTS))),
        "SC": r.normal(0, 1.0, (40, 5)),
    }


def deployment_tables(d: Deployment) -> dict:
    tabs = _base_tables(d.base_seed)
    out = {}
    for k, L in tabs.items():
        L = L.copy()
        if d.quant:
            L += d.quant * np.random.default_rng([d.quant_key, d.base_seed, _stable_hash(k)]).normal(size=L.shape)
        if d.lora:
            L += d.lora * np.random.default_rng([500 + d.lora_dir, d.base_seed, _stable_hash(k)]).normal(size=L.shape)
        if d.sp and k in ("SK", "SS", "SC"):
            L += d.sp * np.random.default_rng([d.sp_key, _stable_hash(k)]).normal(size=L.shape)
        if d.wm_delta:
            green = np.random.default_rng([d.wm_key, _stable_hash(k)]).random(L.shape[1]) < 0.5
            L += d.wm_delta * green[None, :]
        out[k] = L
    return out


def _draw(rng, logits, T, greedy):
    if greedy:
        return int(np.argmax(logits))
    z = logits / T
    p = np.exp(z - z.max())
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def _stable_hash(s: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(s)) % 997


def generate(d: Deployment, n: dict, rng) -> dict:
    """Responses for one deployment: probe_id -> [Response]; n: probe_id -> count."""
    tabs = deployment_tables(d)
    out = {}
    for pid, k in n.items():
        lst = []
        for _ in range(k):
            if pid.startswith("SK"):
                j = int(pid[2:]) - 1
                txt = KOPTS[_draw(rng, tabs["SK"][j], d.T, d.greedy)].capitalize() + "."
            elif pid in ("SS", "SLL"):
                L = tabs[pid]
                alpha = DIGITS if pid == "SS" else LETTERS
                prev = L.shape[0] - 1
                seq = []
                for _ in range(STREAM_LEN):
                    prev = _draw(rng, L[prev], d.T, d.greedy)
                    seq.append(alpha[prev])
                txt = ", ".join(seq)
            elif pid == "SF":
                txt = "\n".join(f"{j + 1}: {FOPTS[_draw(rng, tabs['SF'][j], d.T, d.greedy)]}" for j in range(N_FORM))
            elif pid == "SC":
                words = ["the", "sea", "light", "keeper", "night"]
                txt = " ".join(words[_draw(rng, tabs["SC"][j], d.T, d.greedy)] for j in range(40))
            else:
                continue
            lst.append(Response(text=txt, finish_reason="stop"))
        out[pid] = lst
    return out


def default_counts(n_stream: int = 10, n_form: int = 10, n_k: int = 40, n_free: int = 0) -> dict:
    n = {f"SK{j + 1:02d}": n_k for j in range(N_K)}
    n.update({"SS": n_stream, "SLL": n_stream, "SF": n_form})
    if n_free:
        n["SC"] = n_free
    return n


def make_dataset(deps: list, n: dict | None = None, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    n = default_counts() if n is None else n
    resp = {d.name: generate(d, n, rng) for d in deps}
    return Dataset(synth_probes(), resp)


def ref_like_family(quant: float = 0.25, lora_unit: float = 0.5, T_alt: float = 1.4, wm_delta: float = 2.0,
                    sp: float = 0.6, prefix: str = "syn-", base_seed: int = 0) -> tuple:
    """Nine deployments built like ref-: (deployments, truth endpoint -> (family, model, kind))."""
    p = prefix
    deps = [
        Deployment(f"{p}01", base_seed),
        Deployment(f"{p}02", base_seed, quant=quant),
        Deployment(f"{p}03", base_seed, sp=sp),
        Deployment(f"{p}04", base_seed, lora=0.7 * lora_unit),
        Deployment(f"{p}05", base_seed, sp=sp, sp_key=12),
        Deployment(f"{p}06", base_seed, wm_delta=wm_delta),
        Deployment(f"{p}07", base_seed, lora=1.5 * lora_unit),
        Deployment(f"{p}08", base_seed, T=T_alt),
        Deployment(f"{p}09", base_seed, lora=0.3 * lora_unit),
    ]
    truth = {f"{p}01": ("P", "S", "plain"), f"{p}02": ("P", "T1", "quantized"),
             f"{p}03": ("P", "S", "system prompt"), f"{p}04": ("P", "T3", "LoRA 0.7"),
             f"{p}05": ("P", "S", "persona"), f"{p}06": ("P", "S", "watermark"),
             f"{p}07": ("P", "T4", "LoRA 1.5"), f"{p}08": ("P", "S", "sampling at temperature"),
             f"{p}09": ("P", "T2", "LoRA 0.3")}
    return deps, truth


def write_fake_files(ds: Dataset, truth: dict, directory) -> dict:
    """Write probes.jsonl, ledger.jsonl (client record format) and truth.csv for CLI tests."""
    import json
    from pathlib import Path
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "probes.jsonl", "w") as fh:
        for p in ds.probes.values():
            fh.write(json.dumps({"id": p.id, "category": p.category, "group": p.group, "parser": p.parser,
                                 "text": p.text, "wm_block": p.wm_block, "max_tokens": p.max_tokens,
                                 "priority": 1, "phases": []}) + "\n")
    with open(d / "ledger.jsonl", "w") as fh:
        k = 0
        for e, dd in ds.responses.items():
            for pid, rs in dd.items():
                for rep, r in enumerate(rs):
                    k += 1
                    key = f"{e}|{pid}|{rep}|256"
                    base = {"attempt_id": f"a{k}", "key": key, "endpoint": e, "prompt_id": pid,
                            "prompt": ds.probes[pid].text, "rep": rep, "max_tokens": 256}
                    fh.write(json.dumps(base | {"type": "attempt"}) + "\n")
                    fh.write(json.dumps(base | {"type": "response", "http_status": 200, "text": r.text,
                                                "finish_reason": "stop",
                                                "response": {"choices": [{"message": {"content": r.text}}]}})
                             + "\n")
    with open(d / "truth.csv", "w") as fh:
        fh.write("endpoint,family,model,kind\n")
        for e, (f, m, kd) in sorted(truth.items()):
            fh.write(f"{e},{f},{m},{kd}\n")
    return {"probes": d / "probes.jsonl", "ledger": d / "ledger.jsonl", "truth": d / "truth.csv"}
