"""Hypothesis = weights x dtype x quantization emulation x LoRA scale x template/system prompt.

A hypothesis is a JSON object (one per line in a hypotheses file). Fields:

  name            str, required, used as output sub-directory (safe chars only)
  model           HF repo id or local dir
  revision        optional git revision
  dtype           "bfloat16" | "float16" | "float32"           (default bfloat16)
  attn_impl       "sdpa" | "eager" | ...                        (default sdpa)
  quant           null | "<fmt>/<granularity>" weight fake-quant (round-trip) of nn.Linear:
                    fmt  in fp8_e4m3, fp8_e5m2, int8, int4
                    gran in tensor, channel, none (fp8 only: plain cast), group<N> (e.g. group128)
  act_quant       null | "fp8_e4m3/tensor" | "fp8_e4m3/token"  dynamic activation fake-quant
                    on the inputs of the quantized Linear layers (vLLM fp8 W8A8 style)
  quant_skip      module-name substrings left unquantized       (default ["lm_head"])
  noise           null | {"sigma_rel": float, "seed": int}  W += sigma_rel*std(W)*N(0,1) on Linear
  lora            null | {"path": str, "scale": float, "merge": bool}
  chat_template   null (tokenizer's) | "raw" (no template) | "file:PATH.jinja" | inline jinja
  system_prompt   null (no system message: template default applies) | str ("" = empty system msg)
  template_kwargs {} extra kwargs for apply_chat_template (e.g. {"enable_thinking": false})
  eos_token       null (tokenizer.eos_token) | token string appended when finish_reason == stop
  finish_policy   "auto" | "none" | "stop" | "length"            (default auto)
  topk            int (default 50)
  temps           list of temperatures for stored logsumexp      (default [0.5, 0.7, 1.0, 1.5])
  batch_tokens    padded tokens per forward batch                (default 16384)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Hyp:
    name: str
    model: str
    revision: str | None = None
    dtype: str = "bfloat16"
    attn_impl: str = "sdpa"
    quant: str | None = None
    act_quant: str | None = None
    quant_skip: list = field(default_factory=lambda: ["lm_head"])
    noise: dict | None = None
    lora: dict | None = None
    chat_template: str | None = None
    system_prompt: str | None = None
    template_kwargs: dict = field(default_factory=dict)
    eos_token: str | None = None
    finish_policy: str = "auto"
    topk: int = 50
    temps: list = field(default_factory=lambda: [0.5, 0.7, 1.0, 1.5])
    batch_tokens: int = 16384

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_.+=-]+", self.name):
            raise ValueError(f"bad hypothesis name {self.name!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "Hyp":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown hypothesis fields: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def load_hyps(path: str) -> list[Hyp]:
    txt = open(path).read().strip()
    items = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    hs = [Hyp.from_dict(x) for x in items]
    names = [h.name for h in hs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate hypothesis names")
    return hs


# ---------------------------------------------------------------- fake quantization
def _fp8_dtype(fmt: str):
    import torch
    return {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}[fmt]


def fake_quant(x, spec: str, row_dim: int = -1):
    """Round-trip x (any float tensor) through the format; returns float32.

    For weights [out, in]: 'channel' = one scale per output row; 'groupN' = per row, groups of N
    along the input dim. For activations, 'token' = one scale per row of the flattened input.
    """
    import torch
    fmt, gran = spec.split("/")
    x = x.float()
    shape = x.shape
    if gran.startswith("group"):
        g = int(gran[5:])
        x = x.reshape(-1, g)
        red = dict(dim=-1, keepdim=True)
    elif gran in ("channel", "token"):
        x = x.reshape(-1, shape[-1])
        red = dict(dim=-1, keepdim=True)
    elif gran in ("tensor", "none"):
        red = None
    else:
        raise ValueError(spec)
    amax = (x.abs().amax(**red) if red else x.abs().amax()).clamp_min(1e-12)
    if fmt.startswith("fp8"):
        fdt = _fp8_dtype(fmt)
        fmax = torch.finfo(fdt).max
        if gran == "none":
            y = x.clamp(-fmax, fmax).to(fdt).float()
        else:
            s = amax / fmax
            y = (x / s).clamp(-fmax, fmax).to(fdt).float() * s
    elif fmt in ("int8", "int4"):
        qmax = 127 if fmt == "int8" else 7
        s = amax / qmax
        y = torch.round(x / s).clamp(-qmax - 1, qmax) * s
    else:
        raise ValueError(spec)
    return y.reshape(shape)


def _linear_modules(model, skip):
    import torch.nn as nn
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(s in n for s in skip):
            yield n, m


def apply_noise(model, sigma_rel: float, seed: int = 0, skip=("lm_head",)):
    import torch
    n_mod = 0
    for n, m in _linear_modules(model, skip):
        h = int(hashlib.sha1(f"{seed}:{n}".encode()).hexdigest()[:8], 16)
        g = torch.Generator(device=m.weight.device).manual_seed(h)
        w = m.weight.data
        z = torch.randn(w.shape, generator=g, device=w.device, dtype=torch.float32)
        m.weight.data = (w.float() + sigma_rel * w.float().std() * z).to(w.dtype)
        n_mod += 1
    return n_mod


def apply_weight_quant(model, spec: str, skip):
    n_mod = 0
    for n, m in _linear_modules(model, skip):
        w = m.weight.data
        m.weight.data = fake_quant(w, spec).to(w.dtype)
        n_mod += 1
    return n_mod


def apply_act_quant(model, spec: str, skip):
    def hook(mod, args):
        x = args[0]
        return (fake_quant(x, spec).to(x.dtype),) + tuple(args[1:])
    n_mod = 0
    for n, m in _linear_modules(model, skip):
        m.register_forward_pre_hook(hook)
        n_mod += 1
    return n_mod


def apply_lora(model, path: str, scale: float = 1.0, merge: bool = False):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, path)
    n = 0
    for m in model.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict):
            for k in sc:
                sc[k] = sc[k] * scale
            n += 1
    if merge:
        model = model.merge_and_unload()
    return model, n


def build(h: Hyp, device: str = "cuda"):
    """Return (model, tokenizer, info dict)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = getattr(torch, h.dtype)
    tok = AutoTokenizer.from_pretrained(h.model, revision=h.revision)
    model = AutoModelForCausalLM.from_pretrained(h.model, revision=h.revision, dtype=dt,
                                                 attn_implementation=h.attn_impl)
    model.to(device).eval()
    info: dict[str, Any] = {"model_class": type(model).__name__,
                            "commit": getattr(model.config, "_commit_hash", None)}
    if h.noise:
        info["noise_modules"] = apply_noise(model, float(h.noise["sigma_rel"]), int(h.noise.get("seed", 0)),
                                            tuple(h.quant_skip))
    if h.quant:
        info["quant_modules"] = apply_weight_quant(model, h.quant, h.quant_skip)
    if h.act_quant:
        info["act_quant_modules"] = apply_act_quant(model, h.act_quant, h.quant_skip)
    if h.lora:
        model, n = apply_lora(model, h.lora["path"], float(h.lora.get("scale", 1.0)), bool(h.lora.get("merge", False)))
        info["lora_modules"] = n
        model.eval()
    return model, tok, info
