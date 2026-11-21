"""Shared helpers: ledger reading, tokenizer loading, chat-template rendering, lineages."""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
HF = HERE / "hf"
OUT = HERE / "out"
DEFAULT_LEDGER = HERE.parent.parent / "api" / "ledger" / "ledger.jsonl"


# ----------------------------------------------------------------- ledger
def load_responses(path=DEFAULT_LEDGER) -> list[dict]:
    """Read-only, lock-free reader of 'response' records (tolerates a partial last line).

    Same fields as model_equality.api.ledger.load_responses, plus the raw server body.
    We do not import that module because the package __init__ pulls in `requests`.
    """
    out = []
    with open(path, "rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                r = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or r.get("type") != "response":
                continue
            body = r.get("response") if isinstance(r.get("response"), dict) else {}
            text = r.get("text")
            fr = r.get("finish_reason")
            try:
                ch = body["choices"][0]
                text = ch["message"]["content"] if text is None else text
                fr = ch.get("finish_reason") if fr is None else fr
            except Exception:
                pass
            out.append({
                "endpoint": r.get("endpoint"), "prompt_id": r.get("prompt_id"),
                "prompt": r.get("prompt"), "rep": r.get("rep"), "max_tokens": r.get("max_tokens"),
                "text": text, "finish_reason": fr, "usage": body.get("usage"),
                "body": body, "ts": r.get("ts"),
            })
    return out


def endpoint_matches(ep: str, pats: list[str] | None) -> bool:
    if not pats:
        return True
    return any(re.fullmatch(p.replace("*", ".*"), ep) for p in pats)


# ----------------------------------------------------------------- tokenizers
def models() -> list[str]:
    src = json.loads((HF / "_sources.json").read_text())
    return sorted(n for n, i in src.items() if i.get("repo") and (HF / n / "tokenizer.json").exists())


@lru_cache(maxsize=None)
def tokenizer(name: str) -> Tokenizer:
    return Tokenizer.from_file(str(HF / name / "tokenizer.json"))


def _sha(obj) -> str:
    if isinstance(obj, bytes):
        return hashlib.sha256(obj).hexdigest()[:12]
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def core_hash(name: str) -> str:
    """Hash of what determines how *plain text* is split: model (vocab+merges), normalizer,
    pre_tokenizer, and the non-special added tokens (they are matched inside text)."""
    j = json.loads((HF / name / "tokenizer.json").read_bytes())
    added = sorted((t["content"], t.get("normalized"), t.get("single_word"), t.get("lstrip"), t.get("rstrip"))
                   for t in j.get("added_tokens", []) if not t.get("special"))
    return _sha({"model": j.get("model"), "normalizer": j.get("normalizer"),
                 "pre_tokenizer": j.get("pre_tokenizer"), "added": added})


def bytes_hash(name: str) -> str:
    return _sha((HF / name / "tokenizer.json").read_bytes())


def count(name: str, text: str) -> int:
    return len(tokenizer(name).encode(text, add_special_tokens=False).ids)


def load_lineages() -> dict:
    """atlas.json -> {'lineages': {lid: {...members...}}, 'models': {...}}"""
    return json.loads((OUT / "atlas.json").read_text())


def lineage_rep(lineage: dict) -> str:
    return lineage["representative"]


# ----------------------------------------------------------------- chat templates
def chat_template(name: str) -> str | None:
    d = HF / name
    if (d / "chat_template.jinja").exists():
        return (d / "chat_template.jinja").read_text()
    if (d / "chat_template.json").exists():
        ct = json.loads((d / "chat_template.json").read_text()).get("chat_template")
        if ct:
            return ct
    cfg = json.loads((d / "tokenizer_config.json").read_text())
    ct = cfg.get("chat_template")
    if isinstance(ct, list):  # named templates
        ct = next((t["template"] for t in ct if t.get("name") == "default"), ct[0]["template"])
    return ct


def _special(cfg: dict, key: str) -> str:
    v = cfg.get(key)
    if isinstance(v, dict):
        v = v.get("content")
    return v or ""


def render_chat(name: str, user: str, system: str | None = None, add_generation_prompt=True) -> str | None:
    """Render the chat template like transformers' apply_chat_template(tokenize=False)."""
    tpl = chat_template(name)
    if tpl is None:
        return None
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        import datetime
        return datetime.datetime(2026, 11, 21).strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    cfg = json.loads((HF / name / "tokenizer_config.json").read_text())
    smap = HF / name / "special_tokens_map.json"
    sm = json.loads(smap.read_text()) if smap.exists() else {}
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": user}]
    tpl = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", tpl)  # transformers-only tag
    return env.from_string(tpl).render(
        messages=msgs, add_generation_prompt=add_generation_prompt,
        bos_token=_special(cfg, "bos_token") or _special(sm, "bos_token"),
        eos_token=_special(cfg, "eos_token") or _special(sm, "eos_token"),
        unk_token=_special(cfg, "unk_token"), pad_token=_special(cfg, "pad_token"),
        tools=None, documents=None, enable_thinking=True, date_string="21 Nov 2026")


def prompt_token_count(name: str, user: str, system: str | None = None) -> int | None:
    """Prompt tokens as vLLM's /chat/completions counts them (and as transformers'
    apply_chat_template(tokenize=True)): template rendered as text, encoded with
    add_special_tokens=False (the template itself carries BOS when the model wants one).
    Cross-checked against transformers 5.x on 9 models (identical)."""
    s = render_chat(name, user, system)
    if s is None:
        return None
    return len(tokenizer(name).encode(s, add_special_tokens=False).ids)
