"""Responses input and context reconstruction.

Accepted input (JSONL, one object per line), either:

* simple rows:  {"id", "endpoint", "prompt" | "messages", "text",
                 "finish_reason"?, "max_tokens"?, "prompt_id"?}
* the API ledger (model_equality/api/ledger/ledger.jsonl): only records with
  type == "response" and http_status == 200 are kept; id = attempt_id.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Iterable

FLAG_EOS_APPENDED = 1   # last token is the end-of-turn token added because finish_reason == stop
FLAG_ALT_SEG = 2        # rank > 1 but the top-1 token also spells the observed text (tokenization ambiguity)
FLAG_TRUNC_DROPPED = 4  # set on the (new) last token of a row whose truncated last token was dropped


def _norm(r: dict) -> dict | None:
    if r.get("type") == "attempt" or r.get("type") == "error":
        return None
    if r.get("type") == "response":
        if r.get("http_status") != 200 or r.get("text") is None:
            return None
        rid = r.get("attempt_id") or r.get("key")
    else:
        rid = r.get("id") or r.get("key")
    if r.get("text") is None or (r.get("prompt") is None and r.get("messages") is None):
        return None
    return {
        "id": str(rid),
        "endpoint": str(r.get("endpoint", "unknown")),
        "prompt_id": r.get("prompt_id"),
        "key": r.get("key"),
        "prompt": r.get("prompt"),
        "messages": r.get("messages"),
        "text": r["text"],
        "finish_reason": r.get("finish_reason"),
        "max_tokens": r.get("max_tokens"),
    }


def load_responses(paths: Iterable[str], endpoints: set[str] | None = None) -> "OrderedDict[str, list[dict]]":
    """Return {endpoint: [rows]} (dedup by id, file order kept)."""
    seen: set[str] = set()
    out: "OrderedDict[str, list[dict]]" = OrderedDict()
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = _norm(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if r is None or r["id"] in seen:
                    continue
                if endpoints and r["endpoint"] not in endpoints:
                    continue
                seen.add(r["id"])
                out.setdefault(r["endpoint"], []).append(r)
    return out


def messages_for(row: dict, system_prompt: str | None) -> list[dict]:
    msgs = list(row["messages"]) if row.get("messages") else [{"role": "user", "content": row["prompt"]}]
    if system_prompt is not None and not (msgs and msgs[0]["role"] == "system"):
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return msgs


def resolve_template(spec: str | None) -> str | None:
    """None -> tokenizer default; 'raw' handled by caller; 'file:PATH' -> file content; else inline jinja."""
    if spec is None or spec == "raw":
        return None
    if spec.startswith("file:"):
        with open(spec[5:]) as fh:
            return fh.read()
    return spec


def prompt_ids(tok, hyp, row: dict, _tpl_cache: dict = {}) -> list[int]:
    if hyp.chat_template == "raw":
        text = row["prompt"] if row.get("prompt") is not None else "\n".join(m["content"] for m in row["messages"])
        if hyp.system_prompt:
            text = hyp.system_prompt + "\n" + text
        return tok(text, add_special_tokens=True)["input_ids"]
    key = hyp.chat_template
    if key not in _tpl_cache:
        _tpl_cache[key] = resolve_template(key)
    tpl = _tpl_cache[key]
    s = tok.apply_chat_template(messages_for(row, hyp.system_prompt), tokenize=False,
                                add_generation_prompt=True, chat_template=tpl,
                                **(hyp.template_kwargs or {}))
    return tok(s, add_special_tokens=False)["input_ids"]


def eos_id(tok, hyp) -> int:
    if hyp.eos_token is not None:
        i = tok.convert_tokens_to_ids(hyp.eos_token)
        if i is None or i == tok.unk_token_id:
            raise ValueError(f"unknown eos_token {hyp.eos_token!r}")
        return int(i)
    return int(tok.eos_token_id)


def response_ids(tok, hyp, row: dict) -> tuple[list[int], list[int], str]:
    """Observed tokens + per-token flags + effective finish policy applied."""
    ids = tok(row["text"], add_special_tokens=False)["input_ids"]
    flags = [0] * len(ids)
    fr = row.get("finish_reason")
    mt = row.get("max_tokens")
    pol = hyp.finish_policy
    eff = "none"
    if pol == "auto":
        if fr == "length" or (mt and len(ids) >= int(mt)):
            eff = "length"
        elif fr == "stop":
            eff = "stop"
    elif pol in ("stop", "length"):
        eff = pol
    if eff == "length" and ids:
        ids, flags = ids[:-1], flags[:-1]
        if flags:
            flags[-1] |= FLAG_TRUNC_DROPPED
    elif eff == "stop":
        ids = ids + [eos_id(tok, hyp)]
        flags = flags + [FLAG_EOS_APPENDED]
    return ids, flags, eff
