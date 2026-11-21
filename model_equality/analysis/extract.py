"""From one response text to observations (METHOD 4 "Parseurs" and 5.1).

``parse(probe, response) -> Parsed`` with
  * ``events``: Counter {(context, option): count} -- the rows of the count
    tables N_X[c, i] of METHOD 5.1;
  * ``ok``: the response is parsable (counts in the conformity rate);
  * ``fmt``: share of the response that is in format (0..1);
  * ``n``: number of draws (events) extracted; ``items``: the extracted sequence.

Context conventions (METHOD 5.1):
  stream, comma/line separator   previous item (order 1), START = '^'
  stream, space separator        two previous items (order 2)
  stream, binary alphabet        three previous items (order 3)
  form / form_open               line number
  first_word / first_match       the prompt (probe id)
  open_list (not in 5.1; choice) 'first' for the first item, 'rest' for the others
Contexts are prefixed by the probe id (several probes may share a unit).
Missing / out-of-option form answers (the '∅' of METHOD 4) are not events;
they lower ``fmt``.

Truncation: the server reports finish_reason="stop" even when cut at
max_tokens, so the last token / item of a stream or list is dropped whenever
the text does not end with a separator (and there are >= 10 of them).

Parsable thresholds (choices, METHOD does not fix them): stream: >= 10 items
and >= 80 % of tokens in the alphabet, or >= 30 items and >= 60 %; open_list: >= 10 items and >= 80 %
plausible items; form: >= 80 % of the n lines answered with a valid option;
first_word / first_match: a valid option found; free: non-empty.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .data import Probe, Response

START = "^"
NULL = "∅"
_STRIP = " \t\r\n*`\"'“”‘’«».,;:!?()[]{}<>"
_TOKEN_SPLIT = re.compile(r"[\s,;|/]+")
_FORM_LINE = re.compile(r"^\s*[*#>\-\s]*(?:q(?:uestion)?|item|n°|no\.?)?\s*(\d{1,3})\s*[:.)\]\-–=]+\s*(.*?)\s*$",
                        re.IGNORECASE)
_WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*|\d+", re.UNICODE)
_THOUSANDS = re.compile(r"(?<=\d)[,\s'](?=\d{3}(?!\d))")
_NUMBERING = re.compile(r"^\s*(?:\d{1,3}\s*[.):\-]|[-*•])\s*")


@dataclass
class Parsed:
    events: Counter = field(default_factory=Counter)
    ok: bool = False
    fmt: float = 0.0
    n: int = 0
    items: list = field(default_factory=list)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().strip(_STRIP).lower()
    return re.sub(r"\s+", " ", s)


def _option_map(options) -> dict | None:
    if not options:
        return None
    m = {}
    for o in options:
        m.setdefault(_norm(str(o)), str(o))
    return m


def _order(parser: dict) -> int:
    if len(parser.get("alphabet") or []) == 2:
        return 3
    return 2 if parser.get("sep") == "space" else 1


def _ctx_events(pid: str, items: list, k: int) -> Counter:
    ev = Counter()
    hist = [START] * k
    for it in items:
        ev[(f"{pid}|{'.'.join(hist[-k:])}", it)] += 1
        hist.append(it)
    return ev


def parse_stream(probe: Probe, resp: Response) -> Parsed:
    p = probe.parser
    amap = _option_map(p.get("alphabet")) or {}
    toks = [t for t in _TOKEN_SPLIT.split(resp.text or "") if t.strip(_STRIP)]
    if len(toks) >= 10 and not re.search(r"[,;\s.]$", resp.text or ""):
        toks = toks[:-1]   # last token possibly cut by max_tokens (finish_reason is unreliable)
    items = [amap[_norm(t)] for t in toks if _norm(t) in amap]
    fmt = len(items) / len(toks) if toks else 0.0
    out = Parsed(events=_ctx_events(probe.id, items, _order(p)), fmt=fmt, n=len(items), items=items)
    # METHOD (rev. 17:45): a stream stays usable at 60 % when valid items are many
    out.ok = (len(items) >= 10 and fmt >= 0.8) or (len(items) >= 30 and fmt >= 0.6)
    return out


def parse_open_list(probe: Probe, resp: Response) -> Parsed:
    sep = probe.parser.get("sep", "comma")
    text = resp.text or ""
    raw = text.splitlines() if sep == "line" else re.split(r"[,\n;]+", text)
    raw = [r for r in raw if r.strip()]
    if len(raw) >= 10 and not re.search(r"[,;\n.]\s*$", text):
        raw = raw[:-1]   # last item possibly cut by max_tokens (finish_reason is unreliable)
    items, bad = [], 0
    for r in raw:
        it = _norm(_NUMBERING.sub("", r))
        if it and len(it) <= 30 and len(it.split()) <= 3 and not it.endswith(":"):
            items.append(it)
        else:
            bad += 1
    ev = Counter()
    for j, it in enumerate(items):
        ev[(f"{probe.id}|{'first' if j == 0 else 'rest'}", it)] += 1
    fmt = len(items) / (len(items) + bad) if (items or bad) else 0.0
    return Parsed(events=ev, ok=len(items) >= 10 and fmt >= 0.8, fmt=fmt, n=len(items), items=items)


def _match_option(ans: str, omap: dict | None, open_: bool) -> str | None:
    a = _norm(ans)
    if not a:
        return None
    if omap is None:   # free form answer, normalised
        return _THOUSANDS.sub("", a)
    if a in omap:
        return omap[a]
    a2 = _THOUSANDS.sub("", a)
    if a2 in omap:
        return omap[a2]
    w = _WORD.findall(a)
    if not open_:
        if w and w[0] in omap:
            # real data (ref-, F01): the model echoes the item, "1: A: tea / B: coffee"; an answer that
            # names two different options is not a choice
            if any(x in omap and x != w[0] for x in w[1:]):
                return None
            return omap[w[0]]
        return None
    # form_open with an option list (tool names): first option occurring in the answer
    best = None
    for k, v in omap.items():
        m = re.search(r"(?<![a-z0-9_])" + re.escape(k) + r"(?![a-z0-9_])", a)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), v)
    return best[1] if best else None


def parse_form(probe: Probe, resp: Response) -> Parsed:
    p = probe.parser
    n = int(p.get("n") or 0)
    open_ = p.get("type") == "form_open"
    omap = _option_map(p.get("options"))
    answers: dict = {}
    for line in (resp.text or "").splitlines():
        m = _FORM_LINE.match(line)
        if not m:
            continue
        j = int(m.group(1))
        if (n and not 1 <= j <= n) or j in answers:
            continue
        answers[j] = _match_option(m.group(2), omap, open_)
    ev = Counter()
    items = []
    for j in sorted(answers):
        o = answers[j]
        if o is not None:
            ev[(f"{probe.id}|{j}", o)] += 1
            items.append((j, o))
    denom = n or max(len(answers), 1)
    fmt = len(items) / denom
    return Parsed(events=ev, ok=fmt >= 0.8, fmt=fmt, n=len(items), items=items)


def parse_first_word(probe: Probe, resp: Response) -> Parsed:
    omap = _option_map(probe.parser.get("options"))
    words = _WORD.findall(unicodedata.normalize("NFKC", resp.text or ""))
    if not words:
        return Parsed()
    w = words[0].lower()
    o = (omap.get(w) if omap is not None else w)
    if o is None:
        return Parsed(fmt=0.0, items=[w])
    nw = len(words)
    return Parsed(events=Counter({(probe.id, o): 1}), ok=True, fmt=1.0 if nw <= 2 else 1.0 / nw, n=1, items=[o])


def parse_first_match(probe: Probe, resp: Response) -> Parsed:
    omap = _option_map(probe.parser.get("options")) or {}
    text = _norm(resp.text or "")
    best = None
    for k, v in omap.items():
        m = re.search(r"(?<![a-z0-9_])" + re.escape(k) + r"(?![a-z0-9_])", text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), v)
    if best is None:
        return Parsed()
    fmt = 1.0 if text == _norm(best[1]) else 0.5
    return Parsed(events=Counter({(probe.id, best[1]): 1}), ok=True, fmt=fmt, n=1, items=[best[1]])


def parse_free(probe: Probe, resp: Response) -> Parsed:
    t = resp.text or ""
    return Parsed(ok=bool(t.strip()), fmt=1.0 if t.strip() else 0.0, items=[t])


_TREE_TOKEN = re.compile(r"[^\W\d_]+|\d|\n+|[^\w\s]", re.UNICODE)


def parse_tree(probe: Probe, resp: Response) -> Parsed:
    """Prefix tree: event (exact prefix of the first j tokens, token j) for j < K (default 12).

    Tokens are words, single digits, punctuation marks and line breaks (one or several), case kept; a response
    shorter than K tokens ends with an ``<eos>`` token (stopping early is a choice of the model).  The context
    is the whole response so far, so these are exact repeated contexts: T_temp is exact for them (no
    marginalisation over a hidden history).  Contexts deeper than the data allows are dropped by the statistics
    themselves (pooled total < 10 for T_raw, common support for T_temp).
    """
    K = int(probe.parser.get("K") or 12)
    toks = _TREE_TOKEN.findall(resp.text or "")[:K]
    toks = [t if not t.startswith("\n") else "\\n" * min(len(t), 2) for t in toks]
    if len(toks) < K:
        toks.append("<eos>")
    ev = Counter()
    for j, t in enumerate(toks):
        ev[(f"{probe.id}|{j}|" + "\x1f".join(toks[:j]), t)] += 1
    return Parsed(events=ev, ok=bool((resp.text or "").strip()), fmt=1.0, n=len(toks), items=toks)


PARSERS = {
    "tree": parse_tree,
    "stream": parse_stream,
    "open_list": parse_open_list,
    "form": parse_form,
    "form_open": parse_form,
    "first_word": parse_first_word,
    "first_match": parse_first_match,
    "free": parse_free,
}


def parse(probe: Probe, resp: Response) -> Parsed:
    return PARSERS.get(probe.ptype, parse_free)(probe, resp)


def parse_dataset(ds) -> dict:
    """endpoint -> probe_id -> [Parsed] (same order as ds.responses)."""
    return {e: {pid: [parse(ds.probes[pid], r) for r in rs] for pid, rs in d.items()}
            for e, d in ds.responses.items()}


def conformity_rows(ds, parsed: dict | None = None) -> list:
    """One row per (endpoint, probe): counts, parsable rate, mean format share, mean draws."""
    parsed = parse_dataset(ds) if parsed is None else parsed
    rows = []
    for e in sorted(parsed):
        for pid in sorted(parsed[e]):
            ps = parsed[e][pid]
            pr = ds.probes[pid]
            k = len(ps)
            rows.append({
                "endpoint": e, "probe": pid, "unit": pr.unit, "category": pr.category, "parser": pr.ptype,
                "n_responses": k, "n_parsable": sum(p.ok for p in ps),
                "parsable_rate": (sum(p.ok for p in ps) / k) if k else float("nan"),
                "mean_fmt": (sum(p.fmt for p in ps) / k) if k else float("nan"),
                "mean_draws": (sum(p.n for p in ps) / k) if k else float("nan"),
            })
    return rows
