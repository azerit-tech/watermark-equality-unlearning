"""Per-endpoint lure labels (METHOD 5.3, 5.4, 5.5).

Labels are read on the pair (X, reference of the family).  Each endpoint gets
one primary label among DET, WM, SP, TEMP, NONE (priority in that order) plus
the list of every criterion met, for review.

* DET: repeated prompts answered identically (>= 90 % of the prompts that
  have >= 2 responses of >= 40 characters, at least one such prompt).
* TEMP: interval of ln kappa_hat outside [-0.15, 0.15] and z_temp <= 2.
* WM: z_temp > 4, z_inv <= 2 and the two-valued offset signature (pairs.wm_signature); or
  ``WM:repeat``: excess of identical / long-common-prefix answers among repetitions of a prompt
  (``repeat_stats``; this is what the ref-06 watermark actually looks like, the green-list signature
  never fired on real data).
* SP: leak (>= 8 word 6-grams (one 13-word span) repeated in >= 2 of X's answers to the leak
  probes and absent from the answers of every other endpoint of the family and
  from the prompt; a version checked only against the reference fired on
  generic phrases of ref-02/04/09), or token count
  (usage.prompt_tokens, if the API ever returns it), or dilution contrast
  z_temp_short > 4 and z_temp_long <= 2, or identity drift (``identity_drift``:
  the proper nouns that the family repeats in its diagnostic answers, e.g. the
  default template identity "SmolLM ... Hugging Face" on ref-, never appear in
  the endpoint's >= 10 diagnostic answers while the family median rate is >=
  15 %; on ref- this separates ref-03 / ref-05 (0/23) from the others, 17-36 %).
"""
from __future__ import annotations

import math
import re
from collections import Counter

LEAK_PROBES = ("D03_leak_repeat", "D04_leak_instructions", "D05_leak_translate", "D06_persona",
               "D09_rules", "D10_first_words")
LNK_BAND = 0.15
Z_SAME, Z_DIFF = 2.0, 4.0


class PairIndex:
    """Symmetric access to pair rows; kappa reported in the requested orientation."""

    def __init__(self, rows: list):
        self.rows = {(r["a"], r["b"]): r for r in rows}

    def get(self, x: str, y: str) -> dict | None:
        r = self.rows.get((x, y))
        if r is not None:
            return r
        r = self.rows.get((y, x))
        if r is None:
            return None
        r = dict(r)
        for k in list(r):
            if k.startswith("kappa") and isinstance(r[k], float) and r[k] > 0:
                r[k] = 1.0 / r[k]
        lo, hi = r.get("lnk_lo"), r.get("lnk_hi")
        if lo is not None and hi is not None:
            r["lnk_lo"], r["lnk_hi"] = -hi, -lo
        r["a"], r["b"] = x, y
        return r


def _f(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def det_flags(ds, min_len: int = 40) -> dict:
    out = {}
    for e in ds.endpoints:
        n_chk = n_same = 0
        for pid, rs in ds.responses[e].items():
            texts = [r.text for r in rs if r.text and len(r.text) >= min_len]
            if len(texts) >= 2:
                n_chk += 1
                n_same += len(set(texts)) == 1
        for pid, rs in ds.other.get(e, {}).items():
            texts = [r.text for r in rs if r.text and len(r.text) >= min_len]
            if len(texts) >= 2:
                n_chk += 1
                n_same += len(set(texts)) == 1
        out[e] = {"det": n_chk > 0 and n_same >= 0.9 * n_chk, "det_checked": n_chk, "det_identical": n_same}
    return out


REPEAT_MIN_LEN, REPEAT_LCP, REPEAT_EXCESS, REPEAT_MIN_PAIRS = 80, 100, 0.08, 60


def _lcp(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def repeat_stats(ds, eps: list, min_len: int = REPEAT_MIN_LEN, lcp_min: int = REPEAT_LCP) -> dict:
    """Partial determinism among repetitions (signature of the ref-06 watermark, found on real data).

    For every endpoint: share of pairs of answers to the same prompt (same max_tokens, both >= ``min_len``
    characters) that are identical or share a prefix of >= ``lcp_min`` characters, and its excess over what
    the family median gives prompt by prompt.  Measured on ref- after R1_b: ref-06 (watermark) 0.149, excess
    +0.129; persona 0.078 (+0.057); the seven others 0.000 to 0.040 (excess <= +0.023).  The watermark of the
    bench is distortion-free (token frequencies unchanged) but makes whole answers repeat; a low default
    temperature would raise the share of every endpoint of the family, hence the comparison to the median.
    Returns endpoint -> {'rep_pairs', 'rep_share', 'rep_excess', 'rep_flag'}.
    """
    per: dict = {}
    for e in eps:
        for pid, rs in ds.responses.get(e, {}).items():
            if pid.endswith("@tree"):
                continue
            by: dict = {}
            for r in rs:
                if r.text and len(r.text) >= min_len:
                    by.setdefault(r.max_tokens, []).append(r.text)
            n = k = 0
            for ts in by.values():
                for i in range(len(ts)):
                    for j in range(i + 1, len(ts)):
                        n += 1
                        k += ts[i] == ts[j] or _lcp(ts[i], ts[j]) >= lcp_min
            if n:
                per[(e, pid)] = (n, k)
    med = {}
    for pid in {p for _, p in per}:
        sh = sorted(per[(e, pid)][1] / per[(e, pid)][0] for e in eps if (e, pid) in per and per[(e, pid)][0] >= 3)
        if len(sh) >= 3:
            med[pid] = sh[len(sh) // 2] if len(sh) % 2 else 0.5 * (sh[len(sh) // 2 - 1] + sh[len(sh) // 2])
    out = {}
    for e in eps:
        keys = [(e, p) for p in med if (e, p) in per]
        n = sum(per[k][0] for k in keys)
        k = sum(per[k_][1] for k_ in keys)
        exp = sum(per[k_][0] * med[k_[1]] for k_ in keys)
        share = k / n if n else float("nan")
        exc = (k - exp) / n if n else float("nan")
        out[e] = {"rep_pairs": n, "rep_share": share, "rep_excess": exc,
                  "rep_flag": bool(n >= REPEAT_MIN_PAIRS and exc >= REPEAT_EXCESS)}
    return out


def _ngrams(text: str, n: int = 6) -> set:
    w = re.findall(r"\w+", (text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def leak_ngrams(ds, X: str, others, n: int = 6) -> list:
    """Word n-grams repeated in >= 2 of X's answers to the leak probes and absent from the answers
    of every other endpoint in ``others`` (the family) and from the prompts: endpoint-specific text."""
    if isinstance(others, str):
        others = [others]
    cnt = Counter()
    seen_elsewhere, prompt = set(), set()
    for pid in LEAK_PROBES:
        if pid in ds.probes:
            prompt |= _ngrams(ds.probes[pid].text, n)
        for r in ds.responses.get(X, {}).get(pid, []):
            cnt.update(_ngrams(r.text, n))
        for o in others:
            if o == X:
                continue
            for r in ds.responses.get(o, {}).get(pid, []):
                seen_elsewhere |= _ngrams(r.text, n)
    return sorted(g for g, c in cnt.items() if c >= 2 and g not in seen_elsewhere and g not in prompt)


def prompt_token_offset(ds, X: str, R: str) -> float:
    """Median difference of usage.prompt_tokens on identical prompts (nan if unavailable)."""
    diffs = []
    for pid in set(ds.responses.get(X, {})) & set(ds.responses.get(R, {})):
        tx = [r.usage.get("prompt_tokens") for r in ds.responses[X][pid] if isinstance(r.usage, dict)]
        tr = [r.usage.get("prompt_tokens") for r in ds.responses[R][pid] if isinstance(r.usage, dict)]
        tx, tr = [t for t in tx if t is not None], [t for t in tr if t is not None]
        if tx and tr:
            diffs.append(sorted(tx)[len(tx) // 2] - sorted(tr)[len(tr) // 2])
    if not diffs:
        return float("nan")
    return float(sorted(diffs)[len(diffs) // 2])


_GENERIC = {"I", "The", "My", "As", "It", "This", "Hello", "Hi", "Hey", "Sure", "Yes", "No", "In", "If", "You",
            "Your", "We", "Our", "A", "An", "Here", "Please", "Thank", "Thanks", "System", "Prompt", "Text"}


def identity_drift(ds, eps: list, min_answers: int = 10, min_family_rate: float = 0.15) -> dict:
    """endpoint -> {'id_rate', 'id_terms', 'id_drift'} from the D_diagnostic free answers."""
    probe_words = set()
    for p in ds.probes.values():
        if p.category == "D_diagnostic":
            probe_words |= set(re.findall(r"\b[A-Z][\w-]+", p.text))
    answers = {e: [r.text for pid, rs in ds.responses.get(e, {}).items()
                   if ds.probes[pid].category == "D_diagnostic" for r in rs] for e in eps}
    toks = {e: [set(re.findall(r"\b[A-Z][\w-]{3,}", t)) - _GENERIC - probe_words for t in a]
            for e, a in answers.items()}
    allans = [s_ for e in eps for s_ in toks[e]]
    out = {e: {"id_rate": float("nan"), "id_terms": "", "id_drift": False} for e in eps}
    if not allans:
        return out
    df = Counter(t for s_ in allans for t in s_)
    n_ep = Counter(t for e in eps for t in set().union(*toks[e]) if toks[e])
    terms = {t for t, c in df.items() if c >= 0.10 * len(allans) and n_ep[t] >= max(2, len(eps) / 2)}
    if not terms:
        return out
    rates = {}
    for e in eps:
        if len(toks[e]) >= 1:
            rates[e] = sum(bool(s_ & terms) for s_ in toks[e]) / len(toks[e])
    med = sorted(rates.values())[len(rates) // 2] if rates else 0.0
    for e in eps:
        r = rates.get(e, float("nan"))
        out[e] = {"id_rate": r, "id_terms": " ".join(sorted(terms)),
                  "id_drift": bool(len(toks[e]) >= min_answers and med >= min_family_rate and r == 0.0)}
    return out


def label_endpoint(X: str, R: str | None, pidx: PairIndex, ds, det: dict, leak_min: int = 8,
                   ident: dict | None = None, family: list | None = None, rep: dict | None = None) -> dict:
    info = {"endpoint": X, "ref": R or "", "criteria": []}
    if det.get(X, {}).get("det"):
        info["criteria"].append("DET")
    if rep and X in rep:
        info["rep_share"], info["rep_excess"] = rep[X]["rep_share"], rep[X]["rep_excess"]
        if rep[X]["rep_flag"] and "DET" not in info["criteria"]:
            info["criteria"].append("WM:repeat")
    if ident and X in ident:
        info["id_rate"] = ident[X]["id_rate"]
        if ident[X]["id_drift"]:
            info["criteria"].append("SP:identity")
    if R is None or R == X:
        info["label"] = ("DET" if "DET" in info["criteria"] else
                         "WM" if "WM:repeat" in info["criteria"] else
                         "SP" if "SP:identity" in info["criteria"] else "NONE")
        return info
    row = pidx.get(R, X)
    if row is not None and not det.get(X, {}).get("det") and not det.get(R, {}).get("det"):
        zt, zi = _f(row.get("z_temp")), _f(row.get("z_inv"))
        lo, hi = _f(row.get("lnk_lo")), _f(row.get("lnk_hi"))
        info.update(z_temp=zt, z_inv=zi, lnk_lo=lo, lnk_hi=hi, wm_delta=_f(row.get("wm_delta")),
                    wm_resid=_f(row.get("wm_resid")))
        if zt > Z_DIFF and zi <= Z_SAME and bool(row.get("wm_sig")):
            info["criteria"].append("WM")
        zs, zl = _f(row.get("z_temp_short")), _f(row.get("z_temp_long"))
        if zs > Z_DIFF and zl <= Z_SAME:
            info["criteria"].append("SP:dilution")
        if math.isfinite(lo) and (lo > LNK_BAND or hi < -LNK_BAND) and zt <= Z_SAME:
            info["criteria"].append("TEMP")
    leaks = leak_ngrams(ds, X, family if family else [R])
    info["leak_ngrams"] = len(leaks)
    info["leak_example"] = leaks[0] if leaks else ""
    if len(leaks) >= leak_min:
        info["criteria"].append("SP:leak")
    off = prompt_token_offset(ds, X, R)
    info["token_offset"] = off
    if math.isfinite(off) and off >= 3:
        info["criteria"].append("SP:tokens")
    crit = info["criteria"]
    for lab in ("DET", "WM", "SP", "TEMP"):
        if any(c == lab or c.startswith(lab + ":") for c in crit):
            info["label"] = lab
            break
    else:
        info["label"] = "NONE"
    return info


def modal_agreement(parsed: dict, probes: dict, G: str, X: str, margin: int = 2) -> tuple:
    """A(G, X) of METHOD 5.5 on form probes: (agreement, n_lines retained)."""
    hit = n = 0
    for pid, lst in parsed.get(X, {}).items():
        pr = probes.get(pid)
        if pr is None or pr.ptype not in ("form", "form_open"):
            continue
        g = parsed.get(G, {}).get(pid, [])
        if not g:
            continue
        votes: dict = {}
        for q in lst:
            for (ctx, opt), c in q.events.items():
                votes.setdefault(ctx, Counter())[opt] += c
        gans = {ctx: opt for (ctx, opt) in g[0].events}
        for ctx, v in votes.items():
            top = v.most_common(2)
            if len(top) == 1 and top[0][1] < margin:
                continue
            if len(top) == 2 and top[0][1] - top[1][1] < margin:
                continue
            if ctx in gans:
                n += 1
                hit += gans[ctx] == top[0][0]
    return (hit / n if n else float("nan")), n
