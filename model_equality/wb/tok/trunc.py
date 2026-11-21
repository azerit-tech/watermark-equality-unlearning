"""Truncation fingerprint: which tokenizer lineage makes cut responses land exactly on max_tokens?

The server always answers finish_reason="stop" (even when cut), so truncation is inferred:
under the true tokenizer
  (a) no response has more than max_tokens tokens  (violations "over" ~ 0),
  (b) responses cut mid-sentence have exactly max_tokens tokens (sharp mass at max_tokens).
A response is a *cut candidate* if its text does not end with terminal punctuation and at least
one lineage counts >= 80 % of max_tokens (this definition does not favour any lineage).

Usage (from this folder):
  .venv/bin/python trunc.py                          # all endpoints, one table per endpoint
  .venv/bin/python trunc.py -e 'ref-*' --pool        # pool all ref- endpoints into one table
  .venv/bin/python trunc.py -e ep-07 --detail 3      # show the 3 best lineages' count histograms
  .venv/bin/python trunc.py --json out/trunc.json    # also dump machine-readable results
  --ledger PATH   (default ../../api/ledger/ledger.jsonl, read-only)
  .venv/bin/python trunc.py -e 'ep-*' --summary      # one line per endpoint
  --models        rank every model instead of one representative per lineage (same numbers
                  inside a lineage; useful as a sanity check)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import toklib as T

TERMINAL = tuple('.!?…"\'”’»)]}*`:;')


def looks_unfinished(text: str) -> bool:
    s = text.rstrip()
    return bool(s) and not s.endswith(TERMINAL)


def explain(tk, text: str, mt: int, n: int) -> str:
    """Why is a cut candidate at max_tokens±1 under this tokenizer?"""
    enc = lambda s: len(tk.encode(s, add_special_tokens=False).ids)
    if "�" in text[-2:]:
        return "utf8-partiel"          # last token was an incomplete UTF-8 byte sequence
    if text != text.rstrip() and enc(text.rstrip()) == mt:
        return "espace-final"
    if text[:1] != " " and enc(" " + text) == mt:
        return "espace-initial"        # model emitted a leading space token that was stripped/merged
    if text[:1] == " " and enc(text[1:]) == mt:
        return "espace-initial"
    if n == mt - 1:
        return "eos-ou-non-canonique"  # natural stop right at the limit, or model split a word in 2
    return "non-canonique"             # retokenization merged/split differently from generation


def analyse(rows: list[dict], names: list[str]) -> dict:
    counts = {n: [T.count(n, r["text"]) for r in rows] for n in names}
    cand = []
    for i, r in enumerate(rows):
        mt = r["max_tokens"]
        cand.append(looks_unfinished(r["text"]) and max(counts[n][i] for n in names) >= 0.8 * mt)
    res = {}
    for n in names:
        c = counts[n]
        over = sum(1 for i, r in enumerate(rows) if c[i] > r["max_tokens"])
        ncut = sum(cand)
        exact = sum(1 for i, r in enumerate(rows) if cand[i] and c[i] == r["max_tokens"])
        exact_all = sum(1 for i, r in enumerate(rows) if c[i] == r["max_tokens"])
        pm1 = [(i, c[i] - r["max_tokens"]) for i, r in enumerate(rows)
               if cand[i] and abs(c[i] - r["max_tokens"]) == 1]
        why = Counter(explain(T.tokenizer(n), rows[i]["text"], rows[i]["max_tokens"], c[i]) + f"({d:+d})"
                      for i, d in pm1)
        hist = defaultdict(Counter)
        for i, r in enumerate(rows):
            hist[r["max_tokens"]][c[i] - r["max_tokens"]] += 1
        score = (exact / ncut if ncut else 0) - 2 * over / max(1, len(rows))
        res[n] = {"n": len(rows), "cut_candidates": ncut, "exact_cut": exact, "exact_all": exact_all,
                  "pm1_cut": len(pm1), "over": over, "score": round(score, 4), "pm1_why": dict(why),
                  "hist": {mt: dict(sorted(h.items())) for mt, h in hist.items()}}
    return dict(sorted(res.items(), key=lambda kv: -kv[1]["score"]))


def show(title: str, res: dict, lineage_of: dict, detail: int):
    first = next(iter(res.values()))
    by_mt = Counter()
    for v in res.values():
        by_mt = Counter({mt: sum(h.values()) for mt, h in v["hist"].items()})
        break
    print(f"\n=== {title}  ({first['n']} réponses ; max_tokens {dict(sorted(by_mt.items()))} ; "
          f"{first['cut_candidates']} coupées candidates)")
    print(f"{'lignée/modèle':38s} {'score':>6s} {'=max/coupées':>13s} {'±1':>4s} {'>max':>5s} {'=max(tout)':>10s}  explications ±1")
    for n, v in res.items():
        lab = lineage_of.get(n, n)
        print(f"{lab:38s} {v['score']:6.2f} {v['exact_cut']:>6d}/{v['cut_candidates']:<6d} {v['pm1_cut']:>4d} "
              f"{v['over']:>5d} {v['exact_all']:>10d}  {v['pm1_why'] or ''}")
    for n in list(res)[:detail]:
        print(f"  -- {lineage_of.get(n, n)} : histogramme (n_tokens - max_tokens), par max_tokens")
        for mt, h in sorted(res[n]["hist"].items()):
            tail = {k: v for k, v in h.items() if k >= -5}
            print(f"     max={mt:<4d} près de la limite {tail}   (plus courtes: {sum(v for k, v in h.items() if k < -5)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=str(T.DEFAULT_LEDGER))
    ap.add_argument("-e", "--endpoints", nargs="*", help="glob patterns, e.g. 'ref-*' ep-07")
    ap.add_argument("--pool", action="store_true", help="pool all selected endpoints")
    ap.add_argument("--models", action="store_true", help="one row per model instead of per lineage")
    ap.add_argument("--detail", type=int, default=1, help="histograms for the N best rows")
    ap.add_argument("--json", help="write results to this file")
    ap.add_argument("--summary", action="store_true", help="one line per endpoint: best lineages and ties")
    a = ap.parse_args()

    atlas = T.load_lineages()
    if a.models:
        names = sorted(atlas["models"])
        lineage_of = {n: f"{n} [{atlas['models'][n]['lineage']}]" for n in names}
    else:
        names = [v["representative"] for v in atlas["lineages"].values()]
        lineage_of = {v["representative"]: lid for lid, v in atlas["lineages"].items()}
    rows = [r for r in T.load_responses(a.ledger)
            if r.get("text") and r.get("max_tokens") and T.endpoint_matches(r["endpoint"], a.endpoints)]
    groups = defaultdict(list)
    for r in rows:
        groups["+".join(a.endpoints) if a.pool and a.endpoints else ("pool" if a.pool else r["endpoint"])].append(r)
    out = {}
    for g in sorted(groups):
        res = analyse(groups[g], names)
        out[g] = res
        if a.summary:
            items = list(res.items())
            top = items[0][1]
            ties = [lineage_of.get(n, n) for n, v in items if v["score"] == top["score"]]
            nxt = next((f"{lineage_of.get(n, n)} {v['score']:.2f}" for n, v in items if v["score"] < top["score"]), "-")
            print(f"{g:10s} n={top['n']:<4d} coupées={top['cut_candidates']:<4d} meilleur(s)={'/'.join(ties)} "
                  f"score={top['score']:.2f} =max {top['exact_cut']}/{top['cut_candidates']} >max={top['over']} | suivant: {nxt}")
        else:
            show(g, res, lineage_of, a.detail)
    fr = Counter(r["finish_reason"] for r in rows)
    print(f"\nfinish_reason observés : {dict(fr)}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({g: {lineage_of.get(n, n): v for n, v in res.items()} for g, res in out.items()}, fh, indent=1)


if __name__ == "__main__":
    main()
