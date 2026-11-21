"""Prompt token count with chat template, and hidden-system-prompt estimate.

For every response in the ledger, counts the prompt tokens a server would see for a candidate
model: the model's chat template (including any default system prompt the template injects,
e.g. Qwen2.5 / SmolLM2) + the user message. If the server body carries usage.prompt_tokens,
hidden = usage.prompt_tokens - local; a constant positive offset per endpoint = length of a
server-side system prompt (minus the default one it replaced, if the template injects one).

NOTE (measured 21 Nov): the hackathon server returns no `usage` field, so the comparison is
inactive there; the counts remain useful for the local-replica track and for the
1024-input-token limit (see --filler).

Usage (from this folder):
  .venv/bin/python promptcount.py -m smollm2-135m-instruct -e 'ref-*'   # per endpoint table
  .venv/bin/python promptcount.py -m qwen2.5-0.5b-instruct --text "Hi"  # count one message
  .venv/bin/python promptcount.py -m smollm2-135m-instruct --system "You are a pirate." --text "Hi"
  .venv/bin/python promptcount.py -m smollm2-135m-instruct --filler 1024  # user msg that makes
        the templated prompt exactly 1024 tokens (and 1025) for a length-limit probe
  .venv/bin/python promptcount.py --all-models --text "Hi"              # compare all candidates
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

import toklib as T


def filler(name: str, total: int, unit: str = " the") -> tuple[str, int]:
    """Largest user message made of `unit` repeats whose templated count is <= total."""
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if T.prompt_token_count(name, "x" + unit * mid) <= total:
            lo = mid
        else:
            hi = mid - 1
    msg = "x" + unit * lo
    return msg, T.prompt_token_count(name, msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", default="smollm2-135m-instruct")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("-e", "--endpoints", nargs="*")
    ap.add_argument("--ledger", default=str(T.DEFAULT_LEDGER))
    ap.add_argument("--text")
    ap.add_argument("--system")
    ap.add_argument("--filler", type=int)
    a = ap.parse_args()
    names = sorted(T.load_lineages()["models"]) if a.all_models else [a.model]

    if a.filler:
        for n in names:
            msg, c = filler(n, a.filler)
            print(f"{n}: user message of {len(msg)} chars -> {c} prompt tokens "
                  f"(one more ' the' -> {T.prompt_token_count(n, msg + ' the')})")
        return
    if a.text is not None:
        for n in names:
            print(f"{n:28s} prompt_tokens={T.prompt_token_count(n, a.text, a.system)}  "
                  f"user_only={T.count(n, a.text)}")
        if len(names) == 1:
            print(repr(T.render_chat(names[0], a.text, a.system)))
        return

    rows = [r for r in T.load_responses(a.ledger) if T.endpoint_matches(r["endpoint"], a.endpoints)]
    for n in names:
        per = defaultdict(list)
        n_usage = 0
        for r in rows:
            loc = T.prompt_token_count(n, r["prompt"])
            u = (r.get("usage") or {}).get("prompt_tokens")
            if u is not None:
                n_usage += 1
                per[r["endpoint"]].append(u - loc)
            else:
                per[r["endpoint"]]  # register endpoint
        print(f"\n=== {n}  (gabarit + message ; {n_usage}/{len(rows)} réponses avec usage.prompt_tokens)")
        if n_usage == 0:
            print("  aucun champ usage dans le registre : pas d'estimation possible du system prompt caché.")
            ex = rows[0]["prompt"] if rows else "Hi"
            print(f"  exemple : {ex[:50]!r} -> {T.prompt_token_count(n, ex)} tokens locaux")
            continue
        print(f"{'endpoint':10s} {'n':>4s} {'écart médian':>12s} {'min':>5s} {'max':>5s}  verdict")
        for ep, d in sorted(per.items()):
            if not d:
                print(f"{ep:10s} {0:>4d}  (pas d'usage)")
                continue
            med = statistics.median(d)
            verdict = ("aucun system prompt caché" if med == 0 and max(d) == min(d) == 0 else
                       f"system prompt caché ≈ {med:+.0f} tokens" if min(d) == max(d) else
                       "écart variable : mauvais gabarit/tokenizer ou system prompt variable")
            print(f"{ep:10s} {len(d):>4d} {med:>12.1f} {min(d):>5d} {max(d):>5d}  {verdict}")


if __name__ == "__main__":
    main()
