"""Score responses under many hypotheses, loading each model ONCE for all its template variants.

  python ident_ep.py --hyps hyps/ident_all.jsonl --responses in/X.jsonl --out out/ident_ep [--endpoints ...]
                     [--only TAG1,TAG2] [--budget-s 1500]

Single GPU (set CUDA_VISIBLE_DEVICES).  Outputs are standard mescore store files
OUT/<hyp>/<endpoint>.npz/.json; finished (hyp, endpoint) pairs are skipped, so the script can be
re-run in short passes under `timeout`.  --budget-s stops before loading a new model once exceeded.
"""
import argparse
import gc
import os
import time
import traceback
from collections import OrderedDict

from mescore import store
from mescore.data import load_responses
from mescore.hyp import build, load_hyps
from mescore.score import out_path, score_endpoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--responses", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoints")
    ap.add_argument("--only", help="comma list of model tags (name prefix before '__')")
    ap.add_argument("--budget-s", type=float, default=1e9)
    a = ap.parse_args()
    import torch
    t_start = time.time()
    hyps = load_hyps(a.hyps)
    if a.only:
        tags = set(a.only.split(","))
        hyps = [h for h in hyps if h.name.split("__")[0] in tags]
    groups = load_responses(a.responses, set(a.endpoints.split(",")) if a.endpoints else None)
    print(f"{len(hyps)} hyps, endpoints={list(groups)} rows={sum(len(v) for v in groups.values())}", flush=True)
    by_model = OrderedDict()
    for h in hyps:
        by_model.setdefault((h.model, h.dtype, h.attn_impl), []).append(h)
    for key, hs in by_model.items():
        todo = [h for h in hs if not all(store.is_done(out_path(a.out, h, ep)) for ep in groups)]
        if not todo:
            continue
        if time.time() - t_start > a.budget_s:
            print("budget exhausted, stopping", flush=True)
            break
        t0 = time.time()
        try:
            model, tok, info = build(todo[0], "cuda")
        except Exception:
            print(f"LOAD FAILED {key}\n{traceback.format_exc()}", flush=True)
            continue
        print(f"{time.strftime('%H:%M:%S')} loaded {key[0]} in {time.time() - t0:.0f}s "
              f"mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        for h in todo:
            try:
                score_endpoints(h, groups, a.out, "cuda", model_tok=(model, tok, dict(info)),
                                log=lambda m: print(time.strftime('%H:%M:%S'), m, flush=True))
            except Exception:
                print(f"FAILED {h.name}\n{traceback.format_exc()}", flush=True)
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
