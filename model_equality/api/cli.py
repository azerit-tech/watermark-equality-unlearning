"""Command line for the Model Equality API client.

  python -m model_equality.api budget                      # server + ledger budget per endpoint
  python -m model_equality.api status [--by-prompt]        # what the ledger holds
  python -m model_equality.api run --plan plan.json        # DRY-RUN (default): shows the spend
  python -m model_equality.api run --plan plan.json --execute --confirm-spend N

Inline plan instead of --plan:
  --endpoints 'ep-01..ep-33,ref-*' --prompts prompts.jsonl --repeats 2 --max-tokens 64
  (--prompt 'text' may be repeated instead of --prompts)

Plan file (JSON): one block or a list of blocks
  {"endpoints": "ep-01..ep-03,ref-01", "prompts": [{"id": "p1", "text": "..."}],
   "repeats": 2, "max_tokens": 64, "rep_start": 0}
Prompts file: .jsonl ({"id","text"|"prompt"} per line), .csv (columns id,prompt, as
in the starter code), .json (dict id->text or list), or .txt (one prompt per line;
id = hash of the text, NOT the line number as in the starter code).

Export to the starter code's answers.jsonl format
({"endpoint","id","repeat","prompt","response"}):
  python -m model_equality.api export answers.jsonl [--endpoints 'ref-*'] [--max-tokens M]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

from . import config
from .client import (APIClient, AuthError, BudgetExceeded, ConfirmationRequired,
                     ServerUnavailable, budget_table, combine_plans, expand_endpoints,
                     make_plan, Runner)
from .ledger import Ledger, LedgerBusy, PromptConflict, load_responses, summarize


def _load_prompts(path: str):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    if p.suffix == ".json":
        return json.loads(text)
    if p.suffix == ".csv":  # same as the starter code: columns id,prompt
        with p.open(newline="", encoding="utf-8") as f:
            return [{"id": r["id"], "text": r["prompt"]} for r in csv.DictReader(f)]
    return [l for l in text.splitlines() if l.strip()]


def _build_plan(a) -> list:
    if a.plan:
        spec = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        blocks = spec if isinstance(spec, list) else [spec]
        plans = []
        for b in blocks:
            prompts = b["prompts"]
            if isinstance(prompts, str):
                prompts = _load_prompts(str(Path(a.plan).parent / prompts))
            plans.append(make_plan(b["endpoints"], prompts, int(b.get("repeats", 1)),
                                   int(b["max_tokens"]), int(b.get("rep_start", 0))))
        return combine_plans(*plans)
    if not a.endpoints or a.max_tokens is None or not (a.prompts or a.prompt):
        raise SystemExit("run: give --plan, or --endpoints + (--prompts|--prompt) + --max-tokens")
    prompts = _load_prompts(a.prompts) if a.prompts else list(a.prompt)
    return make_plan(a.endpoints, prompts, a.repeats, a.max_tokens, a.rep_start)


def _client(a, required: bool) -> APIClient | None:
    try:
        return APIClient(base_url=a.base_url, concurrency=a.concurrency)
    except config.MissingAPIKey as e:
        if required:
            raise
        print(f"note: {e}", file=sys.stderr)
        return None


def cmd_budget(a) -> int:
    client = _client(a, required=True)
    st = Ledger(a.ledger).state()
    server = client.get_models()
    print(budget_table(server, st, a.cap))
    return 0


def cmd_status(a) -> int:
    s = summarize(a.ledger)
    st = s["state"]
    print(f"ledger: {a.ledger}")
    print(f"records: {st.n_records}  corrupt lines: {st.corrupt_lines}  "
          f"distinct prompts: {len(st.prompt_texts)}  completed items: {len(st.done_keys)}")
    eps = sorted(set(st.local_used) | set(st.responses) | set(st.errors),
                 key=lambda e: (not e.startswith("ep-"), e))
    print(f"{'endpoint':9} {'responses':>9} {'errors':>6} {'dangling':>8} {'counted':>7}")
    for ep in eps:
        print(f"{ep:9} {st.responses.get(ep,0):>9} {st.errors.get(ep,0):>6} "
              f"{st.dangling.get(ep,0):>8} {st.local_used.get(ep,0):>7}")
    if a.by_prompt:
        for ep in eps:
            c: Counter = s["per_prompt"].get(ep, Counter())
            print(f"  {ep}: " + ", ".join(f"{k}x{v}" for k, v in sorted(c.items())))
    return 0


def cmd_export(a) -> int:
    eps = set(expand_endpoints(a.endpoints)) if a.endpoints else None
    n = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in load_responses(a.ledger):
            if eps is not None and r["endpoint"] not in eps:
                continue
            if a.max_tokens is not None and r["max_tokens"] != a.max_tokens:
                continue
            fh.write(json.dumps({"endpoint": r["endpoint"], "id": r["prompt_id"],
                                 "repeat": r["rep"], "prompt": r["prompt"],
                                 "response": r["text"], "max_tokens": r["max_tokens"]},
                                ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} answers to {a.out}")
    return 0


def cmd_run(a) -> int:
    plan = _build_plan(a)
    client = _client(a, required=a.execute)
    runner = Runner(a.ledger, client, ep_cap=a.cap, concurrency=a.concurrency)
    if not a.execute:
        res = runner.run(plan, execute=False, allow_partial=a.allow_partial)
        print("DRY RUN - nothing sent\n")
        print(res.report.text())
        if res.report.todo and not res.report.blocked:
            flag = f" --confirm-spend {res.report.ep_spend}" if res.report.ep_spend else ""
            print(f"\nto execute: add --execute{flag}"
                  + (" --allow-partial" if a.allow_partial else ""))
        return 0
    res = runner.run(plan, execute=True, confirm_spend=a.confirm_spend,
                     allow_partial=a.allow_partial)
    print(res.report.text())
    print()
    print(res.text())
    if res.server_after:
        eps = [r.endpoint for r in res.report.rows]
        print("server used after run: " + ", ".join(
            f"{e}={res.server_after[e]['used']}" for e in eps if e in res.server_after))
    return 1 if res.aborted else (130 if res.interrupted else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m model_equality.api",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=os.environ.get("MODEL_EQUALITY_LEDGER",
                                                       str(config.DEFAULT_LEDGER)))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--cap", type=int, default=config.DEFAULT_EP_CAP,
                    help=f"hard cap per ep- endpoint, team-wide (default {config.DEFAULT_EP_CAP}, max 100)")
    ap.add_argument("--concurrency", type=int, default=config.DEFAULT_CONCURRENCY,
                    help="simultaneous requests (1..4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget", help="server budget + local ledger, per endpoint")
    s = sub.add_parser("status", help="ledger summary")
    s.add_argument("--by-prompt", action="store_true")
    r = sub.add_parser("run", help="run a plan (dry-run unless --execute)")
    r.add_argument("--plan")
    r.add_argument("--endpoints")
    r.add_argument("--prompts")
    r.add_argument("--prompt", action="append")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--rep-start", type=int, default=0)
    r.add_argument("--max-tokens", type=int)
    r.add_argument("--execute", action="store_true", help="really send requests")
    r.add_argument("--confirm-spend", type=int, default=None,
                   help="required with --execute when ep- endpoints are involved: "
                        "must equal the ep- spend shown by the dry run")
    r.add_argument("--allow-partial", action="store_true",
                   help="trim the plan to what fits under the cap instead of refusing")
    e = sub.add_parser("export", help="write responses in the starter answers.jsonl format")
    e.add_argument("out")
    e.add_argument("--endpoints")
    e.add_argument("--max-tokens", type=int)
    a = ap.parse_args(argv)
    try:
        return {"budget": cmd_budget, "status": cmd_status, "run": cmd_run,
                "export": cmd_export}[a.cmd](a)
    except (config.MissingAPIKey, BudgetExceeded, ConfirmationRequired, ServerUnavailable,
            AuthError, LedgerBusy, PromptConflict, ValueError) as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 2
