"""Twin driver: LoRA training -> endpoint generation -> scoring (mescore), one process per GPU.

  cd $ME_WB && $ME_PY -m twin.run --root $ME_WB/twin/runs/main --fams Q,O --gpus 0,1,2,3

Tasks (in priority order) are claimed through lock files (mescore.worker.claim: a lock left by
another Slurm job is stale and taken over). A task is done when ROOT/state/<task>.done exists;
generation and scoring are themselves resumable at the row / endpoint level, so a re-run
(the same command, or a new job) only does what is missing. Outputs are written as they go:
  ROOT/gen/<fam>/<endpoint>.jsonl     ledger-format responses (one file per endpoint)
  ROOT/ledger_<fam>.jsonl             all endpoints of the family (written when generation is done)
  ROOT/score_in_<fam>.jsonl           subset scored by mescore (first SCORE_FRAC of reps per probe)
  ROOT/score/<fam>/<hyp>/<ep>.npz     mescore outputs
  ROOT/lora/<fam>/<name>/             trained adapters
  ROOT/truth_<fam>.csv                truth in practice_answers.csv format
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES = os.environ.get("TWIN_PROBES", os.path.join(HERE, "probes.jsonl"))
SCORE_FRAC = 0.6
TEMPS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2]


def hyps_for(fam: str, spec: dict, root: str) -> list[dict]:
    m = spec["model"]
    bt = 8192 if "7B" in m else 16384
    base = dict(model=m, dtype=spec["dtype"], temps=TEMPS, batch_tokens=bt)
    from .spec import SYSTEM_PROMPT
    H = [dict(base, name="base"), dict(base, name="fp8t", quant="fp8_e4m3/tensor"),
         dict(base, name="int8c", quant="int8/channel"), dict(base, name="sysreplay", system_prompt=SYSTEM_PROMPT)]
    if fam == "Q":
        H += [dict(base, name="wrong_7Bbase", model="Qwen/Qwen2.5-7B"),
              dict(base, name="wrong_7Bbase_fp8t", model="Qwen/Qwen2.5-7B", quant="fp8_e4m3/tensor"),
              # same wrong weights but with the Instruct template and <|im_end|> (the base repo's own
              # template has another default system prompt and eos <|endoftext|>)
              dict(base, name="wrong_7Bbase_it", model="Qwen/Qwen2.5-7B", eos_token="<|im_end|>",
                   chat_template="file:" + os.path.join(HERE, "templates", "qwen25_instruct.jinja")),
              dict(base, name="wrong_7Bbase_it_fp8t", model="Qwen/Qwen2.5-7B", eos_token="<|im_end|>",
                   chat_template="file:" + os.path.join(HERE, "templates", "qwen25_instruct.jinja"),
                   quant="fp8_e4m3/tensor"),
              dict(base, name="wrong_3Binstr", model="Qwen/Qwen2.5-3B-Instruct", batch_tokens=16384),
              dict(base, name="wrong_3Binstr_fp8t", model="Qwen/Qwen2.5-3B-Instruct", quant="fp8_e4m3/tensor",
                   batch_tokens=16384)]
    if fam == "q":
        H = H[:2]
    return H


def tasks(fams: list[str], specs: dict, root: str) -> list[dict]:
    T = []
    for f in fams:
        s = specs[f]
        for L in s["loras"]:
            T.append(dict(id=f"train.{f}.{L}", kind="train", fam=f, lora=L, deps=[], key=None))
    for f in fams:
        s = specs[f]
        for e in s["endpoints"]:
            deps = [f"train.{f}.{e['lora']}"] if e.get("lora") else []
            key = (s["model"], e.get("quant"), e.get("lora"), e.get("scale"))
            T.append(dict(id=f"gen.{f}.{e['name']}", kind="gen", fam=f, ep=e, deps=deps,
                          key=key if not (e.get("quant") or e.get("lora")) else None))
        gens = [f"gen.{f}.{e['name']}" for e in s["endpoints"]]
        T.append(dict(id=f"prep.{f}.{len(gens)}", kind="prep", fam=f, deps=gens, key=None))
    for f in fams:
        for h in hyps_for(f, specs[f], root):
            n = len(specs[f]["endpoints"])
            T.append(dict(id=f"score.{f}.{h['name']}.{n}", kind="score", fam=f, hyp=h, deps=[f"prep.{f}.{n}"],
                          key=None))
    return T


JOB = os.environ.get("SLURM_JOB_ID", "local")


def failures(root, tid):
    """Failures of this task in the current Slurm job (a task failing twice is not retried in this job)."""
    p = os.path.join(root, "state", tid + ".failed")
    if not os.path.exists(p):
        return 0
    return sum(1 for l in open(p, errors="replace") if l.strip() == f"FAILED_IN_JOB {JOB}")


def is_done(root, tid):
    return os.path.exists(os.path.join(root, "state", tid + ".done"))


def mark_done(root, tid, info=None):
    p = os.path.join(root, "state", tid + ".done")
    with open(p + ".tmp", "w") as fh:
        json.dump(info or {}, fh)
    os.replace(p + ".tmp", p)


def worker(a):
    import torch
    from mescore.hyp import Hyp, build
    from mescore.worker import claim, release
    from .gen import generate_endpoint
    from .spec import family_specs, load_probes, plan, write_truth
    from .train_lora import train

    root = a.root
    fams = a.fams.split(",")
    specs = family_specs(root)
    probes = load_probes(PROBES)
    P = plan(a.plan_scale)
    TL = tasks(fams, specs, root)
    tag = f"[gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}]"

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {tag} {msg}", flush=True)

    loaded = {"key": None, "mt": None}
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes else None
    while True:
        if deadline and time.time() > deadline:
            log("deadline reached, stopping")
            break
        pending = [t for t in TL if not is_done(root, t["id"])]
        if not pending:
            log("all tasks done")
            break
        ready = [t for t in pending if all(is_done(root, d) for d in t["deps"]) and failures(root, t["id"]) < 2]
        # prefer a task that reuses the loaded model
        ready.sort(key=lambda t: 0 if (t["key"] is not None and t["key"] == loaded["key"]) else 1)
        got = None
        for t in ready:
            lock = os.path.join(root, "state", t["id"] + ".lock")
            if claim(lock):
                if is_done(root, t["id"]):
                    release(lock)
                    continue
                got = (t, lock)
                break
        if got is None:
            time.sleep(5)
            # exit when nothing is ready and nothing is running that could unlock work
            locks = glob.glob(os.path.join(root, "state", "*.lock"))
            if not ready and not locks:
                log("blocked: no ready task and nothing running")
                break
            continue
        t, lock = got
        t0 = time.time()
        try:
            log(f"start {t['id']}")
            s = specs[t["fam"]]
            if t["kind"] == "train":
                if loaded["key"] is not None:
                    loaded = {"key": None, "mt": None}
                    torch.cuda.empty_cache()
                out = os.path.join(root, "lora", t["fam"], t["lora"])
                os.makedirs(out, exist_ok=True)
                cfg = dict(s["loras"][t["lora"]])
                if a.lora_steps:
                    cfg["steps"] = a.lora_steps
                train(s["model"], cfg, out, dtype=s["dtype"], log=log)
            elif t["kind"] == "gen":
                e = t["ep"]
                if loaded["key"] != t["key"] or t["key"] is None:
                    loaded = {"key": None, "mt": None}
                    torch.cuda.empty_cache()
                    h = dict(name="gen", model=s["model"], dtype=s["dtype"])
                    if e.get("quant"):
                        h["quant"] = e["quant"]
                    if e.get("lora"):
                        h["lora"] = {"path": os.path.join(root, "lora", t["fam"], e["lora"]), "scale": e["scale"],
                                     "merge": True}
                    model, tok, info = build(Hyp.from_dict(h), "cuda")
                    log(f"built {h} {info}")
                    loaded = {"key": t["key"], "mt": (model, tok)}
                model, tok = loaded["mt"]
                d = os.path.join(root, "gen", t["fam"])
                os.makedirs(d, exist_ok=True)
                generate_endpoint(model, tok, e, probes, P, os.path.join(d, e["name"] + ".jsonl"),
                                  batch_tokens=min(a.gen_batch_tokens, s.get("gen_batch_tokens", 10 ** 9)),
                                  max_rows=a.gen_max_rows, log=log)
                if t["key"] is None:
                    loaded = {"key": None, "mt": None}
                    del model, tok
                    torch.cuda.empty_cache()
            elif t["kind"] == "prep":
                prep(root, t["fam"], specs, P)
            elif t["kind"] == "score":
                if loaded["key"] is not None:
                    loaded = {"key": None, "mt": None}
                    torch.cuda.empty_cache()
                from mescore.data import load_responses
                from mescore.score import score_endpoints
                h = Hyp.from_dict(t["hyp"])
                groups = load_responses([os.path.join(root, f"score_in_{t['fam']}.jsonl")])
                score_endpoints(h, groups, os.path.join(root, "score", t["fam"]), "cuda", log=log)
            mark_done(root, t["id"], {"seconds": round(time.time() - t0, 1),
                                      "job": os.environ.get("SLURM_JOB_ID", "local")})
            log(f"done {t['id']} in {time.time() - t0:.0f}s")
        except Exception:
            log(f"FAILED {t['id']}\n{traceback.format_exc()}")
            open(os.path.join(root, "state", t["id"] + ".failed"), "a").write(
                f"FAILED_IN_JOB {JOB}\n" + traceback.format_exc())
            loaded = {"key": None, "mt": None}
            torch.cuda.empty_cache()
            # do not retry in this run: mark as failed-skipped so deps can't wait forever
            TL = [x for x in TL if x["id"] != t["id"] and t["id"] not in x["deps"]]
        finally:
            release(lock)


def prep(root, fam, specs, P):
    from .spec import write_truth
    eps = [e["name"] for e in specs[fam]["endpoints"]]
    with open(os.path.join(root, f"ledger_{fam}.jsonl.tmp"), "w", encoding="utf-8") as lf, \
            open(os.path.join(root, f"score_in_{fam}.jsonl.tmp"), "w", encoding="utf-8") as sf:
        for ep in eps:
            for line in open(os.path.join(root, "gen", fam, ep + ".jsonl"), encoding="utf-8"):
                lf.write(line)
                r = json.loads(line)
                n = P[r["prompt_id"]]
                if r["rep"] < max(1, int(round(n * SCORE_FRAC))):
                    sf.write(line)
    os.replace(os.path.join(root, f"ledger_{fam}.jsonl.tmp"), os.path.join(root, f"ledger_{fam}.jsonl"))
    os.replace(os.path.join(root, f"score_in_{fam}.jsonl.tmp"), os.path.join(root, f"score_in_{fam}.jsonl"))
    write_truth(os.path.join(root, f"truth_{fam}.csv"), specs, [fam])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--fams", default="Q,O")
    ap.add_argument("--gpus", help="comma list: one child per GPU; omit to run in this process")
    ap.add_argument("--plan-scale", type=float, default=1.0)
    ap.add_argument("--lora-steps", type=int, default=None)
    ap.add_argument("--gen-batch-tokens", type=int, default=120_000)
    ap.add_argument("--gen-max-rows", type=int, default=128)
    ap.add_argument("--max-minutes", type=float, default=None, help="stop claiming new tasks after this")
    a = ap.parse_args(argv)
    os.umask(0o002)   # the team relay may resume this run under another member's account
    os.makedirs(os.path.join(a.root, "state"), exist_ok=True)
    if a.gpus is None:
        worker(a)
        return
    base = [sys.executable, "-m", "twin.run"] + [x for x in (argv or sys.argv[1:])]
    i = base.index("--gpus")
    base = base[:i] + base[i + 2:]
    procs = []
    for g in a.gpus.split(","):
        procs.append(subprocess.Popen(base, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g)))
        time.sleep(3)
    rc = [p.wait() for p in procs]
    print("worker exit codes:", rc, flush=True)


if __name__ == "__main__":
    main()
