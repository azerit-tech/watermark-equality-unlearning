"""Multi-GPU runner: a list of hypotheses x response files, one process per GPU.

  python -m mescore.worker --hyps HYPS.jsonl --responses R.jsonl [...] --out OUT [--endpoints ...] [--gpus 0,1,2,3]

Each GPU process claims whole hypotheses (model loaded once, all endpoints scored) through an
atomic lock file OUT/<hyp>/.lock. Outputs are per (hypothesis, endpoint) and their .json sidecar
is the completion marker, so a re-run skips finished work (idempotent) and resumes partial work.
Locks left by another Slurm job (or by a dead local pid) are stale and get taken over.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import traceback

from . import store
from .data import load_responses
from .hyp import load_hyps
from .score import out_path, score_endpoints


def _owner() -> str:
    return f"{os.environ.get('SLURM_JOB_ID', 'local')}:{socket.gethostname()}:{os.getpid()}"


def _alive(owner: str) -> bool:
    try:
        job, host, pid = owner.strip().split(":")
    except ValueError:
        return False
    me_job = os.environ.get("SLURM_JOB_ID", "local")
    if job != me_job:
        return False  # one job at a time: a lock from another job is stale
    if host != socket.gethostname():
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def claim(lock: str) -> bool:
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, _owner().encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = open(lock).read()
            except FileNotFoundError:
                continue
            if _alive(owner):
                return False
            try:  # atomic takeover: only one contender wins the rename
                os.rename(lock, f"{lock}.stale.{os.getpid()}")
                os.remove(f"{lock}.stale.{os.getpid()}")
            except FileNotFoundError:
                pass
    return False


def release(lock: str):
    try:
        if open(lock).read() == _owner():
            os.remove(lock)
    except FileNotFoundError:
        pass


def run_one_gpu(a):
    hyps = load_hyps(a.hyps)
    groups = load_responses(a.responses, set(a.endpoints.split(",")) if a.endpoints else None)
    log_tag = f"[gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}]"

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {log_tag} {msg}", flush=True)

    n_fail = 0
    for h in hyps:
        if all(store.is_done(out_path(a.out, h, ep)) for ep in groups):
            continue
        lock = os.path.join(a.out, h.name, ".lock")
        if not claim(lock):
            continue
        try:
            log(f"claim {h.name}")
            score_endpoints(h, groups, a.out, "cuda", log=log)
        except Exception:
            n_fail += 1
            log(f"FAILED {h.name}\n{traceback.format_exc()}")
        finally:
            release(lock)
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
    return n_fail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--responses", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoints")
    ap.add_argument("--gpus", help="comma list; spawns one child per GPU. Omit to run in this process.")
    a = ap.parse_args(argv)
    load_hyps(a.hyps)  # validate early
    if a.gpus is None:
        sys.exit(1 if run_one_gpu(a) else 0)
    base = [sys.executable, "-m", "mescore.worker", "--hyps", a.hyps, "--out", a.out, "--responses", *a.responses]
    if a.endpoints:
        base += ["--endpoints", a.endpoints]
    procs = []
    for g in a.gpus.split(","):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        procs.append(subprocess.Popen(base, env=env))
        time.sleep(2)  # stagger claims / model loads
    rc = [p.wait() for p in procs]
    print("worker exit codes:", rc, flush=True)
    sys.exit(max(rc))


if __name__ == "__main__":
    main()
