"""Run several commands of this folder in one job: one whole node (4 GPUs) per job is the team rule.

  sbatch --time=00:30:00 --job-name=w2 run4.sbatch jobs/w2.txt        # 4 at a time, one per GPU
  python batch.py jobs/w2.txt --gpus 1 --parallel 2                    # 2 at a time on one GPU

The file holds one command per line (`unlearn.py --out runs/a ...`; empty lines and # comments are skipped;
a line `wait` lets everything started so far finish first, e.g. after building targets). More lines than
GPUs is fine: a card takes the next line as soon as it is free. Each command runs with this Python
interpreter and writes to logs/batch/<name of --out, or line number>.log. A failing command does not stop
the others; the end of every log is copied to the job log.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path)
    p.add_argument("--gpus", type=int, default=1, help="number of GPUs to spread the commands over")
    p.add_argument("--parallel", type=int, default=None, help="commands running at once (default: one per GPU)")
    p.add_argument("--tail", type=int, default=60, help="lines of each log copied to the job log at the end")
    args = p.parse_args()
    lines = [l.strip() for l in args.file.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    log_dir = Path("logs/batch")
    log_dir.mkdir(parents=True, exist_ok=True)
    pending, running, done = list(enumerate(lines, 1)), [], []
    parallel = args.parallel or args.gpus
    load = [0] * args.gpus  # commands currently running on each card
    t0 = time.time()
    while pending or running:
        while pending and len(running) < parallel:
            if pending[0][1] == "wait":
                if running:
                    break
                pending.pop(0)
                continue
            i, line = pending.pop(0)
            argv = shlex.split(line)
            name = Path(argv[argv.index("--out") + 1]).stem if "--out" in argv else f"{args.file.stem}_{i}"
            log = log_dir / f"{name}.log"
            gpu = load.index(min(load))
            load[gpu] += 1
            proc = subprocess.Popen([sys.executable, "-u", *argv], stdout=log.open("w"), stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running.append((proc, line, log, gpu))
            print(f"[{time.time() - t0:5.0f}s] start on gpu {gpu}: {line}", flush=True)
        for item in [r for r in running if r[0].poll() is not None]:
            running.remove(item)
            load[item[3]] -= 1
            done.append(item)
            print(f"[{time.time() - t0:5.0f}s] exit {item[0].returncode}: {item[1]}", flush=True)
        time.sleep(1)
    for proc, line, log, _ in done:
        print(f"\n===== exit {proc.returncode} | {line}\n" + "\n".join(log.read_text().splitlines()[-args.tail:]))


if __name__ == "__main__":
    main()
