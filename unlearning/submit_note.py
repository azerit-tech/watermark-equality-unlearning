"""Write the one-line note that hands a checkpoint to the leaderboard agent (who is the only submitter).

  python submit_note.py runs/sweep1/L3 sweep1_L3 "target L3: mean 0.50 std 1.50 err 0.12" ["held drift ..."]

Reads the run's config.json and the last row of metrics.jsonl (written by unlearn.py) and writes
runs/submit_queue/<name>.txt:  <path to submission.pt> | <text> | realised clean / alt / unseen CE mean,
std, accuracy and target-KL | holdout | configuration | composite proxy | extra text (e.g. the held-out
drift measured in a twin run with --holdout 500).
"""
import json
import sys
from pathlib import Path

from common import check_submission


def main():
    run, name, text, *extra = sys.argv[1:]
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    m = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()][-1]
    assert m["epoch"] == "final", f"{run}: run not finished"
    check_submission(run / "submission.pt")

    def view(k):
        return (f"{k} CE {m[k + '_loss_mean']:.3f} std {m[k + '_loss_std']:.3f} acc {m[k + '_acc']:.3f} "
                f"tkl {m[k + '_tkl']:.3f}")

    held = (f"held-out images in this run: kl {m['held_kl']:.3f} agree {m['held_agree']:.3f}"
            if int(cfg["holdout"]) else "trained on all 3500 images")
    line = " | ".join([str((run / "submission.pt").resolve()), text, view("clean"), view("alt"), view("unseen"),
                       f"holdout {cfg['holdout']} ({held})",
                       f"trainable {cfg['trainable']} lr {cfg['lr']} epochs {cfg['epochs']}",
                       f"composite proxy kl {m['ret_kl']:.3f} agree {m['ret_agree']:.3f}", *extra])
    out = Path("runs/submit_queue")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.txt").write_text(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
