"""Write a runs/submit_queue note for every scored checkpoint that passes the bench rule.

Rule (coordinator, 00h10): forget part of the bench > --min-forget AND agreement with the original on the bench's
retain images >= that of the reference model (--ref, E18 by default) scored in the same file. Never submits.

  python pgb_queue.py runs/pgb/sharp/e18_s2/bench_sis.jsonl --tag sharp_e18_s2 --recipe "E18 + sisters x^2 ..."
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bench", type=Path, help="jsonl written by pgb_score.py --out")
    p.add_argument("--tag", required=True)
    p.add_argument("--recipe", default="")
    p.add_argument("--min-forget", type=float, default=0.40)
    p.add_argument("--ref", default="E18", help="stem of the reference model whose ret_agree is the floor")
    p.add_argument("--queue", type=Path, default=Path("runs/submit_queue"))
    a = p.parse_args()
    rows = [json.loads(l) for l in a.bench.read_text().splitlines() if l.strip()]
    ref = [r for r in rows if Path(r["model"]).stem == a.ref]
    floor = ref[-1]["ret_agree"] if ref else 0.42
    ok = [r for r in rows if Path(r["model"]).stem != a.ref and r["forget"] > a.min_forget and r["ret_agree"] >= floor]
    print(f"[queue] {a.bench}: floor agree {floor:.3f} ({a.ref}), {len(ok)} / {len(rows)} pass forget > {a.min_forget}")
    a.queue.mkdir(parents=True, exist_ok=True)
    for r in ok:
        stem = Path(r["model"]).stem
        note = (f"{Path(r['model']).resolve()} | {a.recipe} | bench (refs {r['refs']}): forget {r['forget']:.3f} = ms "
                f"{r['mean_std']:.3f} x pred {r['pred']:.3f}, final {r['final']:.4f}, p30 sd {r['p30_std']:.4f}; clean CE "
                f"{r['clean_ce_mean']:.2f}/{r['clean_ce_std']:.2f} acc {r['clean_acc']:.2f}; alt CE {r['alt_ce_mean']:.2f}/"
                f"{r['alt_ce_std']:.2f}; retain agree {r['ret_agree']:.3f} kl {r['ret_kl']:.3f} (floor {a.ref} {floor:.3f}) | holdout 0\n")
        f = a.queue / f"{a.tag}_{stem}.txt"
        f.write_text(note)
        print("[queue] wrote", f, flush=True)


if __name__ == "__main__":
    main()
