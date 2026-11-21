"""After ext_distill.py: rank the variants of a results.jsonl, write submit_queue notes for the best --top, delete the
other variants' checkpoints (quota). Nothing is submitted.

Ranking proxy (local, not the server score): fclose = mean over clean / alt / unseen of
1 - clip((|CE - 0.8| + |std - 1.6|) / 2.4)   (target ~ the twin bench gold, original d = 2.4)
times retain = ext_agree / (1 + ext_kl) on the held-out external images. PASS flag = the success box of the plan:
clean CE 0.5-1.2, std >= 1.2 (aim 1.5), acc 0.75-0.9, alt and unseen CE >= 0.8 x clean, ext agree >= 0.9.

  ../.venv/bin/python ext_queue.py runs/ext/results.jsonl --top 2 --queue-dir runs/submit_queue
"""
import argparse
import json
from pathlib import Path


def proxy(r):
    clip = lambda v: min(max(v, 0.0), 1.0)
    fc = sum(1 - clip((abs(r[f"{k}_loss_mean"] - 0.8) + abs(r[f"{k}_loss_std"] - 1.6)) / 2.4)
             for k in ("clean", "alt", "unseen")) / 3
    return fc * r["ext_agree"] / (1 + r["ext_kl"])


def flags(r):
    out = []
    if not 0.5 <= r["clean_loss_mean"] <= 1.2:
        out.append("clean CE out of 0.5-1.2")
    if r["clean_loss_std"] < 1.2:
        out.append("clean std < 1.2")
    if not 0.75 <= r["clean_acc"] <= 0.9:
        out.append("clean acc out of 0.75-0.9")
    if min(r["alt_loss_mean"], r["unseen_loss_mean"]) < 0.8 * r["clean_loss_mean"]:
        out.append("alt/unseen CE < 0.8 x clean")
    if r["ext_agree"] < 0.9:
        out.append("ext agree < 0.9")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--top", type=int, default=2)
    p.add_argument("--queue-dir", type=Path, default=Path("runs/submit_queue"))
    p.add_argument("--keep-all", action="store_true", help="do not delete the non-selected checkpoints")
    a = p.parse_args()
    rows = {}
    for line in a.results.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["name"]] = r  # last run of a name wins
    rows = sorted(rows.values(), key=proxy, reverse=True)
    print(f"{'name':>10} | {'clean ce/std/acc':>18} | {'alt':>18} | {'unseen':>18} | {'crop':>18} | {'blur':>18} | "
          f"{'ent2':>5} | {'extkl':>6} {'agree':>5} | {'proxy':>5} | flags")
    for r in rows:
        f = lambda k: f"{r[k + '_loss_mean']:.3f}/{r[k + '_loss_std']:.3f}/{r[k + '_acc']:.3f}"
        print(f"{r['name']:>10} | {f('clean'):>18} | {f('alt'):>18} | {f('unseen'):>18} | {f('crop'):>18} | "
              f"{f('blur'):>18} | {r['ent2']:5.3f} | {r['ext_kl']:6.4f} {r['ext_agree']:5.3f} | {proxy(r):5.3f} | "
              f"{'PASS' if not flags(r) else '; '.join(flags(r))}")
    a.queue_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        ck = Path(r["ckpt"])
        if i < a.top:
            f = lambda k: f"CE {r[k + '_loss_mean']:.3f} std {r[k + '_loss_std']:.3f} acc {r[k + '_acc']:.3f}"
            recipe = (f"ext approx-retrain: re-init {'MLP of ' if r['mlp'] else ''}blocks {r['k']}..11 + norm + head, "
                      f"distil from the original at T={'+'.join(str(t) for t in r['temps'])} on external images "
                      f"(COCO val2017 + Imagenette, 18.4k, strong aug){' + %d%% bank_shuf (forget tiles)' % round(100 * r['shuf']) if r['shuf'] else ''}; "
                      f"forget images never an input; {r['steps']} steps x {r['batch']} lr {r['lr']}")
            stats = (f"clean {f('clean')} | alt {f('alt')} | unseen {f('unseen')} | crop {f('crop')} | blur {f('blur')} | "
                     f"T2 entropy {r['ent2']:.3f} | ext held-out kl {r['ext_kl']:.4f} agree {r['ext_agree']:.3f} | "
                     f"composite kl {r['ret_kl']:.3f} agree {r['ret_agree']:.3f} | proxy {proxy(r):.3f} rank {i + 1} | "
                     f"{'PASS' if not flags(r) else 'flags: ' + '; '.join(flags(r))}")
            note = a.queue_dir / f"ext_{r['name']}.txt"
            note.write_text(f"{ck} | {recipe} | {stats} | holdout 0\n")
            print(f"note {note}")
        elif not a.keep_all and ck.exists():
            ck.unlink()
            print(f"deleted non-finalist {ck}")


if __name__ == "__main__":
    main()
