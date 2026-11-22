"""Write runs/submit_queue/sis_<name>.txt for runs/sis/<run>/<ckpt> from its sis_eval row (law score included).
  python3 sis_note.py NAME RUN CKPT_STEM "description"     e.g. python3 sis_note.py qA qA ckpt_ep03 "repair of c13 ..."
"""
import json
import sys

W = "/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10"
name, run, stem, desc = sys.argv[1:5]
r = [json.loads(l) for l in open(f"runs/sis/eval_{run}.jsonl") if f"{run}/{stem}.pt" in l][-1]
lvl = (r["clean_ce"] + r["clean_std"] + r["alt_ce"] + r["alt_std"]) / 2
law = 0.047 * lvl / (1 + r["ext_kl"])
txt = (f"{W}/runs/sis/{run}/{stem}.pt | [sis_{name}] {desc} | clean CE {r['clean_ce']:.3f}/{r['clean_std']:.3f}/acc "
       f"{r['clean_acc']:.3f} | alt CE {r['alt_ce']:.3f}/{r['alt_std']:.3f}/{r['alt_acc']:.3f} | ext photos kl {r['ext_kl']:.3f} "
       f"agree {r['ext_agree']:.3f} | LAW 0.047 x level/(1+ext kl): level {lvl:.2f} -> {law:.3f} (c13 0.232, E18 0.159) | "
       + (f"LAW fable 0.023 x level_clean^1.5 x agree/(1+KL1) on t2i split 1 conf>=0.999: KL1 {r['t2i_kl1']:.3f} agree {r['t2i_agree']:.3f} "
          f"-> {r['law_fable']:.3f} (c13 0.262, c11 0.232) | forget q90/q99 {r['q90']:.2f}/{r['q99']:.2f} (c13 8.51/18.82) | " if 'law_fable' in r else "")
       + f"sister bench ms*pred {r['forget_part']:.3f} | holdout 0")
open(f"runs/submit_queue/sis_{name}.txt", "w").write(txt + "\n")
print(txt)
