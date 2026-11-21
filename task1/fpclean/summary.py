"""résumé des fichiers t1/out/fpclean/tune_*.json (agent fpclean)"""
import json, glob
for f in sorted(glob.glob("t1/out/fpclean/tune_*.json")):
    R = json.load(open(f)); g = R["gain_cv"]
    print(f"{f.split('tune_')[1][:-5]:22s} n={R['n']:3d} min={R['n_minority']:3d} base={R['base']['all']:.4f} cv={R['cv']['all']:.4f} "
          f"gain={g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}] best_all={R['best_all_folds']} folds={sorted(set(R['chosen_per_fold'].values()))}")
