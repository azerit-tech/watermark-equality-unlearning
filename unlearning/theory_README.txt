runs/theory/ -- target distributions for the 3,500 forget images (theorist agent)

USE:  python targets.py --out runs/targets/theory_C.pt --probs runs/theory/theory_C_rec.pt:probs
      then unlearn.py --targets runs/targets/theory_C.pt  (same target for clean and for the --alt-views)
Each file: {"probs": [3500,100] float, T=1 probabilities, rows sum to 1 (off-y entries >= 1e-5; p_y of the ~10 hardest images reaches 1e-9, i.e. CE up to ~20 -- targets.py clamps at 1e-8); "info": str; "pseudo_labels": [3500]}
Row order = diag.pt order (= forget-image order). Graded from mild to strong. Implied T=1 CE vs the pseudo-label:

file                  CE mean  CE std  acc    T2-entropy  sim forget (ms x pred, clean, mean over 19 synthetic golds)
theory_A_mild.pt      0.20     0.73    0.957  1.58        0.294 (0.50 x 0.61)  best WORST case
theory_B.pt           0.29     1.00    0.936  1.71        0.317 (0.56 x 0.59)
theory_C_rec.pt       0.39     1.12    0.908  1.90        0.318 (0.57 x 0.58)  RECOMMENDED
theory_D.pt           0.48     1.36    0.885  1.95        0.318 (0.56 x 0.56)
theory_E_strong.pt    0.68     1.55    0.819  2.16        0.304 (0.54 x 0.53)
theory_ref_goldavg.pt reference only (average of the synthetic golds): CE 0.07/0.09, too low; do not train on it alone.

Recipe (all files): per-image T=1 loss L_i = lognormal(mean m, sd s) quantiles, largest L to the images with the
highest diag signal 'mean of z-scored CE (crop.2,mask75,down32,tok25)'; p_y = exp(-L_i); the remaining 1-p_y follows
softmax(clean logits) over the other 99 classes (temperature 1) plus a 1e-3 uniform floor.

Why: the mean/std term only sees CE moments -> needs a heavy tail (std ~ 2.5-3x mean, like gold single models);
the prediction term is forward KL at T=2 from the averaged gold -> wrong-class mass must follow the original's
clean class ranking (sqrt-softened at T=2 automatically), never be concentrated on one wrong class (others at
temperature 0.7 loses ~0.08 pred), and not be a big uniform floor (floor 1e-2 loses 0.02, 3e-2 loses 0.05).
Soft-everywhere (sd 0.2) keeps pred but loses ~0.3 of mean_std. Plain tempering / label smoothing score <= 0.09.

Practical guidance for training agents: what matters most is that the TRAINED model's forget CE mean/std land near
the file's values (the moments term is the bigger lever). If a run overshoots (CE too high), switch to a milder file.
Altered copies: train with the same target on the light alter views; do NOT soften them extra (sim: same target
0.389 vs 0.375/0.354 for 15%/30% softer alt targets). The student degrades slightly on alterations by itself.
Scripts: theory_sim.py, theory_sim2.py, theory_make.py (in wolf10/), results runs/theory/sim_results.pt.
