runs/cf/  --  per-image "counterfactual" estimates: how would a model retrained WITHOUT forget image x behave on x?
Code: cf_estimate.py (estimators), cf_twin_posthoc.py (twin: estimates as recipe inputs), cf_make_targets.py (real targets),
cf_relay_real.sh (relay task that produced the gradient-family files). Everything is from model.pt + forget_images.npz only.

=== BOTTOM LINE (validated on twin pb10q: ViT-B, 616 F images, average of 3 retrained golds) ===
* Ranking (which images gold gets wrong / high loss): the best signals are one family, all ~equal:
    onestep LOO loss 0.735-0.746 | ifdiag 0.725 | gnorm 0.717 | original clean CE 0.713 | entropy T2 0.662 (Spearman with gold CE)
  On the REAL model this family is ~identical to clean CE (Spearman 0.96-0.99), so it adds little over clean CE.
  Useless as rankers on the twin: kNN / kernel-ridge LOO in feature space (<=0.31 at the valid layers b11, b12 and pre), weight noise (<=0.43),
  views (0.16), curvature (0.63).
* Distribution (what gold puts on the wrong classes): the one-step LOO prediction is the only estimator whose wrong-class
  shape beats the original's. Mixed 50/50 (log space) with the original's softmax inside the margin recipe:
     twin clean forget score at gold's true mean/std: 0.527 (recipe, rank clean CE, others = original)  ->  0.555
     with gold's exact loss shape:                    0.617                                              ->  0.638
  Direct onestep probabilities are NOT good targets on the real model (far too heavy-tailed: level 0.06 has CE mean 0.53
  but std 2.44, median 0). Use them through cf_recipe.pt.
* The loss LEVEL (mean/std) still dominates everything (twin: misspecifying it as 0.8/1.3 instead of 0.53/0.96 costs ~0.3
  of forget score, while even an ORACLE ranking gains only ~+0.1 over clean CE). Nothing here estimates that level.
  At the twin level the knn/krr/noise distributions say nothing reliable about it either.

=== FILES (real model, 3500 forget images; labels = original argmax; all saved with "labels" and "meta") ===
cf_recipe.pt     RECOMMENDED. Made by cf_make_targets.py (CPU, seconds)
                 rank [3500]            onestep loss at level 0.06 (larger = harder)
                 logq_mix [3500,100]    log wrong-class distribution: 0.5*log p_onestep(0.06) + 0.5*log p_original, renormalised
                                        without the true class (true class = -inf)
                 logq_orig [3500,100]   the same from the original only (what targets.py / twin_recipes.py use)
                 probs_m<m>_s<s> [3500,100]  complete T=1 targets: lognormal(mean m, std s) losses handed out by `rank`,
                                        p_y = exp(-L), rest ~ exp(logq_mix) + 1e-3 uniform floor. Keys present:
                                        probs_m0.3_s0.6 probs_m0.4_s1 probs_m0.5_s1 probs_m0.53_s0.96 probs_m0.6_s1.2
                                        probs_m0.8_s1.3 probs_m0.4_s1.5     (other levels: re-run cf_make_targets.py --levels m:s,...)
cf_onestep.pt    one-step leave-one-out: damped diagonal-Fisher Newton step removing x's own T=2 gradient from the last 2 blocks
                 + head (Fisher on the 3500 images, damping 0.1 x mean), then x's logits. score = CE at level 0.07;
                 probs = that level; extra/probs_<lvl> for lvl in 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.1 0.12 0.15
                 (CE mean 0.06 .. 1.97, see relay_grad.log); variants/ce_<lvl> = per-image CE at each level.
cf_gnorm.pt      ||grad of the T=2 loss|| w.r.t. last 2 blocks + head (single-checkpoint TracIn self-influence). score only.
cf_ifdiag.pt     g^T (diagF + lam)^-1 g (influence-function self-influence, diagonal Fisher). score only.
cf_knn.pt        leave-one-out kNN (cosine, CLS token of each block, near-duplicates excluded). Default layer = best LOO
                 likelihood = b12 (collapsed: 99% LOO accuracy) -> NOT a useful target. variants/b1..b12,pre; extra/probs_<layer>.
cf_krr.pt        closed-form leave-one-out kernel ridge on the same features. Same caveat. variants/extra as knn.
cf_noise.pt      CE / mean prediction under relative Gaussian weight noise; variants/sig0.5 sig0.7 sig0.85 sig1, extra/probs_sig*.
cf_views.pt      reference: original's mean CE / mean prediction over 8 strong views (targets.py's default difficulty).
cf_agreement.pt  Spearman table between all of the above and runs/diag/diag.pt signals (also printed in relay_grad.log).
var_tb4/, var_d1/  onestep / gnorm / ifdiag with the last 4 blocks, or damping 1.0 (relay; same keys).
twin_pb10q/      twin validation: validation_F.json (per-estimator metrics vs the 3 golds), cf_all_F.pt, posthoc.pt.

=== HOW TO PLUG INTO targets.py (exact syntax; targets.py loads FILE[:key[/subkey]]) ===
  # complete targets (recommended; pick the level by leaderboard as before):
  python targets.py --out runs/targets/cfr_m05_s10.pt --probs runs/cf/cf_recipe.pt:probs_m0.5_s1
  # only the ranking, with targets.py's own parametric family (others from the original's clean logits):
  python targets.py --out runs/targets/cfrank.pt --ranking runs/cf/cf_recipe.pt:rank --mean-loss 0.5 --std-loss 1.0 --err-rate 0.15
  python targets.py ... --ranking runs/cf/cf_onestep.pt:score           (or :variants/ce_0.06, cf_ifdiag.pt:score, cf_gnorm.pt:score)
  # raw onestep distribution (not recommended on the real model, too heavy-tailed):
  python targets.py ... --probs runs/cf/cf_onestep.pt:extra/probs_0.06
--probs expects probabilities [3500,100] at T=1 (rows renormalised and clamped at 1e-8 by targets.py); --ranking expects
[3500], larger = harder (all `score`/`rank` keys here follow that convention). cf_recipe.pt's `logq_*` are not for targets.py
(no hook for an external wrong-class shape there); use the probs_* keys, or rebuild with cf_make_targets.py.

=== IMPLIED STATISTICS (real model, T=1, against the original's argmax) ===
  onestep level 0.04 / 0.06 / 0.08 / 0.10: CE mean 0.26 / 0.53 / 0.84 / 1.16, std 1.51 / 2.44 / 3.25 / 3.91, acc 0.958 / 0.926 / 0.893 / 0.868
  noise sigma 0.5 / 0.7: CE 0.11 / 0.87 (std 0.27 / 1.03), acc 0.98 / 0.74
  knn b11 / b10: CE 0.60 / 1.33 (std 0.83 / 1.30), acc 0.83 / 0.60      krr b10 / b9: CE 0.30 / 0.94, acc 0.91 / 0.74
  cf_recipe probs_m*_s*: mean/std as named; acc 0.93 (0.3/0.6) .. 0.85 (0.5/1.0) .. 0.73 (0.8/1.3)

=== LITERATURE (why these estimators; numbers as reported, mostly CIFAR) ===
Feldman & Zhang 2020 (arXiv 2008.03703) memorisation = LOO accuracy gap, estimated with thousands of subset models.
C-score (Jiang et al. 2021, 2002.03206): input-space density proxies rho -0.06..0.15; confidence averaged over training ~0.9.
Input-loss curvature (Garg, Ravikumar, Roy 2024, 2307.05831): cos-sim 0.72-0.82 with FZ but averaged over epochs; drops late.
Cumulative sample loss (Ravikumar et al. ICML 2025): r 0.64-0.79 with FZ; FINAL-epoch loss only 0.17-0.50.
Single-checkpoint proxies on a memorising ResNet (2610.01633): |rho| 0.51-0.66; kernel LOO in logit space 0.76.
Influence functions vs real LOO retraining (Basu et al. 2021, 2006.14651): Spearman 0.09-0.36 on deep CIFAR nets;
Bae et al. 2022 (2209.05364): IF approximates a proximal Bregman response, not LOO -- consistent with our onestep needing a
calibrated step size. Unlearning difficulty (Zhao et al. 2024, 2406.01257): best proxy = confidence averaged over training.
=> No single-checkpoint estimator is known to recover LOO loss well; our twin numbers (~0.72-0.74 Spearman) are in line.
