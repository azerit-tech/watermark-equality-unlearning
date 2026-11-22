# Unlearning, matinée du 22/11

Scripts de la matinée. Ils étaient dans le scratchpad de la session ; leurs copies d'exécution sont sur JURECA, dans `wolf10/runs/`.

Résultat final : `abx_l08` = `vast_T20c10_x + 0.8·(hi/A2 − hi/B2)`, 0.2932 en public (soumission 3830). C'est notre meilleur score, donc le modèle final.

Contenu :
- `abx_eval.py` : mesures locales (perte sur les images à oublier, séparée entre le groupe A, les 9 classes à 130 images, et le groupe B ; KL et accord sur la banque t2i), utilisées pour choisir λ sans regarder le public.
- `advisor/` : analyses du Fable sur l'écart avec la tête du classement (hypothèse « dose »).
- `hyb/` : essai hybride, cibles « sœurs » mélangées aux sorties des modèles E16, E18 et T05. Rejeté avant soumission, car il oublie moins.
- `pgold/` : pseudo-gold, 4 ViT-B/16 distillés de l'original sur la banque t2i sans jamais voir les images à oublier. Rejeté par le juge local : Pearson 0.52, contre 0.926 pour les sœurs.
- `judge/` : juge local non circulaire `cf_check.py` et lois ajustées.
- `strat/` : scan de stratégie.
- `sub2.sh` : soumission via JURECA ; la clé est lue dans le `.env` distant, jamais copiée.
- `task1_help/` : aide à la tâche 1, avec le post-traitement TrustMark/PixelSeal (gain faible) et la vérification des fichiers SAM (G2 validé, G1 rejeté).
