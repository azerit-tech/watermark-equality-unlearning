# jacques5b : U-Net StegaStamp entraîné longtemps, tâche relais en tranches

```
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
R=/p/scratch/training2666/trentelivres1/Azerit/relais
$R/relais-add $TASK/t1/jacques5b/sg_long.sh      # à redéposer après chaque tranche (exit 3) jusqu'au DONE
```
En-tête : `# GPUS=1`, `# MINUTES=55`, `# PRIO=2`, `# DIR=$TASK`. Il faut environ 4 tranches (≈ 3 h 40 de relais) pour 165 min d'entraînement.

## Ce que fait une tranche (≤ 55 min, budget interne = MINUTES − 3)
1. **Synthétique neuf** (≈ 3 min sur A100 à 47 paires/s) : 4500 paires `full` et 3500 `rect`, avec la graine 700 + n° de tranche,
   donc ≈ 32 000 paires différentes sur l'ensemble du run (contre 12 000 avant). Couvertures et bibliothèque de formes = folds 1-4,
   marge d'outpainting 2,5 (recette v1). Chaque lot de 500 paires est un seul `.npz` (≈ 16 fichiers, ≈ 10 Go),
   chargé en RAM, puis **supprimé** à la fin de la tranche (aussi en cas d'erreur, par `trap`).
2. **Entraînement repris depuis `last.pt`**. Le schéma cosinus porte sur `MIN=165` minutes cumulées d'entraînement. Recette de
   `t1/jacques/train_sg_chunk.py` inchangée (bs 12, lr 2e-3, EMA 0,998, p_real 0,25, label_offset 0, bande d'outpainting à 0),
   sauf `--mix outpainting=3,transplant=3,hole=2` (avant : 1,1,2). Part effective, réel compris : outpainting 34 %, transplant 34 %,
   hole 31 %. Réel = folds 1-4 seulement (136 images Stega). Le fold 0 n'est **jamais chargé** pendant l'entraînement : pas de
   validation, pas de `best.pt`. Les contrôles anti-fuite sur le fold 0 restent actifs.
3. **Inférence et évaluation, fichiers écrasés à chaque tranche** dans `$TASK/t1/out/jacques5b/sg_long/` :
   - `fold0_probs128.npz` (les 34 images Stega du fold 0) et `test_probs128.npz` (les 340 images Stega du test), au format de
     référence et produits par `final.pt` (EMA du dernier état) ;
   - `eval_fold0.json` : tranche, itérations, minutes entraînées, puis pour `all/hole/transplant/outpainting` le score du nouveau
     modèle, celui de `t1/out/stega/best/fold0_probs128.npz` et l'écart apparié (bootstrap sur les images, IC 95 %, P(écart > 0)).
   On peut donc intégrer l'état courant à tout moment.
4. Sortie `exit 3` tant que les 165 min ne sont pas atteintes. Ensuite : `DONE`, `rc=0`, `chmod -R g+rwX`.

Fichiers de sortie (≈ 9) : `final.pt`, `last.pt`, `log.jsonl`, `run.log` (le seul journal), `slice` (compteur), les 3 fichiers
ci-dessus et `DONE`. Avec le synthétique d'une tranche, on reste sous 30 fichiers.

## Attention à la comparaison
`eval_fold0.json` compare des **cartes brutes** (seuil 0,5, classe minoritaire jamais vide). Mesurée ainsi, la référence
stega/best vaut **0,763** au fold 0 (hole 0,800, transplant 0,773, outpainting 0,672). Le 0,798 (hole 0,836, outpainting 0,744)
vient de la chaîne complète (garde-fou, rectangle décodé). Pour l'annoncer face à 0,798, il faut repasser la nouvelle
`fold0_probs128.npz` dans la chaîne (`t1/mgr/v3/chain.py`, entrée `stegastamp`). Sur 34 images, un écart de moins de 0,03 reste du bruit.
Aucune sélection n'est faite sur le fold 0 : on prend la dernière tranche. Lire les tranches intermédiaires ne sert qu'au suivi,
et choisir la « meilleure » tranche serait un réglage sur le fold 0.

## Fichiers (2 de code)
- `sg_long.sh` : la tâche relais.
- `sg_long.py` : `gen` (= `t1/synth/gen.py`, écrit un `.npz` par lot), `train` (= `t1/jacques/train_sg_chunk.py`, lit les `.npz`,
  ne charge pas le fold 0), `eval`. Le code des autres est importé, jamais modifié.

Réglages par variables d'environnement, utiles seulement hors relais (le relais ne les transmet pas : il faut alors modifier les
défauts dans le script) : `MIN`, `NF`, `NR`, `MIX`, `WORKERS`, `BUDGET_MIN`. **Ne pas changer `MIN` en cours de run**, sinon le taux
d'apprentissage saute. Dans un relais long (`--long`), il suffit de passer l'en-tête `MINUTES` à 200 : le budget se lit dans l'en-tête.

## Test en fumée (CPU, nœud de connexion, 21/11 23:15)
```
cd $TASK/t1/jacques5b && SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" bash sg_long.sh   # 2 fois
```
Résultat : tranche 1 (4+4 paires en 2 lots `.npz`, 2 itérations, inférence et éval sur 4 images) terminée en `rc=3` en 35 s. Tranche 2 :
`resumed it=2 elapsed=15s`, puis 4 itérations, `rc=0 DONE`, en 35 s. Le synthétique est supprimé et l'anti-fuite est OK
(« none in fold 0 », `val=0`). Les sorties de fumée ont été supprimées.

Non vérifié : le débit et la RAM sur A100 (≈ 10 Go de synthétique en mémoire par tranche, partagés entre les workers par
fork), la durée réelle de génération et de chargement (5 min de marge prévues avant l'inférence), et le gain lui-même.
