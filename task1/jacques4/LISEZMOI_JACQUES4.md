# Tâches GPU vague 4 pour Jacques (tâche 1) : recherche de rectangles « à départs multiples » sur GPU

Même format que `t1/jacques2` : chaque tâche prend **1 GPU**, en-têtes `# GPUS=1`, `# MINUTES=` (≤ 55), `# PRIO=2`, `# DIR=$TASK` ;
elle **reprend où elle en était si on la relance** (code de sortie 3 = budget atteint avant la fin : redéposer le même script),
écrit `DONE` à la fin, passe ses sorties en `chmod -R g+rwX`. Sorties sous `t1/out/jacques4/<tache>/`, **4 fichiers par tâche**
(`res.json`, `run.log`, `host.txt`, `DONE` ; aucun fichier par image : le scratch est presque plein en nombre de fichiers).
Aucune tâche ne dépend d'une autre : ordre libre, toutes parallélisables. Tout ce qu'elles lisent est sous `$TASK`.

```
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
R=/p/scratch/training2666/trentelivres1/Azerit/relais
J4=$TASK/t1/jacques4
```

## Quoi lancer

| Ordre | Script | MINUTES | Images | Durée prévue |
|---|---|---|---|---|
| 1 | `a_sg_fold0.sh` | 30 | 13 (9 transplants + 4 outpaintings ratés, fold 0) | ≈ 8-16 min |
| 1 | `b_ps_fold0.sh` | 30 | 17 (9 transplants + 8 outpaintings ratés, fold 0) | ≈ 6-13 min |
| 2 | `a_sg_test_{0,1,2,3}.sh` | 55 | 113 images test StegaStamp, 4 parts de 28-29 | ≈ 15-35 min chacune |
| 2 | `b_ps_test_{0,1,2}.sh` | 55 | 139 images test PixelSeal, 3 parts de 46-47 | ≈ 15-35 min chacune |

Les tâches `fold0` d'abord : ce sont elles qui disent s'il y a des **faux positifs** (transplant à conf ≥ 0,9). Les durées
sont estimées avec les débits mesurés sur Quadro (StegaStamp ≈ 70-85 décodages/s, PixelSeal ≈ 135/s) et ≤ 6 000 décodages
par image ; si la carte est plus lente, la tâche sort en 3 : la redéposer, elle saute les images déjà faites.

```
for t in a_sg_fold0 b_ps_fold0 a_sg_test_0 a_sg_test_1 a_sg_test_2 a_sg_test_3 b_ps_test_0 b_ps_test_1 b_ps_test_2; do
  $R/relais-add $J4/$t.sh; done
$R/relais-start      # si ta chaîne ne tourne pas déjà
$R/relais-etat
```
Sans relais (sbatch à toi, un seul job, jamais dc-gpu-devel) : 4 tâches par nœud, par exemple
```
cd $TASK && sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 \
  --cpus-per-task=64 --time=00:58:00 --output=t1/out/jacques4/sbatch_%j.out --wrap "
  CUDA_VISIBLE_DEVICES=0 bash $J4/a_sg_test_0.sh > /dev/null 2>&1 &
  CUDA_VISIBLE_DEVICES=1 bash $J4/a_sg_test_1.sh > /dev/null 2>&1 &
  CUDA_VISIBLE_DEVICES=2 bash $J4/a_sg_test_2.sh > /dev/null 2>&1 &
  CUDA_VISIBLE_DEVICES=3 bash $J4/a_sg_test_3.sh > /dev/null 2>&1 &
  wait"
```
(les journaux sont dans `t1/out/jacques4/<tache>/run.log`).

## Ce que fait chaque tâche (`run.sh` → `seedgpu.py`)

Version GPU de la recherche CPU `sgseed` (StegaStamp, 37 → 52 rectangles test ≥ 0,9, 0 faux positif au fold 0) et `psseed`
(PixelSeal), avec ≈ 25 fois plus de décodages :
- **Départs** : boîtes englobantes des composantes des cartes de probabilité déjà calculées, aux seuils **0,3 / 0,5 / 0,7**,
  jusqu'à **4 composantes** par carte (≥ 16 cellules et ≥ 10 % de la plus grande) + la boîte de leur union, + le rectangle
  actuel ; dédoublonnage à 4 px. Cartes StegaStamp (19 en test, 18 au fold 0) : `stegatta`, `stegashift` v0-v7 et moyennes,
  `stegaens/models/*`, `jacques/sg_all_s1`. Cartes PixelSeal (29 en test, 21 au fold 0) : `tiles3/pixelseal/s2_22min` t192-384,
  `tta/pixelseal`, `pstta/pixelseal`, `devel/pixelseal/fold{1,2,3}` et `jacques/tiles_pixelseal_s1` (test).
- Tous les départs décodés une fois (crop R+3, décodeur et objectif de `t1/boxsearch3/rectsearch.Dec` ; StegaStamp en ONNX
  `CUDAExecutionProvider`, vérifiable en tête de `run.log` : « onnx providers »), puis les **10 meilleurs départs distincts**
  (IoU < 0,8) raffinés avec **600 décodages chacun** : recherche locale bord par bord (pas 16 → 1 px, translations, échelle),
  puis perturbations aléatoires des 4 bords (amplitude 12 → 2 px) suivies de recherche locale, jusqu'à épuisement du budget.
  On arrête une image dès qu'un rectangle atteint 0,97.
- Listes d'images figées dans `t1/jacques4/todo_{stegastamp,pixelseal}.json` (faites par `build_lists.py` depuis notre compte) :
  test = régime minoritaire (acc. image entière < 0,755 SG / 0,80 PS) sans rectangle ≥ 0,9 dans `~/sgseed` (SG) ou
  `t1/out/psbox` + `~/psseed` à 22:49 (PS) ; fold 0 = tous les transplants + les outpaintings dont le rectangle actuel est < 0,9.
- **Seuil 0,9 inchangé**, appliqué seulement à la fusion. `res.json` contient par image `rect`, `conf`, `margin`, `old_conf`,
  `old_rect`, `start`, `n_starts`, `n_decodes`, et au fold 0 `edit_type` et `iou` (outpaintings). Fin de `run.log` :
  `SUMMARY ...`, et au fold 0 `FOLD0 transplants ..., FALSE POSITIVES conf >= 0.9: [...]` (+ `WARNING` s'il y en a).

Gain attendu : chaque outpainting retrouvé passe d'≈ 0,5-0,6 à ≈ 0,95 sur l'image. sgseed a trouvé 15 rectangles test de plus
avec ≈ 60 décodages par départ ; avec 600 et 10 départs, on peut espérer quelques rectangles de plus parmi les 113 / 139
restantes, sans garantie (beaucoup sont des transplants, qui ne doivent pas monter à 0,9).

## Fusion (coordinateur, notre compte, CPU, quelques secondes)

```
cd $TASK && source .venv/bin/activate
python ~/psseed/merge.py            # d'abord, si l'agent psseed ne l'a pas fait (base PixelSeal à jour)
python t1/jacques4/merge.py         # -> ~/jacques4m/{stegastamp,pixelseal}_{train,test}_boxes.json
```
Règle : un rectangle ne remplace l'entrée de base que si conf ≥ 0,9 et base < 0,9. **S'il y a un faux positif au fold 0
pour une méthode, son fichier test n'est pas mis à jour (STOP imprimé)**. Les fichiers sortent au format des consommateurs
actuels (`--sgboxes` de la chaîne sub16 pour StegaStamp ; format psbox pour PixelSeal). Les résultats partiels (tâche sortie
en 3) sont pris en compte.

## Vérifier

```
$R/relais-etat
ls $TASK/t1/out/jacques4/*/DONE
grep -h "FOLD0\|WARNING" $TASK/t1/out/jacques4/*_fold0/run.log     # FAUX POSITIFS
grep -h "^SUMMARY\|^elapsed" $TASK/t1/out/jacques4/*/run.log        # rectangles nouveaux, décodages/s
```
Échec avec « budget reached, run the task again » dans `host.txt` : redéposer le même script. Autre échec : `run.log`.
Réglages : `BUDGET_MIN` (défaut MINUTES − 3), `seedgpu.py --K 10 --PER 600`.

## Tests en fumée (CPU, nœud de connexion, 21/11 22:50, jrlogin01)

```
cd $J4 && SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 bash a_sg_fold0.sh   # sorties t1/out/jacques4/smoke/ (effacées ensuite)
```
Mode fumée : 2 images, K = 2 départs, 40 décodages par départ. `a_sg_fold0`, `a_sg_test_0`, `a_sg_test_1`, `b_ps_fold0`,
`b_ps_test_1` : rc = 0, DONE, `res.json` au bon format, 18-29 cartes trouvées, 3-62 départs par image, 95-165 décodages par
image (≈ 10 déc./s SG, 7 PS sur 2 cœurs). `b_ps_fold0` a déjà retrouvé un outpainting raté (4bd2be05e610 : 0,879 → 0,902,
IoU 0,71) ; aucun transplant ≥ 0,9 (2 vus). Reprise testée : `BUDGET_MIN=1` → rc 3, relance → reprend et finit (rc 0),
troisième lancement → « already DONE ». `merge.py --root t1/out/jacques4/smoke` testé (fichiers écrits puis effacés).
Non testé : le chemin GPU (aucun calcul GPU lancé par nous) ; vitesse réelle et `CUDAExecutionProvider` à vérifier dans
`run.log` au premier passage.
