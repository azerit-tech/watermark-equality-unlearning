# Tâches GPU préparées pour Jacques (tâche 1, Watermark Localization)

Notre compte (trentelivres1) n'a plus accès à Slurm. Ces tâches sont prêtes pour **ton** relais (ou un sbatch à toi).
Chaque tâche prend 1 A100, tient dans sa durée annoncée avec marge, **reprend où elle en était si on la relance**
(code de sortie 3 = budget atteint avant la fin : il suffit de la redéposer), et écrit un fichier `DONE` à la fin.
Toutes les sorties sont sous `t1/out/jacques/<tache>/` et passent en `chmod -R g+rwX`.

```
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
R=/p/scratch/training2666/trentelivres1/Azerit/relais
J=$TASK/t1/jacques
```

## Quoi lancer, dans l'ordre

| Ordre | Script | GPU | MINUTES (en-tête) | Durée réelle prévue | Sorties |
|---|---|---|---|---|---|
| 1 | `t1_ps_fold4.sh` (PRIO=1) | 1 | 20 | ≈ 10 min | `t1/out/jacques/ps_oof/fold4/` |
| 2 | `t2_tiles_tm_s1.sh` | 1 | 55 | ≈ 45 min | `t1/out/jacques/tiles_trustmark_s1/` |
| 2 | `t2_tiles_ps_s1.sh` | 1 | 55 | ≈ 45 min | `t1/out/jacques/tiles_pixelseal_s1/` |
| 3 | `t3_sg_all_s1.sh` | 1 | 55 | ≈ 45 min | `t1/out/jacques/sg_all_s1/` |
| 4 | `t2_tiles_tm_s2.sh`, `t2_tiles_ps_s2.sh` | 1 chacun | 55 | ≈ 45 min | `tiles_<methode>_s2/` |
| 4 | `t3_sg_all_s2.sh` | 1 | 55 | ≈ 45 min | `sg_all_s2/` |
| 5 (option) | `t1b_ps_folds123.sh` | 1 | 45 | ≈ 30 min | `ps_oof/fold{1,2,3}/` |

Un passage de relais (4 cartes) suffit pour les lignes 1 à 3 (4 tâches) ; un second pour les lignes 4 et 5.

Commandes exactes (depuis n'importe où, ton compte) :
```
R=/p/scratch/training2666/trentelivres1/Azerit/relais
J=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques
$R/relais-add $J/t1_ps_fold4.sh
$R/relais-add $J/t2_tiles_tm_s1.sh
$R/relais-add $J/t2_tiles_ps_s1.sh
$R/relais-add $J/t3_sg_all_s1.sh
$R/relais-add $J/t2_tiles_tm_s2.sh
$R/relais-add $J/t2_tiles_ps_s2.sh
$R/relais-add $J/t3_sg_all_s2.sh
$R/relais-add $J/t1b_ps_folds123.sh      # optionnel
$R/relais-start                           # si ta chaîne ne tourne pas déjà
$R/relais-etat
```
(Les en-têtes `# GPUS=1`, `# MINUTES=`, `# PRIO=`, `# DIR=$TASK` sont dans les scripts ; CUDA_VISIBLE_DEVICES vient du relais.)

Sans relais (sbatch à toi, un seul job à la fois, jamais dc-gpu-devel) : 4 tâches en parallèle sur un nœud
```
cd $TASK && sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 \
  --cpus-per-task=64 --time=00:58:00 --output=t1/out/jacques/sbatch_%j.out --wrap "
  CUDA_VISIBLE_DEVICES=0 bash $J/t1_ps_fold4.sh > t1/out/jacques/t1.out 2>&1 &
  CUDA_VISIBLE_DEVICES=1 bash $J/t2_tiles_tm_s1.sh > t1/out/jacques/t2tm1.out 2>&1 &
  CUDA_VISIBLE_DEVICES=2 bash $J/t2_tiles_ps_s1.sh > t1/out/jacques/t2ps1.out 2>&1 &
  CUDA_VISIBLE_DEVICES=3 bash $J/t3_sg_all_s1.sh > t1/out/jacques/t3sg1.out 2>&1 &
  wait"
```

## Ce que fait chaque tâche, et ce qu'on en attend

1. **`t1_ps_fold4.sh` : PixelSeal hors-fold, fold 4** (membre manquant de l'ensemble `t1/out/mgr/ens/pixelseal`).
   Le job devel a été tué vers 18 min : la phase a (8 min) est finie, la phase b était à 7,04/14 min. La tâche copie
   `t1/out/devel/pixelseal/fold4/b/last.pt` et **reprend** la phase b jusqu'à 14 min (schéma cosinus conservé), puis prédit
   fold 4 + test (moyenne des échelles 256 et 320, EMA du dernier état). Si l'état devel manquait, elle referait a puis b.
   Sorties : `pred/test_probs128.npz` (340 images PS du test), `pred/fold4_probs128.npz` (carte hors-fold honnête),
   `pred/eval_fold.json` (score fold 4). Pour info, les folds 1-3 de l'ensemble actuel sont des états à 7/14 min.
   Gain attendu : complète l'ensemble test PS (4e membre hors-fold) ; petit (de l'ordre de +0,00x au total), non mesurable
   sur fold 0 (l'ensemble test n'a pas d'équivalent fold 0 propre).
1b. **`t1b_ps_folds123.sh` (option)** : la même reprise pour les folds 1, 2, 3 (à la suite, une carte, ≈ 10 min par fold).
   Les membres de l'ensemble passent de 7 à 14 min de phase b, comme la recette de référence ; les cartes hors-fold
   (`ps_oof/foldK/pred/foldK_probs128.npz`) permettent de vérifier le gain fold par fold face aux anciennes
   (`t1/out/devel/pixelseal/foldK/pred/`).
2. **`t2_tiles_<tm|ps>_s<1|2>.sh` : segmenteurs tiles finaux sur les 5 folds réels** (recette `t1/bigtrain/jobB_tiles.sbatch` :
   TrustMark tiles2, PixelSeal tiles3), 8000 + 8000 paires synthétiques neuves (couvertures et bibliothèque = 5 folds,
   marge outpainting 2,5 ; ≈ 3 min de génération), 36 min d'entraînement, inférence test ≈ 2 min.
   Sorties : `test_probs128.npz` (TM : échelle 320 ; PS : moyenne 256/320), plus `test_t256_…`, `test_t320_…`, `ema_final.pt`.
   Pas de carte fold 0 honnête possible (tous les folds sont vus). Gain attendu : membres d'ensemble supplémentaires,
   avec 25 % de données réelles en plus que les modèles fold 0 ; l'ensemble test a déjà rapporté +0,006 au public.
3. **`t3_sg_all_s<1|2>.sh` : StegaStamp U-Net final sur les 5 folds** (recette v1 de `jobA_stega.sbatch`, `run_all`),
   6000 + 6000 paires neuves (≈ 4,5 min à 47 paires/s), 36 min d'entraînement (la référence fold 0 a tourné 36 min,
   0,761), inférence test ≈ 1 min. Sortie : `test_probs128.npz` (340 images Stega du test), `final.pt`. Pas de carte fold 0.
   Entraînement via `t1/jacques/train_sg_chunk.py` = copie de `t1/bigtrain/train_sg.py` + `--chunk_min` (arrêt propre au
   budget, reprise avec `--resume`).

Réglages par variables d'environnement (facultatif) : `MIN` (minutes d'entraînement), `NF`/`NR` (paires synthétiques),
`BUDGET_MIN` (budget mural de la tâche, défaut = MINUTES − 3), `KEEP_SYNTH=1` (garder le synthétique, sinon effacé à la fin,
≈ 0,8 Mo par paire), `WORKERS` (12).

## Vérifier

```
$R/relais-etat                                   # done/ ou failed/ ; journal : $R/logs/<nom>.log
ls $TASK/t1/out/jacques/*/DONE $TASK/t1/out/jacques/ps_oof/fold*/DONE
cat $TASK/t1/out/jacques/ps_oof/fold4/pred/eval_fold.json     # score hors-fold fold 4 (devel tronqué : ≈ 0,68 en val interne)
cat $TASK/t1/out/jacques/tiles_trustmark_s1/host.txt          # chaque dossier : host.txt = étapes + forme/moyenne des probs
```
Échec en `failed/` avec « run the task again » dans `host.txt` : budget atteint, redéposer le même script (il reprend).
Autre échec : voir `gen.log`, `train*.log`, `pred.log` du dossier de sortie.

## Tests en fumée (CPU, nœud de connexion)

```
cd $J && SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 bash t2_tiles_tm_s1.sh
```
(sorties sous `t1/out/jacques/smoke/…`, 2 itérations, 4 paires synthétiques, peu d'images ; journaux dans `smoke_logs/`).
Résultats du 21/11 21:08, tous rc=0 et DONE en moins de 1,5 min chacun : `t1_ps_fold4` (copie + reprise de devel b, prédiction
fold 4 + test), `t2_tiles_tm_s1`, `t2_tiles_ps_s1`, `t3_sg_all_s1` (génération, 2 itérations, test_probs128 de forme (4,128,128)).
Non testé en fumée : le chemin « budget atteint → exit 3 → relance » (il ne s'active qu'au-delà du mode fumée) et `t1b`
(même worker que `t1`).
