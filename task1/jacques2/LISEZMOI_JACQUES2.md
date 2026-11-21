# Tâches GPU vague 2 pour Jacques (tâche 1, Watermark Localization)

Même format que `t1/jacques` (vague 1) : chaque tâche prend **1 GPU**, en-têtes `# GPUS=1`, `# MINUTES=` (≤ 55), `# PRIO=2`,
`# DIR=$TASK` ; elle **reprend où elle en était si on la relance** (code de sortie 3 = budget atteint avant la fin : redéposer
le même script), écrit `DONE` à la fin, passe ses sorties en `chmod -R g+rwX`. Sorties sous `t1/out/jacques2/<tache>/`.
Aucune tâche ne dépend d'une autre ni de la vague 1 : ordre libre, toutes parallélisables.

```
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
R=/p/scratch/training2666/trentelivres1/Azerit/relais
J2=$TASK/t1/jacques2
```

## Quoi lancer, dans l'ordre conseillé

| Ordre | Script | MINUTES | Durée prévue A100 | Sortie principale (sous `t1/out/jacques2/`) |
|---|---|---|---|---|
| 1 | `c_holes_tm.sh` | 55 | ≈ 48 min (40 min d'entraînement + inférence) | `holes_trustmark/{fold0,test}_probs128.npz`, `eval_fold0.json` |
| 1 | `c_holes_ps.sh` | 55 | ≈ 48 min | `holes_pixelseal/{fold0,test}_probs128.npz`, `eval_fold0.json` |
| 2 | `a_sg_cloud_train.sh` | 40 | ≈ 12-25 min (76 images) | `cloud_stegastamp_train/` : mesure (outpaintings retrouvés, faux positifs) |
| 2 | `a_sg_cloud_test.sh` | 55 | ≈ 25-45 min (128 images) | `cloud_stegastamp_test/stegastamp_test_boxes.json` |
| 3 | `b_ps_cloud_train.sh` | 30 | ≈ 8-15 min (76 images) | `cloud_pixelseal_train/` : mesure |
| 3 | `b_ps_cloud_test.sh` | 45 | ≈ 15-25 min (141 images) | `cloud_pixelseal_test/pixelseal_test_boxes.json` |

C d'abord : c'est la seule tâche mesurable honnêtement au fold 0 (34 images par méthode jamais vues).
Un passage de relais à 4 cartes prend les deux tâches C + les deux tâches A ; un second les deux tâches B. Les durées A et B sont estimées à
partir des débits mesurés sur Quadro (StegaStamp ≈ 70-85 décodages/s, PixelSeal ≈ 135/s) et ≈ 2 300 décodages par image ;
si l'A100 est plus lente que prévu, la tâche sort en 3 et il suffit de la redéposer (elle saute les images déjà traitées).

Commandes exactes (ton compte) :
```
R=/p/scratch/training2666/trentelivres1/Azerit/relais
J2=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1/jacques2
$R/relais-add $J2/c_holes_tm.sh
$R/relais-add $J2/c_holes_ps.sh
$R/relais-add $J2/a_sg_cloud_train.sh
$R/relais-add $J2/a_sg_cloud_test.sh
$R/relais-add $J2/b_ps_cloud_train.sh
$R/relais-add $J2/b_ps_cloud_test.sh
$R/relais-start                           # si ta chaîne ne tourne pas déjà
$R/relais-etat
```

Sans relais (sbatch à toi, un seul job, jamais dc-gpu-devel) : 4 tâches sur un nœud
```
cd $TASK && sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 \
  --cpus-per-task=64 --time=00:58:00 --output=t1/out/jacques2/sbatch_%j.out --wrap "
  CUDA_VISIBLE_DEVICES=0 bash $J2/a_sg_cloud_test.sh  > t1/out/jacques2/a_test.out 2>&1 &
  CUDA_VISIBLE_DEVICES=1 bash $J2/a_sg_cloud_train.sh > t1/out/jacques2/a_train.out 2>&1 &
  CUDA_VISIBLE_DEVICES=2 bash $J2/c_holes_tm.sh       > t1/out/jacques2/c_tm.out 2>&1 &
  CUDA_VISIBLE_DEVICES=3 bash $J2/c_holes_ps.sh       > t1/out/jacques2/c_ps.out 2>&1 &
  wait"
```
(puis un second job pour `b_ps_cloud_test.sh` et `b_ps_cloud_train.sh`).

## Ce que fait chaque tâche, et ce qu'on en attend

**A. `a_sg_cloud_{test,train}.sh` : StegaStamp, recherche « nuage » des rectangles d'outpainting** (`cloud.sh` → `cloud.py`).
Images traitées : test = régime minoritaire (fullacc < 0,755) dont le meilleur rectangle actuel
(`t1/out/stegaedge/stegastamp_test_boxes.json`) a une confiance < 0,9 (128 images) ; train = toutes les entrées < 0,9 de
`stegastamp_train_boxes.json` (76 : outpaintings ratés + transplants/holes minoritaires pour compter les faux positifs).
Pour chaque image : les ≈ 40 candidats stegaedge (`edge_<split>.json`, champ `cands`) + le rectangle actuel ; autour de chacun,
50 rectangles dont les 4 bords sont décalés indépendamment de U{−8..8} px ; tout est décodé sur GPU (même décodeur et même
objectif que `t1/boxsearch3/rectsearch.Dec` / `t1/sgbox2/common.py` : crop R+3, bicubique 400, ONNX CUDAExecutionProvider
via `wmloc.load_watermark`) ; les 3 meilleurs distincts sont raffinés bord par bord (pas 4, 2, 1 px + translations).
Sortie `stegastamp_<split>_boxes.json` au format stegaedge (`{"format","method","boxes":{nom:{rect,conf,src}}}`) : copie de
la base où une entrée n'est remplacée (src `cloud`) que si la confiance du nuage est plus haute. **Seuil 0,9 inchangé**,
appliqué par le consommateur comme avant. Détail par image : `cloud_<split>.json` (ancien/nouveau rect et conf, IoU au train).
Gain attendu : l'outpainting StegaStamp vaut 0,67 au fold 0 ; chaque outpainting retrouvé passe d'≈ 0,5-0,6 à ≈ 0,95
(+0,35-0,45 sur l'image). Si le nuage récupère 10 des 128 images test : ≈ +0,035 sur StegaStamp, ≈ +0,012 au total.
Le train dit combien d'outpaintings ratés sont retrouvés (fin de `cloud.log` : « newly >= 0.9 », IoU, miou du rectangle) et
surtout s'il y a des **faux positifs** (transplants à conf ≥ 0,9) : s'il y en a, ne pas adopter sans relever le seuil
par validation croisée.

**B. `b_ps_cloud_{test,train}.sh` : la même chose pour PixelSeal** (seuil 0,90 de `t1/psbox/cv_build.py`, inchangé).
Base test = `t1/out/psbox/pixelseal_test_boxes.json` (141 images < 0,9) ; base train = fusion pass1 + pass3 comme
`cv_build.py` (`t1/out/boxsearch/pixelseal_train_boxes.json` + `t1/out/boxsearch3/pixelseal_train_pass3.json`, 76 < 0,9).
Il n'y a pas de candidats stegaedge pour PixelSeal : ils sont recalculés par le même score de périmètre
(`t1/stegaedge/perim.py`, top-40 après NMS) + le rectangle actuel + les rectangles issus de la segmentation (`seg_rect`).
Décodeur PixelSeal 256 bilinéaire (rectsearch). Sortie `pixelseal_<split>_boxes.json` au format psbox (`rect`, `box`,
`conf`, `src`, `selected` = conf ≥ 0,9, `thr`). Pour l'utiliser : remplacer l'entrée de `cv_build.py`/de la chaîne v3 par ce
fichier (même règle : régime minoritaire, conf ≥ 0,9, masque = rectangle). Gain attendu : du même ordre que A (outpainting
PixelSeal ≈ 0,59 au fold 0), à confirmer par le train.

**C. `c_holes_{tm,ps}.sh` : segmenteurs tiles réentraînés avec PLUS de trous synthétiques, folds 1-4 seulement**
(`holes_ft.sh`). Recette de `t1/holes/job.sbatch` (non soumise par l'agent holes) : affinage de 40 min à partir des modèles
fold 0 (`t1/out/tiles2/trustmark/last.pt`, `t1/out/tiles3/pixelseal/last.pt`, entraînés val_fold 0), générateur
`t1/holes/train_tiles.py` avec `--mix outpainting=1,transplant=1,hole=4` (deux fois plus de trous que la recette d'origine,
hole=2) et `--hole_src new=1,lib=1` (moitié `sample_hole_mask`, moitié vrais trous des folds 1-4), `--p_focus 0.7`,
`--val_fold 0`. On garde l'EMA du **dernier** état (jamais `best.pt`, choisi sur le fold 0 : il est effacé).
Puis `pred_c.py` : `fold0_probs128.npz` (34 images, honnête : fold 0 jamais vu), `test_probs128.npz` (340 images ; TM échelle
320, PS moyenne 256/320) et `eval_fold0.json` (comparaison appariée avec le modèle de départ, bootstrap 10 000).
Gain attendu : l'essai de 6 min sur Quadro avait donné +0,0054 (PS, IC95 [0,0007 ; 0,0118]) et +0,0056 (TM, IC95
[−0,0002 ; 0,0117]) ; 40 min sur A100 devraient faire au moins autant. En dessous de +0,03 sur 34 images c'est du bruit :
à adopter seulement comme membre d'ensemble, ou si l'IC95 de `eval_fold0.json` exclut 0.

## Vérifier

```
$R/relais-etat                                     # done/ ou failed/ ; journal : $R/logs/<nom>.log
ls $TASK/t1/out/jacques2/*/DONE
tail -n 4 $TASK/t1/out/jacques2/cloud_stegastamp_train/cloud.log   # outpaintings retrouvés, IoU, FAUX POSITIFS
tail -n 4 $TASK/t1/out/jacques2/cloud_pixelseal_train/cloud.log
grep "^fold0" $TASK/t1/out/jacques2/holes_*/pred.log                # base -> new, IC95, par type
```
Échec en `failed/` avec « run the task again » ou « budget reached » dans `host.txt` : redéposer le même script (il reprend).
Autre échec : `cloud.log`, `train.log`, `pred.log` du dossier de sortie.

Réglages par variables d'environnement (facultatif) : `BUDGET_MIN` (budget mural, défaut MINUTES − 3), `MIN` (minutes
d'affinage C, 40), `WORKERS` (12). Pour A/B, `cloud.py --K 40 --N 50 --J 8` (candidats, taille du nuage, amplitude en px).

## Tests en fumée (CPU, nœud de connexion)

```
cd $J2 && SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 bash a_sg_cloud_test.sh
```
(sorties sous `t1/out/jacques2/smoke/…` ; A/B : 2 images, 4 candidats × 5 rectangles ; C : 4 images fold 0, 2 images test ;
journaux dans `smoke_logs/`).
Résultats du 21/11 21:34-21:37 (nœud jrlogin05, CPU), les 6 scripts rc=0 et DONE, chacun en moins d'1 min :
- `a_sg_cloud_test`, `a_sg_cloud_train`, `b_ps_cloud_test`, `b_ps_cloud_train` : 2 images chacune (≈ 170-265 décodages,
  16-25 s par image sur CPU), fichier `<methode>_<split>_boxes.json` écrit au bon format, bilan train imprimé (pas de faux
  positif sur les 2-3 images non-outpainting vues). Le chemin « candidats perimètre recalculés » (PixelSeal) est testé.
- `c_holes_tm`, `c_holes_ps` : init depuis tiles2/tiles3, `pred_c.py` écrit `fold0_probs128.npz`, `test_probs128.npz`,
  `eval_fold0.json`. NB : en mode fumée `train_tiles.py` s'arrête avant le 1er pas (son test `it >= 2` voit
  l'it 6745 du point de départ). Le chemin réel a donc été testé à part sur CPU (TrustMark, bs 2, `--minutes 0.6
  --stop_min 0.3` puis relance `--resume`) : 40 pas d'entraînement, last.pt sauvé (elapsed 36 s), reprise OK, rc=0 les 2 fois.
Non testé : la vitesse réelle sur A100 (durées A/B estimées) et le chemin shell « exit 3 → relance » (même logique que la
vague 1). Attention : le temps compté par `train_tiles.py` inclut la validation fold 0 de fin de tranche (≈ 10 s sur GPU).
