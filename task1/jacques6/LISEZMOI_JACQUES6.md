# Tâches GPU vague 6 pour Jacques (tâche 1) : recettes gagnantes réentraînées sur TOUT le train réel

À lancer **APRÈS les tâches longues de jacques5 / jacques5b** (une seule allocation GPU par équipe).
Même format que `t1/jacques2` / `t1/jacques3` : chaque tâche prend **1 GPU**, en-têtes `# GPUS=1`, `# MINUTES=55`,
`# PRIO=2`, `# DIR=$TASK` ; elle **reprend où elle en était si on la relance** (code de sortie 3 = budget atteint avant la
fin : redéposer le même script), écrit `DONE` à la fin, passe ses sorties en `chmod -R g+rwX`.
Sorties sous `t1/out/jacques6/<tache>/`, **4 fichiers par tâche** : `last.pt`, `run.log`, `test_probs128.npz`, `DONE`
(`log.jsonl`, `best.pt`, `ema_last.pt` des scripts d'entraînement sont effacés à chaque sortie).

```
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
R=/p/scratch/training2666/trentelivres1/Azerit/relais
J6=$TASK/t1/jacques6
$R/relais-add $J6/informed_ps_all.sh
$R/relais-add $J6/informed_tm_all.sh
$R/relais-add $J6/holes_ps_all.sh
$R/relais-start          # si ta chaîne ne tourne pas déjà
$R/relais-etat
```

| Ordre | Script | MINUTES | Durée prévue A100 | Recette d'origine (gain fold 0, folds 1-4) | Sortie |
|---|---|---|---|---|---|
| 1 | `informed_ps_all.sh` | 55 | ≈ 45 min (40 min d'affinage + test) | jacques3 `informed_pixelseal` (+0,014) | `informed_ps_all/test_probs128.npz` (340, PS, échelles 256/320) |
| 2 | `informed_tm_all.sh` | 55 | ≈ 45 min | jacques3 `informed_trustmark` (+0,014) | `informed_tm_all/test_probs128.npz` (340, TM, échelle 320) |
| 3 | `holes_ps_all.sh` | 55 | ≈ 45 min | jacques2 `holes_pixelseal` (+0,012) | `holes_ps_all/test_probs128.npz` (340, PS, échelles 256/320) |

Les trois tâches sont indépendantes (parallélisables) ; l'ordre ci-dessus est l'ordre de priorité.

## Ce qui change par rapport aux recettes d'origine

- Code : `ft6.sh <tache>` (shell commun) → `run6.py train` lance **sans le modifier** le script d'origine
  (`t1/jacques3/train_informed.py` ou `t1/holes/train_tiles.py`), mêmes hyperparamètres, même départ
  (`t1/out/tiles3/pixelseal/last.pt`, `t1/out/tiles2/trustmark/last.pt`), même durée (`MIN` = 40 min, cosinus sur 40 min).
- Réel : **les 5 folds** (`--val_fold -1` : tous les folds ≠ −1 ; 170 images par méthode au lieu de 136).
- Bibliothèque de formes du synthétique (trous / silhouettes tirés des vrais masques) : folds 0-4 au lieu de 1-4
  (`run6.py` remplace la valeur par défaut `(1,2,3,4)` de `SynthDS`, `MsgHoleSynth`, `Library`). Les couvertures
  synthétiques pré-générées (`t1/out/synth/v0`) restent celles d'origine.
- **Aucune sélection sur une validation** : on garde l'EMA du dernier état. NB `train_tiles.py` (holes) appelle encore
  `validate()` en fin de tranche ; avec `--val_fold -1` elle porte sur le fold 4 (vu à l'entraînement, ≈ 10 s, sans effet) et
  le `best.pt` qu'elle écrit est effacé.
- `run6.py pred` : **seulement** `test_probs128.npz` (format habituel : `names`, `probs` float16 [340,128,128]) ; pas de fold 0
  possible (tout le réel est vu). TM échelle 320, PS moyenne 256/320, comme jacques2/3.

Usage prévu : membres supplémentaires des moyennes de test, **sans réglage** (même procédé que celui qui a fait +0,006 au
public), par exemple `informed_ps_all` dans la moyenne PixelSeal à côté de `informed_pixelseal` (jacques3). Aucun score
honnête n'est mesurable pour ces modèles : ne pas les utiliser pour régler quoi que ce soit.

## Vérifier
```
$R/relais-etat
ls $TASK/t1/out/jacques6/*/DONE
cat $TASK/t1/out/jacques6/*/DONE                 # "trained 2400s / 2400s", "wrote ... (340) ... mean ..."
```
Échec en `failed/` avec « run the task again » dans `run.log` / le journal du relais : redéposer le même script (reprend de
`last.pt`, sauvé toutes les 5 min pour informed, en fin de tranche pour holes). Autre échec : `run.log` du dossier de sortie.
Réglages (facultatif) : `BUDGET_MIN` (52), `MIN` (minutes d'affinage, 40), `WORKERS` (12).

## Tests en fumée (CPU, nœud de connexion)
```
cd $TASK/t1/jacques6 && SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 bash informed_ps_all.sh
```
(sorties sous `t1/out/jacques6/smoke/<tache>/`, à effacer après ; bs 2, workers 0 ; informed : 2 images réelles (`--smoke`),
2 pas puis 2 pas repris de `last.pt` ; holes : `train_tiles.py --smoke` s'arrête avant le 1er pas (compteur du point de
départ), le test passe donc par le vrai chemin `--stop_min 0.15` deux fois (2e fois = reprise) ; prédiction sur 2 images test).
Résultats du 21/11 23:17-23:20 (nœud jrlogin02, CPU, les 3 en parallèle), les 3 scripts rc=0 et DONE, chacun < 3 min,
sorties de fumée effacées ensuite :
- `informed_ps_all`, `informed_tm_all` : `lib_folds=(0, 1, 2, 3, 4)` confirmé, `(folds != -1)`, entrées `[2, 7, 256|320, ...]`,
  init tiles3 / tiles2, 2 pas, `last.pt`, puis `RESUME it=2` et 2 pas de plus (reprise OK), `test_probs128.npz` (2, finies).
- `holes_ps_all` : `real_train=170` (5 folds), `lib_folds=(0, 1, ...)`, init tiles3 it 3526 → 3534 (8 pas) → reprise → 3542,
  `best.pt`/`ema_last.pt`/`log.jsonl` effacés, `test_probs128.npz` (2, échelles 256/320). Dossier final : 4 fichiers.
Non testé : la vitesse sur A100 (mêmes recettes que jacques2/3, 40 min d'affinage + ≈ 2-3 min d'inférence test), le chemin
« exit 3 → relance » (même logique que jacques2/3).
