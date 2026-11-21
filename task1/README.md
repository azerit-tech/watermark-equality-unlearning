# Tâche 1 : localisation de watermark

État au samedi 21 nov, 18:50. Une soumission est faite : 0,5528 au leaderboard public (0,557 mesuré en local).

## Où est le code

Ce dossier est la copie de `t1/` sur le cluster, dans le dossier de la tâche :

```
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
```

C'est la copie du cluster qui tourne. Rien ne lit ce dépôt au moment d'un job.

| Dossier | Contenu |
|---|---|
| `common/` | Folds de validation, index du train par type de retouche, score officiel |
| `synth/` | Générateur de données synthétiques (embedders officiels du starter kit) |
| `train/` | U-Net pleine résolution, prédiction, garde-fou de type d'image |
| `dense/` | Segmenteur bâti sur les extracteurs officiels (TrustMark, PixelSeal) |
| `mgr/` | Jobs regroupés et construction du fichier de soumission |
| `signal/`, `exp1/`, `exp2/`, `edges/`, `fidelity/`, `frozen/`, `explore/` | Mesures et expériences, avec leurs résultats |

## Lancer le job regroupé

Un job, 4 GPU, 59 minutes. À lancer depuis le dossier de la tâche :

```
cd $TASK
sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 \
  --cpus-per-task=64 --time=00:59:00 --job-name=t1_combo --output=t1/out/mgr/%j.out t1/mgr/run_combo.sbatch
```

| GPU | Calcul | Sorties |
|---|---|---|
| 0 | TrustMark, segmenteur sur l'extracteur officiel | `t1/out/dense/d0/tm_ft/` |
| 1 | PixelSeal, idem, entrée 256 | `t1/out/dense/d0/ps256/` |
| 2 | StegaStamp v1, fold de validation 0 | `t1/out/train/v1/stegastamp_f0/` |
| 3 | StegaStamp v1, fold de validation 1 | `t1/out/train/v1/stegastamp_f1/` |

Le même job est déjà en file sous le compte `trentelivres1` (15902239). Si tu le lances depuis ton compte, préviens : on annule le nôtre pour ne pas occuper deux nœuds.

## Lire les résultats

```
cd $TASK && source .venv/bin/activate
tail -1 t1/out/dense/d0/tm_ft/log.jsonl          # champ "val" : score sur le fold 0 réel
tail -1 t1/out/dense/d0/ps256/log.jsonl
python t1/train/oof_report.py --root t1/out/train/v1 --method stegastamp
```

Chaque run écrit `fold0_probs128.npz` (ou `fold_probs128.npz`) et `test_probs128.npz` à la fin.

## Ne pas faire

- Ne rien envoyer au leaderboard depuis ces scripts : la soumission se décide à part.
- Ne pas modifier `t1/synth` ni `t1/train` pendant qu'un job attend : il lit ce code à son démarrage.

## Scores mesurés sur de vraies images (fold 0, 34 images par méthode)

| Méthode | Score | Méthode employée |
|---|---|---|
| StegaStamp | 0,763 | U-Net entraîné sur synthétique |
| TrustMark | 0,455 soumis ; 0,557 après 1,6 min d'essai du segmenteur sur extracteur | En cours |
| PixelSeal | 0,455 soumis ; 0,535 après 1,6 min d'essai | En cours |
