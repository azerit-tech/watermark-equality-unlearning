# Tâches GPU vague 3 pour Jacques (tâche 1) : segmenteur tiles « informé » (TrustMark, PixelSeal)

Même format que `t1/jacques2` : chaque tâche prend **1 GPU**, en-têtes `# GPUS=1`, `# MINUTES=55`, `# PRIO=2`, `# DIR=$TASK` ;
elle **reprend où elle en était si on la relance** (code de sortie 3 = budget atteint avant la fin : redéposer le même
script), écrit `DONE` à la fin, passe ses sorties en `chmod -R g+rwX`. Sorties sous `t1/out/jacques3/<tache>/`.
Les deux tâches sont indépendantes (entre elles, et des vagues 1-2) : parallélisables.

```
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
R=/p/scratch/training2666/trentelivres1/Azerit/relais
J3=$TASK/t1/jacques3
$R/relais-add $J3/d_informed_tm.sh
$R/relais-add $J3/d_informed_ps.sh
$R/relais-start          # si ta chaîne ne tourne pas déjà
$R/relais-etat
```

| Script | MINUTES | Durée prévue A100 | Sortie (sous `t1/out/jacques3/`) |
|---|---|---|---|
| `d_informed_tm.sh` | 55 | ≈ 48 min (40 min d'affinage + ≈ 5 min d'inférence fold 0 ×2 modèles + test) | `informed_trustmark/{fold0,test}_probs128.npz`, `eval_fold0.json` |
| `d_informed_ps.sh` | 55 | ≈ 48 min | `informed_pixelseal/{fold0,test}_probs128.npz`, `eval_fold0.json` |

Sans relais (un seul job, 2 cartes) :
```
cd $TASK && sbatch --account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:2 \
  --cpus-per-task=32 --time=00:58:00 --output=t1/out/jacques3/sbatch_%j.out --wrap "
  CUDA_VISIBLE_DEVICES=0 bash $J3/d_informed_tm.sh > t1/out/jacques3/d_tm.out 2>&1 &
  CUDA_VISIBLE_DEVICES=1 bash $J3/d_informed_ps.sh > t1/out/jacques3/d_ps.out 2>&1 &
  wait"
```

## Ce que fait la tâche D (`informed_ft.sh` → `train_informed.py` → `pred_informed.py`)

Idée (agent papers) : le modèle tiles (`TileSeg`, encodeur = extracteur officiel + logits par position signés par le message)
ne voit ni le résidu de ré-insertion ni la JND. On lui ajoute 4 canaux d'entrée pleine résolution, **calculés à la volée sur
le GPU** à partir de l'image observée x et de son message connu m (`informed.py`) :

| Canal | Calcul | Échelle |
|---|---|---|
| HF(Y) | luma − flou gaussien 5×5 | ×255/8 |
| r'(m) | luma de Embed(x, m) − x (ré-insertion du MÊME message, résidu flottant avant arrondi) | ×255/2 |
| d | r'(m) − r'(1−m) (retire la part liée au contenu) | ×255/2 |
| JND(Y) | carte JND de videoseal (`modules/jnd.py`, luma) | ×255/10 |

- Embed : TrustMark = mêmes étapes que `trustmark.encode` en lot torch (bilinéaire 256, encodeur, clamp, moyenne par canal
  retirée, bilinéaire retour) ; PixelSeal = `videoseal model.embed` (embedder Y seul + atténuation JND). Les canaux sont calculés
  sur l'image 512 entière (l'insertion est globale), puis découpés avec la tuile RGB (entraînement et inférence).
- Modèle `InformedTileSeg` : un petit tronc lit [HF, r', d, JND, HF·d, HF·r'] à la résolution de la tuile → 32 canaux sur la
  grille T/4 (convolutions dilatées pour agréger la corrélation sur ≈ 30 px), ajoutés à l'entrée de la tête. **Les poids
  de la tête pour ces 32 canaux sont nuls au départ** : le modèle démarre exactement comme tiles2/tiles3.
- Affinage 40 min depuis `t1/out/tiles2/trustmark/last.pt` (TM : bs 16, lr 3e-4, lr_enc 1,5e-5, lr du tronc 1e-3) /
  `t1/out/tiles3/pixelseal/last.pt` (PS : bs 24, lr 1,5e-4, lr_enc 1e-5), réel folds 1-4 (p_real 0,25) + synthétique
  `t1/out/synth/v0` (recette tiles3 : outpainting 1, transplant 1, hole 2). Optimiseur neuf + montée de lr sur 200 pas.
  **Aucune validation sur le fold 0 pendant l'entraînement, pas de best.pt** : on garde l'EMA du dernier état.
- `pred_informed.py` : `fold0_probs128.npz` (34 images jamais vues), `test_probs128.npz` (340 images ; TM échelle 320,
  PS moyenne 256/320 comme jacques2) et `eval_fold0.json` (comparaison appariée avec le modèle de départ, mêmes échelles,
  bootstrap 10 000, par type). Une ligne `fold0 all {...}` est copiée dans `host.txt`/`DONE`.

Gain attendu : inconnu (idée nouvelle). Sur 34 images, < +0,03 = bruit : n'adopter que si l'IC95 de `eval_fold0.json` exclut 0,
sinon seulement comme membre d'ensemble.

## Vérifier
```
ls $TASK/t1/out/jacques3/*/DONE
grep "^fold0" $TASK/t1/out/jacques3/informed_*/pred.log        # base -> new, IC95, par type
grep aux_head_w $TASK/t1/out/jacques3/informed_trustmark/train.log | tail -n 3   # > 0 : la tête utilise les nouveaux canaux
```
Échec en `failed/` avec « run the task again » dans `host.txt` : redéposer le même script (reprend de `last.pt`, sauvé toutes
les 5 min et à la fin de chaque tranche). Autre échec : `train.log`, `pred.log` du dossier de sortie.
Réglages (facultatif) : `BUDGET_MIN` (52), `MIN` (minutes d'affinage, 40), `WORKERS` (12).

## Tests en fumée (CPU, nœud de connexion)
```
cd $J3 && SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 WORKERS_SMOKE=0 bash d_informed_tm.sh
```
(sorties sous `t1/out/jacques3/smoke/…` ; 2 images réelles + synthétique, bs 2, 2 pas, puis 2 pas repris de `last.pt` ;
prédiction sur 2 images fold 0 et 2 images test ; journaux dans `smoke_logs/`).
Résultats du 21/11 21:59-22:00 (nœud jrlogin10, CPU), `d_informed_tm.sh` et `d_informed_ps.sh` rc=0 et DONE, < 1,5 min chacun :
- entrées de forme `[2, 7, 256, 256]` / `[2, 7, 320, 320]` (tailles de tuile alternées), canaux auxiliaires d'ordre 1
  (|HF|, |r'|, |d|, |JND| moyens ≈ 0,3-1,3), pertes finies, pas de NaN ; init depuis tiles2 (TM, encodeur déjà dégelé) et
  tiles3 (PS) ; 2 pas, `last.pt`, puis `RESUME it=2` et 2 pas de plus (reprise OK) ; les poids de la tête pour les nouveaux
  canaux quittent 0 (1e-5 après 4 pas à lr de montée : le gradient passe).
- `pred_informed.py` : `fold0_probs128.npz` (2), `test_probs128.npz` (2, forme (2,128,128), finies), `eval_fold0.json`.
- `check_embed.py` (CPU) : le résidu de ré-insertion en lot corrèle à 0,99 (TM) et 0,92-0,98 (PS) avec `wm.embed(x,m) − x`
  officiel (arrondi uint8 compris) ; sur 2 couvertures synthétiques entièrement watermarkées, corr(r'(m), vrai résidu) =
  0,79-0,86 (TM) et 0,73-0,78 (PS), corr(d, vrai résidu) ≈ 0,62-0,66 : conforme aux chiffres de l'agent papers.
Non testé : la vitesse sur A100 (≈ 2 ré-insertions à 256 + JND 512 par image en plus du modèle tiles : surcoût estimé
faible), le chemin « exit 3 → relance » côté shell (même logique que jacques2), et bien sûr le gain réel.
