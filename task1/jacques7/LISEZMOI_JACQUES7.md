# jacques7 : U-Net StegaStamp « informé » (r'(m) et d en entrée), tâche relais

```
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
R=/p/scratch/training2666/trentelivres1/Azerit/relais
$R/relais-add $TASK/t1/jacques7/sg_informed.sh      # 1 GPU, une tranche suffit normalement
```
En-tête : `# GPUS=1`, `# MINUTES=55`, `# PRIO=2`, `# DIR=$TASK`. Durée prévue sur A100 : ≈ 1 min de chargement, 40 min d'affinage
puis ≈ 2-3 min d'inférence (7 min réservées). Indépendante de jacques5b (autre dossier, même modèle de départ, lu seulement).
Code de sortie 3 : budget atteint avant la fin des 40 min. Redéposer le même script : il reprend de `last.pt` (sauvé toutes
les 5 min et à la fin). `DONE` à la fin, sorties en `chmod -R g+rwX`.

## Ce que fait la tâche
- Modèle : le U-Net pleine résolution de `t1/train/unet.py`, entrée 3 + 6 canaux. Les 6 canaux sont calculés sur le GPU à
  partir de l'image observée x et de son message CONNU m (métadonnées réelles ou `msg` des paires synthétiques) :
  - `r'(m) = Embed(x, m) − x` (3 canaux couleur, flottant, avant l'arrondi uint8), ×255/8, borné à ±8 ;
  - `d = r'(m) − r'(1−m)` (3 canaux : la part du résidu qui dépend du contenu est retirée), même échelle.
  Embed reproduit `wmloc` `StegaStamp.embed` en lot : bicubique (antialias) 512 → 400, sortie `residual` de l'ONNX
  (`models/stegastamp/stega_stamp_logits.onnx`, onnxruntime CUDA, assertion si CUDA manque), moyenne par canal retirée,
  bilinéaire 400 → 512, ajout, borne [0, 1].
- Départ : `t1/out/train/v0/stegastamp/best.pt` (= stega/best). Les poids de la première convolution pour les 6 nouveaux
  canaux sont **nuls** : le modèle démarre exactement comme stega/best.
- Affinage : 40 min (cosinus sur 40 min cumulées, montée de lr sur 200 pas, AdamW lr 5e-4, bs 12, EMA 0,998), recette de
  stega/best : réel folds 1-4 (136 images Stega, p_real 0,25) + synthétique `t1/out/synth/v0/stegastamp` (mix outpainting 1,
  transplant 1, hole 2). Le fold 0 n'est **jamais chargé** pendant l'entraînement : pas de validation, pas de best.pt, on garde
  l'EMA du dernier état (`last.pt["ema"]`).
- Inférence et évaluation (`eval`) dans `$TASK/t1/out/jacques7/sg_informed/` :
  - `fold0_probs128.npz` (34 images Stega du fold 0) et `test_probs128.npz` (340 images Stega du test), format de référence ;
  - `eval_fold0.json` : par type (`all/hole/transplant/outpainting`), score officiel des cartes brutes (seuil 0,5, classe
    minoritaire jamais vide) de `new` (nouveau modèle), `ref` (`t1/out/stega/best/fold0_probs128.npz`), `base` (poids de
    départ recalculés ici, contrôle) et `new_noaux` (nouveaux poids, canaux r'/d mis à zéro = ablation), avec les écarts
    appariés `new − ref`, `new − base`, `new − noaux` (bootstrap 10 000, IC 95 %, P(écart > 0)), et `stem_w_abs`
    (poids moyens |w| de la première convolution, RGB et nouveaux canaux).

Fichiers de sortie (6) : `last.pt`, `run.log` (seul journal), `fold0_probs128.npz`, `test_probs128.npz`, `eval_fold0.json`, `DONE`.

## Lire le résultat
```
grep "^fold0" $TASK/t1/out/jacques7/sg_informed/run.log
python -c "import json; j=json.load(open('$TASK/t1/out/jacques7/sg_informed/eval_fold0.json')); print(j['stem_w_abs']); [print(g, r['new'], r['ref'], r['new_minus_ref'], r['new_minus_noaux']) for g, r in j['scores'].items()]"
grep '"it"' $TASK/t1/out/jacques7/sg_informed/run.log | tail -n 3     # w_aux > 0 : les nouveaux canaux sont utilisés
```
- Référence en cartes brutes : stega/best = 0,763 au fold 0 (hole 0,800, transplant 0,773, outpainting 0,672). Le 0,798 vient
  de la chaîne complète (`t1/mgr/v3/chain.py`, entrée `stegastamp`) : pour comparer à 0,798, repasser `fold0_probs128.npz`
  dans la chaîne.
- `new − ref` mélange deux effets : les canaux informés ET 40 min d'entraînement de plus. `new − noaux` isole la part des
  canaux (même poids). Le contrôle propre « 40 min de plus sans canaux » est le run jacques5b (recette un peu différente :
  synthétique neuf, mix 3/3/2).
- stega/best a été choisi sur le fold 0 (best.pt de `t1/train`) : `ref` est un peu optimiste, ce qui joue contre `new`.
- Sur 34 images, un écart < 0,03 est du bruit. N'adopter que si l'IC 95 % de `new − ref` exclut 0, sinon seulement comme
  membre d'ensemble. Aucun réglage n'est fait sur le fold 0.

## Fichiers (2 de code)
- `sg_informed.sh` : la tâche relais (réglages par variables d'environnement hors relais : `MIN` 40, `WORKERS` 12,
  `BUDGET_MIN` = MINUTES − 3, `EVAL_MIN` 7).
- `sg_informed.py` : `train` (reprise `--resume`), `eval`, `check` (fidélité de l'Embed en lot). Le code des autres
  (`t1/train/unet.py`, `t1/synth/dataset.py`, `t1/synth/shapes.py`, `t1/common/evaluate.py`, `wmloc`) est importé, jamais modifié.

## Tests en fumée (CPU, nœud jrlogin05, 21/11 23:25-23:27)
```
cd $TASK && source .venv/bin/activate && CUDA_VISIBLE_DEVICES="" python t1/jacques7/sg_informed.py check
cd $TASK/t1/jacques7 && SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" bash sg_informed.sh     # 2 fois (sorties t1/out/jacques7/smoke)
```
- `check` : le résidu en lot corrèle à **0,998-0,999** avec `wm.embed(x, m) − x` officiel (arrondi uint8 compris) sur
  2 images réelles ; résidu officiel RMS 5,6-6,5 niveaux de gris ; canaux mis à l'échelle d'ordre 1 (|r'| ≈ 0,5, |d| ≈ 0,7).
- Fumée (2 images réelles + synthétique, bs 2) : 1er passage 2 pas, `last.pt`, `rc=3` ; 2e passage `RESUME it=2`, 2 pas,
  inférence sur 2 images fold 0 + 2 test, `eval_fold0.json`, `rc=0 DONE`, ≈ 20 s chacun. Pertes finies, `w_aux` passe de 0
  à 1e-5 (le gradient atteint les nouveaux poids). Contrôle : `base` recalculé = 0,8164 contre `ref` 0,8165 sur ces 2 images
  (le chemin d'inférence reproduit stega/best). Les sorties de fumée ont été supprimées.

Non vérifié : onnxruntime CUDA dans le job (assertion qui échoue tôt sinon ; repli CPU impossible en 40 min), le débit sur
A100 (2 passages de l'encodeur ONNX à 400 × 400 par image en plus du U-Net, transfert CPU-GPU par numpy : estimé 6-10 it/s,
soit ≈ 15-25 k itérations), le chemin « exit 3 → relance » en vrai relais, et bien sûr le gain.
