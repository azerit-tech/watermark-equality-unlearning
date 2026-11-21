# Tâches GPU vague 5 pour Jacques (tâche 1) : segmenteur « informé », variantes longues, combinées, graines

Même format que `t1/jacques3` : 1 GPU par tâche, en-têtes `# GPUS=1`, `# MINUTES=55`, `# PRIO=2`, `# DIR=$TASK` ; reprise
automatique depuis `last.pt` si on redépose le même script (code de sortie 3 = budget atteint avant la fin) ; `DONE` à la fin ;
`chmod -R g+rwX` des sorties. Sorties sous `t1/out/jacques5/<tache>/`, **6 fichiers au plus par tâche** :
`last.pt`, `run.log` (entraînement + prédiction + bilan), `fold0_probs128.npz`, `test_probs128.npz`, `eval_fold0.json`, `DONE`.
Aucune tâche ne dépend d'une autre ; seule `e_long_ps.sh` lit `t1/out/jacques3/informed_pixelseal/last.pt` (à ne pas effacer).

```
TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task
R=/p/scratch/training2666/trentelivres1/Azerit/relais
J5=$TASK/t1/jacques5
```

## Ordre conseillé

| Ordre | Script | Tâche (sortie) | Ce que c'est | Durée A100 |
|---|---|---|---|---|
| 1 | `f_longcomb_tm.sh` | `longcomb_trustmark` | informé + plus de trous, **160 min** d'entraînement en 4 tranches | 4 passages de ≈ 52 min |
| 1 | `f_longcomb_ps.sh` | `longcomb_pixelseal` | idem PixelSeal | 4 passages de ≈ 52 min |
| 2 | `e_comb_ps.sh` | `comb_pixelseal` | (b) informé + plus de trous, 40 min | ≈ 45 min |
| 3 | `e_seed_ps.sh` | `seed1_pixelseal` | (a) informed_pixelseal, graine 1, 40 min | ≈ 45 min |
| 4 | `e_seed_tm.sh` | `seed1_trustmark` | (a) informed_trustmark, graine 1, 40 min | ≈ 45 min |
| 5 | `e_long_ps.sh` | `long_pixelseal` | (c) informed_pixelseal de jacques3 prolongé de 40 min | ≈ 45 min |

```
$R/relais-add $J5/f_longcomb_tm.sh
$R/relais-add $J5/f_longcomb_ps.sh
$R/relais-add $J5/e_comb_ps.sh
$R/relais-add $J5/e_seed_ps.sh
$R/relais-add $J5/e_seed_tm.sh
$R/relais-add $J5/e_long_ps.sh
$R/relais-start            # si la chaîne ne tourne pas déjà
$R/relais-etat
```
**Tâches longues : à chaque sortie en `failed/` avec « run the task again (resumes) » (code 3), redéposer LE MÊME script**
(`$R/relais-add $J5/f_longcomb_tm.sh`) ; il reprend de `last.pt` (modèle, EMA, optimiseur, position dans le cosinus).
Fin quand `t1/out/jacques5/longcomb_<m>/DONE` existe (4 passages prévus).

## Ce que fait chaque tâche (`ft5.sh` → `train5.py` → `t1/jacques3/pred_informed.py`)

`train5.py` = copie de `t1/jacques3/train_informed.py` (modèle `InformedTileSeg` : RGB + HF(Y), r'(m), d, JND(Y) ; réel folds
1-4 avec p_real 0,25 + synthétique) avec trois options : `--seed`, `--hole_src` (synthétique `t1/holes` MsgHoleSynth, recette
jacques2), `--cont` (continuer un `last.pt` informé), `--fresh_full N` (synthétique neuf en mémoire). **Le fold 0 ne sert à
rien pendant l'entraînement** (pas de validation, pas de best.pt) : on garde l'EMA du dernier état. `pred_informed.py` écrit
les cartes fold 0 (34 images jamais vues) et test (340), et `eval_fold0.json` (comparaison appariée avec `REF`, bootstrap
10 000, par type ; la ligne `fold0 all` est recopiée dans `DONE`).

- **longcomb_{trustmark,pixelseal}** : recette combinée (b) (`--mix outpainting=1,transplant=1,hole=4 --hole_src new=1,lib=1
  --p_focus 0.7`, hyperparamètres informed de jacques3, départ tiles2 / tiles3, graine 4), cosinus sur **160 min** (≈ 40 min
  d'entraînement par passage de 52 min ; 4 tranches). **Synthétique renouvelé à chaque tranche** : au démarrage de chaque
  passage, 1 500 nouvelles insertions pleine image (au plus 240 s) sont fabriquées EN MÉMOIRE sur le GPU, avec la même recette
  que `t1/synth/gen.py` (couvertures = images du régime partiel des folds 1-4 seulement, augmentation flip/crop, message
  aléatoire, `wm.embed`), et ajoutées au pool v0 (4 002) : **aucun fichier écrit** (mieux que le .npz par tranche demandé).
  Elles alimentent les trous et transplants (5/6 du synthétique). **À la fin de CHAQUE tranche**, `fold0_probs128.npz`,
  `test_probs128.npz` et `eval_fold0.json` sont réécrits (écrasés) avec l'état courant ; `run.log` dit lequel
  (« chunk maps written: state Xs / 9600s »). Réf. : tiles2 / tiles3 (comme jacques2/3). Variables : `MIN_LONG` (160),
  `FRESH` (1500).
- **comb_pixelseal** (b) : même recette combinée, 40 min, sans synthétique neuf, graine 2. Réf. tiles3.
- **seed1_pixelseal / seed1_trustmark** (a) : exactement la recette de `d_informed_ps.sh` / `d_informed_tm.sh`, graine 1
  (jacques3 avait une graine tirée de l'horloge : init du tronc auxiliaire et découpes différentes). Réf. tiles3 / tiles2.
  Utilité : mesurer la variance entre graines, et moyenner les deux graines (ensemble) sans aucun réglage.
- **long_pixelseal** (c) : charge `t1/out/jacques3/informed_pixelseal/last.pt` (modèle, EMA, optimiseur, it 6001) et repart
  pour 40 min avec un **nouveau cosinus à mi-lr** (lr 7,5e-5, lr_enc 5e-6, lr_aux 5e-4 ; montée sur 200 pas), recette
  jacques3 inchangée. Réf. = le `last.pt` de jacques3 : `eval_fold0.json` mesure directement le gain du prolongement
  (base attendue 0,6692).

## Honnêteté de la carte fold 0, et comment intégrer
- Toutes les tâches s'entraînent sur les folds 1-4 seulement ; la carte fold 0 n'a servi à aucun choix.
- Tâches longues : les cartes intermédiaires sont des états en milieu de cosinus (lr encore haute), en général moins bons
  que l'état final. **Intégrer le DERNIER état disponible, pas celui qui a le meilleur fold 0** : choisir une tranche sur son
  score fold 0 serait régler sur les images qui servent à l'annoncer (règle de surapprentissage).
- Sur 34 images par méthode, < +0,03 est du bruit : n'adopter seul que si l'IC95 exclut 0 ; sinon comme membre d'ensemble.
- Rappel : holes_trustmark (jacques2) avait dégradé transplant dans la chaîne ; surveiller la ligne `fold0 transplant` de
  `longcomb_trustmark` avant de l'adopter.

## Vérifier
```
ls $TASK/t1/out/jacques5/*/DONE
grep "^fold0\|chunk maps\|fresh_full\|CONT\|RESUME" $TASK/t1/out/jacques5/*/run.log
```
Échec avec « run the task again » : redéposer le même script. Autre échec : `run.log`.

## Tests en fumée (CPU, nœud de connexion, 21/11 23:10-23:15, jrlogin11)
```
cd $J5 && SMOKE=1 DEV=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 bash f_longcomb_tm.sh
```
(sorties sous `t1/out/jacques5/smoke/…`, effacées après le test ; 2 images réelles, bs 2, 2 pas, puis 2 pas repris de `last.pt` ;
prédiction sur 2 images fold 0 et 2 test.)
- `e_comb_ps.sh` : rc=0, DONE ; MsgHoleSynth (mix hole 2/3, hole_src new/lib, lib_folds 1-4), init tiles3, 2 pas, RESUME it=2,
  2 pas, cartes et `eval_fold0.json` écrits, 6 fichiers.
- `f_longcomb_tm.sh` : rc=0, DONE ; `fresh_full` 108 puis 109 insertions neuves en 20 s (CPU, 204 couvertures folds 1-4),
  cartes écrites après la 1re tranche puis réécrites à la fin ; la tête commence à utiliser les canaux informés.
- `e_long_ps.sh` : rc=0, DONE ; `CONT` depuis jacques3 (it 6001, 2 400 s), it repart à 6002, montée de lr OK, RESUME
  it_cont=6001.
- Non testés en fumée : `e_seed_ps.sh`, `e_seed_tm.sh`, `f_longcomb_ps.sh` (mêmes chemins de code que ci-dessus, seuls les
  arguments changent) ; le chemin réel « tranche partielle → cartes → exit 3 → reprise » (même logique que jacques3, plus
  l'appel de prédiction) ; la vitesse de `fresh_full` sur A100 (plafond 240 s, compté dans le budget de la tranche).
