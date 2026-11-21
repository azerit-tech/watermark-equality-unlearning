# Unlearning : reprise par Jacques (compte trentelivres1)

Le compte wolf10 ne peut plus lancer de job Slurm depuis 21h14 (« Invalid account »).
Tout le calcul GPU et les soumissions passent donc par ton compte. Ce fichier dit quoi lancer, dans quel ordre.

Dossier de travail sur JURECA (tout y est, ouvert à l'équipe) :

```
cd /p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
```

Python : `../.venv/bin/python`. Relais : `R=/p/scratch/training2666/trentelivres1/Azerit/relais`.

## Règles à respecter (on a déjà eu deux comptes bloqués dans l'équipe)

- Un seul job Slurm par personne, relais compris. Tout le GPU passe par `$R/relais-add <script>`.
- Aucun calcul GPU sur le nœud de connexion, même court.
- Une soumission à la fois : le serveur impose ~10 min entre deux soumissions. Ne pas lancer deux chaînes.
- Le disque scratch est partagé et sature par moments : supprimer les checkpoints inutiles (345 Mo chacun).

## 1. Où on en est

Meilleur score public : **0.1236** avec `runs/oracle/E13.pt` (21h32 ; = original + 1.3 × le changement de poids de SCRUB lr 1e-5, époque 2). E12 (× 1.2) avait fait 0.1046. Avant : 0.0653 avec `runs/oracle/l5e3_a3.pt`.
Oublier plus fort a payé : le score suit l'erreur sur les copies altérées et n'a pas encore saturé.

Tous les scores : `submissions.jsonl` (une ligne par soumission, avec les statistiques locales dans `note`).

Ce qu'on a appris (détails dans les notes de `runs/submit_queue/` et dans `runs/twin/`) :

| modèle | erreur clean (moy/éc.) | erreur copies altérées | score |
|---|---|---|---|
| SCRUB ep2 | 0.50 / 1.63 | 0.43 | 0.064 |
| l5e3_a3 (meilleur) | 0.68 / 2.62 | 0.54 | 0.0653 |
| modèles « distillés vers une cible » (G0, G1, G2…) | 0.5–0.7 / 1.0–1.4 | 0.35–0.49 | 0.031–0.053 |

- Les modèles de la famille SCRUB oublient de façon qui tient sur des déformations jamais vues (recadrage fort : erreur 0.6–1.2) ; les modèles distillés non (0.3–0.4). C'est la meilleure explication de l'écart.
- Hypothèse d'un relecteur indépendant : le frein principal est la forme des réponses (nos modèles SCRUB restent aussi sûrs d'eux que l'original sur ~88 % des images). D'où les versions « adoucies » ci-dessous.

## 2. Soumissions : modèles prêts (aucun calcul nécessaire)

Une chaîne tourne encore sous wolf10 jusqu'à ~22h00 (`runs/oracle/chain6.txt` : **E14** = ep2 × 1.4 vers 21h43, puis **E13_tau1.3** = E13 adouci vers 21h54).
**Attends qu'elle soit finie** (`tail runs/oracle/chain6.log` doit afficher deux « Score »), puis prends le relais.
Règle simple pour la suite : tant que monter d'un cran (× 0.1 sur ep2) fait gagner plus de 0.01, continuer (E15 = × 1.5, etc., construit comme E14) ; dès que ça baisse, revenir au meilleur et tester l'adoucissement dessus.

Modèles prêts, tous vérifiés par `check_submission` (chemins relatifs au dossier de travail) :

| fichier | ce que c'est | clean moy/éc./préc. | copies altérées | recadrage fort (jamais vu) |
|---|---|---|---|---|
| runs/oracle/E11.pt | ep2 × 1.1 | 0.70/1.97/0.84 | 0.65 | 0.81 |
| runs/oracle/E12.pt | ep2 × 1.2 (en chaîne) | 0.92/2.29/0.80 | 0.85 | 1.00 |
| runs/oracle/E13.pt | ep2 × 1.3 | 1.17/2.58/0.75 | 1.07 | 1.22 |
| runs/oracle/A35.pt | l5e3 × 3.5 | 0.92/3.08/0.86 | 0.79 | 0.87 |
| runs/oracle/A4.pt | l5e3 × 4 | 1.17/3.44/0.82 | 1.02 | 1.10 |
| runs/oracle/a3_tau1.3.pt | meilleur + sortie adoucie τ 1.3 (en chaîne) | 0.56/2.02/0.89 | 0.48 | 0.55 |
| runs/oracle/a3_tau1.5.pt | meilleur + τ 1.5 | 0.51/1.76/0.89 | 0.44 | 0.51 |
| runs/oracle/E12_tau1.3.pt | E12 + τ 1.3 | 0.79/1.77/0.80 | 0.73 | 0.86 |
| runs/soft/a3_affb1_0.2.pt | meilleur, adouci vers les classes voisines (léger) | 0.57/2.10/0.89 | 0.49 | – |
| runs/soft/a3_affb1_0.3.pt | idem, plus fort | 0.52/1.84/0.89 | 0.45 | – |
| runs/soft/E12_affb1_0.2.pt | E12 adouci vers les classes voisines | 0.80/1.84/0.80 | 0.74 | – |

### Quel modèle soumettre ensuite (selon les scores de E13 et E12_tau1.3)

1. Si **E13 > E12 + 0.01** (oublier encore plus aide) : soumettre **A4**, puis construire E14 (voir plus bas).
2. Si **E13 ≈ E12 ou moins** : le bon niveau est autour de E12. Soumettre **A35** (même niveau, autre direction, moins de dérèglement).
3. Si **E12_tau1.3 > E12 + 0.01** (adoucir aide) : soumettre **E12_affb1_0.2**, et adoucir aussi le meilleur suivant.
4. Sinon, pas d'adoucissement.

Construire un nouveau niveau (≈ 20 s, se lance via le relais, pas sur le nœud de connexion) :
`../.venv/bin/python oracle_interp.py --ckpt runs/cal_lr1e-5/ckpt_ep02.pt --alphas 1.4 --save runs/oracle/E14.pt`
(lire l'en-tête de `oracle_interp.py` ; mettre la commande dans un script relais avec les en-têtes `# GPUS=1 # MINUTES=5 # PRIO=1 # DIR=<dossier de travail>`).

Écart de moins de ~0.01 = bruit (le score public ne porte que sur ~30 % des données) : ne pas choisir le modèle final sur un écart aussi petit.

### Comment soumettre

Clé : `export HACKATHON_API_KEY=<clé de l'équipe>` dans ton shell (le fichier `.env` du dossier appartient à wolf10).
Une soumission isolée :

```
../.venv/bin/python submit.py runs/oracle/A4.pt --note "A4: l5e3 x4"
```

Ou une chaîne qui respecte le délai du serveur (une ligne par fichier, format `chemin|note`, on peut ajouter des lignes pendant qu'elle tourne) :

```
echo "$PWD/runs/oracle/A4.pt|[A4] l5e3 x4" > runs/oracle/chainJ.txt
setsid nohup ./oracle_submit2.sh runs/oracle/chainJ.txt > runs/oracle/chainJ.log 2>&1 < /dev/null &
tail runs/oracle/chainJ.log
```

Pour l'arrêter : `touch runs/oracle/chainJ.txt.stop` (elle s'arrête quand la liste est épuisée), ou `kill <pid>` (pid via `ps -u $USER -o pid,args | grep oracle_submit2`).

## 3. Calculs GPU à lancer via le relais (tous testés sur CPU)

Lance d'abord ta chaîne relais si elle est arrêtée : `$R/relais-start` (refusé si tu as déjà un job). État : `$R/relais-etat`.

Par ordre de priorité :

1. **Réentraînement des dernières couches sur images extérieures** (la piste la plus prometteuse : imite un vrai réentraînement sans les images à oublier).
   Scripts et commandes exactes : voir la section 4 (ajoutée dès que la préparation des images est finie).
2. **Test sur le jumeau, réentraînement sans images à oublier** (~27 min) :
   ```
   $R/relais-add reinit_relay_twin5.sh
   ```
   Résultats : `runs/reinit/logs/twin_*.log`. Comparer à : modèle de référence = erreur 0.53, 83 % de bonnes réponses ; plafond 0.71 ; ancienne méthode 0.32.
3. **« Faux comité » (distillation de la moyenne de plusieurs modèles oubliés)** et **SCRUB + images mosaïques** : voir la section 4.
4. **(Priorité basse, la nuit) jumeau moins biaisé** : 4 modèles, se relancent tout seuls, ~50 min par passage :
   ```
   $R/relais-add twin_relay_pb6_a.sh
   $R/relais-add twin_relay_pb6_b.sh
   ```
   Résultats dans `logs/twin/pb6_*.log`. Bilan complet du jumeau actuel : `runs/twin/pb10q/` et `logs/twin/`.

Chaque calcul qui produit un modèle intéressant écrit une note dans `runs/submit_queue/` (format `chemin | recette | stats | holdout 0`).
Ne soumettre que les modèles dont l'erreur sur recadrage fort (« crop ») est d'au moins ~0.8.

## 4. Commandes des nouvelles méthodes

### 4a. Réentraînement des dernières couches sur images extérieures (PRIORITÉ 1)

Les 18 370 photos extérieures (COCO val2017 + Imagenette, sans étiquette) sont déjà prêtes : `runs/ext/bank_ext.npz`.
Les scripts ont été testés sur CPU de bout en bout. Rien n'a encore tourné sur GPU.

```
R=/p/scratch/training2666/trentelivres1/Azerit/relais
W=/p/scratch/training2666/trentelivres1/Azerit/unlearning/wolf10
$R/relais-add $W/jobs/ext_retrain.sh   # le calcul principal, ~25 min sur un A100
$R/relais-add $W/jobs/ext_logits.sh    # diagnostic facultatif, ~3 min
$R/relais-etat
```

Ce que fait `ext_retrain.sh` : il efface les blocs k..11 + norm + head, puis les reconstruit en imitant le modèle d'origine sur les photos extérieures fortement augmentées. Les images à oublier ne sont jamais utilisées. Il y a 4 variantes : k10 et k8, chacune sans et avec 15 % d'images mosaïques.

À lire après :
- `runs/ext/retrain.log` : une ligne `RESULT` par variante, puis un tableau classé.
- Les 2 meilleurs modèles sont dans `runs/ext/<nom>.pt`, avec leurs notes dans `runs/submit_queue/ext_<nom>.txt`.
- **Lire le champ `flags:` de chaque note avant de soumettre.** Un modèle « PASS » a une erreur sur les images à oublier entre 0.5 et 1.2 (écart ≥ 1.2, 75–90 % de bonnes réponses), une erreur sur les copies et sur les déformations jamais vues au moins égale à 0.8 × l'erreur sur les originales, et ≥ 90 % d'accord avec l'original sur les photos non utilisées.
- Si les variantes `_shuf` oublient nettement moins que les `_ext`, les mosaïques réapprennent les images : ne garder que les `_ext`.

Soumettre en premier le meilleur « PASS ».
