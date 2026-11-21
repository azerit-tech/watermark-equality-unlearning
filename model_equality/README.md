# Projet 2 : Model Equality Testing

État au samedi 21 nov. 2026, vers 18:00. Ce fichier dit où on en est et quoi faire ensuite.

## La tâche en cinq lignes

- **Objectif** : 33 endpoints d'API (`ep-01` à `ep-33`) servent des LLM en boîte noire. Il faut les regrouper par famille (même modèle de base) et par modèle. Une quantification ou un LoRA différent fait un modèle différent ; un watermark, un system prompt ou une stratégie de décodage sont des leurres à ignorer.
- **Livrable** : un CSV `endpoint,family,model`, une ligne par endpoint.
- **Score** : `0.1 × MCC_family + 0.9 × MCC_model`, calculé sur les paires d'endpoints. Le leaderboard n'en montre qu'environ 30 % ; le classement final porte sur le reste.
- **Budget** : 100 réponses par endpoint `ep-` pour toute la compétition, pour toute l'équipe, irrécupérables. Les 9 endpoints d'entraînement `ref-01` à `ref-09` sont gratuits et leur vérité terrain est connue.
- **Échéance** : dimanche 22 nov. à 12:00. Une soumission toutes les 30 minutes.

## Où on en est

**Fait et vérifié**

| Élément | Où | État |
|---|---|---|
| Clé d'équipe | variable `API_KEY_BOARD` du `.env` à la racine (ignoré par git) | valide, en-tête `X-API-Key` |
| Budget `ep-` | lu sur le serveur | intact : 0 utilisé sur 100 pour les 33 endpoints |
| Client API | `api/` | 31 tests verts contre un faux serveur |
| Score local, validateur, soumission | `scoring/` | 42 tests verts ; reproduit au caractère près le scoreur des organisateurs |
| Code de départ des organisateurs | `starter/` | copie identique au cluster |

Le projet ne dépend plus de JURECA : tout tourne depuis un poste local, via l'API.

**En cours**

- Validation du client contre le vrai serveur, sur les `ref-` seulement, avec mesure du débit. Tant qu'elle n'est pas rendue, le format réel des réponses n'est pas confirmé.
- Conception de la batterie de sondes et de la méthode statistique. Elle arrivera dans `probes/` (`probes.jsonl` et `METHOD.md`) ; le dossier est encore vide.
- Plan de la piste GPU : répliques locales des modèles sur A100.

**Leaderboard de la tâche 2, relevé à 17:23**

| Rang | Équipe | Score public |
|---|---|---|
| 1 | SLZ | 0,20 |
| 2 | BlitzTeam | 0,05 |
| 3 ex æquo | quatre équipes | 0,00 |

Azerit n'a encore soumis sur aucune des trois tâches.

## Prochaines étapes

Plan de budget par endpoint `ep-` : environ 15 réponses pour le diagnostic, 55 à 60 pour la batterie retenue, le reste en réserve.

| # | Étape | Produit | Dépend de | Coût `ep-` |
|---|---|---|---|---|
| 1 | Campagne massive sur les `ref-` et sélection des sondes | Les 15 à 25 prompts qui séparent les vrais modèles sans réagir aux leurres, avec leurs seuils | Client validé, `probes/` livré | 0 |
| 2 | Diagnostic des familles sur les `ep-` | Partition en familles, carte des leurres par endpoint, endpoints à décodage déterministe | Client validé, sondes de diagnostic | ≈ 15 par endpoint |
| 3 | Première soumission | Un score public non nul et la preuve que la chaîne d'envoi marche | Étape 2, accord de Jacques | 0 |
| 4 | Batterie retenue sur les `ep-`, famille par famille | Les réponses qui servent à distinguer les modèles dans chaque famille | Étapes 1 et 2 | ≈ 55 à 60 par endpoint |
| 5 | Clustering final et usage de la réserve | La partition finale ; la réserve sert à trancher les paires incertaines | Étape 4 | ≈ 25 à 30 par endpoint |
| 6 | Piste GPU, en parallèle | Répliques locales d'une famille identifiée, pour calibrer et tester par vraisemblance | Étape 2 (famille identifiée comme modèle ouvert), créneau GPU | 0 |
| 7 | Présentation de 6 minutes | Les diapositives, si l'équipe finit dans les 6 premières (dimanche 13:00) | Étape 5 | 0 |

Précisions :

- **Étape 1.** Le critère de sélection joue dans les deux sens : ne pas séparer `ref-01/03/05/06/08` (même modèle sous leurres), et séparer `ref-02` (quantifié) et `ref-09` (LoRA faible), les cas les plus proches de la base.
- **Étape 2.** En cas de doute sur une frontière de famille, fusionner plutôt que scinder : scinder une vraie famille coûte aussi sur les 90 % du score.
- **Étape 3.** Soumettre peu, et seulement des partitions auxquelles on croit.
- **Étape 6.** Bornée dans le temps. Elle utilise le GPU dont les tâches 1 et 3 ont aussi besoin.

## Commandes utiles

Toutes se lancent depuis la racine du dépôt, avec `python3.12`.

**Budget et requêtes**

```bash
# Budget par endpoint (serveur et registre local)
python3.12 -m model_equality.api budget

# Essai à blanc d'une campagne : affiche la dépense, n'envoie rien
python3.12 -m model_equality.api run --endpoints 'ep-01..ep-33' --prompts prompts.jsonl --repeats 2 --max-tokens 64

# Campagne réelle sur les ref- (gratuit)
python3.12 -m model_equality.api run --endpoints 'ref-01..ref-09' --prompts prompts.jsonl --repeats 5 --max-tokens 64 --execute

# Campagne réelle sur les ep- : N est la dépense affichée par l'essai à blanc
python3.12 -m model_equality.api run --endpoints 'ep-01..ep-33' --prompts prompts.jsonl --repeats 2 --max-tokens 64 --execute --confirm-spend N

# État du registre des réponses
python3.12 -m model_equality.api status --by-prompt

# Export au format answers.jsonl du code de départ
python3.12 -m model_equality.api export answers.jsonl --endpoints 'ref-*'
```

Le plafond par endpoint `ep-` est de 50 par défaut. Pour le relever, placer `--cap` avant `run` (maximum 100). `--concurrency` se place au même endroit.

**Score et partition**

```bash
# Valider un CSV de soumission
python3.12 -m model_equality.scoring.csvio SUB.csv

# Scorer une prédiction contre une vérité terrain (ici, celle des ref-)
python3.12 -m model_equality.scoring.metrics PRED.csv model_equality/scoring/data/practice_answers.csv

# Passer de scores par paires à une partition au format du livrable
python3.12 -m model_equality.scoring.partition --pairs PAIRES.csv --threshold 0.7 --out SUB.csv
```

**Soumission**

```bash
# Essai à blanc : valide le CSV et vérifie le délai, n'envoie rien
python3.12 -m model_equality.scoring.submit SUB.csv

# Envoi réel
python3.12 -m model_equality.scoring.submit SUB.csv --confirm-submit

# État du délai de 30 minutes et journal des envois
python3.12 -m model_equality.scoring.submit --status
```

**Leaderboard** : ouvrir `http://34.147.229.55/leaderboard_page`, onglet « Task 2 Model Equality ».

**Tests**

```bash
python3.12 -m unittest model_equality.api.tests.test_api
python3.12 -m unittest discover -s model_equality/scoring/tests -t .
```

## Règles à ne pas enfreindre

- **Un seul registre des réponses** : `model_equality/api/ledger/ledger.jsonl`. Toutes les requêtes passent par le client, jamais par un script à part. Ne pas supprimer ce fichier : c'est la seule trace des réponses achetées.
- **4 requêtes simultanées au total** pour toute l'équipe, `ref-` et `ep-` confondus. Une seule campagne à la fois.
- **Le budget `ep-` ne se récupère pas.** Toujours faire l'essai à blanc avant, et vérifier la dépense annoncée.
- **Une soumission toutes les 30 minutes, même si elle échoue.** La garde du script ne voit que les envois faits depuis ce poste : se coordonner avant d'en lancer une d'ailleurs.
- **Ne pas régler la partition sur le score public.** Les organisateurs préviennent qu'une soumission ajustée au leaderboard peut finir proche de zéro au classement final. On juge la méthode sur les `ref-`.
- **La clé ne s'affiche pas et ne se commite pas.** Elle reste dans le `.env`.
- **Aucune soumission sans l'accord de Jacques.**

## Questions ouvertes

**À poser aux organisateurs (Discord)**

1. Quelle soumission compte pour le classement final : la meilleure au score public, ou la dernière ?
2. Les modèles locaux et autres ressources externes sont-ils autorisés ?
3. La limite de 4 requêtes simultanées est-elle commune aux `ref-` et aux `ep-` ?

**À trancher dans l'équipe**

1. Azerit n'a de score sur aucune tâche, et le classement général ne retient que les équipes qui en ont un sur les trois. Il faut au moins une soumission valide par tâche.
2. Le classement général fait la moyenne des trois tâches, chacune ramenée entre 0 et 1 par rapport aux autres équipes. Quelle part de l'effort et du GPU donner à chacune ?
3. Qui déclenche les soumissions de la tâche 2, et depuis quel poste ?
4. Le registre des réponses se partage-t-il par git ? Il n'est pas dans le `.gitignore` et rien de `model_equality/` n'est encore commité.
5. La piste GPU vaut-elle un nœud, sachant que les tâches 1 et 3 en ont besoin ?

## Références

Voir `REFERENCES.md` à la racine. La description détaillée de la tâche cite aussi deux articles sur l'audit d'API en boîte noire : arXiv 2506.06975 (test de rang, avec un modèle local de référence) et arXiv 2609.00052 (sondes à décisions discrètes, écrit par des auteurs de la tâche).
