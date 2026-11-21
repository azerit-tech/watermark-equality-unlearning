# Piste GPU — Model Equality Testing : plan d'exécution

> **État au 21 nov. (faits établis après la rédaction du plan)**
>
> - L'utilisateur a autorisé les requêtes réelles sur `ref-` et sur `ep-`.
> - Le registre des réponses est `model_equality/api/ledger/ledger.jsonl`. Il contient déjà de vraies réponses `ref-`.
> - La batterie en boîte noire est livrée dans `model_equality/probes/` : `METHOD.md`, `probes.jsonl`, 138 sondes dont 15 de diagnostic et une sonde de dépassement `X01_oversize`.
> - Un code d'analyse est en cours d'écriture dans `model_equality/analysis/`.
> - Aucun jeton Hugging Face n'est disponible pour l'instant : utiliser des miroirs non restreints (du type `unsloth/*`, mêmes poids).
>
> Le reste du document est le plan tel qu'il a été conçu, en lecture seule, avant ces faits. Là où il dit « non vérifié » ou « à tester d'abord sur `ref-` », le registre permet désormais de trancher : commencer par là.

Statut du plan : conception seule, en lecture seule. Aucune requête API, aucun job, aucun téléchargement de modèle, aucune écriture n'ont été faits pour le rédiger. Les comportements de l'API cités plus bas (champ `usage`, décodage par défaut, message de rejet) ne sont pas vérifiés et sont à tester d'abord sur `ref-`.

## 1. Verdict

**Oui, mais comme piste de certification bornée dans le temps, jamais sur le chemin critique de la collecte.** Gain attendu sur le score final : +0,05 à +0,25, médiane autour de +0,10. Je donne environ 50 % de chances que la chaîne complète tourne à temps sur au moins la moitié des familles, et environ 80 % aux sous-produits sans GPU (points 3 et 4).

- **Ce que vaut une erreur.** Sur la structure du banc `ref-` (5 endpoints identiques + 4 singletons), une seule erreur d'endpoint fait tomber le MCC à 0,72–0,73. Extrapolé à 33 endpoints en 4 familles, chaque erreur coûte environ 0,08 de score. Si la boîte noire fusionne toutes les variantes quantifiées avec leur base, le MCC modèle tombe vers 0,70.
- **Où la boîte noire échoue, d'après les articles des organisateurs.** RUT rapporte une puissance de 0,05 à 0,13 sur le 8 bits (100 prompts × 30 tokens, T=0,5). AgentProv, écrit par les auteurs de la tâche, indique que MET, RUT et AgentProv acceptent tous le 8 bits, et que MET rejette à tort 33 modèles sur 55 avec un simple message système vide. Ces chiffres viennent d'un résumé automatique des pages HTML, à recouper sur les PDF.
- **Concurrence.** Dans la file Slurm le 21 nov. à 17 h 25 : `mirande-ney1` (`me-atlas-lora-gen`, 2 jobs de 2 h en attente) et `destors1` (`t2calib`, `t2calib16`, `t2exp2`, `t2mining`, `t2wm` en cours). Les noms suggèrent de la calibration par simulation, donc l'idée (d). Notre différenciateur serait la certification par vraisemblance, pas la simulation seule.

**Jugement des cinq idées :**

| Idée | Jugement |
|---|---|
| (a) Rejeu à l'identique | Illusoire en génération libre : première divergence numérique, pile de service inconnue, dtype, batching. Solide sous la forme « cohérence argmax en teacher forcing » : on rejoue le texte de l'endpoint dans le modèle local et on regarde si chaque token observé est l'argmax, avec la marge. Auto-certifiant, mais exige le contexte exact (template, system prompt, pénalité de répétition). |
| (b) Rapport de vraisemblance | Solide. Décisif pour LoRA, 4 bits, system prompt rejoué. Limité pour fp8 sous échantillonnage : z ≈ √(N·KL/2), donc z ≥ 4 avec 25 600 tokens demande une perturbation RMS des logits d'au moins 0,05. À mesurer sur `ref-02` avant tout. Sous greedy, quelques centaines de tokens suffisent. |
| (c) Extraction du system prompt puis rejeu | Tient si l'extraction est exacte, et se vérifie par l'ajustement obtenu. Une extraction partielle ne sert à rien à l'échelle fp8. |
| (d) Simulation des leurres | Solide et facile, mais probablement déjà faite par les concurrents. Dépend de la fidélité de nos leurres simulés à ceux des organisateurs. |
| (e) LoRA comme écart inexpliqué | Solide seulement si les leurres sont expliqués d'abord, sinon persona et watermark passent pour un LoRA. |

**Conditions :**

- un seul nœud à la fois, 9 à 12 heures-nœud au total ;
- un jeton Hugging Face disponible, ou des miroirs non restreints ;
- les portes d'arrêt de la section 8 respectées.

## 2. Ce que les sources révèlent

- **Structure du banc.** `practice_answers.csv` : 5 endpoints sur 9 sont le modèle de base sous leurre (plain `ref-01`, system prompt `ref-03`, persona `ref-05`, watermark `ref-06`, sampling T=1,0 `ref-08`), et 4 sont des singletons (fp8 `ref-02`, LoRA 0,3 / 0,7 / 1,5 = `ref-09` / `ref-04` / `ref-07`). Le même LoRA à trois forces donne trois modèles. C'est un a priori fort, mais pas une règle : l'exemple OLMo du PDF décrit un endpoint cumulant watermark et system prompt, donc les combinaisons existent.
- **Décodage.** « sampling at temperature 1.0 » est listé comme un leurre, donc le décodage par défaut n'est pas T=1. La diapositive p. 36 cite « greedy vs beam search ». Le commentaire de `task_template.py` dit « answers are sampled ». C'est la première chose à trancher, gratuitement, par des répétitions sur `ref-01`. Hypothèse à tester en priorité : les valeurs du `generation_config.json` du modèle de base (pour Qwen : T=0,7, top_p=0,8, top_k=20, pénalité de répétition 1,05).
- **Quantification.** « fp8 » est un vocabulaire de vLLM (`--quantization fp8`) et d'OpenRouter. AgentProv dit que leurs déploiements quantifiés sont « served through vLLM », en 8 et 4 bits. Candidats à émuler : aller-retour `torch.float8_e4m3fn` (sans échelle, par tenseur, par canal ; avec ou sans activations), `optimum-quanto` float8, et en second rang bitsandbytes int8 / nf4.
- **LoRA.** « strength » correspond à un facteur d'échelle sur la mise à jour, soit W + s·BA. Aucun adaptateur n'est publié sous l'organisation SprintML (2 modèles sans rapport).
- **Familles candidates.** Le zoo d'AgentProv : Qwen2.5 (0,5 à 7B), Qwen3 (0,6 à 4B), Llama-3.1-8B, Llama-3.2 (1B, 3B), Mistral-7B-Instruct, Nemo, Ministral-8B, DeepSeek-R1-Distill, Hermes-3, Phi-4-mini, Gemma-3-1B-it, SmolLM3-3B. S'y ajoutent OLMo (cité dans l'énoncé) et les modèles de RUT (Llama-3.2-3B, Mistral-7B-v0.3, Gemma-2 2B et 9B). Pour le « small model » du banc : SmolLM2 (135M, 360M, 1,7B), Qwen2.5-0.5B, Qwen3-0.6B, Gemma-3 (270m, 1b), Llama-3.2-1B, OLMo-2-1B, TinyLlama.
- **Watermark.** Aucun indice sur le schéma. Les fiches des jeux SprintML sur le watermark renvoient 404.
- **Canal latéral documenté par les organisateurs.** AgentProv compare le `prompt_tokens` renvoyé au compte local pour révéler un system prompt caché. Il faut regarder si l'API renvoie `usage`. Le registre de `model_equality/api/` conserve déjà le corps complet des réponses, `elapsed_s` et le corps des erreurs 4xx.
- **Intention des organisateurs.** Le `requirements.txt` de la tâche n'a pas torch : la solution attendue est en boîte noire. Mais RUT, dans leur bibliographie, suppose un modèle de référence local, donc les bases sont très probablement des modèles ouverts.

## 3. Identifier le modèle de base

Presque sans requête supplémentaire. Sondes à ajouter au diagnostic, toutes à valider d'abord sur `ref-` :

1. **Empreinte par troncature (0 requête).** Une réponse avec `finish_reason=length` fait exactement `max_tokens` tokens sous le vrai tokenizer. On retokenise les réponses longues de la batterie avec une cinquantaine de tokenizers candidats ; seul le bon donne 256 à chaque fois.
2. **Oracle de dépassement (gratuit selon l'énoncé).** Envoyer un prompt de plus de 1 024 tokens et lire le message de rejet. S'il donne un compte, on obtient le tokenizer et la longueur du system prompt caché. Sinon, une dichotomie sur le seuil d'acceptation avec des chaînes dont le compte diverge entre tokenizers. Vérifier sur un endpoint que `budget.used` ne bouge pas avant de généraliser.
3. **`usage.prompt_tokens`, s'il existe.** L'écart au compte local (template + message) donne la présence et la longueur d'un system prompt. Cela trie les endpoints « à extraire ».
4. **Question d'identité.** Une réponse courte par endpoint (« quelle entreprise t'a entraîné, quel est ton nom de modèle »). Bruité sur les petits modèles, mais cohérent par famille.
5. **Checkpoint exact parmi ceux qui partagent le tokenizer (GPU, 0 requête).** Vraisemblance en teacher forcing des réponses déjà collectées sous chaque candidat. Le bon se détache nettement : rang 1 sur environ 99 % des tokens sous greedy, ou NLL égale à l'entropie attendue sous échantillonnage.
6. **Contrôle positif par famille.** Au moins un endpoint doit coller au modèle local nu. Sinon la réplique locale est fausse et la famille retourne en boîte noire.
7. **Sondes à tokens spéciaux.** `<|im_end|>`, `<|eot_id|>`, `<end_of_turn>` dans le message utilisateur : le comportement trahit le template.
8. **Latence par token.** Déjà dans le registre (`elapsed_s`). Elle signale le beam search, bitsandbytes ou un LoRA non fusionné.

## 4. La méthode

**Sortie du scoreur.** Par couple (endpoint, hypothèse) et par token : log-probabilité du token observé, rang, marge au premier, entropie, et les 50 premiers logits. Une hypothèse = poids × dtype × template et system prompt × décodage. Stocker les 50 logits permet d'ajuster T, top-p, top-k et la pénalité de répétition hors ligne, sans repasser sur GPU.

Détails de reconstruction du contexte, à traiter dans le scoreur :

- si `finish_reason=stop`, ajouter le token de fin comme dernier token observé (il est informatif) ;
- si `finish_reason=length`, écarter le dernier token (UTF-8 possiblement tronqué) ;
- la retokenisation d'un texte échantillonné n'est pas toujours celle que le modèle a produite : écarter ou signaler les positions aberrantes plutôt que de les compter comme violations ;
- le dtype de service (fp32, bf16, fp16) fait partie de l'hypothèse ;
- certains templates insèrent un system prompt par défaut quand il n'y en a pas (Qwen) : c'est une variante de template à tester.

| Cas | Test local | Décision | Réponses par endpoint |
|---|---|---|---|
| Endpoint déterministe, base ou quantification | Taux de rang 1 et nombre de violations dures (marge > 0,1–0,25). Bruit numérique calibré par rejeu local (dtype, taille de batch). | Certifié si une hypothèse atteint ≥ 99,5 % de rang 1 et que les autres montrent des violations. | 10 à 20 réponses de 256 tokens |
| Beam search (leurre) | Texte déterministe, de log-probabilité jointe supérieure à celle du greedy local. Rejeu en beam k = 2 à 5, comparaison exacte. | Même modèle si le rejeu reproduit une majorité de prompts. | 5 à 10 |
| Échantillonnage (leurre) | Ajustement de T, top-p, top-k. Test z « surprise − entropie » et uniformité des PIT par token. Rapport de vraisemblance entre candidats, bootstrap par réponse. | Base si \|z\| < 3 ; candidat retenu si LLR/erreur-type > 4 ; sinon indécis, renvoyé à la batterie et à l'a priori. | 60 à 80 |
| System prompt, persona | Extraction en 1 à 3 requêtes, ciblée par l'écart de tokens, puis rejeu dans le template. | Certifié si l'ajustement redevient parfait. Sinon « écart structuré non résolu ». | 1 à 3 de plus |
| Watermark | Détecteurs à clés publiques par défaut : `WatermarkDetector` de transformers, KGW et Unigram façon MarkLLM (clé 15485863, γ 0,25 / 0,5, δ 2). Si z > 6 sur `ref-06`, appliquer partout et intégrer le watermark exact à la vraisemblance. | Sinon : signature « excès de surprise non structuré » (résidus non prédictibles depuis les états cachés), à l'opposé d'un LoRA. | 0 de plus |
| LoRA inconnu | Écart à la base une fois les leurres expliqués. Amplitude = excès de surprise par token, qui croît à peu près comme s². Direction = vecteur de résidus projeté en basse dimension. | Deux endpoints sont identiques seulement si amplitude et direction concordent. Une même direction avec une amplitude différente signifie une autre force, donc un autre modèle. | 20 à 40 |
| Inconnu contre inconnu | Vraisemblance croisée : petit correctif de logits ajusté sur la moitié des réponses de A, évalué sur B. | Même gain sur B que sur A tenu à l'écart : même modèle. | prompts identiques, tirés de la réserve |

**Sélection de sondes par simulation.** C'est le complément propre à la batterie : classer un vivier de prompts par le rapport entre la sensibilité aux poids (fp8, LoRA de substitution) et la sensibilité aux leurres (bibliothèque de system prompts, T, watermark), en gardant ceux qui atteignent toujours 256 tokens.

**Jumeau local.** Un registre simulé au même format JSONL que le vrai, avec vérité terrain, pour recalibrer les seuils de la batterie sur la bonne famille sans toucher à son code.

**Fusion.** La piste GPU sort un fichier de paires `a,b,p_model,p_family` au format de `model_equality/scoring/partition.py`, plus un drapeau « certifié ». Les décisions certifiées priment, la batterie décide ailleurs, l'a priori structurel départage les indécis.

**Budget `ep-` proposé :**

- diagnostic : 12 au plus ;
- prompts longs communs : 40 (génériques si la simulation n'est pas prête, car le rapport de vraisemblance exploite n'importe quelle réponse a posteriori) ;
- ciblé : 20 à 30 ;
- réserve : au moins 15, intouchée jusqu'à 9 h dimanche.

## 5. Plan d'exécution sur JURECA

**Constaté sur le cluster le 21 nov. vers 17 h 25 :**

- Aucun des trois `.venv` ne contient transformers, tokenizers, peft, accelerate ni bitsandbytes. Le cache uv non plus. Aucun module système ne les fournit (les modules PyTorch listés sont des 1.x).
- `unlearning/.venv` : Python 3.12, torch 2.15.0 (PyPI), huggingface_hub 2.4.0, safetensors, numpy, pyyaml, tqdm, filelock.
- `Hackathon_Watermark_Localisation_Task/.venv` : Python 3.11, torch 2.5.1, numpy 1.26.4, scipy.
- `HF_HOME` vaut `/p/scratch/training2666/trentelivres1/Azerit/.cache`. Aucun modèle dedans. 95 To libres.
- Aucun jeton Hugging Face (ni dans le home, ni dans `HF_HOME`, ni dans l'environnement).
- Nœud de login : huggingface.co et PyPI joignables, 256 cœurs, 1 To de RAM, 2 Quadro RTX 8000 de 46 Go inoccupées.
- Réservation `cispahack` : 30 nœuds jusqu'à dimanche 14 h. 24 jobs en cours, 11 en attente, démarrage estimé sous 5 à 40 minutes. Durée par défaut 1 h, mode exclusif.
- Le job de test voit les 4 A100 même avec `--gres=gpu:1`.
- Non vérifié : l'accès Internet des nœuds de calcul. Le plan n'en dépend pas.
- Mac local : M1, 8 Go, sans torch. Bon pour les tokenizers seulement.

**Environnement.** Ne rien reconstruire. Réutiliser `unlearning/.venv` tel quel et lui superposer un dossier d'ajout via `PYTHONPATH` :

```
uv pip install --python /p/scratch/training2666/trentelivres1/Azerit/unlearning/.venv/bin/python \
  --target /p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/overlay \
  --no-deps transformers tokenizers regex accelerate psutil peft
```

Ajouter `optimum-quanto` ou bitsandbytes seulement si une quantification le demande. La compatibilité du dernier transformers avec torch 2.15 n'est pas vérifiée : un test d'import sur le login puis un job de 5 minutes la tranchent. Repli : un nouveau venv dédié `model_equality/.venv-gpu` avec torch==2.15.0, que le cache uv fournit sans retéléchargement. Ne jamais modifier les `.venv` existants.

**Modèles.** Téléchargement depuis le nœud de login vers `$HF_HOME/hub`, puis `HF_HUB_OFFLINE=1` dans les jobs. Tokenizers et configurations des ~50 candidats d'abord (quelques Mo). Les poids seulement après identification, environ 15 Go par modèle de 7B. Sans jeton : miroirs non restreints.

**Dossier de travail sur le cluster.** `/p/scratch/training2666/trentelivres1/Azerit/model_equality/wb/` avec `overlay/`, `jobs/`, `in/` (exports du registre), `out/` (scores), `sim/` (jumeau local).

**Forme des jobs.** `--account=training2666 --partition=dc-gpu --reservation=cispahack --nodes=1 --gres=gpu:4 --time=00:30:00` à `00:45:00` (en Slurm, `00:30` signifie 30 secondes), pour passer en backfill. Quatre processus, un par GPU (`CUDA_VISIBLE_DEVICES`), tirant d'un manifeste partagé. Jobs idempotents et reprenables, un fichier de sortie par couple (endpoint, hypothèse). Les requêtes API restent sur le Mac ; le registre est copié vers `wb/in/`.

**Ordre et chiffrage (heures-nœud) :**

| Étape | Coût |
|---|---|
| Test de l'environnement | 0,1 |
| Validation `ref-` | 1 |
| Identification des bases `ep-` | 1 |
| Simulation et sélection de sondes | 2 |
| Jumeau local et calibration de la batterie | 1,5 |
| Scoring `ep-` (toutes hypothèses) | 2 |
| Aléas | 1,5 |
| **Total** | **≈ 9, plafond 12** |

Le scoring lui-même est léger : environ 4 minutes par hypothèse pour 3 300 réponses sur un 7B (ordre de grandeur estimé, pas mesuré). Placer les jobs lourds entre 22 h et 6 h. Un seul nœud à la fois : les coéquipiers ont besoin du GPU pour les tâches 1 et 3. Si le modèle du banc fait moins de 2B, la validation `ref-` peut tourner en CPU sur le login, à valider avec les règles JSC.

## 6. Validation minimale sur `ref-`

Avant toute dépense `ep-` guidée par la piste, avec au plus 100 réponses par endpoint pour rester à parité de budget :

- **E0 (sans GPU).** 30 à 100 prompts longs × 9 endpoints, plus répétitions, oracle de dépassement et question d'identité. On en tire le tokenizer et le régime de décodage.
- **E1.** Checkpoint, dtype, template et décodage qui maximisent l'ajustement de `ref-01`. Critère : ≥ 99 % de rang 1, ou |z| < 3.
- **E2.** `ref-08` : T estimé à 1,0 ± 0,03 et ajustement correct. `ref-03` et `ref-05` : extraction puis rejeu qui restaure l'ajustement. `ref-06` : grille de détecteurs à clés par défaut.
- **E3.** `ref-02` : trouver l'émulation fp8 qui reproduit l'endpoint, et mesurer le taux de bascules d'argmax et la KL par token entre base et fp8. `ref-09`, `ref-04`, `ref-07` : amplitudes ordonnées 0,3 < 0,7 < 1,5, toutes différentes de la base.
- **E4.** Partition de `ref-` par la piste GPU seule, puis fusionnée, notée par `score_practice.py`. Cible : MCC = 1,0 avec 100 réponses, et au moins 0,9 avec 50.

## 7. Découpage en tâches

« Maintenant » = peut démarrer sans attendre l'identification des familles.

| Tâche | Produit | Dépend de | Agent | Maintenant |
|---|---|---|---|---|
| T0 (humain) | Jeton Hugging Face, licences Llama / Gemma / Mistral acceptées, quota GPU convenu avec les tâches 1 et 3 | — | humain | oui |
| T1 | Dossier d'ajout, script d'environnement, test GPU | — | Opus | oui |
| T2 | Atlas de ~50 tokenizers ; outils : empreinte par troncature, comptes de prompt avec template, détecteurs de watermark à clés par défaut | T1 pour les détecteurs torch | Opus | oui |
| T3 | Addendum de sondes de diagnostic pour la batterie (section 3), ordre de test sur `ref-` | — | Fable | oui |
| T4 | Scoreur en teacher forcing (dtype, émulations fp8, échelle LoRA, 50 logits). Test unitaire : texte auto-généré → 100 % de rang 1, z ≈ 0 | T1 | Opus | oui, sur un petit modèle libre |
| T5 | Jumeau local : génération par variante et leurre, registre simulé avec vérité | T1 | Opus | oui |
| T6 | Règles de décision et seuils, test structuré / non structuré, vraisemblance croisée, règle de fusion | format de T4 | Fable | oui pour la conception, finalisée après T7 |
| T7 | Validation `ref-` (E0 à E4), rapport de passage ou d'arrêt | T1, T2, T4, réponses `ref-` | Opus | dès que T4 tourne |
| T8 | Base exacte par famille `ep-` et endpoint de contrôle positif ; téléchargement des poids | diagnostic `ep-`, T2, T4, T0 | Opus | non |
| T9 | Liste de sondes à fort rapport signal / leurre | T5, T8 | Fable puis Opus | non (repli : prompts génériques) |
| T10 | Scoring `ep-`, postérieurs par endpoint, fichier de paires | T4, T8, réponses `ep-` | Opus | non |
| T11 | Partition finale et soumissions | T6, T10, batterie | Fable | non |

## 8. Risques et points d'arrêt

**Points d'arrêt :**

- **Samedi 21 h.** Tokenizer ou checkpoint du banc non identifié : abandonner la validation `ref-`. Si en plus aucune famille `ep-` n'est identifiée à 23 h : abandonner la piste et ne garder que les sous-produits sans GPU.
- **Après 2 heures-nœud.** Aucune hypothèse locale ne reproduit `ref-01` (moins de 97 % de rang 1, ou |z| > 5) : réplique infidèle. Abandonner la certification, garder seulement le jumeau local.
- **Après E3.** Si base contre fp8 donne z < 2 avec 100 réponses sous le décodage réel : la piste ne sépare pas non plus la quantification. La restreindre au LoRA, au system prompt et au watermark.
- **Dimanche 4 h.** Aucune famille `ep-` avec contrôle positif certifié : arrêt, tout l'effort passe à la boîte noire.
- **File d'attente.** Plus de 60 minutes d'attente à répétition, ou tâches 1 et 3 bloquées : arrêt des jobs.

**Risques :**

- modèle sous licence restreinte sans jeton (repli : miroirs non restreints du type `unsloth/*`, mêmes poids) ;
- base hors de notre liste de candidats ;
- pile de service qui modifie les logits (pénalité de répétition, paramètres par défaut du `generation_config`) ;
- extraction de system prompt inexacte ;
- retokenisation non canonique des textes échantillonnés (écarter les positions aberrantes) ;
- leurres simulés différents de ceux des organisateurs ;
- sur-confiance dans l'a priori « un seul écart par endpoint ».
