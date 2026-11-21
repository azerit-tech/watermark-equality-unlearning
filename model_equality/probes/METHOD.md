# Model Equality Testing : sondes et méthode statistique

Ce document suffit à exécuter la campagne sans autre contexte. Il accompagne `probes.jsonl` (138 sondes, générées par `build_probes.py`).

Statut : conception faite sans aucun appel aux endpoints. Tout seuil chiffré est une valeur par défaut à régler sur `ref-` ; les points non vérifiés sont marqués « à vérifier ».

## 1. Le problème en bref

33 endpoints `ep-01`..`ep-33` servent des LLM en boîte noire (texte seulement). Il faut rendre, pour chaque endpoint, une famille (même modèle de base) et un modèle.

| Différence entre deux déploiements | Verdict |
|---|---|
| Modèle de base (checkpoint) différent | familles différentes, modèles différents |
| Configuration de quantification différente | même famille, modèles différents |
| Adaptateur LoRA différent, ou même LoRA à une force différente | même famille, modèles différents |
| Watermark de décodage | même modèle (leurre) |
| System prompt ou persona, ajouté côté serveur | même modèle (leurre) |
| Stratégie de décodage (greedy, beam, autre température) | même modèle (leurre) |

Le piège est symétrique : les leurres changent beaucoup le texte, alors que deux quantifications ou deux LoRA peuvent donner des textes presque identiques. Un test qui mesure « à quel point les textes diffèrent » classe donc à l'envers.

### Métrique et coût des erreurs

`score = 0,1 × MCC_family + 0,9 × MCC_model`, chaque MCC étant ramené à 0 s'il est négatif. `MCC_model` porte sur les paires de même famille vraie, toutes familles confondues : une famille de taille n pèse n(n−1)/2 paires. Le score public couvre environ 30 % des paires, le classement final les 70 % restants.

Exemple pour une famille de 9 endpoints construite comme `ref-` (un groupe de 5 endpoints du même modèle, 4 variantes isolées : 10 paires « même », 26 paires « différent ») :

| Réponse rendue | MCC_model |
|---|---|
| Tout différent, ou tout identique | 0 |
| Parfaite | 1,00 |
| Deux variantes fusionnées entre elles | 0,94 |
| Une variante fusionnée à tort dans le groupe de 5 | 0,73 |
| Un leurre laissé seul hors du groupe | 0,72 |
| Groupe de 5 scindé en 3 + 2 | 0,57 |
| Modèle de base confondu avec la variante quantifiée (le groupe est accroché à la mauvaise) | 0,45 |

Conséquences :
- Les seules paires positives sont celles du groupe « modèle de base + ses leurres ». Sans elles le score vaut 0. Retrouver ce groupe compte autant que d'en exclure les variantes.
- Les deux erreurs sur le groupe coûtent à peu près autant ; il n'y a pas de sens de l'erreur à privilégier, la bonne règle est « fusionner si la probabilité d'identité dépasse 0,5 ».
- Une erreur entre deux variantes isolées coûte peu : ne pas y dépenser la réserve.
- Frontière de famille douteuse : fusionner les familles plutôt que les scinder. Une scission rend « différentes » d'office toutes les paires à cheval, donc des faux négatifs dans les 90 % ; une fusion ne coûte que dans les 10 %.

## 2. Faits sur l'API

Source : `starter/task_template.py`, `starter/query_reference.py`, description détaillée de la tâche.

- `POST {base}/chat/completions` avec `model`, `messages` (un seul message `user`), `max_tokens`. Pas de paramètre `n`. `temperature` et `seed` sont ignorés, les messages système sont rejetés.
- 100 réponses par `ep-` pour toute la compétition, 256 tokens générés au plus, 1 024 tokens d'entrée au plus, 4 requêtes simultanées par équipe (limite commune à `ref-` et `ep-`).
- Les requêtes invalides sont rejetées et non décomptées.
- `ref-01`..`ref-09` : un autre petit modèle, une seule famille, non notés, limite de 1 000 000 réponses chacun.

| Endpoint | Nature | Modèle |
|---|---|---|
| ref-01 | plain | S |
| ref-03 | system prompt | S |
| ref-05 | persona | S |
| ref-06 | watermark | S |
| ref-08 | sampling à température 1,0 | S |
| ref-02 | quantifié fp8 | T1 |
| ref-09 | LoRA force 0,3 | T2 |
| ref-04 | LoRA force 0,7 | T3 |
| ref-07 | LoRA force 1,5 | T4 |

Deux déductions à vérifier : la température par défaut n'est pas 1,0 (sinon ref-08 serait identique à ref-01) ; `ref-` ne contient ni décodage déterministe ni beam search, alors que la consigne les cite.

À journaliser pour chaque requête (le client le fait déjà) : corps complet de la réponse, `finish_reason`, `usage` s'il existe, `elapsed_s`, et pour les rejets le code HTTP et le champ `detail`.

## 3. Idée directrice

Trois constats fixent la méthode.

**1. Le budget impose des contextes répétés.** Une quantification déplace les logits d'une petite quantité ε, différente dans chaque contexte. Avec n tirages d'un même contexte, l'écart devient visible quand n·ε² dépasse quelques dizaines. Cent réponses à cent prompts différents ne donnent qu'un tirage par contexte et ne montrent rien ; cent premières réponses à un seul prompt donnent n = 100, insuffisant si ε < 0,7. Il faut donc que chaque réponse contienne beaucoup de tirages dans des contextes quasi identiques. Deux formes de sondes le font :
- les **flux** (S, SL) : « écris 120 chiffres au hasard » donne 100 à 250 tirages par réponse ;
- les **formulaires** (F) : 20 à 30 décisions proches de l'indifférence dans une seule réponse, chacune dans un contexte identique d'une réponse à l'autre.

**2. Chaque leurre a une signature paramétrique dans l'espace des log-probabilités, qu'une modification des poids n'a pas.** Soit ℓ(c, i) le log de la probabilité de l'option i dans le contexte c.

| Leurre | Effet sur ℓ(c, i) | Ce qui le retire |
|---|---|---|
| Température T | ℓ centré sur les options est multiplié par 1/T ; l'ordre des options est conservé | ajuster une pente κ unique pour la paire |
| Greedy, beam | limite T → 0 : on n'observe que l'option modale | comparer à la réponse majoritaire des endpoints échantillonnés |
| Top-k, top-p | la queue est coupée, le reste suit la loi de température | ne comparer que le support commun |
| Watermark de type liste verte | ajoute δ aux options d'un ensemble pseudo-aléatoire qui ne dépend que des h derniers tokens et de l'option, pas du sens | comparer des contextes qui partagent leurs h derniers tokens : l'effet est alors un décalage fixe par option, retiré par centrage |
| System prompt, persona | change le contexte, donc tout, mais c'est du texte en tête de conversation : il peut fuir, et son influence s'efface quand le message utilisateur est long et contraignant | détecter (fuite, compte de tokens), puis comparer sur des sondes à contexte long |
| Quantification, LoRA | résidu qui reste après avoir retiré tout ce qui précède, présent aussi en contexte long | c'est le signal cherché |

**3. Les leurres s'accrochent au modèle de base.** La consigne présente les leurres comme « base model + watermark », « base model + system prompt », « base model + decoding strategy », et `ref-` est construit ainsi : une seule modification par endpoint. Hypothèse de structure, notée H-struct : dans chaque famille, un groupe « base + leurres », et des variantes isolées. Elle sert de prior et de départage, jamais de règle absolue : la statistique invariante de la section 5 reste valable si un leurre est posé sur une variante.

Les trois articles cités par les organisateurs vont dans ce sens :
- MET (Gao et al.) : test à deux échantillons, MMD à noyau de Hamming. Il détecte aussi température et watermark, donc ici il scinderait tous les leurres. Il sert de test brut.
- AgentProv (Wang et al., dont deux auteurs de la tâche) : empreinte catégorielle sur des choix entre options équivalentes, test par permutation ; 7 % de faux positifs sous system prompt injecté contre 67 % pour MET, mais une quantification 8 bits n'y est pas détectée. Les sondes F20, F21 et KT en sont l'émulation en texte. L'article exploite aussi le compte de tokens d'entrée pour repérer un system prompt caché.
- RUT (Zhu et al.) : compare une réponse d'API à 100 tirages d'un modèle de référence local. C'est la piste B (section 10).

## 4. La banque de sondes

`probes.jsonl` : une sonde par ligne, texte complet prêt à envoyer.

| Champ | Sens |
|---|---|
| `id`, `group` | identifiant ; `group` relie les variantes d'une même sonde |
| `category` | voir tableau suivant |
| `text` | message `user` complet |
| `max_tokens` | valeur conseillée |
| `est_prompt_tokens` | estimation pessimiste (1 token pour 2,6 caractères) ; tout est sous 760 sauf X01, volontairement invalide |
| `parser` | comment transformer la réponse en événements (ci-dessous) |
| `wm_block` | contextes qui partagent le même contexte de hachage de watermark |
| `phases` | `ref_R0`, `ref_R1`, `ref_R2`, `ep_P0`, `ep_P1`, `ep_battery` (candidat), `ep_det` (une fois sur un endpoint déterministe), `ep_domain` |
| `priority` | 1 = à lancer d'abord |
| `hypothesis`, `resists` | ce que la sonde doit révéler, et à quel leurre elle doit résister |

| Catégorie | Nombre | Rôle |
|---|---|---|
| `D_diagnostic` | 15 | identité, fuite du system prompt, persona, signature de tokenizer |
| `X_free_sidechannel` | 1 | prompt trop long, rejeté sans coût : lire le message d'erreur |
| `K_single` | 42 | réponse d'un mot, par blocs à options communes ; calibration massive sur `ref-` |
| `S_stream` | 12 | flux courts sur alphabet fermé |
| `S_stream_open` | 8 | listes à vocabulaire ouvert, dont trois hors anglais |
| `SL_stream_diluted` | 8 | flux précédés de 150 à 200 exemples en contexte (dilution du system prompt) |
| `F_form` | 18 | formulaires de 12 à 30 décisions, dont choix d'outils à la AgentProv |
| `C_continuation` | 10 | continuations à la MET, cinq langues, code, deux versions longues |
| `L_domain` | 24 | révélateurs du domaine d'un LoRA |

Parseurs :
- `stream` : extraire dans l'ordre les items de `alphabet` (expression régulière sur les séparateurs `sep`) ; ignorer le reste ; noter la part de texte hors format.
- `open_list` : items séparés par `sep`, en minuscules, sans ponctuation.
- `form` : lignes `numéro: option` ; une réponse absente ou hors `options` vaut `∅`.
- `form_open` : même chose, la réponse est une chaîne libre normalisée.
- `first_word`, `first_match` : premier mot de la réponse, ou première option trouvée.
- `free` : pas de parsing (lecture, expressions régulières, stylométrie).

## 5. Features et statistiques

### 5.1 Tables de comptage

Chaque réponse parsée donne des événements (contexte c, option i). Pour un endpoint X et une sonde, N_X[c, i] est le nombre d'événements, cumulé sur les réponses.

| Sonde | Contexte c | Option i |
|---|---|---|
| Flux à séparateur virgule ou ligne | item précédent (ordre 1) ; début de réponse = contexte `START` | item suivant |
| Flux à séparateur espace | les deux items précédents (ordre 2) | item suivant |
| Formulaire | numéro de la ligne | réponse |
| Bloc K | prompt | premier mot |

Pour les flux binaires (S01, S04, S08, SL01), prendre les trois items précédents comme contexte : 8 contextes.

### 5.2 Trois statistiques emboîtées

Pour une paire (X, Y), sur un ensemble de sondes :

**T_raw, test brut.** Pour chaque contexte dont le total cumulé atteint 10, test G du tableau 2 × m (fusionner les options d'effectif attendu inférieur à 3) ; T_raw est la somme des G. Il détecte toute différence, leurres compris. Pour les sondes `free`, le test brut est le MMD de MET : noyau de Hamming au niveau caractère sur les 200 premiers caractères, complétés par un caractère de bourrage, sommé par prompt.

**T_temp, invariant à la température.**
1. ℓ_X[c, i] = log((N_X[c, i] + 0,5) / (N_X[c, ·] + 0,5 m)), variance v_X[c, i] ≈ 1/(N_X[c, i] + 0,5).
2. Ne garder que les cellules où N_X[c, i] ≥ 5 et N_Y[c, i] ≥ 5 (support commun : neutralise top-k et top-p).
3. Centrer ℓ sur les options dans chaque contexte : Θ[c, i] = ℓ[c, i] − moyenne sur i.
4. κ̂ = argmin sur κ ∈ [0,25 ; 4] de Q(κ) = Σ (Θ_Y − κ Θ_X)² / (κ² v_X + v_Y), un seul κ pour la paire, toutes sondes réunies. T_temp = Q(κ̂).

**T_inv, invariant à la température et au watermark.** Comme T_temp, mais à l'étape 3 on fait un double centrage dans chaque bloc `wm_block` : on retire de ℓ l'effet du contexte et l'effet de l'option (résidus de l'ajustement additif ℓ ≈ a_c + b_i, par moindres carrés pondérés quand des cellules manquent). Il ne reste que l'interaction contexte × option, sur laquelle un watermark dont le contexte de hachage est constant dans le bloc n'a aucune prise. Pour un formulaire binaire, cela revient à centrer sur les lignes le logit de chaque ligne.

T_inv a moins de degrés de liberté que T_temp, donc moins de puissance : on ne l'emploie que si une signature de watermark est trouvée (5.4).

**Calibration par permutation.** Les tirages d'une même réponse ne sont pas indépendants : ne jamais lire T sur une table du χ². Pour chaque paire, permuter les réponses entières entre X et Y à l'intérieur de chaque sonde (B = 500), recalculer T de bout en bout, et rendre z = (T − moyenne des permutations) / écart-type des permutations. On obtient z_raw, z_temp, z_inv. Contrôle : la même procédure entre deux moitiés aléatoires d'un même endpoint doit donner z centré sur 0 ; sinon la sonde est instable et on l'écarte.

L'unité « écart-type du bruit de l'endpoint lui-même » est ce qui se transporte d'une famille à l'autre ; les seuils en unités absolues ne se transportent pas.

### 5.3 Grandeurs par endpoint

- **Déterminisme** : deux réponses identiques à un même prompt long (D08, C01, chacun répété deux fois). Étiquette DET.
- **Diversité** : nombre de premiers mots distincts, entropie des tables de flux, longueur du préfixe commun entre réponses à un même prompt.
- **Compte de tokens** : `usage.prompt_tokens` si l'API le renvoie, ou le nombre de tokens cité dans le message de rejet de X01. À texte égal, un écart constant entre deux endpoints d'une famille est la longueur d'un system prompt caché.
- **Latence** : `elapsed_s` divisé par 256 pour les réponses coupées à la longueur maximale (`finish_reason = length`). Beam search et adaptateur non fusionné ralentissent. Signal d'appoint, jamais décisif seul.

### 5.4 Étiquettes de leurre

Chaque étiquette se lit sur la paire (X, référence de la famille), la référence étant le candidat « base » de la section 9.

| Étiquette | Critère par défaut |
|---|---|
| DET | réponses répétées identiques |
| TEMP | intervalle de confiance bootstrap de ln κ̂ hors de [−0,15 ; 0,15], et z_temp ≤ 2 |
| WM | z_temp > 4 et z_inv ≤ 2, et les écarts par option b_i^Y − κ̂ b_i^X prennent deux valeurs : après un 2-means à une dimension, écart entre centres δ̂ ≥ 1, variance résiduelle inférieure à 20 % de la variance initiale (une loi normale en laisse 36 %), δ̂ stable à ±30 % d'un bloc à l'autre, au moins 8 options au total. Les flux de chiffres, de lettres et de dés sont les détecteurs |
| SP | texte de fuite propre à l'endpoint (D03 à D06, D09, D10), ou écart de compte de tokens, ou contraste de dilution : z > 4 sur les sondes courtes et z ≤ 2 sur les sondes longues (SL, formulaires, C09) |
| NONE | aucune des précédentes |

### 5.5 Endpoint déterministe contre endpoint échantillonné

Un endpoint déterministe G ne donne qu'une réponse par prompt : ne jamais répéter un prompt dessus, et ne pas lui envoyer de flux (ils bouclent). On lui envoie une fois chaque sonde `ep_det` (formulaires, continuations), soit 28 réponses.

- **Accord modal sur formulaires.** Pour chaque ligne j, m_X(j) est la réponse majoritaire de l'endpoint échantillonné X, retenue si elle devance la seconde d'au moins 2 voix. A(G, X) est la part des lignes retenues où la réponse de G vaut m_X(j). L'argmax ne dépend pas de la température : pour le même modèle, A est proche de 1.
- **Préfixe commun sur continuations.** ρ(G, X) = longueur moyenne du préfixe commun entre la réponse de G et celles de X, divisée par la longueur moyenne du préfixe commun entre deux réponses de X. Pour le même modèle ρ ≥ 1, car le chemin greedy est le plus probable.
- **Deux endpoints déterministes** : comparer les réponses prompt par prompt. Tout identique = même modèle et même décodage.

Ces deux mesures séparent G d'un LoRA fort, pas d'un fp8. La décision par défaut vient donc de H-struct : G rejoint le groupe de base, sauf si A ou ρ désigne nettement une variante (écart d'au moins 0,15 sur A, confirmé par ρ).

## 6. Sélection sur `ref-`

Mesurer le débit en R0 et ajuster les répétitions : les nombres ci-dessous supposent 5 000 à 8 000 réponses longues par heure.

| Étape | Sondes | Répétitions par endpoint | Volume | But |
|---|---|---|---|---|
| R0 | toutes, sur ref-01 seul ; X01 sur les 9 | 2 | 300 | conformité des formats, débit, contenu des rejets, présence de `usage` |
| R1 | K priorité 1 (12 sondes), puis les 30 autres | 300, puis 100 | 32 000 + 27 000 courtes | machinerie des leurres et taille des effets |
| R2 | S, SL, F, C priorité 1 (19) ; autres S, F, C (37) ; D (15) ; L (24) | 40 ; 12 ; 6 ; 8 | 13 400 | criblage |
| R3 | les 10 meilleures sondes | compléter à 150 | 10 000 | 10 budgets disjoints de 12 à 15 réponses pour l'évaluation finale |

### R1 : ce qu'il faut en tirer

1. **Température par défaut.** Pente κ̂ entre ref-01 et ref-08 sur les blocs KD, KB, KY : T_défaut ≈ κ̂ si ref-08 est à 1,0. Vérifier que z_temp(01, 08) ≤ 2 : c'est la validation de l'invariance à la température.
2. **Troncature.** Sur KW, la queue de ref-08 contient-elle des mots absents de ref-01 ? Si oui, top-k ou top-p par défaut.
3. **Watermark.** Sur KD (10 options) et KW, les écarts par option entre ref-06 et ref-01 prennent-ils deux valeurs ? Relever δ̂ et la part d'options favorisées. Les options favorisées sont-elles les mêmes dans les 8 prompts de KD ? Si oui, le contexte de hachage du premier token est la fin du template. Puis, sur les flux de R2 : l'effet du watermark dépend-il de l'item précédent seulement (contexte 1) ou de plus loin ? Cela fixe la profondeur de contexte à tenir constante, donc l'ordre des tables et le choix entre F01 et F01k. Si le watermark rend les réponses déterministes (schéma de type Gumbel), il se traite comme DET.
4. **Taille des effets.** Écart quadratique moyen des log-cotes centrées, après ajustement de κ, entre ref-01 et ref-02 (fp8), ref-09, ref-04, ref-07 : ε_fp8, ε_0,3, ε_0,7, ε_1,5. Même mesure pour ref-03 et ref-05 : ε_SP en contexte court.
5. **Faisabilité.** Un écart ε se voit à 4 écarts-types avec environ 50/ε² tirages par table de 10 options : 1 300 tirages pour ε = 0,2 (10 réponses de flux), 5 000 pour ε = 0,1 (40 réponses), 20 000 pour ε = 0,05 (hors budget). Si ε_fp8 < 0,07, la boîte noire ne séparera pas fp8 de la base : passer à la piste B ou s'en remettre au prior.
6. **Colinéarité du LoRA.** Les écarts de ref-09, ref-04, ref-07 à ref-01 doivent être alignés et ordonnés 0,3 < 0,7 < 1,5. Si c'est vrai, un alignement de ce type dans une famille notée signale une série de forces d'un même LoRA.

### R2 : critère pour garder une sonde

Calculer, pour chaque sonde et chaque paire parmi les 36, z_raw, z_temp, z_inv sur des sous-échantillons de 12 réponses par endpoint (20 tirages bootstrap). Une sonde est gardée si elle remplit les trois conditions :

1. **Conformité** : au moins 85 % de réponses parsables sur chacun des 9 endpoints (70 % toléré pour ref-05, la persona).
2. **Sûreté face aux leurres** : pour les 10 paires « même modèle », la statistique prévue pour la paire (5.2 et 5.4) vaut z ≤ 2,5 dans au moins 90 % des tirages. Une sonde qui échoue sur un seul type de leurre reste utilisable si ce leurre est étiquetable et qu'on l'exclut alors de la sonde.
3. **Sensibilité** : z > 3 dans au moins 50 % des tirages pour au moins une paire difficile : (01, 02) fp8, (01, 09) LoRA 0,3, (09, 04) LoRA 0,3 contre 0,7.

La sûreté est obligatoire, la sensibilité peut être partielle : deux sondes se complètent. Garder au moins trois classes de sondes différentes, et préférer une sonde qui a une justification mécanique à une sonde qui gagne de peu sur 36 paires.

Relever aussi, pour la section 9 : la paire de sondes courte et diluée qui donne le meilleur contraste pour ref-03 et ref-05 ; les formulations de fuite qui marchent ; les sondes L où ref-07 s'écarte le plus (nature du LoRA).

### R3 : validation de bout en bout

Rejouer le protocole `ep-` complet (sections 7 à 9) sur 10 budgets disjoints de 100 réponses par endpoint tirés du stock, et scorer chaque partition avec `score_practice.py`. Régler les seuils (2 ; 4 pour les zones, et la table de poids) pour maximiser le MCC moyen, pas le meilleur. Objectif : MCC moyen ≥ 0,7 et aucun tirage sous 0,5. Noter la puissance sur fp8 à ce budget, π_fp8 : elle sert en 9.3.

**Greedy simulé.** `ref-` n'a pas d'endpoint déterministe. Construire un pseudo-greedy de ref-01, ref-04 et ref-02 : pour chaque ligne de formulaire, la réponse majoritaire sur 150 réponses. Vérifier que l'accord modal de 5.5 rattache le pseudo-greedy de ref-01 au groupe S et celui de ref-04 à ref-04.

## 7. Campagne sur `ep-`

Avant tout appel `ep-` : le client plafonne par défaut à 50 réponses par endpoint (`DEFAULT_EP_CAP`), il faudra relever ce plafond pour la vague B et la réserve.

### Phase 0 : gratuite

1. Envoyer X01 (trop long) à un seul `ep-`, relire le budget : `used` ne doit pas avoir bougé. Si c'est confirmé, l'envoyer aux 33.
2. Lire `detail`. S'il contient un nombre de tokens : regrouper les endpoints par nombre (tokenizer et template, donc famille) ; un endpoint qui dépasse ses voisins d'une constante porte un system prompt.
3. Si `detail` ne dit rien, chercher par dichotomie sur `ref-` si la limite de 1 024 tokens porte sur le message seul ou sur l'entrée complète, system prompt compris (comparer ref-01 et ref-03 avec un prompt fait de N répétitions de `hello `). Si elle porte sur l'entrée complète : sur `ep-`, trouver N* sur un endpoint par famille (au plus 6 réponses acceptées, `max_tokens` = 1), puis envoyer N* + 1 et N* aux autres. Un rejet de N* signale un system prompt, sans coût.

### Phase 1 : 12 réponses par endpoint

| Sonde | Réponses | Sert à |
|---|---|---|
| D01, D03, D04, D06, D11 | 1 chacune | famille, fuite du system prompt, persona, tokenizer |
| D08 | 2 | déterminisme, style, persona |
| C01 | 2 | déterminisme, test brut, famille |
| flux principal retenu en R3 (par défaut SL02) | 2 | conformité dans la famille ; compte dans la batterie |
| formulaire principal retenu en R3 (par défaut F01) | 1 | conformité ; compte dans la batterie |

Envoyer D08 et C01 en premier : un endpoint dont les deux réponses sont identiques est déterministe, et on remplace alors ses 2 réponses de flux par 2 sondes `ep_det`. Les trois dernières lignes du tableau sont déjà des réponses de batterie ; le coût propre du diagnostic est de 7 réponses.

### Familles

Variables, par ordre de fiabilité : point de coupure de D11 et comportement sur D12 (tokenizer) ; compte de tokens de la phase 0 ; auto-identification D01 ; profil de réponses de F05 ; stylométrie (cosinus TF-IDF de trigrammes de caractères) sur D08 et C01. Classification hiérarchique à lien moyen, coupée au plus grand saut.

- Une persona ou un LoRA fort peut dépasser en style l'écart entre deux modèles de base : la signature de tokenizer prime sur le style.
- Deux tailles d'une même lignée ont le même tokenizer et la même identité déclarée : les séparer par F05 et par la latence.
- En cas de doute, fusionner les familles (section 1).

### Batterie en deux vagues

Par famille, la batterie est la même pour tous les endpoints échantillonnés. Les endpoints DET reçoivent seulement les sondes `ep_det`, une fois.

| Étape | Cumul par endpoint | Contenu |
|---|---|---|
| Vague A | ≈ 48 | flux dilué principal porté à 14 réponses, flux court associé 12, formulaire principal 8, second formulaire 5 |
| Analyse | | étiquettes, z par paire, choix de la vague B |
| Vague B | ≈ 70 | 22 réponses sur les 2 sondes qui séparent le mieux dans cette famille ; si aucune variante n'apparaît, 12 réponses sur les 6 meilleures sondes L |
| Réserve | 100 | section 9.4 |

Choix des sondes par famille, sans vérité terrain : parmi les sondes validées sur `ref-`, garder celles dont le taux de parsing dépasse 80 % dans la famille et dont l'entropie n'est pas dégénérée ; après la vague A, classer les sondes par le rapport entre la dispersion des z entre endpoints et le bruit entre moitiés d'un même endpoint. Si le flux principal échoue dans une famille (modèle trop petit pour tenir le format), essayer deux autres candidats sur deux endpoints, une réponse chacun.

Concentrer plutôt que disperser : à nombre de tirages égal, trois sondes à 12 réponses donnent un meilleur z que douze sondes à 3 réponses, parce que le bruit croît avec le nombre de cellules.

## 8. Ordre des analyses

1. Parser, calculer le taux de conformité par endpoint et par sonde.
2. Étiquette DET, puis familles.
3. Par famille : z_raw, z_temp, κ̂ pour toutes les paires d'endpoints échantillonnés ; signature de watermark ; z_inv pour les paires concernées ; z séparés sur sondes courtes et sur sondes longues.
4. Étiquettes TEMP, WM, SP.
5. Accord modal et ρ pour les endpoints DET.
6. Décision (section 9), puis choix des sondes de la vague B ou de la réserve.

## 9. Décision

### 9.1 Statistique retenue pour une paire

- Paire de deux endpoints échantillonnés : z* = z_temp. Si z_temp > 2 et que la signature de watermark de 5.4 est présente pour la paire, z* = z_inv.
- Si l'un des deux est étiqueté SP, calculer z* sur les sondes longues seulement.
- Si la persona ne se dilue pas, même en contexte long (à voir sur ref-05) : la paire est jugée sur le prior seul.

### 9.2 Candidat « base »

Dans chaque famille, le candidat base est l'endpoint NONE dont la somme des z* vers les endpoints étiquetés leurre est la plus faible. S'il n'y a aucun leurre, il n'y a pas de groupe à former : tous les endpoints restent seuls, sauf paire à z* ≤ 2.

### 9.3 Partition

Poids d'une paire :

| z* | Poids |
|---|---|
| ≤ 2 | +2 |
| 2 à 3 | +0,5 |
| 3 à 4 | −0,5 |
| 4 à 6 | −2 |
| > 6 | −4 |

Corrections de prior (H-struct) :
- +1 si l'un est étiqueté leurre et que l'autre est le candidat base ou un autre leurre ;
- −1 si les deux sont NONE ;
- paire avec un endpoint DET : poids +1 vers le candidat base et vers les leurres, −2 vers une variante, inversés si l'accord modal désigne nettement cette variante.

La partition retenue maximise la somme des poids des paires placées dans un même groupe. Jusqu'à 10 endpoints par famille, énumérer toutes les partitions ; au-delà, regroupement agglomératif à lien complet puis déplacements d'un endpoint à la fois. Cette formulation règle la transitivité : A proche de B, B proche de C et A loin de C ne fusionnent que si le total reste positif, et un poids de −4 agit presque comme un veto.

**Deux endpoints NONE indiscernables.** Avec π_fp8 mesurée en R3 :
- π_fp8 ≥ 0,8 : une différence aurait été vue. Le poids +2 − 1 = +1 les fusionne.
- π_fp8 < 0,5 : l'absence de différence ne prouve rien. Ramener le poids de z* ≤ 2 à +0,5 pour les paires NONE–NONE, ce qui les sépare. S'il faut alors choisir lequel des deux porte le groupe des leurres sans indice, mieux vaut les mettre tous deux dans le groupe (0,73 attendu, contre 0,72 en moyenne pour un choix au hasard qui vaut 1,00 ou 0,45).
- Entre les deux : réserve.

### 9.4 Incertitude et réserve

Une paire est grise si 2 < z* ≤ 4, ou si la partition change quand on retire une sonde. Ordre de priorité de la réserve :
1. paires grises entre le candidat base et un endpoint NONE (décide une variante contre un doublon) ;
2. paires grises entre un leurre et le candidat base ;
3. choix du candidat base quand deux NONE sont proches ;
4. jamais les paires entre deux variantes isolées.

Règle séquentielle : ajouter 10 réponses aux deux endpoints de la paire sur les 2 sondes de plus fort z dans la famille, recalculer, arrêter quand z* ≤ 1,5 ou z* > 4,5, ou à épuisement. Les réponses ajoutées à un endpoint servent à toutes ses paires.

Un budget non dépensé à la fin ne vaut rien. Après résolution des paires grises, dépenser le reste sur les sondes principales de chaque famille, en gardant 5 réponses par endpoint pour un incident.

Le score public n'est pas un oracle : il ne couvre que 30 % des paires, le tableau n'affiche que le meilleur score de l'équipe, une soumission en erreur consomme aussi les 30 minutes. Usage prévu : une soumission « familles seules, tous les modèles différents » doit donner 0,1 × MCC_family, ce qui valide les familles ; ensuite, contrôle de vraisemblance uniquement.

## 10. Piste B, optionnelle : boîte blanche locale

Si le modèle de base d'une famille est identifié (la description cite OLMo comme exemple) et téléchargeable sur le cluster, chaque token généré devient un tirage dont la loi de référence est connue, et les 25 000 tokens d'un endpoint deviennent exploitables.

1. **Vérifier l'identité** : log-vraisemblance par token des réponses sous le candidat, avec le template exact. Le bon modèle donne une valeur proche de l'entropie attendue, un mauvais plusieurs nats de plus.
2. **Température** : ajuster T par maximum de vraisemblance sur les logits du modèle de base.
3. **Watermark** : avec le tokenizer seul, lancer les détecteurs de liste verte standards avec leurs clés par défaut. Un z élevé identifie le watermark sans rien d'autre. À essayer d'abord sur ref-06.
4. **System prompt** : comparer la vraisemblance avec et sans le texte obtenu par fuite.
5. **Quantification** : quantifier localement (fp8 de vLLM, 8 bits, 4 bits) et calculer le rapport de vraisemblance entre base et candidat quantifié sur les réponses de chaque endpoint. Ne marche que si la quantification locale reproduit celle du serveur.
6. **LoRA** : écart résiduel après ajustement de T, par un test de rang à la RUT.

Décider de la lancer après R1 : elle vaut l'effort si ε_fp8 < 0,1 et si au moins une grande famille est identifiée avec certitude. Les sondes C et L servent aux deux pistes.

## 11. Risques de transfert de `ref-` vers `ep-`

| Risque | Ce qui le limite |
|---|---|
| Autres modèles, autres capacités : un format tenu par le modèle `ref-` ne l'est pas ailleurs | conformité mesurée par famille en phase 1 ; plusieurs classes de sondes ; choix non supervisé par famille |
| Seuils réglés sur 36 paires d'une seule famille | z par permutation, en unités du bruit propre de chaque endpoint ; réglage sur le MCC moyen de 10 budgets, pas sur le meilleur |
| Une seule quantification (fp8) dans `ref-` ; ailleurs 4 bits, 8 bits ou autre | le test est générique (tout déplacement des logits) ; une quantification plus forte est plus facile ; π_fp8 borne le cas le plus fin |
| Un seul LoRA à trois forces ; ailleurs d'autres adaptateurs, d'autres domaines | 24 sondes de domaine ; flux et formulaires sensibles à tout déplacement ; colinéarité des forces |
| Greedy et beam absents de `ref-` | greedy simulé en R3 ; décision par prior, corrigée seulement sur évidence nette |
| Watermark, température par défaut ou system prompts différents de ceux de `ref-` | signatures retestées dans chaque famille (5.4), jamais supposées |
| Leurre posé sur une variante (absent de `ref-`) | T_inv et le contraste de dilution restent valables ; H-struct n'est qu'un prior de poids 1 |
| Persona qui ne se dilue pas | étiquette SP par fuite ou compte de tokens, puis prior |
| Familles de tailles inégales : une erreur dans une grande famille pèse plus | traiter les grandes familles d'abord, y dépenser la réserve en priorité |
| Deux tailles d'une même lignée prises pour une famille | F05 et latence ; en cas de doute, fusionner |

## 12. Paramètres par défaut

| Paramètre | Valeur | Réglé en |
|---|---|---|
| Effectif minimal d'une cellule | 5 | R2 |
| Bornes de κ | [0,25 ; 4] | R1 |
| Permutations par paire | 500 | fixe |
| Zones de z* | ≤ 2 compatible ; 2 à 4 gris ; > 4 différent | R3 |
| Étiquette TEMP | intervalle de ln κ̂ hors de ±0,15 | R1 |
| Signature WM | δ̂ ≥ 1 ; variance résiduelle < 20 % ; ≥ 8 options | R1 |
| Conformité | ≥ 85 % sur `ref-`, ≥ 80 % dans la famille | R2 |
| Sûreté d'une sonde | z ≤ 2,5 dans ≥ 90 % des tirages sur les 10 paires « même » | R2 |
| Sensibilité d'une sonde | z > 3 dans ≥ 50 % des tirages sur une paire difficile | R2 |
| Budget `ep-` | 12 phase 1, 36 vague A, 22 vague B, 30 réserve | à revoir après R3 |
| Poids de prior | ±1 | R3 |
