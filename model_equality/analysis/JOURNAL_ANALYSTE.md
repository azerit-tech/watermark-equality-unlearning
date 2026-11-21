# Journal de l'analyste (Model Equality, ep-)

Une entrée par événement : heure (Paris), candidat, changements, preuves, score attendu, score obtenu.
Scores attendus : toujours `python3.12 model_equality/analysis/expected_score.py <csv> <leurres supposés vrais>`,
donnés pour un groupe de base vrai de 5 / 6 / 7 endpoints par famille. Le score public porte sur 30 % des paires :
un écart de quelques centièmes entre deux soumissions n'est pas une preuve.

## Samedi 21 novembre

### 19 h 34 — soumission 1 : partition_families_v1.csv (sha 6fdf597b…)
- Contenu : 3 familles de 11, un seul modèle par famille.
- Score public obtenu : 0,100. Lecture : MCC_famille = 1 (familles confirmées), MCC_modèle = 0, donc calculé sur
  les seules paires intra-famille.

### 20 h 05 — reprise par le nouvel analyste, vérification de la file
- NEXT.csv (sha db78e7f7…) et NEXT2.csv (sha f7ebdb83…) régénérés dans le scratchpad avec les commandes de leurs
  .txt : sha identiques. File gelée, non touchée.
- Scores attendus recalculés (identiques à ceux du prédécesseur) :
  - v2 (partition_lures_v2.csv, sha b06cee59…, 8 leurres groupés) : 0,50 / 0,41 / 0,34 ;
  - NEXT = v2 + ep-02 : 0,56 / 0,45 / 0,38 ; si une étiquette est fausse : 0,43 / 0,35 / 0,28 ;
  - NEXT2 = NEXT + ep-30, ep-31, ep-33 : 0,77 / 0,61 / 0,50 ; si une étiquette est fausse : 0,63 / 0,50 / 0,39.
- Réserve relevée dans le code : en mode `--labels-only`, un endpoint sans étiquette forcée garde son étiquette
  AUTOMATIQUE (labels.py : DET, WM:repeat, SP par fuite ou identité). Quand E2 remplit le registre, une étiquette
  automatique peut donc apparaître et changer la partition régénérée : « ne dépend pas du registre » n'est vrai
  qu'à données constantes. Parade : je revérifie le sha régénéré avant chaque créneau ; à partir de NEXT3 le fichier
  d'étiquettes liste les 33 endpoints (NONE explicite), ce qui rend la partition indépendante du registre.
- Hypothèse de structure à garder en tête (banc ref- : 1 nu + 4 leurres + fp8 + 3 LoRA) : un leurre peut aussi être
  posé sur une variante ; seule la boîte blanche peut le voir.

### 20 h 07 — soumission 2 : v2 = partition_lures_v2.csv (sha b06cee59…), id 1423
- Contenu : 8 leurres groupés (llama3 ep-10, 16, 29 ; qwen_A ep-04, 23 ; qwen_B ep-06, 08, 20), 25 endpoints seuls.
- Attendu sur toutes les paires : 0,50 / 0,41 / 0,34 (groupe de base 5 / 6 / 7). Obtenu (public) : 0,35.
- Lecture, avec la vraie formule MCC et un tirage public de 30 % des 528 paires (scratchpad public_sim.py, 4000 tirages
  par hypothèse) :
  - v2 ne prédit que 7 paires « même modèle » sur 165 paires intra-famille. Il n'en tombe qu'environ 2 dans le
    public (0 : 8 %, 1 : 25 %, 2 : 32 %, 3 : 23 %). Le score public de v2 a donc un écart-type de 0,09 à 0,19 selon
    l'hypothèse : il mesure surtout le tirage.
  - Combinaisons entières compatibles avec 0,35 sans aucune erreur d'étiquette : 1 paire groupée publique et 10 à 11
    paires « même » vraies dans le public (attendu 9 si le groupe de base fait 5) ; ou 2 paires groupées et 16 à 19
    paires vraies (attendu 13,5 pour 6 et 18,9 pour 7) ; ou 3 paires et 20 à 25 paires vraies (25 pour 8).
  - Probabilité d'observer 0,35 (à l'arrondi près) : groupe 5 sans erreur 0,9 % ; 6 sans erreur 2,3 % ; 7 sans
    erreur 4,7 % ; 8 sans erreur 3,5 % ; 5 avec une étiquette fausse 2,4 % ; 6 avec une fausse 3,4 % ; 7 avec une
    fausse 2,5 % ; groupe de 4 sans erreur : 0 %.
- Conclusion : 0,35 penche légèrement (facteur 2 à 5, pas plus) vers « groupe de base de 6 à 8 » ou « une étiquette
  fausse parmi les 8 », contre « groupe de 5 et 8 étiquettes justes ». Il n'exclut que le groupe de 4. Aucune étiquette
  n'est mise en cause par ce seul score. Les 8 étiquettes restent en place.
- Conséquence pour la lecture des soumissions suivantes (même tirage public à chaque fois) :
  - NEXT ajoute 2 paires (ep-02 avec ep-04 et ep-23). Aucune des deux n'est publique dans 49 % des cas : le score
    restera alors exactement 0,35 et ne dira rien sur ep-02. Une hausse = au moins une paire publique et vraie ; une
    baisse = au moins une paire publique et fausse.
  - Surapprentissage : avec si peu de paires publiques par changement, optimiser le score public reviendrait à
    apprendre le tirage. Les décisions restent fondées sur les preuves écrites ; le score public ne sert que de
    contrôle de signe, endpoint par endpoint.

### 20 h 12 — changement de plan (coordinateur) : NEXT2 part à 20 h 40 à la place de NEXT
- NEXT.csv (db78e7f7…, v2 + ep-02) est périmé et ne sera pas soumis ; l'officier est prévenu. Le créneau de 21 h 12
  est réservé à NEXT3 (verdicts boîte blanche + 33 étiquettes forcées), sauté si les verdicts manquent à 21 h 05.
- Si le sha régénéré de NEXT2 diffère de f7ebdb83… à 20 h 40, l'officier ne soumet rien (consigne du coordinateur).
  Revérifié à 20 h 14 après ma correction de cli.py : toujours f7ebdb83….

### 20 h 15 — preuves de répétition relues sur le registre (E1 + début de E2_a, scratchpad reps.py)
- ep-04 (qwen_A) : 3 réponses identiques sur 4 à tp_story (283 caractères), 0 chez les dix autres. Étiquette WM solide.
- ep-06 (qwen_B) : paires identiques sur 3 sondes (C08_story, E01_identity 259 caractères, E03 chinois), 0 chez les
  dix autres hors sondes à réponse quasi unique (pi). Étiquette WM solide.
- ep-10 (llama3) : une seule paire identique (E03 chinois, 53 caractères, texte improbable contenant « named
  LeonidisWakehaven »), 0 sur 6 à tp_story. Une telle paire ne s'obtient pas par hasard à T = 0,7, mais cela reste
  une seule paire : étiquette WM gardée, la plus fragile des douze avec ep-02.
- Fausse alerte écartée : ep-01 a deux réponses identiques à C08_story, mais c'est la réponse modale de qwen_A (elle
  recopie le paragraphe ; 6 endpoints donnent le même texte). Ce n'est pas un signe de watermark.
- Identité (2 réponses par endpoint) : personas tirées à chaque requête confirmées (ep-16 VintageVeg puis Emoji
  Artist ; ep-23 StyleSage puis Inspiring Universe AI ; ep-20 conseiller juridique puis « created by Anthropic ») ;
  ep-08 « Claude » deux fois ; ep-02 sans nom deux fois. Rien de nouveau contre les douze étiquettes.
- ep-33 : style et structure identiques à ceux de la famille, vocabulaire plus rare : cohérent avec une température
  haute. Un bémol : à C08_story, deux réponses courtes (« turned off. », « dark. ») au lieu du paragraphe recopié, comme
  les endpoints à system prompt. Deux réponses seulement, je ne change rien.

### 20 h 18 — correction de `decide --verdicts` (analysis/cli.py, merge_verdicts)
- Défaut : le verdict était lu en 2e colonne ; dans wb_verdicts_ep.csv la 2e colonne est « family » et le verdict est
  en 8e. Tous les verdicts auraient été ignorés sans message.
- Correction : la colonne est cherchée par son nom dans l'en-tête ; une ligne « INFO: verdicts read … » compte les
  verdicts lus (les lignes « NOTE: » restent réservées aux leurres contredits). Testé sur les deux formats ; les 26
  tests unitaires passent ; NEXT2 se régénère à l'identique.
- Nouveau fichier analysis/labels_ep_next3.csv : les 12 leurres + 21 NONE explicites. Sans verdicts il redonne
  exactement NEXT2 (f7ebdb83…) et ne dépend plus du registre.

### 20 h 21 — prévision pour NEXT2 AVANT son envoi (scratchpad oracle_post.py, 600 000 tirages)
- Modèle : même tirage public pour toutes les soumissions ; a priori groupe de base 5 / 6 / 7 / 8 = 0,50 / 0,25 / 0,15
  / 0,10 ; chaque étiquette juste avec probabilité 0,80 à 0,97 ; conditionné sur v1 = 0,10 et v2 = 0,35.
- Après v2 : groupe de base 5 / 6 / 7 / 8 = 0,29 / 0,30 / 0,30 / 0,11. Les étiquettes ne bougent pas (v2 ne les teste
  presque pas).
- Score public prédit pour NEXT2 : moyenne 0,47 ; quartiles 0,38 / 0,46 / 0,55 ; 5 % - 95 % : 0,23 - 0,70.
  Si les 12 étiquettes sont justes : 0,56 en moyenne (0,41 - 0,76). Si au moins une est fausse : 0,43 (0,20 - 0,67).
  Le « 0,77 / 0,61 / 0,50 » du fichier NEXT2.txt vaut pour toutes les paires et 12 étiquettes justes ; sur le public,
  sachant le 0,35 de v2, il faut attendre moins.

### 20 h 26 — candidats préparés à l'avance (demande du coordinateur), aucun n'est encore dans NEXT.csv
- (a) si NEXT2 passe sous 0,35 : décomposition « NEXT2 moins un endpoint », un fichier par endpoint ajouté depuis v2 :
  D_next2_minus_ep33 (88578a90…), D_next2_minus_ep02 (16f28d6e…), D_next2_minus_ep31 (763e2bbf…),
  D_next2_minus_ep30 (b38d7ed4…). Attendu sur toutes les paires : 0,70 / 0,57 / 0,46 chacun ; NEXT2 vaudrait
  0,67 / 0,54 / 0,45 si l'endpoint retiré est faux et 0,77 / 0,61 / 0,50 si tout est juste.
  Ordre provisoire : ep-33, ep-02, ep-31, ep-30 (à remplacer par l'ordre du stratège).
- (b) repli si NEXT2 monte sans verdict boîte blanche à 21 h 05 : B_next2_plus_ep01 (1477a760…), ep-01 ajouté au
  groupe de base de qwen_A. Indices faibles (T_mle 0,70 et excess_z −0,4 sur E1 ; « Qwen » 3 sur 3 ; mêmes réponses
  que le leurre ep-04 sur C08, C01 et pi) ; ep-17, ep-12, ep-25 ont presque le même profil : probabilité environ 0,25.
  Attendu : 0,85 / 0,68 / 0,55 si juste, environ 0,68 / 0,56 / 0,46 sinon ; rentable seulement au-dessus de 0,51.
  C'est un test par le classement, pas un gain attendu.
- Règle de décision fixée avant les verdicts (calcul metrics.py, groupe de base vrai de 5) : ajouter aux 4 leurres
  1 endpoint juste donne MCC 1,00 ; 1 endpoint faux 0,51 ; 2 dont 1 juste 0,77 ; 2 faux 0,35 ; 3 dont 1 juste 0,60 ;
  ne rien ajouter 0,74. Donc on n'ajoute un endpoint que si sa probabilité d'être la base dépasse 0,5, et jamais
  tous les BASE_OR_LURE d'une famille s'ils sont 3 ou plus.

### 20 h 30 — règles fixées AVANT l'arrivée des verdicts boîte blanche
- Probabilités des 4 leurres ajoutés depuis v2, écrites dans strategy/prior_pairs_ep.csv : ep-02 0,80 (était 0,60 :
  identité 0 sur 3 contre 27 sur 27, vocabulaire normal, seul candidat « system prompt » de qwen_A) ; ep-33 0,80
  (était 0,875 : T_mle 1,1, deux réponses atypiques à C08) ; ep-30 et ep-31 0,85.
- Traduction des verdicts en probabilité « sert exactement le modèle de base » pour les 21 endpoints sans étiquette
  (scratchpad wb2prior.py) : a priori 0,25 ; z_excess_Tadj et z_fp8_rel recentrés sur les leurres WM et TEMP de la
  famille (mêmes poids, gabarit par défaut) ; base : N(0,1) ; variante : décalage uniforme sur [0, 8] sur l'une des deux
  statistiques ; verdict VARIANT ou FP8 : plafond 0,03 ; T_mle hors [0,6 ; 0,9] : probabilité multipliée par 0,3.
- Ordre des soumissions : celui de l'outil du stratège (update puis assess / next) sur les scores réels. Un seul
  endpoint incertain par soumission. Les planA_*.csv du stratège et mes D_next2_minus_ep*.csv sont les mêmes
  partitions (sha différents à cause des noms de modèles) : seuls mes fichiers iront dans NEXT.csv.

### 20 h 28 — répétitions à mi-E2_a (environ 27 paires de réponses par endpoint) : les trois WM sont confirmés
- ep-10 (llama3) : 4 paires identiques + 3 à long préfixe commun sur 29, dont une histoire longue de 256 tokens
  rendue deux fois à l'identique ; 0 paire identique chez les dix autres endpoints de llama3. Ce n'est plus l'étiquette
  fragile.
- ep-04 (qwen_A) : 8 identiques + 7 à long préfixe sur 27 (tp_story 6 sur 10). ep-06 (qwen_B) : 8 + 3 sur 27
  (identité 3 sur 3).
- Les paires identiques restantes (ep-01 3, ep-17 1 à C08 ; ep-02 1, ep-20 1 au formulaire F20) sont sur des sondes
  à réponse quasi unique : pas un signe de watermark.
- NEXT2 régénéré à 20 h 27 : toujours f7ebdb83….

### 20 h 30 — premiers verdicts boîte blanche : llama3 (instantané s1 de 20 h 22, 24 réponses, 1 700 tokens)
- Contrôle : base cohérente (référence locale T_mle 0,70, aucun fp8 ≥ 4). Les 4 leurres sortent BASE_OR_LURE, sans
  conflit : ep-10 (excès +0,18), ep-16, ep-29, ep-31 (T_mle 1,0 ; excès +0,65 une fois T ajusté, +5,9 à 0,7).
  L'étiquette TEMP de ep-31 est donc confirmée par un second instantané.
- Non étiquetés : 6 BASE_OR_LURE (ep-05, 09, 11, 13, 15, 19) et ep-27 UNSURE (excès +3,6 à T_mle 0,8, +8,1 à 0,7 :
  meilleur suspect LoRA). Endpoint nu selon la règle pré-enregistrée : ep-13 (score 0,79), « ambigu » avec ep-19
  (1,47) et ep-09 (1,60). Puissance : un LoRA 0,3 donnerait z de 1 à 1,5, donc invisible.
- Révision de ma traduction en probabilités, faite APRÈS avoir vu ces chiffres (exploratoire, je le signale) : ma règle
  de 20 h 30 utilisait z_excess_Tadj et aurait classé ep-09 premier (0,81), parce que l'ajustement de T efface le
  petit décalage de température qu'un LoRA faible produit. L'endpoint nu tourne à T = 0,7 : j'utilise maintenant
  l'excès à 0,7 (z_excess_07), sans recentrage, et la fenêtre T_mle [0,65 ; 0,75] de la boîte blanche. Le classement
  rejoint alors la règle pré-enregistrée de la boîte blanche, écrite avant les résultats : ep-13 premier.
- Probabilités écrites dans strategy/prior_pairs_ep.csv (contre ep-10) : ep-13 0,61 ; ep-09 0,37 ; ep-19 0,36 ;
  ep-11 0,11 ; ep-15 0,10 ; ep-05 0,02 ; ep-27 0,02.
- Candidat construit : C_next2_plus_ep13 (sha 33933a09…) = NEXT2 + ep-13 dans le groupe de base de llama3.
  Attendu (toutes paires, groupe 5 / 6 / 7) : 0,85 / 0,68 / 0,55 si ep-13 est la base, environ 0,68 / 0,56 / 0,46
  sinon ; avec P = 0,61 : environ 0,78 / 0,63 / 0,51, contre NEXT2 0,77 / 0,61 / 0,50. Gain attendu faible, valeur de
  test forte (lisible 3 fois sur 4). Pas encore dans NEXT.csv : j'attends le score de 20 h 40 et assess.
