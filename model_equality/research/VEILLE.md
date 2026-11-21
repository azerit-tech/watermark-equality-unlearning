# Veille ciblée : séparer le LoRA faible sans casser les leurres

Samedi 21 novembre, soir. Aucune requête envoyée à l'API du concours pour ce document.
Lu : l'article des organisateurs (MET, Gao et al.), les deux articles que notre METHOD.md
leur attribue (AgentProv, RUT), `probes/METHOD.md`, `wb/PLAN.md`, les règles R2 et R3.
Le commit 1277819 ne contient que `REFERENCES.md` (des liens, pas de PDF).

Niveaux de preuve : fort = chiffres publiés dans un cadre proche du nôtre ;
moyen = publié mais dans un cadre différent, ou théorie solide ; faible = raisonnement
de notre part, à valider sur `ref-` avant tout usage.


## L'essentiel

1. La littérature confirme notre choix. En boîte noire, le 8 bits ne se voit pas :
   RUT annonce 0,05 à 0,13 de puissance, AgentProv dit que MET, RUT et lui-même
   acceptent tous le 8 bits, MET tombe à 0,06-0,36 sur la plupart des modèles.
   Le rejeu local avec la bonne hypothèse est la seule voie. On ne change pas de cap.

2. Pour le LoRA 0,3, l'idée la plus rentable est un filtre adapté : apprendre le LoRA
   sur les réponses de sa sœur forte (0,7 ou 1,5) de la même famille, puis estimer la
   force s de chaque endpoint. Le signal devient proportionnel à s, au lieu de s² pour
   un test générique. C'est l'idée 1.

3. Alerte sur notre statistique S1 (surprise en excès). Par calcul, S1 = -(1/T) × la
   dérivée de la log-vraisemblance par rapport à 1/T. S1 est donc exactement le test
   du score de la température. Sa moyenne vaut zéro à T_mle, par construction.
   Conséquence : « S1 > 0 à T = 0,7 » veut dire « T_mle un peu au-dessus de 0,7 ».
   Or un system prompt décale T_mle bien plus (ref-03 : 0,75 ; ref-05 : 0,90).
   Sur `ep-`, S1 seul ne distingue pas un LoRA faible d'un system prompt non retrouvé.
   Vérification en dix minutes : T_mle de ref-09 sur grille fine (pas de 0,01) sur P8.

4. Geste gratuit : le budget compte des réponses, pas des tokens. En rejeu, z croît
   comme la racine du nombre de tokens. Les sondes P8 demandent 12 à 200 tokens.
   Tester sur `ref-` si le signal tient à 256 tokens ; si oui, tout passer à 256.
   Prix : 20 s par requête, soit environ 4 h 10 pour 90 réponses sur 33 endpoints.

5. Ordre conseillé ce soir : idée 1 et idée 2 en parallèle sur `ref-` (gratuit),
   puis idées 3 et 4 (deux leurres que le banc calibre mal), le reste si le temps le permet.


## Idée 1. Filtre adapté : distiller le LoRA de la sœur forte, estimer la force

Ce qu'elle corrige : le LoRA faible, notre seule erreur du banc. Elle sépare aussi deux
forces du même LoRA, et elle ne dépend pas de la température.

Principe. À faible force, le LoRA déplace les logits de s × d (d = direction, s = force).
Un test d'énergie (divergence, énergie des résidus) mesure s² : passer de 0,7 à 0,3
divise le signal par 5,4. S1 ne voit, lui, que la part de d alignée sur la température.
Un rapport de vraisemblance le long de la bonne direction mesure s en entier :
z ≈ ρ × s × racine(N × v), où ρ est la qualité de la direction estimée. Passer de 0,7 à
0,3 ne divise z que par 2,3. C'est ce que nous faisons déjà pour fp8, où l'hypothèse
alternative est connue ; ici on la fabrique à partir de la sœur.
Toutes les sondes étant communes aux endpoints d'une famille, le LoRA distillé n'a pas
besoin de généraliser à d'autres prompts : seulement à d'autres tirages.

Gain attendu : MCC du banc de 0,73 à 1,00 si ref-09 sort à z ≥ 4 ; sur `ep-`, environ
0,08 point par LoRA faible récupéré. Probabilité de succès estimée : 50 à 60 %.
Niveau de preuve : moyen. Théorie classique (Neyman-Pearson, linéarité des petits
déplacements de poids : task arithmetic). Indices publiés : RUT détecte un LoRA dès la
première époque ; dans AgentProv, un LoRA distillé sur les sorties d'un modèle (des
choix d'outils) le reproduit sur les sondes d'entraînement (p = 1,00) mais pas ailleurs
(p = 0,013), ce qui est notre cas favorable. Aucun article ne publie ce test tel quel.

Coût : 3 à 4 h. `wb/twin/train_lora.py` et le champ `lora: {path, scale}` de
`mescore.hyp` existent déjà.
Requêtes : 0 sur `ref-` au-delà du registre ; 0 de plus sur `ep-` si la batterie est
commune à la famille. Il faut au moins 40 réponses longues de la sœur forte.

Liens : https://arxiv.org/abs/2506.06975 (RUT), https://arxiv.org/abs/2609.00052
(AgentProv, § 4.7), https://arxiv.org/abs/2212.04089 et https://arxiv.org/abs/2305.12827
(linéarité en la force).

Plan :
1. Sur `ref-`, exporter les réponses de ref-07 (1,5). Garder 20 % de réponses à part.
   Se limiter à 90 réponses pour imiter le budget `ep-`.
2. Adapter `train_lora.py` : le corpus devient (prompt, réponse) de ref-07 ; perte sur les
   tokens de réponse, avec les logits divisés par 0,7 (les réponses sont tirées à T = 0,7,
   sinon on apprend une loi trop piquée). Rang 8, peu de pas, arrêt sur les 20 % gardés.
3. Scorer les 9 endpoints et la référence locale sous base + s × LoRA, pour s dans
   {0 ; 0,1 ; 0,2 ; 0,3 ; 0,5 ; 0,75 ; 1}, avec une grille de T au pas de 0,01.
4. Par endpoint : maximum de vraisemblance conjoint (s, T). Erreur-type par bootstrap
   sur les réponses entières. z = (s_endpoint - s_référence_locale) / erreur combinée.
5. Critères, écrits avant de regarder : ref-09 à z ≥ 4 ; ref-01, 03, 05, 06, 08 et ref-02
   à |z| < 3 ; ordre s(09) < s(04) < s(07), proche de 0,2 : 0,47 : 1.
6. Refaire en n'apprenant que sur ref-04 (0,7) : c'est le cas d'une famille sans LoRA 1,5.
   Sur `ep-` : repérer dans chaque famille les endpoints qui dévient fort, distiller
   chacun, balayer la famille, regrouper par s (fusion si l'écart est sous 2 erreurs-types).

Risque principal : une persona peut se projeter sur la direction apprise. C'est ce que
le critère 5 vérifie sur ref-03 et ref-05.


## Idée 2. Choisir les sondes par simulation locale, et tout demander à 256 tokens

Ce qu'elle corrige : LoRA faible et fp8 (plus de signal par réponse), sans requête.

Principe. Nous avons les poids : pour chaque prompt candidat, on simule la base et
l'alternative (fp8 local, base + 0,3 × LoRA distillé) et on mesure d' = écart de la
statistique de décision entre les deux, en écarts-types par réponse. Le nombre de
réponses pour z = 4 vaut 32 / d'². On garde les prompts à fort d' et à d' nul sous les
leurres. C'est la tâche T9 de `wb/PLAN.md`, jamais lancée.

Gain attendu : facteur 1,5 à 3 sur z à budget égal (notre R3 l'a déjà vu sur 8 sondes).
Niveau de preuve : moyen à fort sur le principe. MET, table 13 (un seul modèle, a priori
Llama-3 8B) : la puissance contre int8 passe de 0,11 (Wikipédia anglais) à 0,24-0,27
(russe, allemand) et 1,00 (français, espagnol) selon les seuls prompts. L'article ne dit
pas pourquoi. B3IT : 5 prompts × 10 tirages battent 1 prompt × 50 ; les entrées
« à la frontière » sont 30 fois moins chères. RESF : 5 prompts optimisés pour la
sensibilité aux poids suffisent pour un LoRA minimal (98,8 %).

Coût : 2 à 3 h, surtout du GPU (le jumeau `wb/twin/` génère déjà).
Requêtes : 0 pour choisir. La validation sur `ref-` est gratuite.

Liens : https://arxiv.org/abs/2410.20247 (MET, annexes C), https://arxiv.org/abs/2602.11083
(B3IT), https://aclanthology.org/2025.emnlp-main.247/ (RESF).

Plan :
1. Vivier de 500 à 1 000 prompts : nos 142 sondes, des continuations hors anglais
   (piste MET ; C04_es est déjà dans P8), identité et persona, domaines que la sœur
   forte trahit dans ses textes, historiques à plusieurs tours.
2. Pour chacun, 30 réponses locales de 256 tokens sous : base, fp8 local, base + 0,3 ×
   LoRA distillé, base + system prompt, base à T = 1,0.
3. Calculer d' par prompt pour fp8 et pour le LoRA ; écarter les prompts où un leurre
   donne d' > 0,3 ou où moins de 80 % des réponses atteignent 200 tokens.
4. Garder 6 à 10 prompts. Vérifier sur `ref-` (10 répétitions sur ref-01, 02, 09 et les
   leurres) que le d' simulé se retrouve.
5. Remplacer `E3_lora_target` par cette liste, à 256 tokens.
6. À tester en passant sur `ref-` : un dernier message de rôle `assistant` est-il
   prolongé par le serveur ? Si oui, on fixe le début de la réponse et donc le contexte.


## Idée 3. Endpoints gloutons : test de marge argmax, puis prompts frontière

Ce qu'elle corrige : le leurre « décodage glouton », absent du banc, et le choix de
l'endpoint qui porte le groupe (base ou fp8), l'erreur la plus chère (MCC 0,45).

Principe. Sous décodage glouton, chaque token est l'argmax du vrai modèle. On rejoue le
texte sous chaque hypothèse et on compte les tokens qui ne sont pas l'argmax, pondérés
par la marge (logit max moins logit observé). Sous la bonne hypothèse il ne reste que du
bruit numérique, à marge minuscule.

Gain attendu : un endpoint glouton rattaché avec certitude, là où nous n'avons que le
prior. Niveau de preuve : fort sur le principe, moyen sur les chiffres. Token-DiFR : plus
de 98 % de tokens identiques entre fournisseur honnête et vérifieur ; 4 bits détecté en
300 tokens (AUC > 0,999) ; écarts de type fp8 en quelques milliers de tokens, soit 4 à
20 réponses de 256 tokens. Ces chiffres sont mesurés avec graine partagée ; à T = 0 la
graine est inutile, mais l'article ne donne pas de chiffre propre à ce cas. B3IT : à
T = 0, les entrées à quasi-égalité sont celles qui révèlent un changement de poids
(AUC 0,87 pour un seul pas de fine-tuning).

Coût : 2 h. `mescore` stocke déjà `rank` et `margin`.
Requêtes : 10 à 20 par endpoint glouton, jamais deux fois le même prompt.

Liens : https://arxiv.org/abs/2511.20621 (DiFR), https://arxiv.org/abs/2602.11083 (B3IT).

Plan :
1. Calibrer le bruit bénin : générer en glouton en local (bf16, fp32, deux tailles de lot),
   rejouer en croisé, relever les marges des désaccords. Seuil τ au-dessus du pire cas.
2. Statistique : violations dures (marge > τ) pour 1 000 tokens, sous base, fp8 local,
   base + s × LoRA distillé. Vérifier dans le jumeau que la bonne hypothèse donne zéro.
3. Hors ligne, chercher des prompts frontière : ceux dont les suites gloutonnes de la
   base et du fp8 local divergent dans les 30 premiers tokens.
4. Envoyer 10 à 20 de ces prompts, une fois chacun, aux endpoints déterministes.
5. Rattacher l'endpoint à l'hypothèse sans violation dure ; si aucune ne se détache,
   garder le prior (glouton = base).


## Idée 4. Watermark sans clé : déterminisme par contexte, puis déduplication

Ce qu'elle corrige : le leurre watermark laissé seul (MCC 0,72), et les faux z du rejeu
quand des réponses reviennent à l'identique.

Principe. Un watermark sans distorsion garde la loi de chaque token, donc nos tests de
rejeu restent centrés. Mais il fixe l'aléa : soit par une suite de clés réutilisée
(Kuditipudi), soit par un hachage des h derniers tokens (Gumbel). D'où nos 15 % de paires
de réponses identiques sur ref-06. Deux signatures lisibles sans clé : le nombre de
réponses distinctes plafonne ; après un même contexte de h tokens, le token suivant est
toujours le même. En boîte blanche, on connaît le taux d'accord attendu sans watermark :
la somme des p² à la température T.

Gain attendu : étiquette WM fondée sur un test, pas sur un seuil réglé à la main ; z de
rejeu corrects sur les endpoints watermarkés. Niveau de preuve : fort pour l'existence de
la signature (Gloaguen et al., trois familles de watermarks détectées en boîte noire),
moyen à notre budget : leur test compte les sorties uniques sur 1 000 requêtes.

Coût : 1 à 2 h. Requêtes : 0.

Liens : https://arxiv.org/abs/2405.20777 (Gloaguen et al.), https://arxiv.org/abs/2307.15593
(Kuditipudi et al.).

Plan :
1. Par endpoint et pour h = 1 à 5 : relever les contextes de h tokens vus dans deux
   réponses différentes ; taux d'accord du token suivant ; comparer à la somme des p².
2. Étiqueter WM si l'accord saute vers 1 à partir d'un h donné, ou si le nombre de
   réponses distinctes par prompt suit n_clés × (1 - (1 - 1/n_clés)^n).
3. Dans toutes les statistiques de rejeu, ne compter qu'une fois un couple (contexte,
   token) répété, et qu'une fois une réponse dupliquée.
4. Vérifier sur ref-06 : étiquette posée, et z de fp8 et de s (idée 1) nuls après
   déduplication. Vérifier qu'aucun autre `ref-` n'est étiqueté.


## Idée 5. System prompt et persona : les expliquer avant de tester les poids

Ce qu'elle corrige : un leurre de contexte pris pour un changement de poids par le rejeu.
C'est le piège documenté : sous system prompt caché, MET se trompe dans 67 % des cas et
RUT (boîte blanche) dans 53 %, selon AgentProv.

Gain attendu : condition pour que les idées 1 et 2 soient utilisables sur `ep-`.
Niveau de preuve : moyen. Prompt Detective décide si un system prompt donné est utilisé,
avec environ 5 % de faux négatifs et aucun faux positif ; en boîte blanche nous avons
mieux, la vraisemblance exacte.

Coût : 2 à 3 h. Requêtes : 0 à 3 par endpoint suspect.

Liens : https://arxiv.org/abs/2609.00052 (AgentProv), https://arxiv.org/abs/2502.09974
(Prompt Detective), https://arxiv.org/abs/2311.13647 (inversion de prompt).

Plan :
1. Candidats : texte obtenu par fuite (D03), system prompt par défaut du gabarit, system
   prompt vide, et une bibliothèque de 50 consignes et personas génériques.
2. Scorer chaque endpoint suspect sous base + chaque candidat (`system_prompt` de
   `mescore.hyp`). Pour une persona tirée à chaque requête : maximum par réponse.
3. Si le meilleur candidat ramène T_mle à 0,70 et s à zéro : leurre confirmé, fusion.
4. Sinon, diluer : un historique fabriqué de 10 à 15 tours (800 tokens, l'API accepte
   32 messages) avant la sonde. Mesurer sur ref-03 et ref-05 si T_mle et s reviennent
   aux valeurs de ref-01. Si oui, tirer la réserve `ep-` sur ces sondes diluées.
5. Tant qu'un endpoint garde un écart « de contexte » inexpliqué, ne pas le déclarer
   LoRA sur la seule foi de S1.


## Idée 6. Test du score générique, pour une famille sans sœur forte

Ce qu'elle corrige : un LoRA faible isolé, sans grand frère pour donner la direction.

Principe. Le test localement le plus puissant contre un petit déplacement de poids est
le test du score : la moyenne du gradient de la log-vraisemblance. S1 en est la
composante « température ». On ajoute les autres : par token, le résidu (token observé
moins probabilités) ramené dans l'espace caché par la matrice de sortie, croisé avec
l'état caché. On projette sur 10 à 20 directions apprises sur des LoRA de substitution
(le jumeau en entraîne déjà), on retire la direction température, on teste contre la
référence locale par permutation.

Gain attendu : incertain, 30 % de chances d'atteindre z ≥ 4 sur ref-09.
Niveau de preuve : faible à moyen. Les tests à classifieur et à noyau appris sont les
plus puissants à petit n quand les features sont bonnes ; aucun résultat publié sur un
LoRA faible vu par ses seuls tirages.

Coût : 4 à 5 h. Requêtes : 0.

Liens : https://arxiv.org/abs/1610.06545 (tests à classifieur),
https://arxiv.org/abs/2002.09116 (noyaux profonds), https://arxiv.org/abs/2306.08777
(MMD-FUSE, sans découpage des données).

Plan :
1. Faire sortir à `mescore` les états cachés de la dernière couche (ou les recalculer).
2. Feature par réponse : somme sur les tokens de (résidu ramené) ⊗ (état caché projeté
   en 16 dimensions), divisée par le nombre de tokens.
3. Directions : composantes principales des features de 4 à 6 LoRA de substitution.
4. Statistique de Hotelling contre 300 réponses locales de la base, aux mêmes sondes ;
   p-valeur par permutation de réponses entières.
5. Valider sur `ref-` avec les critères de l'idée 1. Abandonner si ref-04 (0,7) n'y sort pas.


## Idée 7. Un second avis à la RUT, calibré sur notre référence locale

Ce qu'elle corrige : rien de nouveau ; elle protège d'un faux positif de notre z par un
test de forme différente. Gain attendu : faible. Niveau de preuve : fort sur ce que RUT
vaut (meilleur que MMD et KS à budget égal, mais 0,05 à 0,13 sur le 8 bits).

Coût : 1 h. Requêtes : 0. Lien : https://arxiv.org/abs/2506.06975

Plan :
1. Score par réponse : moyenne du log du rang de chaque token sous la base (déjà stocké).
2. Rang de chaque réponse d'endpoint parmi les 30 réponses locales au même prompt,
   générées à la température estimée de l'endpoint ; égalités tirées au sort.
3. Test de Cramér-von Mises d'uniformité des rangs, une p-valeur par endpoint.
4. Ne l'employer ni sur un endpoint étiqueté system prompt, ni pour séparer fp8.


## Idée 8. Dépenser la réserve par test séquentiel

Ce qu'elle corrige : les paires grises (2 < z < 4), avec moins de réponses.
Gain attendu : 30 à 40 % de réponses économisées sur ces paires. Niveau de preuve :
moyen (théorie de Wald et de Chernoff ; B3IT montre que la répartition du budget compte).

Coût : 1 h. Requêtes : la réserve seulement. Lien : https://arxiv.org/abs/2602.11083

Plan :
1. Lister les paires grises qui touchent le candidat base.
2. Pour chacune, prendre la sonde au plus fort d' simulé (idée 2) pour l'hypothèse en jeu.
3. Envoyer par lots de 5 ; cumuler le rapport de vraisemblance ; s'arrêter à ±4,6
   (1 % d'erreur de chaque côté) ou à 5 réponses de la fin du budget.
4. Sans franchissement : trancher à probabilité 0,5, comme le veut le coût symétrique.


## Ce qui ne marchera pas dans nos contraintes

- Récupérer logits ou logprobs depuis l'API (Carlini et al., Finlayson et al.) : ces
  attaques exigent `logit_bias` ou `logprobs`. Notre API n'accepte que `messages` et
  `max_tokens`. Estimer une loi par tirages : 100 tirages donnent ±0,05 sur une
  probabilité, alors que fp8 la déplace de moins que cela.
- Suivi par logprobs (LT) et « précision du modèle en 20 logprobs » : il faut des logprobs.
- MET tel quel : 250 échantillons par endpoint, 0,06 à 0,36 de puissance sur int8 pour
  5 modèles sur 8, et puissance 1,00 contre un simple changement de température. Il
  scinde tous nos leurres et rate nos cibles. À garder comme test brut seulement.
- RUT tel quel : aveugle au 8 bits, 53 % de faux positifs sous system prompt caché.
- AgentProv tel quel : 1 000 requêtes par endpoint, accepte le 8 bits, et un modèle de
  360 M ne tient pas le format d'appel d'outils. Son canal du compte de tokens est fermé
  (pas de champ `usage`).
- B3IT tel quel : pensé pour T = 0 et le premier token. À T = 0,7, une réponse donne un
  seul tirage. Seul son principe survit (idées 2 et 3).
- Token-DiFR à température non nulle : il faut la graine du serveur.
- Fingerprinting par classifieur (LLMmap, Idiosyncrasies, Stemma) : il reconnaît les
  familles, déjà acquises par la troncature. Idiosyncrasies est au niveau du hasard entre
  deux décodages du même modèle, et Cai et al. notent que ces méthodes ratent la
  quantification. Rien à gagner sur les 90 % du score.
- Tests à deux échantillons « puissants à petit n » en boîte noire (noyaux appris,
  classifieur sur embeddings) : avec moins de 100 réponses, ils ne font pas mieux que la
  vraisemblance exacte que le rejeu nous donne déjà.
- Détecter un watermark à liste verte de clé inconnue (Gloaguen et al.) : plusieurs
  milliers de requêtes. Voler la clé (Jovanović et al.) : dizaines de milliers.
- Inverser le system prompt depuis les sorties (Morris et al.) : il faut entraîner un
  modèle d'inversion. Pas ce soir ; fuite et bibliothèque suffisent (idée 5).
- Attestation matérielle (Cai et al.) : hors sujet.


## Sources

- Model Equality Testing, Gao, Liang, Guestrin, ICLR 2025 : https://arxiv.org/abs/2410.20247
- AgentProv, Wang, Zhao, Backes, Boenisch, Dziedzic, EMNLP 2026 : https://arxiv.org/abs/2609.00052
- RUT, Zhu et al., ICLR 2026 : https://arxiv.org/abs/2506.06975
- Are You Getting What You Pay For?, Cai et al., NeurIPS 2025 : https://arxiv.org/abs/2504.04715
- B3IT, Token-Efficient Change Detection in LLM APIs, Chauvin et al., ICML 2026 : https://arxiv.org/abs/2602.11083
- Log Probability Tracking of LLM APIs, Chauvin et al., ICLR 2026 : https://arxiv.org/abs/2512.03816
- Extracting Model Precision from 20 Logprobs, ICLR 2026 : https://www.iclr.cc/virtual/2026/poster/10012078
- DiFR, Inference Verification Despite Nondeterminism : https://arxiv.org/abs/2511.20621
- Ventor-QTest (surprisal centré, la même statistique que notre S1) : https://arxiv.org/abs/2608.16391
- RESF, Hu et al., EMNLP 2025 : https://aclanthology.org/2025.emnlp-main.247/
- ESF, Bai et al., Findings of ACL 2025 : https://aclanthology.org/2025.findings-acl.546
- Black-Box Detection of Language Model Watermarks, Gloaguen et al., ICLR 2025 : https://arxiv.org/abs/2405.20777
- Robust Distortion-free Watermarks, Kuditipudi et al. : https://arxiv.org/abs/2307.15593
- Watermark Stealing, Jovanović et al. : https://arxiv.org/abs/2402.19361
- Prompt Detective, Levin et al. : https://arxiv.org/abs/2502.09974
- Language Model Inversion, Morris et al. : https://arxiv.org/abs/2311.13647
- Stealing Part of a Production Language Model, Carlini et al. : https://arxiv.org/abs/2403.06634
- Logits of API-Protected LLMs Leak Proprietary Information, Finlayson et al. : https://arxiv.org/abs/2403.09539
- Stealing the Decoding Algorithms of Language Models, Naseh et al. : https://arxiv.org/abs/2303.04729
- LLMmap, Pasquini et al. : https://arxiv.org/abs/2407.15847
- Idiosyncrasies in Large Language Models, Sun et al., ICML 2025 : https://arxiv.org/abs/2502.12150
- Stemma : https://arxiv.org/abs/2607.25880
- Classifier Two-Sample Tests, Lopez-Paz et Oquab : https://arxiv.org/abs/1610.06545
- Deep Kernels for Two-Sample Tests, Liu et al. : https://arxiv.org/abs/2002.09116
- MMD-FUSE, Biggs, Schrab, Gretton : https://arxiv.org/abs/2306.08777
- Task arithmetic, Ilharco et al. : https://arxiv.org/abs/2212.04089 ; dans l'espace tangent, Ortiz-Jimenez et al. : https://arxiv.org/abs/2305.12827

Limites de cette veille : les PDF d'AgentProv et de RESF n'ont pas pu être lus ici (pas
d'outil PDF) ; AgentProv a été lu dans sa version HTML, RESF et ESF dans leurs seuls
résumés. Ce qui est dit de Stemma, du vol de clé de watermark et des deux articles sur la
linéarité des LoRA vient de résumés ou de mémoire, sans relecture. Les estimations de
gain, de probabilité de succès et d'heures sont les nôtres, pas des mesures. L'identité
entre S1 et le score de la température est un calcul de notre part, à vérifier sur ref-09.
