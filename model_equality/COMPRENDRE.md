# Model Equality Testing : ce que nous avons fait

État au samedi 21 novembre, 18 h.

Ce document présente notre travail sur la tâche 2, dans l'ordre d'un exposé.

Trois mots disent où en est chaque élément. « Mesuré » : nous l'avons
observé sur le vrai serveur. « En cours » : cela tourne ou s'écrit en ce
moment. « Prévu » : c'est décidé, pas encore fait.


## 1. Le décor

Un modèle de langue est un énorme tableau de nombres. On appelle ces nombres
les poids. C'est ce tableau qui transforme une question en réponse.

Un modèle ne lit pas des mots, il lit des tokens. Un token est un morceau de
texte : un mot court, un bout de mot, un signe de ponctuation.

Le programme qui découpe le texte en tokens s'appelle le tokenizer. Chaque
lignée de modèles a le sien, et deux tokenizers ne découpent pas la même
phrase de la même façon.

Un modèle écrit sa réponse un token à la fois. À chaque pas, il donne une
probabilité à chaque token possible. Puis il en tire un au sort selon ces
probabilités.

C'est pourquoi la même question posée deux fois donne deux réponses
différentes.

Ce tirage au sort est au cœur de toute notre méthode. Nous ne voyons jamais
les probabilités. Nous ne voyons que les tirages, et nous devons remonter
aux probabilités à partir d'eux.

Un endpoint est un guichet derrière lequel tourne un modèle. On lui envoie
un message, on lit sa réponse, rien d'autre.

On dit que le modèle est en boîte noire. On ne voit ni ses poids ni ses
probabilités.


## 2. La question posée

Les organisateurs ont ouvert 33 guichets, nommés ep-01 à ep-33. Plusieurs
guichets peuvent cacher le même modèle. Il faut dire lesquels.

La réponse a deux niveaux.

Le premier niveau est la famille. Une famille regroupe les guichets
construits à partir du même modèle de départ, qu'on appelle le modèle de
base.

Le second niveau est le modèle. Deux guichets ont le même modèle quand leurs
poids sont exactement les mêmes.

### Ce qui fait un modèle différent

Trois choses font un modèle différent, parce que les poids eux-mêmes
changent.

1. Le modèle de base. Deux modèles de base différents sont deux tableaux de
   nombres sans rapport. Ils ne sont même pas dans la même famille.

2. La quantification. C'est un arrondi des poids, fait pour gagner de la
   mémoire. Un poids qui valait 0,73718 devient par exemple 0,74. Chaque
   nombre bouge très peu, donc le modèle répond presque pareil. Mais ce
   n'est plus le même tableau. Le cas qui nous occupe s'appelle fp8 : chaque
   poids est stocké sur 8 bits, ce qui ne laisse que 256 valeurs possibles.

3. Le LoRA. C'est un petit correctif appris après coup, qu'on ajoute aux
   poids. On l'applique avec une force réglable : poids final = poids de
   base + force × correctif. À force 0,3 le modèle change à peine. À force
   1,5 il change beaucoup. Le même correctif à deux forces différentes
   compte pour deux modèles différents.

### Ce qui est un leurre

Trois autres choses changent les réponses sans toucher aux poids. Ce sont
des leurres. Deux guichets qui ne diffèrent que par un leurre cachent le
même modèle, et il faut les mettre ensemble.

1. Le décodage. C'est la façon de tirer le token suivant. Le réglage
   principal est la température. À température basse, le modèle prend
   presque toujours le token le plus probable. À température haute, il se
   disperse davantage. Les probabilités calculées par le modèle n'ont pas
   changé. Seule la façon d'y piocher a changé.

2. Le system prompt. C'est une consigne cachée, placée par le serveur avant
   notre message. Quand cette consigne donne un rôle au modèle, on parle de
   persona. Le modèle est le même, mais il ne lit pas le même texte.

3. Le watermark. C'est un tatouage invisible du texte. À chaque pas, le
   serveur donne un petit bonus à une partie des tokens, choisie par une
   règle secrète. Le texte reste naturel. Mais celui qui connaît la règle
   peut reconnaître que le texte vient de ce serveur.

### Pourquoi c'est difficile

Un leurre peut changer beaucoup le texte. Une persona se voit à l'œil nu.

Une vraie différence peut le changer très peu. Un arrondi des poids ne se
voit pas à la lecture.

Il faut donc ignorer de grands écarts et détecter de tout petits écarts.


## 3. Le score

Le score ne juge pas les guichets un par un. Il juge les paires de guichets.
Pour chaque paire, la question est : ces deux-là sont-ils ensemble ou
séparés ?

Prenons quatre guichets A, B, C et D. Ils forment six paires : AB, AC, AD,
BC, BD et CD. Avec 33 guichets, cela fait 528 paires.

Supposons que la vérité soit la suivante : A et B cachent le même modèle, C
est seul, D est seul. Une seule paire est « ensemble », la paire AB. Les
cinq autres sont « séparées ».

La mesure utilisée s'appelle le MCC. C'est une note d'accord entre nos
paires et les vraies. Elle vaut 1 quand tout est juste. Elle vaut 0 quand
notre réponse ne contient aucune information.

Voici ce qu'elle donne sur l'exemple.

1. Nous répondons « A et B ensemble, C seul, D seul ». Tout est juste : le
   MCC vaut 1.

2. Nous répondons « tous différents ». Nous n'avons trouvé aucune paire : le
   MCC vaut 0. Ne rien regrouper ne rapporte rien.

3. Nous répondons « A, B et C ensemble, D seul ». Nous annonçons trois
   paires ensemble, dont une seule est vraie : le MCC vaut 0,45. Un seul
   guichet mal placé coûte plus de la moitié de la note.

Le score final combine deux MCC.

score = 0,1 × MCC des familles + 0,9 × MCC des modèles

Le MCC des modèles ne compte que les paires de guichets d'une même famille.

90 % de la note se joue donc à l'intérieur des familles. Il faut séparer la
base de sa version arrondie, et ne pas séparer la base de ses leurres.

Nous avons fait le même calcul à la taille du banc d'entraînement, qui
compte neuf guichets. Un seul guichet mal placé y fait tomber le MCC vers
0,72.

Le score public affiché pendant la compétition ne porte que sur 30 % des
paires environ. Le classement final se fait sur les 70 % restants.

On ne peut soumettre qu'une fois toutes les 30 minutes, soumissions ratées
comprises.


## 4. Les contraintes

Chaque guichet noté accepte 100 réponses, pour toute la compétition. C'est
très peu. Une réponse perdue, c'est 1 % du budget de ce guichet.

À côté, les organisateurs donnent un banc d'entraînement. Ce sont neuf
guichets nommés ref-01 à ref-09, utilisables presque sans limite, dont la
vérité est fournie. C'est là que nous mettons la méthode au point.

Le serveur impose des limites à chaque requête.

1. Une réponse fait au plus 256 tokens.

2. Un message fait au plus 1 024 tokens.

3. On ne peut envoyer que 4 requêtes à la fois.

4. On ne peut pas envoyer de system prompt nous-mêmes.

Nous avons mesuré le temps de réponse. Chaque requête coûte 6,2 secondes
fixes, plus 0,053 seconde par token écrit. Une réponse d'un mot prend 6,4
secondes. Une réponse de 256 tokens prend 20 secondes.

La conséquence est contre-intuitive. Une réponse d'un mot coûte une réponse
du budget, et presque autant de temps qu'une longue. Mais elle ne contient
qu'un seul tirage.

Une réponse longue bien conçue contient plus de cent tirages. Par minute de
serveur, elle rapporte 40 fois plus.

Nous demandons donc des réponses longues, faites de nombreux tirages. Ce
choix structure toute la suite.

Dernier chiffre : utiliser les 3 300 réponses des guichets notés demande
entre 3 et 5 heures de serveur. Le temps est une contrainte aussi dure que
le budget.


## 5. L'outillage

Avant toute mesure, nous avons construit trois outils.

1. Le client. C'est le programme qui parle au serveur. Il écrit chaque envoi
   et chaque réponse dans un registre sur disque. Si la connexion tombe, il
   reprend là où il s'était arrêté, sans rien redemander. Un plafond par
   guichet l'empêche de dépenser le budget par accident. Il a 31 tests. Son
   premier essai réel a fait 218 requêtes sans erreur.

2. Le scoreur local. C'est notre copie du calcul de la note. Sur 45 réponses
   d'essai, il donne la même sortie que le programme des organisateurs, au
   caractère près. Nous pouvons donc noter une méthode sur le banc sans rien
   soumettre.

3. La chaîne de soumission. Elle vérifie le fichier de réponses, respecte le
   délai de 30 minutes, et envoie la même requête que le script des
   organisateurs. Elle a été auditée puis corrigée, et elle a 49 tests.
   Aucune soumission réelle n'a encore été faite.


## 6. Reconnaître le modèle de base

Pour former les familles, il faut savoir quel modèle de base se cache
derrière chaque guichet.

Nous avons trouvé un moyen de le lire dans les réponses déjà reçues, sans
dépenser une requête de plus. Nous l'appelons l'empreinte par troncature.

### L'idée

Quand on demande au plus 64 tokens, le serveur coupe la réponse au 64e token
exactement.

Or ce compte est fait avec le tokenizer du modèle caché. C'est lui qui
trahit le modèle.

### Un exemple

Prenons une réponse coupée en pleine phrase. Nous la recomptons avec le
tokenizer de plusieurs modèles connus.

Avec le bon tokenizer, nous trouvons 64, pile la limite.

Avec un autre, qui découpe les mots autrement, nous trouvons par exemple 59
ou 71. Ces deux chiffres sont une illustration.

71 est impossible : le serveur n'aurait jamais laissé passer plus de
64 tokens. Un seul dépassement suffit donc à éliminer un candidat.

59 n'élimine rien à lui seul. Mais un mauvais candidat ne tombe pile sur la
limite que de temps en temps, par chance. Le bon y tombe presque à chaque
fois.

### Le résultat

Nous avons préparé un atlas de 48 modèles candidats, regroupés par lignée.
Une lignée est un groupe de modèles qui partagent le même tokenizer.

Mesuré sur le banc : sur 370 réponses, 163 étaient coupées. Avec le
tokenizer de la lignée SmolLM, 158 tombent pile sur la limite et aucune ne
dépasse.

La deuxième meilleure lignée n'en aligne que 44, et elle dépasse 8 fois.

Une confirmation indépendante est arrivée d'un guichet du banc. Il a récité
mot pour mot la consigne par défaut de SmolLM2 : « You are a helpful AI
assistant named SmolLM, trained by Hugging Face ».

Le modèle du banc est donc un SmolLM2.

### Les limites

La méthode donne la lignée, pas la taille. SmolLM2 existe en trois tailles
qui partagent le même tokenizer. Nous n'avons pas encore tranché laquelle.

Certaines lignées comptent pareil sur de l'anglais, par exemple Llama 3 et
OLMo 2. Il faut alors des réponses en chiffres ou en chinois pour les
séparer.

État : mesuré sur le banc. Prévu sur les 33 guichets notés, dès leurs
premières réponses.


## 7. La méthode en boîte noire

C'est le cœur du travail. Elle n'utilise que les réponses des guichets.

### Pourquoi répéter le même contexte

Rappel : nous ne voyons que des tirages, jamais les probabilités.

Imaginons deux pièces de monnaie. L'une tombe sur pile 50 fois sur 100,
l'autre 52 fois sur 100. Pour les distinguer, dix lancers ne servent à rien.
Il en faut plusieurs milliers de chacune.

Un arrondi des poids fait la même chose à un modèle. Dans une situation
donnée, la probabilité de chaque token bouge d'un petit écart.

Plus l'écart est petit, plus il faut de tirages. Un écart dix fois plus
petit demande cent fois plus de tirages.

Notre ordre de grandeur est 50 divisé par le carré de l'écart. Un écart de
0,1 correspond à une probabilité qui change d'environ 10 % de sa valeur. Il
demande de l'ordre de 5 000 tirages.

Le point décisif : ces tirages doivent venir de la même situation.

On appelle contexte le texte que le modèle a sous les yeux au moment de
choisir un token. L'écart dû à l'arrondi change d'un contexte à l'autre, en
taille et en sens. Si on mélange des contextes différents, les écarts se
compensent et on ne voit rien.

Cent questions différentes donnent cent contextes vus une seule fois. Cela
ne montre rien.

Il faut au contraire revoir le même contexte des milliers de fois, avec
seulement cent réponses.

### Les sondes : flux et formulaires

Une sonde est une question que nous posons à tous les guichets.

Deux formes de sondes font tenir beaucoup de tirages comparables dans une
seule réponse.

1. Le flux. La consigne est du type « écris 120 chiffres au hasard ». Chaque
   chiffre écrit est un tirage. Le contexte utile se résume au chiffre
   précédent. Il n'y a donc que dix contextes, et chacun revient sans cesse.
   Une réponse donne plus de cent tirages.

2. Le formulaire. C'est une liste de 20 à 30 petites questions à choix
   serré, où le modèle hésite vraiment. Chaque ligne est un tirage. La ligne
   numéro 7 est le même contexte d'une réponse à l'autre.

Notre banque compte 138 sondes. Le banc sert à choisir celles qui séparent
le mieux les vrais modèles.

### La table de comptage

Pour chaque guichet, on compte. Après le chiffre 3, combien de fois a-t-il
écrit 0, combien de fois 1, et ainsi de suite.

Cela donne une table, avec une ligne par contexte et une colonne par option.

Comparer deux guichets, c'est comparer leurs deux tables. Nous faisons cette
comparaison de trois façons, de plus en plus sélectives.

### Première mesure : l'écart brut

La première mesure s'appelle T_raw. Elle dit simplement si les deux tables
diffèrent, par un test statistique classique sur des comptages.

Elle voit tout. Un arrondi des poids la fait monter, mais une température
différente aussi.

Seule, elle séparerait tous les leurres. C'est exactement l'erreur à ne pas
faire.

C'est la limite de la méthode de référence citée par les organisateurs :
elle détecte aussi la température et le watermark. Nous la gardons comme
test brut, pas comme verdict.

### Deuxième mesure : retirer la température

La température a un effet très particulier. Elle ne change pas l'ordre des
préférences du modèle. Elle les étire ou les écrase toutes ensemble.

Un exemple. Après le chiffre 3, le modèle préfère le 7 au 5 avec un certain
écart, et le 5 au 1 avec un écart deux fois plus petit. Si on monte la
température, les deux écarts rétrécissent, mais du même facteur. Le premier
reste le double du second.

Ce facteur est le même dans tous les contextes et pour toutes les options.

La mesure T_temp exploite cela. On cherche le facteur unique qui rapproche
le mieux les deux tables. Puis on regarde ce qui reste.

Si les deux guichets ne diffèrent que par la température, il ne reste rien.

Si les poids diffèrent, il reste un écart. Un arrondi pousse certaines
options et pas d'autres, sans facteur commun.

On dit que T_temp est invariante à la température. Elle donne le même
résultat quelle que soit la température de chaque guichet.

Le facteur trouvé est lui-même une information. S'il est loin de 1, nous
savons que ce guichet a une température différente, et nous l'étiquetons
ainsi.

### Troisième mesure : retirer le watermark

Le watermark le plus courant est dit à liste verte. À chaque pas, une règle
secrète désigne une partie des tokens, la liste verte. Le serveur leur
ajoute un bonus.

La liste verte ne dépend que des derniers tokens écrits.

Conséquence utile : si on compare des contextes qui se terminent par les
mêmes tokens, la liste verte est la même. Le watermark ajoute alors un bonus
fixe à certaines options, toujours les mêmes.

La mesure T_inv retire cet effet par un double centrage. On retire de la
table la moyenne de chaque ligne, puis la moyenne de chaque colonne.

Le bonus fixe d'une option part avec la moyenne de sa colonne. Il ne reste
que ce qui dépend à la fois du contexte et de l'option. Un watermark n'a
aucune prise là-dessus.

Cette mesure a un prix. En retirant plus de choses, elle garde moins de
signal et devient moins sensible.

Nous ne l'utilisons donc que si la signature d'un watermark apparaît. Cette
signature est nette : les écarts par option se rangent en deux paquets, les
tokens favorisés et les autres.

### Le cas du system prompt

Un system prompt ne se retire pas par un calcul. Il change le texte que le
modèle lit, donc il peut tout changer.

Nous avons deux prises sur lui.

1. Il fuit. En posant les bonnes questions, le modèle récite parfois sa
   consigne.

2. Il se dilue. Quand notre message est long et très contraignant, une
   consigne placée loin devant pèse de moins en moins.

Un guichet soupçonné d'avoir un system prompt est donc comparé aux autres
sur les sondes longues seulement.

Un écart présent sur les sondes courtes et absent sur les longues désigne un
system prompt.

### Calibrer par permutation

Une mesure brute ne dit rien seule. Un écart de 300, est-ce beaucoup ? Il
faut savoir ce que donnerait la même mesure entre deux guichets identiques.

Les tables des manuels de statistique ne conviennent pas. Elles supposent
des tirages indépendants. Or les chiffres d'une même réponse dépendent les
uns des autres, parce que le modèle se relit en écrivant.

Nous fabriquons donc la référence nous-mêmes.

On prend les réponses des deux guichets. On les mélange comme un jeu de
cartes. On les redistribue en deux tas au hasard. On refait la mesure sur
ces deux tas.

Par construction, ces deux tas n'ont aucune vraie différence. La mesure
obtenue montre ce que le hasard seul produit.

On recommence 500 fois. On obtient 500 valeurs de hasard pur.

On exprime alors la vraie mesure dans cette unité. C'est le score z. Un z de
0 veut dire « comme le hasard ». Un z de 4 veut dire que l'écart observé
dépasse le hasard de quatre fois son ampleur habituelle.

Un détail compte beaucoup : on mélange des réponses entières, jamais des
chiffres isolés. C'est ce qui respecte la dépendance à l'intérieur d'une
réponse.

Nos repères par défaut sont les suivants. En dessous de 2, rien ne distingue
les deux guichets. Au-dessus de 4, la différence est nette. Entre les deux,
la paire est douteuse.

Un contrôle accompagne chaque sonde. On coupe les réponses d'un même guichet
en deux moitiés et on les compare. Le z doit être proche de 0. Sinon la
sonde est instable et on l'écarte.

Cette unité a un dernier avantage. Elle se mesure toujours par rapport au
bruit du guichet lui-même. Un seuil réglé sur le banc garde donc un sens sur
les guichets notés.

### Lire les trois z ensemble

Pour une paire, nous avons trois z : le brut, celui sans température, celui
sans température ni watermark. Leur motif donne le diagnostic.

1. Les trois sont petits. Rien ne distingue les deux guichets : même modèle.

2. Le z brut est grand, le z sans température est petit. La différence
   s'explique entièrement par la température : même modèle.

3. Le z sans température est grand, le z sans watermark est petit, et la
   signature en deux paquets est là. C'est un watermark : même modèle.

4. L'écart est grand sur les sondes courtes et disparaît sur les longues.
   C'est un system prompt : même modèle.

5. L'écart résiste à tout. Ce sont les poids qui diffèrent : modèles
   différents.

### Décider par partition

Il reste à transformer ces verdicts par paire en groupes.

Les verdicts par paire peuvent se contredire. On peut trouver A pareil que
B, B pareil que C, et pourtant A différent de C.

Or la réponse à rendre est faite de groupes. Dans un groupe, tout le monde
est pareil que tout le monde. Un découpage en groupes s'appelle une
partition.

Nous ne tranchons donc pas paire par paire. Chaque paire reçoit un poids. Il
est positif si les deux guichets se ressemblent, négatif s'ils diffèrent,
proche de zéro si on ne sait pas.

Puis on cherche, dans chaque famille, la partition qui réunit le plus de
poids positif et sépare le plus de poids négatif. Une paire douteuse est
ainsi tranchée par ses voisines.

Quand les mesures ne tranchent pas, nous nous appuyons sur une hypothèse
tirée du banc et de la consigne. Dans chaque famille, il y a un groupe formé
du modèle de base et de ses leurres. Chaque vraie variante est seule.

Cette hypothèse départage les cas douteux. Elle ne passe jamais devant une
mesure nette.

### État de la boîte noire

Le code d'analyse est écrit. Il est testé de bout en bout sur des données
synthétiques, avec 24 tests. Ce sont des réponses que nous avons fabriquées
nous-mêmes et dont nous connaissons la vérité.

Il n'est pas encore validé sur les vraies réponses du banc. C'est l'objet de
la soirée.

Notre propre estimation reste prudente. Nous donnons environ 60 % de chances
aux flux et aux formulaires de fonctionner comme prévu. Nous donnons environ
40 % de chances à la détection du fp8 en boîte noire.


## 8. La méthode en boîte blanche, sur GPU

La boîte noire devine les probabilités à partir de tirages. La boîte blanche
les lit directement.

### Le principe

Elle part du résultat de la section 6. Si le modèle de base d'un guichet est
public, nous pouvons le télécharger et le faire tourner sur le
supercalculateur auquel nous avons accès.

Nous avons alors une copie locale, dont nous voyons tout.

Nous ne demandons pas à cette copie d'écrire. Nous lui faisons relire une
réponse du guichet, token par token.

À chaque pas, nous lui demandons : quelle probabilité aurais-tu donnée au
token que le guichet a réellement écrit ? Cette technique s'appelle le
teacher forcing.

Si la copie est le bon modèle, elle trouve la réponse plausible. Les tokens
écrits sont ceux qu'elle attendait.

Si le guichet cache une variante, la copie est un peu surprise, un peu
partout.

### Comparer des hypothèses

Nous préparons plusieurs copies, une par hypothèse : la base, la base
arrondie en fp8, la base avec la consigne retrouvée, et ainsi de suite.

La copie qui trouve les réponses du guichet les plus plausibles désigne la
bonne hypothèse. Cette comparaison s'appelle un rapport de vraisemblance.

L'avantage est grand. Chaque token de chaque réponse devient une mesure,
même dans du texte libre. Il n'y a plus besoin de répéter un contexte.

### Ce qu'elle peut et ne peut pas

Ce qu'elle peut certifier : un LoRA, un arrondi fort sur 4 bits, un system
prompt que nous avons retrouvé et rejoué.

Sa limite est le fp8. L'arrondi sur 8 bits bouge si peu les probabilités que
même la lecture directe peut manquer de signal.

Le signal grandit comme la racine carrée du nombre de tokens. Nos 100
réponses de 256 tokens font 25 600 tokens.

Avec ce volume, il faut que le fp8 déplace d'au moins 0,05 en moyenne les
scores internes que le modèle donne aux tokens. Ces scores s'appellent les
logits. En dessous, nous n'atteignons pas un z de 4.

Nous ne savons pas encore si le fp8 atteint ce seuil. C'est la première
chose à mesurer, sur le guichet fp8 du banc.

Autre limite : la méthode ne vaut que pour les familles dont le modèle de
base est public et identifié.

### État de la boîte blanche

Le plan est écrit. L'outil de rejeu est en cours d'écriture. Rien n'est
encore mesuré.

Nous estimons à une chance sur deux que la chaîne complète tourne à temps.
Le gain espéré va de 0,05 à 0,25 point de score. Ce sont des estimations,
pas des mesures.


## 9. Ce que le banc nous a appris

Le banc compte neuf guichets dont la vérité est donnée.

Un guichet sert le modèle nu. Quatre lui ajoutent un leurre chacun : un
system prompt, une persona, un watermark, une température de 1,0. Un guichet
sert la version fp8. Trois servent le même LoRA, aux forces 0,3, 0,7 et 1,5.

La bonne réponse sur le banc est donc un groupe de cinq guichets, et quatre
guichets seuls.

Voici ce que nous y avons mesuré.

1. Tout est échantillonné. Aucun guichet ne répète une réponse, même le
   modèle nu. Nous avons donc partout des tirages à comparer.

2. Trois écarts se voient à l'œil nu : la persona, la température de 1,0 et
   le LoRA à force 1,5. Deux de ces trois sont des leurres.

3. Le fp8 et le LoRA à force 0,3 ne se distinguent pas du modèle nu à la
   lecture. Ce sont pourtant deux vrais modèles différents. C'est sur eux
   que le score se joue.

4. Le system prompt simple ne se voit pas non plus à la lecture.

5. Le serveur ne donne aucune aide. Il ne renvoie pas le nombre de tokens.
   Il ne dit pas si une réponse a été coupée.

6. Quand un message est trop long, le serveur le refuse avec le même texte
   pour tous les guichets, sans aucun chiffre. Ce refus n'est pas décompté
   du budget. Il reste une piste prévue : chercher à quelle longueur exacte
   chaque guichet commence à refuser. Une consigne cachée prend de la place
   et avancerait ce seuil.

7. Le modèle du banc est un SmolLM2, identifié par l'empreinte de la section
   6.


## 10. Où nous en sommes

Samedi, 18 h.

Mesuré : le débit du serveur, le comportement des neuf guichets du banc,
l'identité du modèle du banc.

Construit et testé : le client, le scoreur, la chaîne de soumission, le code
d'analyse.

En cours : la collecte sur le banc. Le registre contient 706 réponses du
banc. La vague 0 se termine. La vague 1, environ 1 300 réponses, s'enchaîne
seule pendant près de deux heures.

Pas encore touché : les 33 guichets notés. Aucune de leurs 3 300 réponses
n'est dépensée. Aucune soumission n'est faite.

À 17 h 23, le meilleur score public du tableau était 0,20.


## 11. Ce qui reste

### Le plan

Le budget de chaque guichet noté est réparti ainsi : 12 réponses de
diagnostic, 58 de batterie, 30 de réserve.

1. Samedi vers 20 h. Premier contact avec les 33 guichets : 9 réponses
   courtes chacun. Elles servent à former les familles.

2. Samedi de 20 h 30 à minuit. Deuxième vague sur le banc, environ 2 500
   réponses. Nous y validons la méthode et choisissons les sondes.

3. Dans ce même créneau, première soumission, avec l'accord de Jacques. Nous
   y mettons les familles seules et déclarons tous les modèles différents.
   Le score vaut alors 0,1 × le MCC des familles. Il nous dit si nos
   familles sont justes.

4. Dimanche de 0 h 30 à 4 h. La batterie sur les 33 guichets, en deux temps.
   D'abord 36 réponses par guichet. Puis une analyse. Puis 22 réponses sur
   les sondes qui séparent le mieux dans chaque famille.

5. Dimanche de 4 h 30 à 6 h. La réserve, dépensée sur les paires encore
   douteuses.

6. Dimanche de 6 h à 7 h 30. Partition finale et soumission.

7. Dimanche à 11 h au plus tard, dernière soumission. La clôture est à midi.
   L'oral est à 13 h : six minutes, pour les six premières équipes.

### Les deux inconnues

1. Le fp8 se laisse-t-il séparer ? D'après notre première lecture des
   articles fournis, les méthodes en boîte noire le détectent mal. La vague
   1 nous donnera notre propre chiffre ce soir. Si la réponse est non, il
   reste la boîte blanche. Si elle échoue aussi, il reste l'hypothèse de
   structure de la section 7.

2. La méthode se transporte-t-elle ? Le banc ne contient qu'un seul petit
   modèle. Rien ne garantit que les sondes validées sur lui fonctionnent sur
   les familles notées. C'est pour cela que nous mesurons en unités de
   bruit, et que nous gardons 30 réponses de réserve par guichet.


## En cinq phrases, pour l'oral

1. Le problème : regrouper 33 guichets par modèle, en ignorant trois
   leurres, avec 100 réponses par guichet.

2. Le choix de départ : des réponses longues faites de nombreux tirages,
   parce que le temps de serveur compte autant que le budget.

3. La trouvaille : la coupure d'une réponse révèle le tokenizer, donc le
   modèle de base, sans requête de plus.

4. La méthode : trois mesures emboîtées qui retirent la température puis le
   watermark, calibrées par permutation, puis une décision par partition.

5. Le renfort : rejouer les réponses dans une copie locale sur GPU, pour
   lire les probabilités au lieu de les deviner.
