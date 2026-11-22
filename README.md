# Filigranes, Model Equality, Unlearning : l'équipe Azerit à l'EuroAIChampionship Paris 2026

Ce dépôt rassemble le travail de l'équipe Azerit pendant l'EuroAIChampionship, à Paris, du 21 au 22 novembre 2026. Le défi comportait trois tâches de sécurité et de traçabilité des modèles d'IA. Chacune avait un classement public calculé sur environ 30 % des données ; le classement final porte sur les 70 % restants, et seule la meilleure soumission de chaque équipe compte.

Le sujet officiel est dans EuroAIChampionshipParis2026-11-21.pdf, et les articles conseillés par les organisateurs dans REFERENCES.md.


## Tâche 1 : localiser un filigrane dans une image (dossier task1/)

Le problème : des images portent un filigrane invisible (TrustMark, PixelSeal ou StegaStamp), souvent après une retouche (trou, extension, greffe d'un morceau d'une autre image). Il faut produire, pour chaque image de test, un masque 128×128 de la zone filigranée.

Ce qu'on a fait : des segmenteurs entraînés par type de filigrane, bâtis en partie sur les extracteurs officiels, et des données synthétiques produites avec les encodeurs du kit de départ. Un assemblage final combine des rectangles prédits par plusieurs modèles. Pour finir, SAM (Segment Anything, ViT-H) redessine les masques des images greffées, uniquement quand les deux méthodes concordent.

Résultat : 0,7326 au classement public (de 0,5528 au premier essai à 0,7326 au final). Le détail des soumissions est dans task1/SOUMISSIONS.md.


## Tâche 2 : Model Equality, reconnaître des modèles derrière une API (dossier model_equality/)

Le problème : 33 points d'accès servent des modèles de langage en boîte noire. Il faut les regrouper par famille et par modèle exact. Une quantification ou un LoRA différent compte comme un autre modèle ; un filigrane, un prompt système ou un réglage de décodage sont des leurres à ignorer. Chaque point d'accès ne répond que 100 fois pour toute la compétition.

Ce qu'on a fait : un client d'API avec un registre de toutes les réponses, et un scoreur local identique à celui des organisateurs. Nous avons écrit une batterie de sondes qui sépare les vrais changements de modèle des leurres. Les règles ont d'abord été calibrées sur 9 points d'entraînement gratuits dont la réponse était connue. Le journal d'analyse, avec chaque décision et ses preuves, est dans model_equality/analysis/JOURNAL_ANALYSTE.md.

Résultat : 1,0 au classement public.


## Tâche 3 : faire oublier des images à un classifieur (dossier unlearning/)

Le problème : un ViT-B/16 classe 100 classes d'images synthétiques. Il doit oublier 3 500 images d'entraînement et se comporter comme des modèles réentraînés sans elles, que nous ne voyons jamais. Il doit garder intact tout le reste. Le score multiplie un terme d'oubli (distribution des pertes et sorties image par image) par un terme de rétention.

Ce qu'on a fait : nous ne disposions que du modèle et des images à oublier, pas des données d'entraînement. Nous avons donc estimé les sorties des modèles de référence avec des « sœurs » : des variantes de chaque image générées par SDXL-Turbo. La recette finale entraîne vers ces cibles, ajoute un terme de queue qui fait oublier fortement environ 20 % des images, puis répare la rétention sur une banque d'images générées et extrapole les poids. Un juge local, qui recoupe 28 de nos soumissions avec leurs scores publics, a servi à trier les idées sans surajuster au classement public.

Résultat : 0,2932 au classement public (de 0,008 au premier essai), avec le modèle abx_l08. Le récit de la dernière matinée et les pistes abandonnées sont dans unlearning/morning_1122/README.md.


## Infrastructure (dossiers tools/ et GPU_EQUIPE.md)

Les calculs ont tourné sur le supercalculateur JURECA (nœuds de 4 A100, un job par personne). Un « relais » y faisait passer les tâches de toute l'équipe dans un seul job. Quelques GPU ont aussi été loués sur vast.ai pour les entraînements longs. Mode d'emploi : GPU_EQUIPE.md.

Les secrets (clés d'API) restent dans des fichiers .env ignorés par git. Les modèles et les grosses données (.pt, .npz) ne sont pas versionnés : ils sont sur JURECA ou sur les postes de l'équipe.
