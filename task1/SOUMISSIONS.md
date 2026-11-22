# Tâche 1 : soumissions de la nuit du 21 au 22/11

Meilleure soumission : subC2_sam_G2.pt, score public 0,7326 (id 3886). Seule la meilleure compte.

Les fichiers .pt sont dans task1/subs/ (ignoré par git, à garder sur le Mac). Le journal complet de chaque envoi, avec les contrôles et la réponse du serveur, est dans task1/submissions_log.jsonl. Le fold 0 désigne nos 102 images de validation (34 par méthode).

## Les soumissions, dans l'ordre

**sub20.pt — 0,7228.** Assemblage de Lumine (integrate20) : rectangles multi-cartes, modèles TM/PS/SG et assemblage final. C'est la base de tout le reste.

**sub20_A_lumine.pt — 0,7241 (id 2539).** sub20 plus les modèles informed_tm_all et informed_ps_all sur le test, retenus avec la règle de Lumine : gain d'au moins +0,01 au fold 0, avec un IC95 bas positif.

**subA_C2.pt — 0,7246 (id 2848).** A plus 6 modèles de la nuit (divcov, graines, comb, long), retenus avec la règle « membre », plus souple. Gain au fold 0 : +0,0010. Il modifie 555 images.

**subA_D1.pt — 0,7244 (id 3210).** C2 plus 4 modèles PixelSeal (divcov_ps_cont, seed1, divcovxc_s13, divcovx_s13). Gain au fold 0 : +0,0010. Au public, le résultat est du bruit.

**subC2_sam_G2.pt — 0,7326 (id 3886), la meilleure.** C2, où SAM ViT-H redessine le masque des images greffées (transplants). La porte G2 n'applique SAM que si aucune boîte n'est confiante et si le masque de SAM s'accorde à au moins 0,7 avec celui de C2. Elle touche 166 images. Gain au fold 0 : +0,0110 ; au public : +0,008.

**subC2_sam_G2_pp.pt — 0,7308 (id 3916).** G2 plus une retouche des masques TrustMark/PixelSeal. Elle supprime les petites composantes et agrandit les zones minoritaires trop petites, hors des images SAM. Gain au fold 0 : +0,0028. Au public, il n'y a aucun gain.

## Refaire la meilleure

1. Lancer SAM sur les images greffées : task1/sam_vast/sam_run.py, ou la version JURECA sam_run_j.py. Sur vast, 1 A100 met 18 min.
2. Appliquer sur C2 : task1/sam_vast/apply_test.py, avec la porte G2 et subA_C2.pt comme base (4e argument).
3. Vérifier et envoyer : python3 task1/submit_t1.py <fichier.pt> --ref task1/subs/subA_C2.pt, puis --send. La clé est lue dans le .env.

C2 lui-même se reconstruit sur JURECA avec ~/t1w/run_integ.sh et integrate20 (option --base-extra, liste dans ~/t1w/integ20/C2_extra.txt).

## Ce qui n'a pas marché

- SAM sur les trous fait perdre environ 0,15 par image, et il fait aussi perdre sur les outpaintings.
- La porte G1, sans condition d'accord, étend les masques sur de grands fonds (mur, neige, ciel).
- La porte G3, avec un filtre sur le ratio de couverture, fait perdre sur les outpaintings.
- Les modèles divcovx/divcovxc (couvertures externes) et rgb ne donnent aucun gain TrustMark.
- La retouche morphologique des trous ne donne rien : les pertes viennent de zones mal placées, pas de masques troués.
- L'écart avec la tête (0,79) vient surtout des trous TrustMark et PixelSeal, qui demanderaient des entraînements bien plus longs.
