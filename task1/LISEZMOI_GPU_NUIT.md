# Tâche 1 — toutes les tâches GPU de la nuit, par ordre de priorité

Une seule page pour Jacques. Tout est testé sur CPU, tout passe par le relais de l'équipe
(1 A100 par tâche, `# MINUTES` ≤ 55), jamais `dc-gpu-devel`. Chaque tâche n'écrit que 4 à 9 fichiers
(quota de fichiers du scratch).

```
R=/p/scratch/training2666/trentelivres1/Azerit/relais
T=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task/t1
```

## Déjà dans la file (déposées le 21/11 à 23:51, rien à faire)

| Tâche | Durée | Remarque |
|---|---|---|
| `jacques5/f_longcomb_tm.sh` | ≈ 3 h 40 en 4 tranches | sort en code 3 à chaque tranche : un script le redépose automatiquement |
| `jacques5/f_longcomb_ps.sh` | ≈ 3 h 40 en 4 tranches | idem |
| `jacques7/sg_informed.sh` | ≈ 45 min | a besoin d'onnxruntime avec CUDA |
| `jacques5b/sg_long.sh` | ≈ 3 h 40 en 4 tranches | ≈ 10 Go de RAM par tranche ; redéposé automatiquement |

## À déposer ensuite, dans cet ordre (si le nœud a de la place)

1. Variantes courtes de la recette qui a gagné (mesurables au fold 0) :
   ```
   $R/relais-add $T/jacques5/e_comb_ps.sh
   $R/relais-add $T/jacques5/e_seed_ps.sh
   $R/relais-add $T/jacques5/e_seed_tm.sh
   $R/relais-add $T/jacques5/e_long_ps.sh
   ```
2. Recettes gagnantes entraînées sur toutes les images (membres de moyenne pour le test) :
   ```
   $R/relais-add $T/jacques6/informed_ps_all.sh
   $R/relais-add $T/jacques6/informed_tm_all.sh
   $R/relais-add $T/jacques6/holes_ps_all.sh
   ```
3. Recherche de rectangles StegaStamp sur GPU (d'abord le contrôle des faux positifs) :
   ```
   $R/relais-add $T/jacques4/a_sg_fold0.sh
   $R/relais-add $T/jacques4/a_sg_test_0.sh   # puis _1, _2, _3
   ```
4. Seulement s'il reste du temps : `jacques4/b_ps_*` (PixelSeal : on a mesuré que les rectangles
   de moins de 40 px de côté ne se décodent pas, gain attendu faible) et les `*_s2` de la vague 1.

## À NE PAS lancer

- `jacques2/a_sg_cloud_*`, `jacques2/b_ps_cloud_*` : mesuré inutile.
- Rien sur `dc-gpu-devel`, et pas de second job d'équipe (une seule allocation par équipe).

## Vérifier

- `$R/relais-etat` : état de la file.
- Sorties dans `$T/out/jacques{4,5,5b,6,7}/<tache>/`, fichier `DONE` à la fin.
- jacques4 : la ligne « onnx providers » de `run.log` doit afficher CUDAExecutionProvider.

L'intégration des résultats est faite de notre côté (scripts d'intégration avec contrôle automatique :
un modèle n'est gardé que s'il gagne au moins +0,01 sur nos images de validation).
