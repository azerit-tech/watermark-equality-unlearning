# Brief commun aux agents, tâche 1 (à lire en entier avant de commencer)

## Le problème
- Localiser la zone watermarkée d'images PNG 512×512. Sortie : une grille binaire 128×128 par image (cellule 4×4 px, 1 si ≥ 50 % de la cellule est watermarkée).
- Score officiel : moyenne par image de (IoU classe 0 + IoU classe 1)/2, avec IoU = 1 si l'union est vide. Implémenté dans `wmloc/utils.py` (`downsample_mask`, `miou_image`, `score`) et appelé par `t1/common/evaluate.py` (`evaluate(pred_dict, names=None, return_per_image=False)`, CLI `python t1/common/evaluate.py preds.npz [--fold k]`).
- Trois méthodes à parts égales : pixelseal (256 bits), trustmark (variante Q, 100 bits), stegastamp (100 bits, ONNX). Méthode et message embarqué sont CONNUS pour chaque image, au train comme au test.
- Train : 510 images (170 par méthode) avec masques. Test : 1020 images (340 par méthode). Public 30 % / privé 70 %. Fin des soumissions : dimanche 22 nov 12:00.

## Les types de retouche (champ `edit_type` de `t1/common/train_index.json`), par méthode
| Type | Part | Géométrie du watermark | Masque |
|---|---|---|---|
| `hole` | 50 % | `embed` sur l'image entière, puis quelques taches restaurées à l'original | image entière moins des blobs ; zone watermarkée majoritaire |
| `transplant` | 25 % | `embed` sur l'image entière, seule la silhouette d'un objet de la scène le garde | silhouette d'objet ; zone watermarkée minoritaire |
| `outpainting` | 25 % | `embed` sur un rectangle = masque élargi de 2 à 5 px par côté ; la bande élargie a été régénérée (pixels propres) | un rectangle exact |

- Le « régime » (zone watermarkée majoritaire ou minoritaire) se devine à ≈ 98 % par le décodage de l'image entière : caches `t1/out/signal/fullacc_<methode>_{train,test}.npz` (`names`, `acc`), seuils pixelseal 0,80, trustmark 0,76, stegastamp 0,755 ; fonctions `guard`, `load_regimes` dans `t1/train/regime_guard.py`.
- Score d'une image ≈ 0,5 + 0,5 × IoU de la région minoritaire. Plancher « régime + constante » : 0,455. Une erreur de 1 px sur les bords coûte ≈ 3 points par image, 4 px ≈ 10-14 points.
- Bords : pas de décalage entre masque et zone watermarkée pour `hole` ; ≈ +1 px (zone watermarkée plus grande) pour `transplant`.

## Ce qui existe et ce que ça vaut (fold 0 réel, 34 images par méthode)
- StegaStamp : U-Net pleine résolution entraîné sur synthétique (`t1/train/`), 0,763 (outpainting 0,67, transplant 0,77, hole 0,80).
- TrustMark, PixelSeal : le U-Net n'apprend pas (résidu de ±1 niveau de gris). Ce qui marche : un segmenteur dont l'encodeur est l'extracteur officiel (`t1/dense/` : `dense_model.py`, `train_dense.py`), 1,6 min d'entraînement sur une carte du nœud de connexion → TrustMark 0,557, PixelSeal 0,535. Une sonde linéaire sur les étages gelés donnait une AUC de 0,81 (TrustMark layer2/3) et 0,75 (PixelSeal stages.2 à 256).
- Générateur synthétique fidèle : `t1/synth/` (`gen.py`, `dataset.py`, `shapes.py`), données prêtes dans `t1/out/synth/v0/<methode>/{full,rect}` et validation synthétique `t1/out/synth/v0/val`.
- Folds : `t1/common/splits_5fold.json` (5 folds de 102, stratifiés méthode × régime × rectangle). Le fold 0 sert de validation par défaut.
- Format des prédictions : `np.savez(path, names=np.array([...]), probs=float16 [N,128,128])`, noms = `image_name` sans extension. Fichiers de référence : `fold0_probs128.npz`, `test_probs128.npz`.
- Meilleure soumission : 0,5528 public. Le coordinateur construit et envoie les soumissions : tu ne soumets JAMAIS rien.

## Accès au cluster JURECA
- `TASK=/p/scratch/training2666/trentelivres1/Azerit/Hackathon_Watermark_Localisation_Task`. Exécution depuis TASK avec `source .venv/bin/activate` (Python 3.11, torch 2.5.1+cu124, timm, numpy, scipy, opencv, pillow ; PAS de scikit-learn, matplotlib, skimage, segmentation-models-pytorch). N'installe aucun paquet.
- SSH : le cluster coupe les connexions quand on en ouvre trop, et le nœud de connexion change à chaque connexion. Utilise TOUJOURS une connexion maître persistante, avec un chemin qui t'est propre (remplace `<agent>` par le nom de ton dossier) :
  `ssh -o BatchMode=yes -o ConnectTimeout=25 -o ControlMaster=auto -o ControlPath=/private/tmp/claude-501/cm-<agent>-%C -o ControlPersist=20m jureca '...'`
  Tu restes ainsi sur le même nœud (tes processus `nohup` restent visibles) et tu n'ouvres qu'une connexion. Si elle tombe, attends 20 s et recommence.
- Copie de fichiers : `scp`/`rsync` avec les mêmes options `-o ControlPath=...`.
- Calcul sur le nœud de connexion :
  - CPU : 8 cœurs au plus pour toi (`OMP_NUM_THREADS=8`, `torch.set_num_threads(8)`, ONNX `intra_op_num_threads`).
  - GPU : chaque nœud a 2 Quadro RTX 8000 (46 Go) partagées avec tout JURECA. Tu peux en prendre UNE si elle est libre (`nvidia-smi` : utilisation < 30 % et < 4 Go occupés), pour des calculs de 10 minutes au plus à la fois, puis la rendre. Pas de bf16 rapide sur ces cartes : fp32 ou fp16.
  - Les longs calculs passent par `nohup ... > log 2>&1 &` avec un fichier marqueur de fin sur le disque partagé.
- File GPU (A100) : aucun `sbatch` ni `scancel`. Le coordinateur gère la file. Si ton travail mérite un job GPU, prépare le script et décris-le dans ton rapport.

## Fichiers : quota de NOMBRE de fichiers presque plein (consigne des organisateurs, 21 nov 23:00)
- Tout le hackathon partage un quota de nombre de fichiers sur /p/scratch/training2666, plein à 99 %. Chaque fichier et chaque dossier compte 1.
- `export PYTHONDONTWRITEBYTECODE=1` avant toute commande Python. Réutiliser le .venv existant, aucune installation.
- Une sortie = un conteneur : un seul .npz / .json / .jsonl par run, jamais un fichier par image. Checkpoints : last et best seulement. Un journal par run.
- Une étape ne doit pas créer plus de quelques centaines de fichiers. Avant et après : `df --output=iavail /p/scratch/training2666` et `find <dossier> -xdev | wc -l`.
- Supprimer en fin de travail ce que tu as créé et qui ne sert plus (tests en fumée, sorties intermédiaires). Ne jamais supprimer ce qu'un autre a créé.
- GPU : une seule allocation par ÉQUIPE (relais compris). Aucun sbatch de ta part.

## Règles
- Budget : 40 minutes au plus. Rends un travail FINI, mesuré, avec des fichiers de prédiction si tu en produis, plutôt qu'un brouillon ambitieux.
- Écris uniquement dans ton dossier : local `/Users/lumine/Desktop/azerit-hackathon/task1/<agent>/`, cluster `$TASK/t1/<agent>/` (code) et `$TASK/t1/out/<agent>/` (sorties). Le code des autres (`t1/common`, `t1/synth`, `t1/train`, `t1/dense`, `t1/signal`, `t1/mgr`…) se LIT et se COPIE, il ne se modifie pas.
- Aucune soumission au leaderboard, aucun commit git, aucun job Slurm, aucune installation. N'affiche aucun secret.
- Aucun processus ne doit tourner après ton rapport.
- Évalue toujours sur de vraies images que le modèle n'a pas vues (fold 0 par défaut), par type de retouche. Sur 34 images par méthode, un écart < 0,03 est du bruit : dis-le.
- SURAPPRENTISSAGE, règle stricte : ne choisis pas un checkpoint, un seuil, un poids ou un réglage sur les mêmes images que celles qui servent à annoncer le score. Si tu dois régler quelque chose, fais-le par validation croisée (sous-folds dans le fold 0, ou sur les folds 1-4 réels) et annonce le score sur des images qui n'ont servi à rien. Moins de paramètres réglés, c'est mieux : une option qui ne gagne pas ≥ 0,03 de façon stable (bootstrap) ne s'adopte pas. Ne te sers jamais du leaderboard public pour régler quoi que ce soit.

## Rapport final
Français, court, chiffré : ce qui a été fait, scores sur vraies images (par type et total) face à la référence, chemins des fichiers produits (prédictions fold 0 et test au format ci-dessus), commande pour relancer, ce qui n'a pas marché ou pas été vérifié, et la prochaine étape que tu recommandes.
