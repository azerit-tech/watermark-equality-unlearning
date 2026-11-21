# GPU sur JURECA : mode d'emploi de l'équipe

La situation (samedi 18 h 30)
- Toutes les équipes du hackathon partagent 30 nœuds de 4 A100 (120 GPU), jusqu'à dimanche 14:00.
- Tout est pris : 30 jobs tournent, environ 35 attendent. Le reste de la partition (hors réservation) est plein aussi.
- Un job reçoit toujours un nœud entier (4 cartes), même s'il n'en demande qu'une.

Pourquoi c'est long alors qu'il y a « beaucoup de GPU » ?
Parce qu'un job = un nœud : il ne peut tourner que 30 jobs à la fois pour tout le hackathon.
Et la moitié d'entre eux n'utilisent qu'une carte sur quatre : les trois autres dorment.

Les règles
1. Un job = 4 calculs. Demandez --gres=gpu:4 et lancez un calcul par carte (modèle plus bas).
   Raison : vous attendez une fois et vous obtenez quatre résultats.
2. Testez d'abord sur le nœud de connexion, quelques minutes, sur une carte libre.
   Raison : un job qui plante pour une faute de frappe vous a coûté toute l'attente.
3. Pas plus d'un job en attente par personne ; regroupez plutôt.
   Raison : chaque job fait la queue séparément ; dix petits jobs, c'est dix attentes.
4. Annulez tout de suite ce qui ne sert plus (scancel).
   Raison : un job inutile en file ou en cours bloque un nœud entier.
5. Mettez une durée --time réaliste (sans --time, c'est 1 h ; le job est tué à la limite).
   Raison : l'heure de démarrage estimée de tout le monde est calculée avec ces durées.
6. Prévenez sur la discussion avant tout job de plus d'une heure.

L'ordre de passage
Tout le monde a la même priorité de base. L'ordre suit surtout l'heure de soumission,
corrigée un peu par ce que chacun a déjà consommé. Soumettre plus de jobs ne fait pas passer devant.
L'heure de démarrage affichée est pessimiste : aujourd'hui nos jobs sont souvent partis bien avant.

Les commandes
```
./gpu.sh                    # état de la réservation et des jobs de l'équipe (dans le dossier d'équipe)
squeue --me --start         # mes jobs en attente et leur heure de démarrage estimée
scancel 12345678            # annuler un job (son numéro)
nvidia-smi                  # sur le nœud de connexion : quelle carte est libre
CUDA_VISIBLE_DEVICES=1 timeout 10m python mon_script.py --petit   # essai court sur la carte 1
```
Le nœud de connexion a des Quadro RTX 8000 (48 Go, plus anciennes que les A100, pas de bf16 rapide).
Elles sont partagées avec tous les utilisateurs de JURECA : essais de quelques minutes seulement.

Le modèle de job : un calcul par carte
```
#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python run.py --seed 0 > logs/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python run.py --seed 1 > logs/gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python run.py --seed 2 > logs/gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python run.py --seed 3 > logs/gpu3.log 2>&1 &
wait
```
À lancer depuis le dossier de la tâche (mkdir -p logs ; sbatch job.sbatch). Changez les quatre lignes python.

Le partage entre nos trois tâches
On n'est classés que si on a un score sur les trois tâches. Donc :
- Une tâche qui n'a encore aucun score passe en premier : les autres attendent avant de soumettre un nouveau job.
- Une fois les trois scores obtenus : chacun un job en cours et un en attente au plus, à tour de rôle.
- Dimanche à partir de 10:00 : le GPU va aux soumissions finales (limite 12:00).

Une autre file, à confirmer avant usage
La partition dc-gpu-devel (12 nœuds, 2 h max, 2 jobs par personne) est hors réservation.
Notre compte y a accès et d'autres équipes s'en servent, avec peu d'attente.
On n'a pas vérifié que les organisateurs l'autorisent ni ce que ça coûte au budget commun : demandez d'abord.
