"""Critere local des cibles par classe (serveur : eval, fp16). Par checkpoint, sur les images a oublier (clean) et une copie alteree :
  CE moyenne / ecart-type / accord avec l'etiquette d'origine, ms = 1 - (|moy - 2.6| + |ect - 4.4|) / 7.0,
  pred_cls = 1 - KL(or_classe || modele) / KL(or_classe || original) a T=2, ou or_classe = gold_cls_T2.pt.
References (memes vues) : b2 clean ms 0.948 pred_cls 0.411, alt 0.894 / 0.418 ; c13 0.982 / 0.404 ; sM 0.306 / 0.719 ; E18 0.813 / 0.440.
  python measure_cls.py --gold /workspace/data/gold_cls_T2.pt CKPT [CKPT ...]      (a lancer depuis le dossier du code : common.py, augment.py)
"""
import argparse
import torch
import torch.nn.functional as F
from augment import alter
from common import load_forget_images, load_model, preprocess

p = argparse.ArgumentParser()
p.add_argument("--orig", default="/workspace/model.pt")
p.add_argument("--forget", default="/workspace/forget_images.npz")
p.add_argument("--gold", required=True)
p.add_argument("ckpts", nargs="+")
a = p.parse_args()
dev = torch.device("cuda")
X = load_forget_images(a.forget)
gd = torch.load(a.gold, map_location="cpu", weights_only=False)
G, y_ref = gd["gold_cls_T2"].to(dev).double(), gd["labels"].to(dev)
g = torch.Generator(device=dev).manual_seed(4321)
views = {"clean": [preprocess(X[i:i + 250].to(dev)) for i in range(0, len(X), 250)]}
views["alt"] = [alter(x, g) for x in views["clean"]]


@torch.no_grad()
def logits(m, xs):
    m.eval(); out = []
    for x in xs:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(m(x).float())
    return torch.cat(out).double()


def kl(Pg, z):
    return (Pg * (Pg.clamp_min(1e-12).log() - F.log_softmax(z / 2, 1))).sum(1).mean().item()


orig = load_model(a.orig, dev)
Zo = {k: logits(orig, v) for k, v in views.items()}
y = Zo["clean"].argmax(1)
assert (y == y_ref).all(), "ordre des images different de celui des cibles"
for c in a.ckpts:
    m = load_model(c, dev); parts = []
    for k, v in views.items():
        z = logits(m, v); ce = F.cross_entropy(z, y, reduction="none")
        ms = 1 - min(1.0, (abs(ce.mean().item() - 2.6) + abs(ce.std().item() - 4.4)) / 7.0)
        pr = 1 - kl(G, z) / kl(G, Zo[k])
        parts.append(f"{k} CE {ce.mean():.2f}/{ce.std():.2f} acc {(z.argmax(1) == y).float().mean():.3f} ms {ms:.3f} pred_cls {pr:.3f}")
    print(c, "|", " | ".join(parts), flush=True)
    del m; torch.cuda.empty_cache()
