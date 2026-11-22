"""Collage SAM sur le test à partir de la variante A.
usage: python apply_test.py <masks_dir> <gate: G1|G2> [out.pt]
Porte : régime minoritaire dans A ET pas de boîte test confiante (tmseed 0.79, sgseed2 0.9, psseed 0.9)
        [G2 : + accord (score officiel espéré du candidat sous le masque A) >= 0.7].
Choix : masque SAM 'perm' seul maximisant l'IoU espérée sous le masque A remonté en 512 (plus proche voisin),
        puis réduction officielle (wmloc.utils.downsample_mask)."""
import collections, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/Users/jacquesmougeot/Desktop/Azerit_t1_mirror")
from wmloc.utils import downsample_mask

D = Path("/Users/jacquesmougeot/Desktop/Azerit_t1_mirror/sam_test/ind")
mdir, gate = Path(sys.argv[1]), sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "/Users/jacquesmougeot/Desktop/Azerit-hack/task1/subs/subA_sam.pt"
BASE = sys.argv[4] if len(sys.argv) > 4 else "/Users/jacquesmougeot/Desktop/Azerit-hack/task1/subs/sub20_A_lumine.pt"
A = torch.load(BASE, weights_only=True)
print("base", BASE)
names = list(A["image_names"]); M = A["masks"].numpy().copy()
meta = {x["image_name"]: x["method"] for x in json.load(open(D / "test_metadata.json"))}
CONF = {"stegastamp": 0.9, "trustmark": 0.79, "pixelseal": 0.9}
box = {}
for f, m in (("sg", "stegastamp"), ("tm", "trustmark"), ("ps", "pixelseal")):
    d = json.load(open(D / f"{f}_test_boxes.json")); d = d.get("boxes", d)
    box.update({n: b.get("conf", 1) >= CONF[m] for n, b in d.items()})


def escore(c, q):
    s = 0
    for cc, qq in ((c, q), (1 - c, 1 - q)):
        i = (cc * qq).sum(); u = cc.sum() + qq.sum() - i; s += i / max(u, 1e-6)
    return s / 2


stats = collections.Counter(); cells = collections.Counter(); missing = []; modified = []
for i, n in enumerate(names):
    base = M[i]
    if base.mean() >= 0.5 or box.get(n, False):
        continue
    f = mdir / f"{n}_perm.npz"
    if not f.exists():
        missing.append(n); continue
    z = np.load(f); k = len(z["piou"])
    if k == 0:
        continue
    S = np.unpackbits(z["seg"], axis=1)[:, :512 * 512].reshape(k, -1).astype(np.float32)
    q512 = np.repeat(np.repeat(base.astype(np.float32), 4, 0), 4, 1).reshape(-1)
    a = S.sum(1); iq = S @ q512; Q = q512.sum()
    b = int(np.argmax(iq / np.maximum(a + Q - iq, 1e-6)))
    c128 = downsample_mask(S[b].reshape(512, 512)).astype(bool)
    eB = escore(c128.astype(np.float32), base.astype(np.float32))
    if gate == "G2" and eB < 0.7:
        stats[(meta[n], "rejeté_accord")] += 1; continue
    d = int((c128 != base).sum())
    M[i] = c128; stats[(meta[n], "modifié")] += 1; cells[meta[n]] += d
    if d: modified.append(n)
print("manquants:", len(missing))
for k, v in sorted(stats.items()): print(k, v)
print("cellules modifiées par méthode:", dict(cells), "total", sum(cells.values()))
torch.save({"image_names": names, "masks": torch.from_numpy(M).bool().contiguous()}, out)
cand = set(json.load(open(D / "test_candidates.json")))
print("images modifiées", len(modified), "hors candidats", sum(n not in cand for n in modified))
cov = M.reshape(len(M), -1).mean(1); print("masques vides", int((cov == 0).sum()), "pleins", int((cov == 1).sum()))
print("manquants (minoritaires sans boîte sans masque SAM):", missing[:5])
json.dump(modified, open(D / (Path(out).name + ".modified.json"), "w"))
print("écrit", out)
