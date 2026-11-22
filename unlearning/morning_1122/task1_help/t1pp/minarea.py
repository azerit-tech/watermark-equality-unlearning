from base import *
from rules import *
import torch
allm = {n: gt[n] for n in gt}
def minfrac(m): c = float(np.asarray(m).mean()); return min(c, 1 - c)
print("GT minority fraction quantiles on the whole train set (per method x regime): q05 q10 q25 q50 q75")
for meth in ("trustmark", "pixelseal", "stegastamp"):
    for full in (True, False):
        v = np.array([minfrac(gt[n]) for n in gt if meta[n]["method"] == meth and bool(meta[n]["full_watermark"]) == full])
        print(f"  {meth:10s} {'full' if full else 'part'} n={len(v):3d} ", np.round(np.quantile(v, [.05, .1, .25, .5, .75]), 4), " share<0.5%:", round(float((v < .005).mean()), 3), "<1%:", round(float((v < .01).mean()), 3), "<2%:", round(float((v < .02).mean()), 3))
sub = torch.load(D / "sub20i.pt", weights_only=True); tn = sub["image_names"]; tm_ = sub["masks"].numpy()
print("predicted minority fraction: fold0 final masks / C2 test masks")
for s, meth in (("tm", "trustmark"), ("ps", "pixelseal")):
    d = {n: binarize(p) for n, p in load(s).items()}
    for full in (True, False):
        v = np.array([minfrac(d[n]) for n in d if (d[n].mean() >= .5) == full])
        w = np.array([minfrac(tm_[i]) for i, n in enumerate(tn) if tmeta[n]["method"] == meth and (tm_[i].mean() >= .5) == full])
        g = np.array([minfrac(gt[n]) for n in d if (d[n].mean() >= .5) == full])
        for tag, x in (("fold0 pred", v), ("fold0 GT  ", g), ("test pred ", w)):
            print(f"  {meth:10s} {'full' if full else 'part'} {tag} n={len(x):3d} ", np.round(np.quantile(x, [.05, .1, .25, .5, .75]), 4), " share<0.5%:", round(float((x < .005).mean()), 3), "<1%:", round(float((x < .01).mean()), 3), "<2%:", round(float((x < .02).mean()), 3))
