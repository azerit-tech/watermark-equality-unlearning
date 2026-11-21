"""Analyse des masques hole / transplant du train. Depuis TASK (CPU, ~1 min) :
    python t1/shapes/analyze.py            # -> t1/out/shapes/desc.json, montage.png, dup.json
"""
import json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from desc import mask_desc, comp_desc  # noqa: E402

OUT = TASK / "t1/out/shapes"; OUT.mkdir(parents=True, exist_ok=True)
IDX = json.loads((TASK / "t1/common/train_index.json").read_text())


def read_mask(n):
    return (np.asarray(Image.open(TASK / f"release/train/masks/{n}.png")) > 127).astype(np.uint8)


def read_img(n):
    return np.asarray(Image.open(TASK / f"release/train/images/{n}.png").convert("RGB"))


def edge_alignment(img, mino, rng, nshift=8):
    """gradient moyen le long du contour du masque / même contour translaté aléatoirement."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    k = np.ones((3, 3), np.uint8)
    cont = (cv2.dilate(mino, k) - cv2.erode(mino, k)).astype(bool)
    cont[:2] = cont[-2:] = False; cont[:, :2] = cont[:, -2:] = False
    if cont.sum() < 20:
        return None
    on = gm[cont].mean(); offs = []
    for _ in range(nshift):
        dy, dx = rng.integers(-60, 61, 2)
        offs.append(np.roll(gm, (dy, dx), (0, 1))[cont].mean())
    # contenu : gradient dans le trou vs hors du trou
    inside = gm[mino.astype(bool)].mean() if mino.any() else np.nan
    outside = gm[~mino.astype(bool)].mean()
    return dict(edge_ratio=float(on / (np.mean(offs) + 1e-6)), grad_in=float(inside), grad_out=float(outside))


def norm_shape(c, s=48):
    ys, xs = np.nonzero(c); cr = c[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    return (cv2.resize(cr, (s, s), interpolation=cv2.INTER_AREA) > 0.5)


def main():
    rng = np.random.default_rng(0)
    rows, comps = [], []
    for n, v in sorted(IDX.items()):
        if v["edit_type"] not in ("hole", "transplant"):
            continue
        m = read_mask(n); mino = (1 - m) if v["full_watermark"] else m
        d = mask_desc(mino); d.update(name=n, kind=v["edit_type"], method=v["method"])
        ea = edge_alignment(read_img(n), mino, rng)
        if ea: d.update(ea)
        rows.append(d)
        k, lab, st, _ = cv2.connectedComponentsWithStats(mino, connectivity=8)
        for i in range(1, k):
            if st[i, cv2.CC_STAT_AREA] >= 400:
                c = (lab == i).astype(np.uint8)
                cd = comp_desc(c); cd.update(name=n, kind=v["edit_type"])
                comps.append((cd, norm_shape(c), c))
    (OUT / "desc.json").write_text(json.dumps(rows))
    (OUT / "comps.json").write_text(json.dumps([c[0] for c in comps]))

    # ---- doublons : IoU des formes normalisées (bbox -> 48x48) sous les 8 symétries, plus comparaison exacte des masques entiers
    shp = np.stack([c[1] for c in comps]).reshape(len(comps), -1).astype(np.float32)
    best = []
    D8 = []
    for c in comps:
        s = c[1]; vs = []
        for t in range(8):
            u = np.rot90(s, t % 4); u = u[:, ::-1] if t >= 4 else u
            vs.append(u.reshape(-1))
        D8.append(np.stack(vs).astype(np.float32))
    for i in range(len(comps)):
        inter = D8[i] @ shp.T                                         # [8, N]
        a = D8[i].sum(1, keepdims=True); b = shp.sum(1)[None]
        iou = (inter / (a + b - inter + 1e-6)).max(0); iou[i] = 0
        j = int(iou.argmax())
        # ne compter que les autres images
        same = [comps[x][0]["name"] == comps[i][0]["name"] for x in range(len(comps))]
        iou2 = np.where(same, 0, iou); j = int(iou2.argmax())
        best.append(dict(i=i, name=comps[i][0]["name"], kind=comps[i][0]["kind"], area=comps[i][0]["area"],
                         match=comps[j][0]["name"], match_kind=comps[j][0]["kind"], iou=float(iou2[j]),
                         area_ratio=float(comps[j][0]["area"] / comps[i][0]["area"])))
    (OUT / "dup.json").write_text(json.dumps(best))
    # masques entiers identiques (à symétrie près) entre images
    whole = {}
    for r in rows:
        whole[r["name"]] = None
    print("composantes analysées :", len(comps))
    ious = np.array([b["iou"] for b in best])
    for kind in ("hole", "transplant"):
        s = np.array([b["kind"] == kind for b in best])
        print(kind, "IoU meilleure forme d'une autre image : quantiles", np.quantile(ious[s], [.5, .9, .99, 1]).round(3),
              "n>0.9:", int((ious[s] > 0.9).sum()), "n>0.8:", int((ious[s] > 0.8).sum()))

    # ---- planche 24 masques (16 hole, 8 transplant), image + masque en rouge, 2 lignes par masque
    sel_h = [r["name"] for r in rows if r["kind"] == "hole"][::16][:16]
    sel_t = [r["name"] for r in rows if r["kind"] == "transplant"][::16][:8]
    tiles = []
    for n in sel_h + sel_t:
        img = cv2.resize(read_img(n), (192, 192), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = read_mask(n); mino = (1 - m) if IDX[n]["full_watermark"] else m
        mm = cv2.resize(mino, (192, 192), interpolation=cv2.INTER_NEAREST).astype(bool)
        ov = img.copy(); ov[mm] = 0.45 * ov[mm] + 0.55 * np.array([255, 0, 0])
        bw = np.repeat((mm * 255).astype(np.uint8)[..., None], 3, 2)
        t = np.concatenate([ov.astype(np.uint8), bw], 0)
        t = cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 120, 40) if IDX[n]["edit_type"] == "hole" else (40, 40, 160))
        tiles.append(t)
    rows_img = [np.concatenate(tiles[i:i + 8], 1) for i in range(0, 24, 8)]
    Image.fromarray(np.concatenate(rows_img, 0)).save(OUT / "montage.png")


if __name__ == "__main__":
    main()
