"""gradrect: recherche du rectangle d'outpainting par optimisation par gradient à travers le décodeur.
Paramètres continus par départ : bords (y0, y1, x0, x1) du rectangle R ; crop R+3 px (comme boxsearch2) obtenu par
affine_grid + grid_sample (sortie S x S) puis redimensionnement antialiasé à la résolution du décodeur.
Objectif : log-vraisemblance moyenne du message connu, mean_b logsigmoid(L_b * s_b), maximisée par Adam, en batch.
Final : meilleur point de chaque trajectoire, arrondi, puis (option) raffinement discret RS.refine, confiance =
précision de décodage (décodeur de référence RS.Dec) du rectangle final.
usage (depuis TASK) :
  python ~/gradrect/gradrect.py --method trustmark --mode basin  --names a,b,c --out f.json
  python ~/gradrect/gradrect.py --method pixelseal --mode random --names a,b --starts 64 --iters 60 --out f.json
  python ~/gradrect/gradrect.py --method pixelseal --mode random --split test --names @liste.json --device cuda --out f.json
Méthodes : trustmark, pixelseal (StegaStamp = ONNX, non différentiable ici)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
TASK = Path(os.getcwd()); sys.path[:0] = [str(TASK), str(TASK / "t1/boxsearch2")]
import rectsearch as RS  # noqa
PAD = 3
THR = {"trustmark": 0.79, "pixelseal": 0.90}
ASPECT = {"trustmark": 2, "pixelseal": 16}


class GDec:
    def __init__(self, wm, dev, S=512):
        self.wm, self.dev, self.S = wm, dev, S
        if wm.name == "trustmark":
            self.net = wm.net; self.res = 245; self.f = lambda z: self.net(z * 2 - 1)
        else:
            self.net = wm.det; self.res = 256; self.f = lambda z: self.net(z)[:, 1:]
        for p in self.net.parameters(): p.requires_grad_(False)
        self.net.eval()

    def logits(self, x, E):
        """x [1,3,512,512], E [B,4] float (y0,y1,x0,x1) of R. Crop R+PAD clipped to the image, differentiable."""
        y0 = (E[:, 0] - PAD).clamp(min=0); y1 = (E[:, 1] + PAD).clamp(max=512)
        x0 = (E[:, 2] - PAD).clamp(min=0); x1 = (E[:, 3] + PAD).clamp(max=512)
        B = E.shape[0]; th = torch.zeros(B, 2, 3, device=E.device)
        th[:, 0, 0] = (x1 - x0) / 512; th[:, 0, 2] = (x0 + x1) / 512 - 1
        th[:, 1, 1] = (y1 - y0) / 512; th[:, 1, 2] = (y0 + y1) / 512 - 1
        g = F.affine_grid(th, (B, 3, self.S, self.S), align_corners=False)
        c = F.grid_sample(x.expand(B, -1, -1, -1), g, mode="bilinear", padding_mode="border", align_corners=False)
        c = F.interpolate(c, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True)
        return self.f(c).float()


def project(E, A):
    with torch.no_grad():
        E[:, 0].clamp_(0, 512 - 8); E[:, 2].clamp_(0, 512 - 8)
        E[:, 1].copy_(torch.maximum(E[:, 1].clamp(max=512), E[:, 0] + 8))
        E[:, 3].copy_(torch.maximum(E[:, 3].clamp(max=512), E[:, 2] + 8))


def optimise(G, x, msg, E0, iters, lr, A, bs):
    """returns best (acc, rect) per start along the trajectory + history of acc."""
    s = torch.tensor(2.0 * msg - 1, dtype=torch.float32, device=x.device)
    out_r, out_a = [], []
    for i in range(0, len(E0), bs):
        E = torch.tensor(E0[i:i + bs], dtype=torch.float32, device=x.device, requires_grad=True)
        opt = torch.optim.Adam([E], lr=lr)
        best_a = torch.full((len(E),), -1.0); best_r = E.detach().cpu().clone()
        for it in range(iters):
            L = G.logits(x, E); m = L * s
            acc = (m > 0).float().mean(1).detach().cpu()
            upd = acc > best_a; best_a[upd] = acc[upd]; best_r[upd] = E.detach().cpu()[upd]
            loss = -F.logsigmoid(m).mean(1).sum()
            opt.zero_grad(); loss.backward(); opt.step(); project(E, A)
        out_r.append(best_r.numpy()); out_a.append(best_a.numpy())
    return np.concatenate(out_r), np.concatenate(out_a)


def rint(r):
    r = [int(round(v)) for v in r]; r[0] = max(0, r[0]); r[2] = max(0, r[2]); r[1] = min(512, r[1]); r[3] = min(512, r[3]); return tuple(r)


def prior_starts(n, rng, method):
    P = json.loads((TASK / "t1/out/rectprior/outpaint_rects_train.json").read_text())
    P = [p for p in P if p[1] == method] or P
    out = []
    for _ in range(n):
        p = P[rng.integers(len(P))]; h = (p[3] - p[2]) * np.exp(rng.normal(0, 0.25)); w = (p[5] - p[4]) * np.exp(rng.normal(0, 0.25))
        h = float(np.clip(h, 12, 500)); w = float(np.clip(w, 12, 500))
        y0 = rng.uniform(0, 512 - h); x0 = rng.uniform(0, 512 - w); out.append([y0, y0 + h, x0, x0 + w])
    return np.array(out, np.float32)


def shifted_starts(gt, d, n, rng):
    out = []
    for _ in range(n):
        q = np.array(gt, np.float32) + rng.choice([-1, 1], 4) * d   # every edge moved by d px, random sign
        q[0] = max(0, q[0]); q[2] = max(0, q[2]); q[1] = min(512, max(q[1], q[0] + 12)); q[3] = min(512, max(q[3], q[2] + 12))
        out.append(q)
    return np.array(out, np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method"); ap.add_argument("--mode", default="random")
    ap.add_argument("--split", default="train"); ap.add_argument("--names"); ap.add_argument("--out")
    ap.add_argument("--starts", type=int, default=64); ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2.0); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--S", type=int, default=512); ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--shifts", default="10,20,40"); ap.add_argument("--nshift", type=int, default=8)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(a.threads); M = a.method; A = ASPECT[M]
    wm = RS.load_watermark(M, device=a.device); G = GDec(wm, a.device, a.S); D = RS.Dec(wm, a.device)
    names = json.loads(Path(a.names[1:]).read_text()) if a.names.startswith("@") else a.names.split(",")
    meta = {m["image_name"]: m for m in json.loads((TASK / f"release/{a.split}/metadata.json").read_text())}
    idx = json.loads((TASK / "t1/common/train_index.json").read_text()) if a.split == "train" else {}
    fout = Path(a.out); res = json.loads(fout.read_text()) if fout.exists() else {}
    for k, nm in enumerate(names):
        if nm in res: continue
        t0 = time.time(); rng = np.random.default_rng(a.seed + k)
        msg = np.array([int(c) for c in meta[nm]["message"]], np.float64)
        x = torch.from_numpy(np.array(Image.open(TASK / f"release/{a.split}/images/{nm}.png").convert("RGB"))).permute(2, 0, 1)[None].float().div(255).to(a.device)
        gt = idx.get(nm, {}).get("bbox512"); rec = {"gt": gt}
        if gt:
            with torch.no_grad():
                rec["gt_acc_ref"] = float(D.score(x, [tuple(gt)], msg)[1][0])
                rec["gt_acc_diff"] = float(((G.logits(x, torch.tensor([gt], dtype=torch.float32)) > 0).numpy()[0] == msg.astype(bool)).mean())
        if a.mode == "basin":
            rec["basin"] = {}
            for d in [int(v) for v in a.shifts.split(",")]:
                E0 = shifted_starts(gt, d, a.nshift, rng)
                with torch.no_grad(): a0 = D.score(x, [rint(q) for q in E0], msg)[1]
                R, acc = optimise(G, x, msg, E0, a.iters, a.lr, A, a.bs)
                rr = [rint(q) for q in R]; _, aref, _ = D.score(x, rr, msg)
                rec["basin"][d] = {"acc0": a0.round(3).tolist(), "acc_end": aref.round(3).tolist(), "iou0": [round(RS.iou(rint(q), gt), 3) for q in E0],
                                   "iou_end": [round(RS.iou(q, gt), 3) for q in rr], "rects": [list(q) for q in rr]}
                RS.log(nm, "d", d, "acc0", np.round(np.mean(a0), 3), "acc_end", np.round(np.mean(aref), 3),
                       "iou_end", np.round([RS.iou(q, gt) for q in rr], 2).tolist())
        else:
            E0 = prior_starts(a.starts, rng, M)
            R, acc = optimise(G, x, msg, E0, a.iters, a.lr, A, a.bs)
            order = np.argsort(-acc); top = []
            for j in order:
                q = rint(R[j])
                if all(RS.iou(q, t) < 0.5 for t in top): top.append(q)
                if len(top) >= a.topk: break
            _, atop, _ = D.score(x, top, msg)
            fin = []
            for q in top:   # discrete refine (same as boxsearch2) from the gradient result
                o0 = D.score(x, [q], msg)[0][0]; r2, _ = RS.refine(D, x, msg, q, o0, A if RS.ok(q, A) else 999)
                fin.append(r2)
            _, afin, _ = D.score(x, fin, msg); j = int(np.argmax(afin))
            rec.update({"top": [list(q) for q in top], "top_acc": atop.round(4).tolist(), "rect": list(map(int, fin[j])), "conf": float(afin[j]),
                        "grad_rect": list(top[0]), "grad_conf": float(atop[0]),
                        "frac_starts_iou50": float(np.mean([RS.iou(rint(q), gt) > 0.5 for q in R])) if gt else None})
            if gt: rec["iou"] = RS.iou(fin[j], gt); rec["grad_iou"] = RS.iou(top[0], gt)
        rec["sec"] = time.time() - t0; res[nm] = rec; fout.write_text(json.dumps(res))
        RS.log(k, nm, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rec.items() if kk not in ("basin", "top")})
    Path(str(fout) + ".done").write_text("ok"); RS.log("DONE")
