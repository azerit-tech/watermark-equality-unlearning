"""TrustMark remover as a LOCAL detector (agent tmrm). CPU only. Run from TASK:
    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python t1/tmrm/rm_local.py synth|real
d = RM(y) - y at 256 (y = bilinear 256 of the 512 image, in [-1,1]); z = y + d = RM(y);
r = Enc(z, m) - z (channel mean removed, known message m).  Maps (higher = watermarked):
  mag_s  = sqrt(G_s * |d|^2)            energy of the removal
  a_s    = -G_s*(d.r) / G_s*(r.r)        local projection of d on the expected residual (~1 where watermarked)
  c_s    = -G_s*(d.r) / sqrt(G_s*|d|^2 G_s*|r|^2)   local correlation
synth: 20 synthetic full embeddings (t1/out/synth/v0/trustmark/full/s0_0000: C, Cw from fold 1-4 covers),
       2-3 elliptical blobs restored to C (each 1-5 % of the image) -> per-image AUC at 256, blob vs rest.
real : maps for the 170 trustmark train images (cached d256 of t1/out/tmofficial/raw_train.npz) ->
       per-image AUC at 128 vs true masks, by edit type, folds 1-4 and fold 0. Saves t1/out/tmrm/real_maps128.npz."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

torch.set_num_threads(4)
TASK = Path(os.getcwd())
OUT = TASK / 't1/out/tmrm'; OUT.mkdir(parents=True, exist_ok=True)
SIG = (2, 4, 8)


def auc(s, g, ign=None):
    s = np.asarray(s, np.float64).ravel(); g = np.asarray(g).ravel().astype(bool)
    if ign is not None:
        k = ~ign.ravel(); s, g = s[k], g[k]
    p = g.sum(); n = g.size - p
    if p == 0 or n == 0: return np.nan
    r = rankdata(s); return float((r[g].sum() - p * (p + 1) / 2) / (p * n))


def G(a, s): return gaussian_filter(a, s)


def load_tm():
    from trustmark import TrustMark
    return TrustMark(use_ECC=False, verbose=False, secret_len=100, model_type='Q', device='cpu', loadRemover=True)


def to256(img512):
    return torch.from_numpy(np.asarray(Image.fromarray(img512).resize((256, 256), Image.BILINEAR), np.float32)
                            .transpose(2, 0, 1).copy())[None] / 127.5 - 1


@torch.no_grad()
def residual(tm, z, bits):
    sec = torch.from_numpy(np.asarray(bits, np.float32))[None]
    st, _ = tm.encoder(z, sec)
    r = st.clamp(-1, 1) - z
    return (r - r.mean(dim=(2, 3), keepdim=True))[0].numpy()


def maps(d, r):
    out = {}
    dd = (d ** 2).sum(0); rr = (r ** 2).sum(0); dr = (d * r).sum(0)
    for s in SIG:
        gd, gr, gdr = G(dd, s), G(rr, s), G(dr, s)
        out[f'mag{s}'] = np.sqrt(gd)
        out[f'a{s}'] = -gdr / (gr + 1e-8)
        out[f'c{s}'] = -gdr / (np.sqrt(gd * gr) + 1e-10)
    return out


def blobs(rng, k):
    yy, xx = np.mgrid[:512, :512]; m = np.zeros((512, 512), bool)
    for _ in range(k):
        area = rng.uniform(0.01, 0.05) * 512 * 512; ar = rng.uniform(0.5, 2.0)
        a = np.sqrt(area / np.pi * ar); b = area / np.pi / a; th = rng.uniform(0, np.pi)
        cy, cx = rng.uniform(60, 452, 2)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th); v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        m |= (u / a) ** 2 + (v / b) ** 2 <= 1
    return m


def synth():
    tm = load_tm(); D = TASK / 't1/out/synth/v0/trustmark/full/s0_0000'
    C = np.load(D / 'C.npy', mmap_mode='r'); Cw = np.load(D / 'Cw.npy', mmap_mode='r')
    msg = np.load(D / 'msg.npy'); old = np.load(D / 'old.npy')
    rng = np.random.default_rng(0); res = {}; t = time.time()
    for i in range(20):
        hole = blobs(rng, int(rng.integers(2, 4)))
        y = np.where(hole[..., None], C[i], Cw[i]).astype(np.uint8)
        y256 = to256(y)
        with torch.no_grad():
            z = tm.removal(y256).clamp(-1, 1)
        d = (z - y256)[0].numpy(); r = residual(tm, z, msg[i])
        M = maps(d, r)
        # oracle residual actually present (Cw - C) for reference
        w = (to256(np.asarray(Cw[i])) - to256(np.asarray(C[i])))[0].numpy()
        lab = ~np.asarray(Image.fromarray(hole.astype(np.uint8) * 255).resize((256, 256), Image.NEAREST), bool)
        ign = np.unpackbits(old[i])[:512 * 512].reshape(512, 512)[::2, ::2].astype(bool)
        for k, v in M.items():
            res.setdefault(k, []).append(auc(v, lab, ign))
        res.setdefault('corr_r_w', []).append(float((r * w).sum() / np.sqrt((r * r).sum() * (w * w).sum())))
        res.setdefault('corr_d_negw', []).append(float(-(d * w).sum() / np.sqrt((d * d).sum() * (w * w).sum())))
        print(i, f'{time.time() - t:.1f}s', 'mag4', round(res['mag4'][-1], 3), 'a4', round(res['a4'][-1], 3),
              'corr(r,w)', round(res['corr_r_w'][-1], 3), 'corr(d,-w)', round(res['corr_d_negw'][-1], 3), flush=True)
    summ = {k: float(np.nanmean(v)) for k, v in res.items()}
    print('SYNTH mean per-image AUC (label = watermarked, blobs restored):', json.dumps(summ, indent=1))
    (OUT / 'synth.json').write_text(json.dumps({'mean': summ, 'per_image': res}, indent=1))


def real():
    sys.path.insert(0, str(TASK / 't1/common'))
    from evaluate import load_train
    meta, gt = load_train()
    idx = json.load(open(TASK / 't1/common/train_index.json'))
    folds = json.load(open(TASK / 't1/common/splits_5fold.json'))['folds']
    fold_of = {n: k for k, f in enumerate(folds) for n in f}
    tm = load_tm(); z0 = np.load(TASK / 't1/out/tmofficial/raw_train.npz')
    names = [str(n) for n in z0['names']]; R = {}; t = time.time()
    for i, n in enumerate(names):
        x = to256(np.asarray(Image.open(TASK / f'release/train/images/{n}.png').convert('RGB')))
        d = z0['d256'][i].astype(np.float32); z = (x + torch.from_numpy(d)[None]).clamp(-1, 1)
        bits = [int(c) for c in meta[n]['message']]
        M = maps(d, residual(tm, z, bits))
        R[n] = {k: v.reshape(128, 2, 128, 2).mean((1, 3)) for k, v in M.items()}
        if i % 40 == 0: print('real', i, f'{time.time() - t:.1f}s', flush=True)
    keys = list(R[names[0]])
    np.savez_compressed(OUT / 'real_maps128.npz', names=np.array(names), keys=np.array(keys),
                        maps=np.stack([np.stack([R[n][k] for k in keys]) for n in names]).astype(np.float16))
    oof = np.load(TASK / 't1/out/oof/trustmark/oof_probs128.npz'); P = {str(a): b.astype(np.float32) for a, b in zip(oof['names'], oof['probs'])}
    rep = {}
    for lab, sel in (('folds1-4', lambda n: fold_of[n] != 0), ('fold0', lambda n: fold_of[n] == 0)):
        for et in ('hole', 'transplant', 'outpainting'):
            ns = [n for n in names if sel(n) and idx[n]['edit_type'] == et]
            row = {k: float(np.nanmean([auc(R[n][k], gt[n]) for n in ns])) for k in keys}
            row['oof_map'] = float(np.nanmean([auc(P[n], gt[n]) for n in ns if n in P]))
            row['n'] = len(ns); rep[f'{lab}/{et}'] = row
            print(lab, et, 'n=%d' % len(ns), ' '.join(f'{k}={row[k]:.3f}' for k in keys + ['oof_map']), flush=True)
    (OUT / 'real_auc.json').write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    {'synth': synth, 'real': real}[sys.argv[1]]()
