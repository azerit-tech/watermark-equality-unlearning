"""Remover maps from raw_{split}.npz (d256 = rm(x256)-x256). Variants: |d| smoothed, local corr(d, HF(x)), product.
Stage 1: AUC per image (mean) by edit type on folds 1-4 -> pick best variant (a priori pool). Stage 2: fold 0 AUC + combo w=0.5 with sources.
Run from TASK: python t1/tmofficial/rm_eval.py"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
TASK = Path(os.getcwd())
for p in ("", "t1/common", "t1/train", "t1/mgr", "t1/synth", "t1/reembed"): sys.path.insert(0, str(TASK / p))
from combo import auc, comb, score, boot  # noqa
from evaluate import load_train  # noqa
from regime_guard import load_regimes  # noqa
idx = json.load(open('t1/common/train_index.json')); folds = json.load(open('t1/common/splits_5fold.json'))['folds']
fold_of = {n: k for k, f in enumerate(folds) for n in f}
TYPES = ('hole', 'transplant', 'outpainting')

def G(a, s): return gaussian_filter(a, (0, s, s)) if a.ndim == 3 else gaussian_filter(a, s)
def pool(m): return m.reshape(128, 2, 128, 2).mean((1, 3))  # 256 -> 128

def maps_for(split, n, d):
    x = np.asarray(Image.open(f'release/{split}/images/{n}.png').convert('RGB').resize((256, 256), Image.BILINEAR), np.float32).transpose(2, 0, 1) / 127.5 - 1
    d = d.astype(np.float32)
    r = x - G(x, 1.0)                       # image HF residual
    dh = d - G(d, 1.0)                      # remover HF
    out = {}
    for s in (4, 8, 16):                    # sigma at 256 = 8/16/32 px at 512
        mag = np.sqrt(G((d ** 2).sum(0), s)); out[f'mag{s}'] = mag
        out[f'maghf{s}'] = np.sqrt(G((dh ** 2).sum(0), s))
        num = G((d * r).sum(0), s); den = np.sqrt(G((d ** 2).sum(0), s) * G((r ** 2).sum(0), s)) + 1e-8
        c = -num / den; out[f'corr{s}'] = c   # d ~ -w, so -corr expected positive in watermarked zones
        out[f'prod{s}'] = mag * np.clip(c, 0, None)
    return {k: pool(v) for k, v in out.items()}

if __name__ == '__main__':
    meta, gt = load_train(); reg = load_regimes('train')
    z = np.load('t1/out/tmofficial/raw_train.npz'); names = [str(n) for n in z['names']]
    M = {n: maps_for('train', n, z['d256'][i]) for i, n in enumerate(names)}
    keys = list(M[names[0]])
    def table(sel, lab):
        print(f'\n{lab}  (mean per-image AUC; hole/transplant/outpainting/all)')
        res = {}
        for k in keys:
            a = {t: np.nanmean([auc(M[n][k], gt[n]) for n in sel if idx[n]['edit_type'] == t]) for t in TYPES}
            a['all'] = np.nanmean([auc(M[n][k], gt[n]) for n in sel]); res[k] = a
            print(f'  {k:9s} ' + ' '.join(f'{a[t]:.3f}' for t in TYPES + ('all',)))
        return res
    tr = [n for n in names if fold_of[n] != 0]; f0 = [n for n in names if fold_of[n] == 0]
    r14 = table(tr, 'folds 1-4 (n=%d)' % len(tr))
    best = max(r14, key=lambda k: r14[k]['all']); print('\nchosen on folds 1-4:', best, r14[best])
    r0 = table(f0, 'fold 0 (n=%d)' % len(f0))
    # combo on fold 0
    src_cfg = json.loads((TASK / 't1/mgr/sources.json').read_text())
    et = np.array([idx[n]['edit_type'] for n in f0])
    for sname, path in (('frozen_tm', 't1/out/frozen/trustmark/fold0_probs128.npz'), ('agent_tmdense', 't1/out/tmdense/best/fold0_probs128.npz')):
        zz = np.load(path); d = {str(n): p.astype(np.float32) for n, p in zip(zz['names'], zz['probs'])}
        for w in (0.0, 0.5):
            sc = np.array([score(d[n] if w == 0 else comb(d[n], M[n][best], w), reg[n], gt[n]) for n in f0])
            au = np.array([auc(d[n] if w == 0 else comb(d[n], M[n][best], w), gt[n]) for n in f0])
            if w == 0: b0 = sc
            lo, hi = boot(sc - b0) if w else (0, 0)
            print(f'  {sname:14s} w={w} score {sc.mean():.4f} [' + ' '.join(f'{t[0]}{sc[et == t].mean():.3f}' for t in TYPES) + f'] auc {np.nanmean(au):.3f} d={sc.mean() - b0.mean():+.4f} CI[{lo:+.3f},{hi:+.3f}]')
