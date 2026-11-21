"""Evaluate official TrustMark bbox detector boxes (raw_train.npz) vs masks; compare with boxsearch. Run from TASK."""
import json, numpy as np
from pathlib import Path
from PIL import Image
TASK = Path('.').resolve()
idx = json.load(open('t1/common/train_index.json'))
z = np.load('t1/out/tmofficial/raw_train.npz'); names = [str(n) for n in z['names']]
bs = json.load(open('t1/out/boxsearch/trustmark_train_boxes.json'))['boxes']
def m512(n): return np.array(Image.open(f'release/train/masks/{n}.png').convert('L')) > 127
def rmask(r):
    m = np.zeros((512, 512), bool); y0, y1, x0, x1 = [int(round(v)) for v in r]; m[max(y0,0):y1, max(x0,0):x1] = True; return m
def iou(a, b): u = (a | b).sum(); return (a & b).sum() / u if u else 1.0
rows = {t: [] for t in ('outpainting', 'transplant', 'hole')}
for i, n in enumerate(names):
    t = idx[n]['edit_type']; S = z['det_scores'][i]; B = z['det_boxes'][i]; L = z['det_labels'][i]
    M = m512(n); k = int(np.argmax(S))
    x0, y0, x1, y1 = B[k]; r = [y0, y1, x0, x1]
    full = (S[k] <= 0) or ((x1 - x0) * (y1 - y0) > 0.97 * 512 * 512)
    ys, xs = np.where(M); gb = [ys.min(), ys.max() + 1, xs.min(), xs.max() + 1] if len(ys) else [0, 0, 0, 0]
    b2 = bs.get(n)
    rows[t].append(dict(n=n, s=float(S[k]), lab=int(L[k]), full=bool(full), iou=iou(rmask(r), M), err=np.array(r) - np.array(gb),
                        cov=M.mean(), bs_iou=iou(rmask(b2['rect']), M) if b2 and b2['rect'] else np.nan, bs_conf=b2['conf'] if b2 else np.nan, nbox=int((S > 0).sum())))
for t, R in rows.items():
    s = np.array([r['s'] for r in R]); io = np.array([r['iou'] for r in R]); full = np.array([r['full'] for r in R]); cov = np.array([r['cov'] for r in R])
    print(f'\n== {t} n={len(R)}  mean cov {cov.mean():.2f}  non-full boxes {np.sum(~full)}  score>0.5 non-full {np.sum((~full)&(s>0.5))}')
    nf = ~full
    if nf.any():
        print(f'  non-full: IoU median {np.median(io[nf]):.3f} mean {io[nf].mean():.3f}  scores median {np.median(s[nf]):.3f}')
    if t == 'outpainting':
        print('  name          score lab  IoU_det  edge_err(y0,y1,x0,x1)   bs_IoU bs_conf  nbox')
        for r in sorted(R, key=lambda r: -r['s']):
            print(f"  {r['n']} {r['s']:.3f} {r['lab']} {r['iou']:.3f} {str(np.round(r['err']).astype(int).tolist()):22s} {r['bs_iou']:.3f} {r['bs_conf']:.2f} {r['nbox']}")
        for thr in (0.05, 0.1, 0.3, 0.5, 0.9):
            sel = nf & (s >= thr); print(f'  thr {thr}: {sel.sum()} boxes, IoU median {np.median(io[sel]) if sel.any() else np.nan:.3f}, IoU>0.5: {(io[sel] > 0.5).sum()}')
        e = np.array([r['err'] for r in R])[nf & (io > 0.5)]
        if len(e): print('  edge err (IoU>0.5) median', np.median(e, 0), 'MAE', np.abs(e).mean(0))
    else:
        print('  score dist of non-full boxes:', np.round(np.sort(s[nf])[::-1][:15], 3).tolist())
        print('  IoU of those vs mask:', np.round(io[nf][np.argsort(-s[nf])][:15], 3).tolist())
