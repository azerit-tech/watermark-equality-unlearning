"""python t1/stegagrid/gridcost.py : size distribution of StegaStamp train outpaintings + dense grid size (step 4% of side)."""
import json, numpy as np, sys
sys.path.insert(0, 't1/stegagrid')
idx = json.load(open('t1/common/train_index.json')); F = json.load(open('t1/common/splits_5fold.json'))['folds']
fo = {n: k for k, f in enumerate(F) for n in f}
O = [(n, v['bbox512'], fo[n]) for n, v in idx.items() if v['method'] == 'stegastamp' and v['edit_type'] == 'outpainting']
H = np.array([b[1] - b[0] for _, b, _ in O]); W = np.array([b[3] - b[2] for _, b, _ in O]); A = np.maximum(H / W, W / H)
print('n', len(O), 'h min/med/max', H.min(), np.median(H), H.max(), 'w', W.min(), np.median(W), W.max(), 'aspect max %.1f' % A.max(), 'area frac min %.4f max %.3f' % ((H * W).min() / 512**2, (H * W).max() / 512**2))
print('sorted min side', sorted(np.minimum(H, W))[:10], 'sorted max side', sorted(np.maximum(H, W))[-5:])
def grid(smin, smax, step=0.04, A=16, amin=None, amax=None):
    sz = [smin]
    while sz[-1] * (1 + step) <= smax: sz.append(sz[-1] * (1 + step))
    sz = sorted(set(int(round(s)) for s in sz)); n = 0; ns = 0
    for h in sz:
        for w in sz:
            if max(h / w, w / h) > A: continue
            if amin is not None and not (amin <= h * w <= amax): continue
            py = len(range(0, 512 - h + 1, max(1, int(round(step * h))))); px = len(range(0, 512 - w + 1, max(1, int(round(step * w)))))
            n += py * px; ns += 1
    return n, len(sz), ns
for st in (0.04, 0.06, 0.08):
    for smin in (8, 16, 24):
        n, k, ns = grid(smin, 512, st, A=16, amin=(H * W).min() * 0.8, amax=(H * W).max() * 1.2)
        print(f'step {st} min side {smin}: {k} sizes/axis, {ns} (h,w) pairs, {n:,} rects/img -> {n/160/60:.0f} min at 160/s')
