"""Compare t1/out/tmofficial/trustmark_train_boxes.json (boxsearch + detector starts) with boxsearch alone. Run from TASK."""
import json, sys, os, numpy as np
sys.path.insert(0, os.getcwd())
from wmloc.utils import miou_image, downsample_mask
from PIL import Image
b = json.load(open('t1/out/tmofficial/trustmark_train_boxes.json'))['boxes']
f0 = set(json.load(open('t1/common/splits_5fold.json'))['folds'][0])
gz = np.load('release/train/masks_128.npz'); gt = dict(zip(gz['names'].tolist(), gz['masks']))
def m128(r):
    m = np.zeros((512, 512), np.uint8); m[r[0]:r[1], r[2]:r[3]] = 1; return downsample_mask(m) if True else m
for t in ('outpainting', 'transplant', 'hole'):
    R = {n: v for n, v in b.items() if v['edit_type'] == t}
    print('==', t, len(R))
    for n, v in sorted(R.items(), key=lambda kv: -kv[1]['conf']):
        if v['src'] == 'det' or t == 'outpainting':
            print(f"  {n} f0={int(n in f0)} src={v['src']:9s} conf {v['conf']:.2f} (bs {v['boxsearch_conf']:.2f}) iou {v['iou']:.3f} (bs {v['boxsearch_iou']:.3f}) det_score {v['det_score']}")
    for thr in (0.8, 0.9, 0.95):
        new = [v for v in R.values() if v['conf'] >= thr]; old = [v for v in R.values() if v['boxsearch_conf'] >= thr]
        print(f"  conf>={thr}: new {len(new)} (IoU med {np.median([v['iou'] for v in new]) if new else 0:.3f}, IoU>0.5: {sum(v['iou'] > 0.5 for v in new)}) | boxsearch {len(old)} (IoU med {np.median([v['boxsearch_iou'] for v in old]) if old else 0:.3f}, >0.5: {sum(v['boxsearch_iou'] > 0.5 for v in old)})")
    # official per-image score if rect used as mask when conf>=0.9 (vs boxsearch), only on images where they differ
    for thr in (0.9,):
        dn = [(miou_image(m128(v['rect']), gt[n]), miou_image(m128(v['boxsearch_rect']), gt[n])) for n, v in R.items() if v['conf'] >= thr or v['boxsearch_conf'] >= thr]
        if dn: d = np.array(dn); print(f"  rect-as-mask score (images with conf>={thr} in either, n={len(d)}): new {d[:,0].mean():.3f} boxsearch {d[:,1].mean():.3f}")
