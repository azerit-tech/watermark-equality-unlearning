"""Official TrustMark bbox detector boxes as starting points for the decoding rectangle search (t1/boxsearch/rectsearch.refine).
Scope = same as boxsearch (minority regime: fullacc < 0.76). For each image: refine each non-full detector box (score>0) by decoding,
then keep the best of {boxsearch rect, refined det rects} by the boxsearch objective (acc + 0.1*soft).
Writes t1/out/tmofficial/trustmark_{train,test}_boxes.json. Run from TASK: python t1/tmofficial/det_refine.py --device cuda"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
TASK = Path(os.getcwd()); sys.path.insert(0, str(TASK)); sys.path.insert(0, str(TASK / 't1/boxsearch'))
from rectsearch import Dec, refine, ok, iou, ASPECT, THR  # noqa
from wmloc.watermarks import load_watermark  # noqa
ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cuda'); a = ap.parse_args(); torch.set_num_threads(8)
M = 'trustmark'; A = ASPECT[M]; OUT = TASK / 't1/out/tmofficial'
D = Dec(load_watermark(M, device=a.device), a.device)
idx = json.loads((TASK / 't1/common/train_index.json').read_text())
for split in ('train', 'test'):
    meta = {m['image_name']: m for m in json.loads((TASK / f'release/{split}/metadata.json').read_text()) if m['method'] == M}
    bs = json.loads((TASK / f't1/out/boxsearch/{M}_{split}_boxes.json').read_text())['boxes']
    z = np.load(OUT / f'raw_{split}.npz'); names = [str(n) for n in z['names']]
    res = {}; t0 = time.time()
    for i, n in enumerate(names):
        if n not in bs: continue
        msg = np.array([int(c) for c in meta[n]['message']], np.float64)
        x = torch.from_numpy(np.asarray(Image.open(TASK / f'release/{split}/images/{n}.png').convert('RGB'))).to(a.device).permute(2, 0, 1)[None].float() / 255
        cands = []
        r0 = tuple(bs[n]['rect']); o0, a0, _ = D.score(x, [r0], msg); cands.append(('boxsearch', r0, float(o0[0]), float(a0[0]), None, None))
        S = z['det_scores'][i]; B = z['det_boxes'][i]
        for k in range(len(S)):
            if S[k] <= 0: continue
            x0, y0, x1, y1 = [int(round(v)) for v in B[k]]
            s = (max(0, y0), min(512, y1), max(0, x0), min(512, x1))
            if (s[1] - s[0]) * (s[3] - s[2]) > 0.97 * 512 * 512 or not ok(s, 999): continue
            os_, _, _ = D.score(x, [s], msg)
            r, ob = refine(D, x, msg, s, float(os_[0]), A if ok(s, A) else 999)
            _, ac, _ = D.score(x, [r], msg)
            cands.append(('det', tuple(map(int, r)), float(ob), float(ac[0]), float(S[k]), list(s)))
        best = max(cands, key=lambda c: c[2])
        e = {'rect': list(map(int, best[1])), 'conf': best[3], 'src': best[0], 'obj': best[2], 'det_score': best[4], 'det_start': best[5],
             'boxsearch_rect': list(r0), 'boxsearch_conf': bs[n]['conf'],
             'det_top': {'score': float(S.max()), 'xyxy': [float(v) for v in B[int(np.argmax(S))]]}}
        if split == 'train':
            e['edit_type'] = idx[n]['edit_type']; e['bbox512'] = idx[n]['bbox512']; e['iou'] = iou(best[1], idx[n]['bbox512'])
            e['boxsearch_iou'] = bs[n].get('iou')
        res[n] = e
    (OUT / f'{M}_{split}_boxes.json').write_text(json.dumps({'format': 'rect=[y0,y1,x0,x1] px 512 exclusive; conf=bit acc of crop rect+3px', 'method': 'boxsearch + official bbox_Q detector starts refined by decoding', 'boxes': res}, indent=0))
    print(split, len(res), 'images', f'{time.time() - t0:.0f}s', 'det chosen', sum(v['src'] == 'det' for v in res.values()), flush=True)
(OUT / 'det_refine.done').write_text('ok')
