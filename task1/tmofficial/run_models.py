"""Run official TrustMark remover (rm_Q) and bbox detector (bbox_Q) on all trustmark train+test images.
Outputs t1/out/tmofficial/raw_{split}.npz : names, d256 (N,3,256,256) f16 = removal(x256)-x256, det_boxes (N,K,4) px512 xyxy, det_scores (N,K), det_labels.
Run from TASK with PYTHONPATH=t1/tmofficial/pkg."""
import json, sys, time, numpy as np, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from trustmark import TrustMark

TASK = Path('.').resolve()
OUT = TASK / 't1/out/tmofficial'; OUT.mkdir(parents=True, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(8)
tm = TrustMark(model_type='Q', loadRemover=True, loadBBoxDetector=True, device=dev, verbose=False)
for split in ['train', 'test']:
    meta = json.loads((TASK / f'release/{split}/metadata.json').read_text())
    names = [m['image_name'] for m in meta if m['method'] == 'trustmark']
    D = np.zeros((len(names), 3, 256, 256), np.float16)
    K = 8; B = np.zeros((len(names), K, 4), np.float32); S = np.zeros((len(names), K), np.float32) - 1; L = np.zeros((len(names), K), np.int16)
    t = time.time()
    for i, n in enumerate(names):
        img = Image.open(TASK / f'release/{split}/images/{n}.png').convert('RGB')
        with torch.no_grad():
            x = transforms.ToTensor()(img.resize((256, 256), Image.BILINEAR)).unsqueeze(0).to(dev) * 2 - 1
            y = tm.removal(x).clamp(-1, 1)
            D[i] = (y - x)[0].float().cpu().numpy()
            full = [transforms.ToTensor()(img).to(dev) * 2 - 1]
            preds, _ = tm.detector.yolo(full)
            b = preds['boxes'][0].float().cpu().numpy(); s = preds['scores'][0].float().cpu().numpy()
            lab = preds['labels'][0].cpu().numpy() if 'labels' in preds else np.zeros(len(s))
            o = np.argsort(-s)[:K]; k = len(o)
            B[i, :k] = b[o]; S[i, :k] = s[o]; L[i, :k] = lab[o]
        if i % 50 == 0: print(split, i, time.time() - t, flush=True)
    np.savez(OUT / f'raw_{split}.npz', names=np.array(names), d256=D, det_boxes=B, det_scores=S, det_labels=L)
    print('done', split, time.time() - t, flush=True)
(OUT / 'raw.done').write_text('ok')
