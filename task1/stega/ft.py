"""Short fine-tune of the v0 StegaStamp U-Net (run from TASK, 1 GPU, fp16):
  python t1/stega/ft.py --init t1/out/train/v0/stegastamp/best.pt --out t1/out/stega/ft1 --minutes 8
Real images = folds 1-4 only (p_real), synthetic = v0 chunks with transplant label offset +1 px and residual
attenuation (x atten with prob p_atten, transplant/hole only), loss = bce + dice_w * dice. Validation fold 0 real."""
import argparse, copy, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
TASK = Path(__file__).resolve().parents[2]
for p in (TASK / "t1/train", TASK / "t1/synth", TASK / "t1/common", TASK):
    sys.path.insert(0, str(p))
from train import loss_fn, predict_probs, binarize, group_scores
from predict import load_model
from dataset import SynthDS, RealDS, MixDS, offset_mask, IGN_R, _unpack, dilate
from shapes import names_in_folds, read_img, sample_minority
from evaluate import evaluate


class HardSynth(SynthDS):
    def __init__(self, *a, atten=0.7, p_atten=0.5, t_offset=1, **k):
        super().__init__(*a, **k); self.atten, self.p_atten, self.t_offset = atten, p_atten, t_offset

    def sample(self, rng, typ=None, info=None):
        typ = typ or self._choice(rng, self.mix)
        if typ == "outpainting":
            return super().sample(rng, typ)
        A, i = self.full[rng.integers(len(self.full))]
        C, Cw, oldC = np.asarray(A["C"][i]), np.asarray(A["Cw"][i]), _unpack(A["old"][i])
        if rng.random() < self.p_atten:
            Cw = np.clip(C.astype(np.float32) + self.atten * (Cw.astype(np.float32) - C), 0, 255).round().astype(np.uint8)
        if typ == "transplant":
            M = sample_minority(self.lib, rng, "patch", avoid=dilate(oldC, IGN_R))
            x = np.where(offset_mask(M, self.t_offset)[..., None].astype(bool), Cw, C)
            return x, M.astype(np.uint8), dilate(oldC, IGN_R), 1
        H = sample_minority(self.lib, rng, "full"); M = (1 - H).astype(np.uint8)
        x = np.where(M[..., None].astype(bool), Cw, C)
        return x, M, dilate(oldC, IGN_R), 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--synth", default="t1/out/synth/v0"); ap.add_argument("--p_real", type=float, default=0.5)
    ap.add_argument("--minutes", type=float, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--dice_w", type=float, default=2.0)
    ap.add_argument("--atten", type=float, default=0.7); ap.add_argument("--p_atten", type=float, default=0.5)
    ap.add_argument("--t_offset", type=int, default=1); ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.995); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = True
    real_names = names_in_folds([1, 2, 3, 4], "stegastamp"); val_names = names_in_folds([0], "stegastamp")
    assert not set(real_names) & set(names_in_folds([0]))
    synth = HardSynth(a.synth, "stegastamp", atten=a.atten, p_atten=a.p_atten, t_offset=a.t_offset)
    cn = synth.cover_names(); assert cn is None or not (cn & set(names_in_folds([0]))), "leak"
    dl = DataLoader(MixDS(synth, RealDS(real_names), a.p_real, 10 ** 8), batch_size=a.bs, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_imgs = [read_img(n) for n in val_names]
    model = load_model(a.init).train(); ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); scaler = torch.cuda.amp.GradScaler()
    log = open(out / "log.jsonl", "w"); t0 = time.time(); it = 0; best = -1

    def validate():
        nonlocal best
        torch.backends.cudnn.allow_tf32 = False
        probs = predict_probs_fp16(ema, val_imgs)
        res = evaluate({n: binarize(p) for n, p in zip(val_names, probs)}, val_names, return_per_image=True)
        g = group_scores(res["per_image"], val_names)
        rec = {"it": it, "min": round((time.time() - t0) / 60, 2), "val": round(res["all"][0], 4),
               **{k.split("/")[1]: v[0] for k, v in g.items() if k.startswith("edit/")}}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save({"model": ema.state_dict(), "args": {"stem_depth": 2}, "it": it, "val": rec["val"]}, out / f"ema_it{it}.pt")
        torch.save({"model": ema.state_dict(), "args": {"stem_depth": 2}, "it": it, "val": rec["val"]}, out / "last.pt")

    validate()
    for x, lab, ign, _ in dl:
        frac = (time.time() - t0) / (a.minutes * 60)
        if frac >= 1:
            break
        lr = a.lr * min(1, (it + 1) / 100) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.9 + 0.1)
        for g in opt.param_groups: g["lr"] = lr
        x = x.cuda(non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last)
        lab = lab.cuda(non_blocking=True); ign = ign.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x)
        import torch.nn.functional as F
        valid = (ign == 0).float(); labf = lab.float(); lgf = lg.float()
        bce = (F.binary_cross_entropy_with_logits(lgf, labf, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
        p = torch.sigmoid(lgf) * valid; dice = 0
        for pc, gc in ((p, labf * valid), ((1 - p) * valid, (1 - labf) * valid)):
            inter = (pc * gc).sum((1, 2)); den = pc.sum((1, 2)) + gc.sum((1, 2))
            dice = dice + (1 - (2 * inter + 1) / (den + 1)).mean() / 2
        loss = bce + a.dice_w * dice
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); scaler.step(opt); scaler.update()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
        it += 1
        if it % 100 == 0:
            print(f"it {it} {(time.time()-t0)/60:.1f}min bce={bce.item():.4f} dice={dice.item():.4f}", flush=True)
        if it % a.eval_every == 0:
            validate(); model.train()
    validate(); (out / "DONE").write_text("ok")


@torch.no_grad()
def predict_probs_fp16(model, imgs, bs=4):
    import torch.nn.functional as F
    model.eval(); outs = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).cuda().float().div(255).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x)
        outs.append(F.avg_pool2d(torch.sigmoid(lg.float())[:, None], 4)[:, 0].cpu().numpy())
    return np.concatenate(outs)


if __name__ == "__main__":
    main()
