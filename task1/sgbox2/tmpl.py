"""Template matching: StegaStamp residual for the known message (encoder on a cover), resized to (h,w), NCC vs image high-pass.
Usage: python tmpl.py <split> name1 name2 ...  prints rank of GT among top peaks (train)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.ndimage import gaussian_filter
SIZES = sorted(set(int(round(12 * 1.07 ** k)) for k in range(60) if 12 * 1.07 ** k <= 512))


def residual(sess, cover400, msg):
    r = sess.run(["residual"], {"image": cover400[None].astype(np.float32), "secret": msg[None].astype(np.float32)})[0][0]
    return r - r.mean((0, 1), keepdims=True)  # 400x400x3


def hp(img):
    x = img.astype(np.float32) / 255
    return x - np.stack([gaussian_filter(x[..., c], 1.5) for c in range(3)], 2)


def search(sess, img, msg, cover="gray", topn=30):
    c400 = (np.full((400, 400, 3), 0.5, np.float32) if cover == "gray" else
            np.asarray(Image.fromarray(img).resize((400, 400), Image.Resampling.BICUBIC), np.float32) / 255)
    R = torch.from_numpy(residual(sess, c400, msg)).permute(2, 0, 1)[None]  # 1,3,400,400
    I = torch.from_numpy(hp(img)).permute(2, 0, 1).double()  # 3,512,512
    FI = torch.fft.rfft2(I)
    II = torch.zeros(513, 513, dtype=torch.float64); II[1:, 1:] = (I ** 2).sum(0).cumsum(0).cumsum(1)
    out = []
    for h in SIZES:
        for w in SIZES:
            if max(h / w, w / h) > 16: continue
            T = F.interpolate(R, size=(h, w), mode="bilinear", align_corners=False, antialias=True)[0].double()
            T = T - T.mean((1, 2), keepdim=True); T = T - gaussian_filter_t(T)
            Tp = torch.zeros(3, 512, 512, dtype=torch.float64); Tp[:, :h, :w] = T
            C = torch.fft.irfft2(FI * torch.conj(torch.fft.rfft2(Tp)), s=(512, 512)).sum(0)[:513 - h, :513 - w]
            E = II[h:, w:][:513 - h, :513 - w] - II[:513 - h, w:][:, :513 - w] - II[h:, :513 - w][:513 - h] + II[:513 - h, :513 - w]
            N = C / torch.sqrt(E.clamp(min=1e-9) * (T ** 2).sum())
            v, ix = torch.topk(N.flatten(), 3)
            for vv, j in zip(v.tolist(), ix.tolist()):
                y, x = divmod(j, 513 - w); out.append((vv, (y, y + h, x, x + w)))
    out.sort(key=lambda t: -t[0]); res = []
    for v, r in out:
        if all(iou(r, q) < 0.5 for _, q in res): res.append((v, r))
        if len(res) >= topn: break
    return res


def gaussian_filter_t(T):
    return torch.from_numpy(np.stack([gaussian_filter(T[c].numpy(), 1.5) for c in range(3)]))


if __name__ == "__main__":
    split = sys.argv[1]; M = meta(split); idx = json.loads((TASK / "t1/common/train_index.json").read_text())
    D = Dec()
    for nm in sys.argv[2:]:
        img = np.asarray(Image.open(TASK / f"release/{split}/images/{nm}.png").convert("RGB")); msg = msgof(M[nm])
        for cover in ("gray", "img"):
            t0 = time.time(); res = search(D.sess, img, msg, cover)
            g = idx[nm]["bbox512"]; ious = [iou(r, g) for _, r in res]
            print(nm, cover, "%.1fs" % (time.time() - t0), "best iou %.2f at rank %d" % (max(ious), int(np.argmax(ious))), "top:", [(round(v, 3), r, round(i, 2)) for (v, r), i in list(zip(res, ious))[:4]], flush=True)
