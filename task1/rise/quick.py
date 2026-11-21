import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "t1", "rise"))
from maps import masks, map_cov, auc_line, IDX
from scipy.ndimage import gaussian_filter
gz = np.load("release/train/masks_128.npz"); gt = dict(zip(gz["names"].tolist(), gz["masks"]))
for f in sys.argv[1:]:
    z = np.load(f); names = [str(n) for n in z["names"]]; S = z["scores"]; M = masks(S.shape[1]); a = M.mean((1, 2))
    print(f, np.isfinite(S).all())
    def resid(s):
        A = np.stack([np.ones_like(a), a, a ** 2], 1); return s - A @ np.linalg.lstsq(A, s, rcond=None)[0]
    base = np.stack([map_cov(s, M) for s in S]); auc_line("cov", base, names, gt)
    auc_line("cov smooth4", np.stack([gaussian_filter(m, 4) for m in base]), names, gt)
    r = np.stack([map_cov(resid(s), M) for s in S]); auc_line("cov area-resid", r, names, gt)
    auc_line("cov area-resid smooth4", np.stack([gaussian_filter(m, 4) for m in r]), names, gt)
    rk = np.stack([map_cov(np.argsort(np.argsort(resid(s))).astype(np.float64), M) for s in S]); auc_line("cov rank(resid)", rk, names, gt)
