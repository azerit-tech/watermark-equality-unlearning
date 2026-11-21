"""Basin width: random edge perturbations of GT (max |d| = D) -> acc / margin, for some train outpaintings."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
D = Dec(); M = meta("train"); idx = json.loads((TASK / "t1/common/train_index.json").read_text())
rng = np.random.default_rng(0)
for nm in ["57bee312c5bd", "9f251d379b53", "1256bb953ed7", "bd90e4da0f52", "0f30c4fb22ca"]:
    g = idx[nm]["bbox512"]; x = load("train", nm); msg = msgof(M[nm]); print(nm, g)
    for d in (2, 4, 6, 8, 12, 16, 24, 32):
        rs = []
        while len(rs) < 12:
            q = tuple(int(np.clip(g[e] + rng.integers(-d, d + 1), 0, 512)) for e in range(4))
            if q[1] - q[0] >= 8 and q[3] - q[2] >= 8: rs.append(q)
        o, acc, mar = D.score(x, rs, msg)
        print("  d=%2d acc mean %.2f max %.2f min %.2f | margin mean %.2f | mean iou %.2f" % (d, acc.mean(), acc.max(), acc.min(), mar.mean(), np.mean([iou(r, g) for r in rs])), flush=True)
    # random rects elsewhere
    rs = []
    while len(rs) < 24:
        y0, x0 = rng.integers(0, 448, 2); h, w = rng.integers(16, 300, 2)
        q = (int(y0), int(min(512, y0 + h)), int(x0), int(min(512, x0 + w)))
        if iou(q, g) == 0: rs.append(q)
    o, acc, mar = D.score(x, rs, msg); print("  random disjoint acc mean %.2f max %.2f margin %.2f" % (acc.mean(), acc.max(), mar.mean()))
print("s/dec %.3f" % (D.t / D.n))
