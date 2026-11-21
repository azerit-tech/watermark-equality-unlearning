"""PixelSeal/any: candidate discriminating features for found boxes vs transplant/hole false positives."""
import json, numpy as np, sys
M = sys.argv[1]; D = json.load(open(f"t1/out/boxsearch2/diag_{M}.json")); W = 16
def iou(a,b):
    iy=max(0,min(a[1],b[1])-max(a[0],b[0])); ix=max(0,min(a[3],b[3])-max(a[2],b[2])); i=iy*ix
    return i/((a[1]-a[0])*(a[3]-a[2])+(b[1]-b[0])*(b[3]-b[2])-i)
rows = []
for k, v in D.items():
    if v["conf"] < 0.7: continue
    pos = v["edit_type"] == "outpainting" and iou(v["rect"], v["gt"]) > 0.3
    prof = np.array([v["prof"][str(e)]["acc"] for e in range(4)])          # [4, 2W+1]
    drop8 = v["conf"] - np.mean([prof[:, W - 8], prof[:, W + 8]])           # acc drop when one edge moves by 8 px
    drop16 = v["conf"] - np.mean([prof[:, 0], prof[:, -1]])
    padc = min(v["acc_p2"], v["acc_p4"]); pad6 = v["acc_p6"]
    rows.append((pos, v["edit_type"], v["conf"], drop8, drop16, padc, pad6, k))
rows.sort(key=lambda r: -r[2])
print("pos  type        conf  drop8  drop16  min(p2,p4)  p6")
for r in rows: print(f"{int(r[0])}  {r[1]:11s} {r[2]:.3f} {r[3]:+.3f} {r[4]:+.3f}   {r[5]:.3f}     {r[6]:.3f}")
