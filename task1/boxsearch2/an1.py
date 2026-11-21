import json, numpy as np, sys
M = sys.argv[1]; D = json.load(open(f"t1/out/boxsearch2/diag_{M}.json"))
def r128(r):
    m = np.zeros((512,512)); m[r[0]:r[1], r[2]:r[3]] = 1
    return (m.reshape(128,4,128,4).mean((1,3)) >= .5)
def sc(r, g):
    a, b = r128(r), r128(g); s = 0
    for c in (0,1):
        A, B = (a == c), (b == c); u = (A|B).sum(); s += 1 if u == 0 else (A&B).sum()/u
    return s/2
def iou(a,b):
    iy=max(0,min(a[1],b[1])-max(a[0],b[0])); ix=max(0,min(a[3],b[3])-max(a[2],b[2])); i=iy*ix
    return i/((a[1]-a[0])*(a[3]-a[2])+(b[1]-b[0])*(b[3]-b[2])-i)
out = {k:v for k,v in D.items() if v["edit_type"]=="outpainting"}
print(M, "outpaint", len(out))
for k,v in sorted(out.items(), key=lambda kv: (kv[1]["gt"][1]-kv[1]["gt"][0])*(kv[1]["gt"][3]-kv[1]["gt"][2])):
    g=v["gt"]; h,w=g[1]-g[0],g[3]-g[2]
    print(f"{k} {h:3d}x{w:3d} conf {v['conf']:.2f} iou {iou(v['rect'],g):.2f} oracle_resize {np.round(v['oracle_resize'],2).tolist()} refl {np.round(v.get('oracle_reflect',[]),2).tolist()} gray {np.round(v.get('oracle_gray',[]),2).tolist()}")
