import json, os, numpy as np
from scipy.stats import chisquare
from collections import Counter
rows=json.load(open(os.path.expanduser("~/rectprior_rows.json")))
A=np.array([r[2:6] for r in rows]); me=np.array([r[1] for r in rows])
h=A[:,1]-A[:,0]; w=A[:,3]-A[:,2]
V={"y0":A[:,0],"y1":A[:,1],"x0":A[:,2],"x1":A[:,3],"h":h,"w":w}
for q in (2,4,8,16,32,64):
    ps=[]
    for k,v in V.items():
        ps.append("%s p=%.2f"%(k,chisquare(np.bincount(v%q,minlength=q)).pvalue))
    print("mod",q," ".join(ps))
pooled=np.concatenate([A.ravel()])
for q in (4,8,16): print("pooled coords mod",q,"p=%.3f"%chisquare(np.bincount(pooled%q,minlength=q)).pvalue)
print("squares |ar-1|<=2%:", int((np.abs(w/h-1)<=0.02).sum()), "/129")
print("dup (h,w):", [k for k,c in Counter(zip(h,w)).items() if c>1])
print("min margin to border pct", np.percentile(np.minimum.reduce([A[:,0],512-A[:,1],A[:,2],512-A[:,3]]),[0,10,25,50,75,90]))
for m in ("pixelseal","stegastamp","trustmark"):
    s=me==m; print(m,"h med %d w med %d area med %.3f ar med %.2f ar>3 or<1/3: %d/43"%(np.median(h[s]),np.median(w[s]),np.median((h*w)[s])/512**2,np.median((w/h)[s]),((w/h)[s]>3).sum()+((w/h)[s]<1/3).sum()))
# entropy of rect -> number of candidates needed to cover 90% with all edges within 5px if drawn from empirical prior: approx via kde-free counting
# Count distinct "cells" of the 4D space at 10px resolution occupied: equals n -> no clustering
cells=Counter(tuple(r) for r in (A//10)); print("distinct 10px cells", len(cells), "of", len(A))
# nearest-neighbour Linf distance between rects (if small set of templates, NN would be ~0)
D=np.abs(A[:,None,:]-A[None,:,:]).max(-1); np.fill_diagonal(D,9999); print("NN Linf dist pct", np.percentile(D.min(1),[0,10,25,50,75]))
