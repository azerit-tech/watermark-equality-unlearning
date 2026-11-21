import json, numpy as np
from scipy import ndimage
idx=json.load(open("t1/common/train_index.json"))
P=np.load("t1/out/stegatta2/stegastamp/fold0_probs128.npz"); names=list(P["names"]); probs=P["probs"].astype(np.float32)
B=json.load(open("t1/out/stegaedge/stegastamp_train_boxes.json"))["boxes"]
for nm,p in zip(names,probs):
    d=idx[nm]
    if d["edit_type"]!="outpainting": continue
    g=np.array(d["bbox512"]); m=p>0.5
    lab,n=ndimage.label(m)
    if n==0: print(nm,"empty"); continue
    sz=ndimage.sum(m,lab,range(1,n+1)); k=np.argmax(sz)+1; ys,xs=np.where(lab==k)
    r=np.array([ys.min()*4,(ys.max()+1)*4,xs.min()*4,(xs.max()+1)*4])
    c=B.get(nm,{}).get("conf",-1)
    print(nm,"conf %.2f"%c,"gt",g,"unet-bbox err",r-g,"maxabs",np.abs(r-g).max(), "ncomp",n, "frac>0.5 %.3f"%m.mean())
