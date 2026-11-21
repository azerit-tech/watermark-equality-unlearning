import json, os, numpy as np
from PIL import Image
from collections import Counter
T=os.getcwd()
idx=json.load(open("t1/common/train_index.json"))
F=json.load(open("t1/common/splits_5fold.json"))["folds"]
f0=set(F[0])
rows=[]
for nm,d in sorted(idx.items()):
    if d["edit_type"]!="outpainting": continue
    m=np.array(Image.open(f"release/train/masks/{nm}.png"))
    if m.ndim==3: m=m[...,0]
    b=m>127
    ys=np.where(b.any(1))[0]; xs=np.where(b.any(0))[0]
    y0,y1,x0,x1=ys[0],ys[-1]+1,xs[0],xs[-1]+1
    exact = b[y0:y1,x0:x1].all() and b.sum()==(y1-y0)*(x1-x0)
    rows.append((nm,d["method"],int(y0),int(y1),int(x0),int(x1),exact,nm in f0))
print(len(rows), Counter(r[1] for r in rows), "nonexact", sum(not r[6] for r in rows))
for r in rows:
    nm,me,y0,y1,x0,x1,ex,f=r; h=y1-y0; w=x1-x0
    print(f"{nm} {me[:5]} {'F0' if f else '  '} y {y0:3d}-{y1:3d} x {x0:3d}-{x1:3d} h{h:3d} w{w:3d} cy{(y0+y1)/2:6.1f} cx{(x0+x1)/2:6.1f} area{h*w/512/512:.3f} ar{w/h:.3f} marg t{y0} b{512-y1} l{x0} r{512-x1}")
A=np.array([r[2:6] for r in rows]); h=A[:,1]-A[:,0]; w=A[:,3]-A[:,2]
for name,v in [("y0",A[:,0]),("y1",A[:,1]),("x0",A[:,2]),("x1",A[:,3]),("h",h),("w",w)]:
    print(name, "mod8", Counter(v%8).most_common(4), "mod16", Counter(v%16).most_common(3), "mod64",Counter(v%64).most_common(3))
print("touch edge any:", sum(((A[:,0]==0)|(A[:,1]==512)|(A[:,2]==0)|(A[:,3]==512))))
print("area", np.percentile(h*w/512**2,[0,10,25,50,75,90,100]))
print("ar", np.percentile(w/h,[0,10,25,50,75,90,100]))
print("cy-256", np.percentile((A[:,0]+A[:,1])/2-256,[0,10,25,50,75,90,100]))
print("cx-256", np.percentile((A[:,2]+A[:,3])/2-256,[0,10,25,50,75,90,100]))
print("h", np.percentile(h,[0,10,25,50,75,90,100]), "w", np.percentile(w,[0,10,25,50,75,90,100]))
print("sym tb |t-b|", np.percentile(np.abs(A[:,0]-(512-A[:,1])),[0,10,25,50,75,90,100]))
print("sym lr |l-r|", np.percentile(np.abs(A[:,2]-(512-A[:,3])),[0,10,25,50,75,90,100]))
json.dump([list(map(lambda z: z if not isinstance(z,(np.integer,np.bool_)) else z.item(), r)) for r in rows], open(os.path.expanduser("~/rectprior_rows.json"),"w"))
