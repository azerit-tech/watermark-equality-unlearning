import sys, json, os, random
import numpy as np, torch
from PIL import Image, ImageDraw
SUBS="/Users/jacquesmougeot/Desktop/Azerit-hack/task1/subs/"
IMG="/Users/jacquesmougeot/Desktop/Azerit_t1_mirror/release/test/images/"
OUT="/private/tmp/claude-501/-Users-jacquesmougeot-Desktop-Azerit-hack/052a6bcb-a0cb-4fe9-b432-408500ce5ac5/scratchpad/t1verify/png/"
os.makedirs(OUT,exist_ok=True)
cp=sys.argv[1]; tag=os.path.basename(cp).replace(".pt","")
B=torch.load(SUBS+"subA_C2.pt",weights_only=False); C=torch.load(cp,weights_only=False)
names=B["image_names"]; mb=B["masks"].numpy(); mc=C["masks"].numpy()
meta={x["image_name"]:x["method"] for x in json.load(open("/Users/jacquesmougeot/Desktop/Azerit_t1_mirror/sam_test/ind/test_metadata.json"))}
diff=(mb!=mc).reshape(len(names),-1).sum(1); idx=np.nonzero(diff)[0]
order=idx[np.argsort(-diff[idx])]
big=list(order[:6]); rest=[i for i in idx if i not in big]; random.seed(0); rnd=random.sample(rest,min(6,len(rest)))
def outline(m):
    m=m.astype(bool); e=m.copy()
    e[1:]&=m[:-1]; e[:-1]&=m[1:]; e[:,1:]&=m[:,:-1]; e[:,:-1]&=m[:,1:]
    return m&~e
def panel(i):
    im=Image.open(IMG+names[i]+".png").convert("RGB").resize((384,384))
    a=np.array(im)
    up=lambda m: np.kron(m.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    ob=outline(up(mb[i])); oc=outline(up(mc[i]))
    # dilate outline a bit
    def thick(o):
        t=o.copy(); t[1:]|=o[:-1]; t[:,1:]|=o[:,:-1]; return t
    ob=thick(ob); oc=thick(oc)
    left=a.copy(); left[ob]=[0,255,0]; left[oc]=[255,0,0]
    # fill view: green tint C2-only, red tint cand-only, yellow both
    f=a.copy().astype(float)*0.4
    U=up(mb[i]); V=up(mc[i])
    f[U&V]+= [150,150,0]; f[U&~V]+=[0,150,0]; f[~U&V]+=[150,0,0]
    p=np.concatenate([a,left,np.clip(f,0,255).astype(np.uint8)],1)
    P=Image.fromarray(p); d=ImageDraw.Draw(P)
    d.rectangle([0,0,383,14],fill=(0,0,0)); d.text((2,1),f"{names[i]} {meta[names[i]]} C2cov={mb[i].mean():.3f} new={mc[i].mean():.3f} d={diff[i]}",fill=(255,255,255))
    return P
for k,grp in (("big",big),("rnd",rnd)):
    ps=[panel(i) for i in grp]
    W=ps[0].width; H=ps[0].height
    S=Image.new("RGB",(W,H*len(ps)))
    for j,p in enumerate(ps): S.paste(p,(0,j*H))
    # split into 2 sheets of 3
    for s in range(0,len(ps),3):
        T=Image.new("RGB",(W,H*len(ps[s:s+3])))
        for j,p in enumerate(ps[s:s+3]): T.paste(p,(0,j*H))
        T.save(OUT+f"{tag}_{k}_{s//3}.png")
print("ok", [names[i] for i in big], [names[i] for i in rnd])
