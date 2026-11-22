import sys, json, os, random
import numpy as np, torch
SUBS="/Users/jacquesmougeot/Desktop/Azerit-hack/task1/subs/"
IND="/Users/jacquesmougeot/Desktop/Azerit_t1_mirror/sam_test/ind/"
OUT="/private/tmp/claude-501/-Users-jacquesmougeot-Desktop-Azerit-hack/052a6bcb-a0cb-4fe9-b432-408500ce5ac5/scratchpad/t1verify/"
cand_path=sys.argv[1]
B=torch.load(SUBS+"subA_C2.pt",map_location="cpu",weights_only=False)
C=torch.load(cand_path,map_location="cpu",weights_only=False)
print("file", cand_path, "size MB", os.path.getsize(cand_path)/1e6)
print("type", type(C), "keys", list(C.keys()) if isinstance(C,dict) else None, "C2 keys", list(B.keys()))
for k in B:
    print(" key",k, type(C.get(k)), getattr(C.get(k),'shape',None), getattr(C.get(k),'dtype',None), "contig", getattr(C.get(k),'is_contiguous',lambda:None)())
assert list(C["image_names"])==list(B["image_names"]), "NAMES DIFFER"
print("names identical & same order:", True, len(C["image_names"]), "dup", len(set(C["image_names"]))!=len(C["image_names"]))
mb=B["masks"].numpy(); mc=C["masks"].numpy()
assert mc.shape==(1020,128,128) and C["masks"].dtype==torch.bool
names=list(B["image_names"])
meta={x["image_name"]:x["method"] for x in json.load(open(IND+"test_metadata.json"))}
cand=set(json.load(open(IND+"test_candidates.json")))
diff=(mb!=mc).reshape(1020,-1).sum(1)
ch=[names[i] for i in np.nonzero(diff)[0]]
print("changed", len(ch), "cells", int(diff.sum()), "outside candidates", [n for n in ch if n not in cand][:20], sum(n not in cand for n in ch))
print("candidates unchanged", len(cand-set(ch)), "of", len(cand))
import collections
print("changed by method", collections.Counter(meta[n] for n in ch))
cb=mb.reshape(1020,-1).mean(1); cc=mc.reshape(1020,-1).mean(1)
print("C2 empty/full", int((cb==0).sum()), int((cb==1).sum()), " cand empty/full", int((cc==0).sum()), int((cc==1).sum()))
idx=np.nonzero(diff)[0]
r=cc[idx]/np.maximum(cb[idx],1e-9)
print("C2 cov of changed: min %.4f med %.4f max %.4f"%(cb[idx].min(),np.median(cb[idx]),cb[idx].max()))
print("cand cov of changed: min %.4f med %.4f max %.4f"%(cc[idx].min(),np.median(cc[idx]),cc[idx].max()))
print("ratio cand/C2: min %.3f p10 %.3f med %.3f p90 %.3f max %.3f"%(r.min(),np.percentile(r,10),np.median(r),np.percentile(r,90),r.max()))
ext=[(names[i],round(float(cb[i]),4),round(float(cc[i]),4)) for i,rr in zip(idx,r) if rr<0.2 or rr>5]
print("extremes (ratio<0.2 or >5):", len(ext), ext[:30])
# IoU between C2 and cand masks on changed images
def iou(a,b):
    u=(a|b).sum(); return 1.0 if u==0 else (a&b).sum()/u
ious=np.array([iou(mb[i],mc[i]) for i in idx])
print("fg IoU C2 vs cand on changed: min %.3f med %.3f max %.3f"%(ious.min(),np.median(ious),ious.max()))
# changed images where C2 was not minority
print("changed where C2 cov>=0.5:", int((cb[idx]>=0.5).sum()))
json.dump({"changed":ch,"diff":{names[i]:int(diff[i]) for i in idx}},open(OUT+os.path.basename(cand_path)+".changed.json","w"))
