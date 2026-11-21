import json, numpy as np
idx=json.load(open("t1/common/train_index.json")); f0=set(json.load(open("t1/common/splits_5fold.json"))["folds"][0])
for src in ["t1/out/stegaedge/stegastamp_train_boxes.json","t1/out/sgbox2/stegastamp_train_boxes.json","t1/out/stegaedge/edge_train.json"]:
    try: B=json.load(open(src))
    except Exception as e: print(src,e); continue
    B=B.get("boxes",B)
    print("==",src,len(B))
    errs=[]
    for nm,d in idx.items():
        if d["method"]!="stegastamp" or d["edit_type"]!="outpainting": continue
        if nm not in B: print("  missing",nm); continue
        b=B[nm]; r=b["rect"] if isinstance(b,dict) and "rect" in b else None
        if r is None: print("  keys",list(b)[:8] if isinstance(b,dict) else type(b)); break
        g=d["bbox512"]; e=np.array(r)-np.array(g); c=b.get("conf",-1)
        errs.append((nm in f0,c,e))
        print("  %s %s conf %.2f err %s"%(nm,"F0" if nm in f0 else "  ",c,e))
    E=np.array([e for f,c,e in errs if c<0.9 and not f]); 
    if len(E): print(" failed nonF0 n=%d |err| max-edge pct"%len(E), np.percentile(np.abs(E).max(1),[10,25,50,75,90]), "frac all edges<=16:", (np.abs(E).max(1)<=16).mean(), "<=32:",(np.abs(E).max(1)<=32).mean())
