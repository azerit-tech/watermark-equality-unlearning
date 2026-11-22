import numpy as np, torch, sys
S=sys.argv[1]
d=np.load(S+'/an2/logits.npz'); ks=sorted(set(k.split('::')[0] for k in d.files)); print(len(ks), ks); print({v:d['..__model::'+v].shape for v in ('clean','alt','t2i','ext')})
s=np.load(S+'/tomo/sis.npz'); print({k:s[k].shape for k in s.files})
c=np.load('/Users/jacquesmougeot/Desktop/Azerit-hack/unlearning/runs_vast/cf/cf_probs_k7.npz'); print({k:c[k].shape for k in c.files})
t=torch.load(S+'/vast/jur/sis_probs.pt',map_location='cpu',weights_only=False); print({k:(getattr(v,'shape',v) if not isinstance(v,(dict,list)) else (type(v),len(v), (list(v.keys())[:10] if isinstance(v,dict) else ''))) for k,v in t.items()})
