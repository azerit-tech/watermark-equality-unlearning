import numpy as np, torch, sys, json
S=sys.argv[1]
t=torch.load(S+'/vast/jur/sis_probs.pt',map_location='cpu',weights_only=False)
ps=t['per_strength']; y=t['labels'].numpy(); I=np.arange(3500)
for k,v in ps.items():
    v=v.numpy().astype(np.float64); print(k, v.shape, 'rowsum',v.sum(1).mean(),'p_y',v[I,y].mean(),'flip',(v.argmax(1)!=y).mean(),'CE',(-np.log(np.maximum(v[I,y],1e-12))).mean(), (-np.log(np.maximum(v[I,y],1e-12))).std())
m=sum(v.numpy().astype(np.float64) for v in ps.values())/4
print('mean4 vs probs',np.abs(m-t['probs'].numpy()).max(), 'vs probs_T2',np.abs(m-t['probs_T2'].numpy()).max())
m2=sum(np.sqrt(v.numpy().astype(np.float64)) for v in ps.values()); 
D=json.load(open(S+'/tomo/feat.json')); r=D['rows'][0]; print([k for k in r.keys() if 'clean' not in k and 'alt' not in k]); print(D['oa'])
for r in D['rows']: print(r['n'],r['pub'],'cl %.2f/%.2f alt %.2f/%.2f t2i kl999 %.3f dis999 %.3f kl %.3f dis %.3f ext kl %.3f dis %.3f'%(r['clean_m'],r['clean_s'],r['alt_m'],r['alt_s'],r['t2i_kl999'],r['t2i_dis999'],r['t2i_kl'],r['t2i_dis'],r['ext_kl'],r['ext_dis']))
