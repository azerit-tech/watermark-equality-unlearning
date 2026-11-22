import json, sys, numpy as np, torch
S=sys.argv[1]
exec(open(S+'/tomo/feat.py').read().split("rows=[]")[0].replace("S=sys.argv[1]",""))
D=json.load(open(S+'/tomo/feat.json')); rows=D['rows']; OA=D['oa']
SIS={"sM_ep08","c11","c13","c16","e2","qB","wA"}
names=[r['n'] for r in rows]; pub=np.array([r['pub'] for r in rows]); fam=np.array([n in SIS for n in names]); M=len(rows)
LQ={v:np.stack([lsm(d[key[n]+'::'+v]) for n in names]) for v in ('clean','alt')}   # [M,N,100]
LO={v:lsm(o[v]) for v in ('clean','alt')}
RETv=np.array([max(0,1-0.3*r['t2i_dis999'])/(1+0.3*r['t2i_kl999']) for r in rows])
def MSv(mg,sg):
    out=[]
    for r in rows:
        c=max(0.0,1-(abs(r['clean_m']-mg)+abs(r['clean_s']-sg))/(mg+sg)); a=max(0.0,1-(abs(r['alt_m']-mg)+abs(r['alt_s']-sg))/(abs(OA[0]-mg)+abs(OA[1]-sg))); out.append((c+a)/2)
    return np.array(out)
MSL={'sis':MSv(2.31,3.77),'law':MSv(2.4,4.4)}
def pred(G):
    G=np.maximum(G,1e-12); G=G/G.sum(1,keepdims=True); ent=(G*np.log(G)).sum()/N; pr=np.zeros(M)
    for v in ('clean','alt'):
        ks=ent-np.einsum('nc,mnc->m',G,LQ[v])/N; ko=ent-(G*LO[v]).sum()/N
        pr+=np.maximum(0,1-ks/ko)/2
    return pr
def stats(pre):
    sc=lambda m:(pub[m]*pre[m]).sum()/(pre[m]**2).sum()
    c=sc(np.ones(M,bool)); return dict(pear=np.corrcoef(pub,pre)[0,1],scale=c,ratio=sc(fam)/sc(~fam),rms=np.sqrt(np.mean((pub-c*pre)**2)))
def loo(P):   # P [n_theta, M] predicted (MS*PRED*RET); nested choice of theta + scale on the other 27
    e=np.zeros(M); ch=[]
    for j in range(M):
        m=np.ones(M,bool); m[j]=False
        c=(pub[m]*P[:,m]).sum(1)/(P[:,m]**2).sum(1); r=((pub[m]-c[:,None]*P[:,m])**2).mean(1); t=r.argmin(); ch.append(t)
        e[j]=pub[j]-c[t]*P[t,j]
    return e,ch
t=torch.load(S+'/vast/jur/sis_probs.pt',map_location='cpu',weights_only=False)
PS=np.stack([t['per_strength'][k].numpy().astype(np.float64) for k in ('0.50','0.60','0.65','0.70')])
def temp(p,T):
    l=np.log(np.maximum(p,1e-12))/T; l-=l.max(-1,keepdims=True); e=np.exp(l); return e/e.sum(-1,keepdims=True)
base=temp(PS,2).mean(0); print('recon probs_T2 err',np.abs(base-sp).max())
po=np.exp(LO['clean']); cf=np.load('/Users/jacquesmougeot/Desktop/Azerit-hack/unlearning/runs_vast/cf/cf_probs_k7.npz'); oof=cf['probs_T2'].astype(np.float64)
cf8=np.load('/Users/jacquesmougeot/Desktop/Azerit-hack/unlearning/runs_vast/cf/cf_probs.npz')['probs_T2'].astype(np.float64)
def lab(G,k):
    G=G.copy(); G[I,y]*=k; return G/G.sum(1,keepdims=True)
sl=-np.log(np.maximum(PS.mean(0)[I,y],1e-12)); rk=np.argsort(np.argsort(-sl))/N   # 0 = hardest by sisters
def tailk(G,f,k):
    G=G.copy(); m=rk<f; G[m,y[m]]*=k; return G/G.sum(1,keepdims=True)
FAM={
 'A temp per sister (T, base 2)':[(T,temp(PS,T).mean(0)) for T in (1,1.25,1.5,1.75,2,2.5,3,4)],
 'A2 power on mean (beta, base 1)':[(b,sp**b) for b in (0.5,0.7,0.85,1,1.2,1.5,2,3)],
 'B strength tilt (gamma, base 0)':[(g,np.tensordot(np.exp(g*np.arange(4))/np.exp(g*np.arange(4)).sum(),temp(PS,2),1)) for g in (-3,-1,-0.5,0,0.5,1,3)],
 'B2 n sisters used (first k)':[(k,temp(PS[:k],2).mean(0)) for k in (1,2,3,4)],
 'C linear mix with orig (lam, base 0)':[(l,np.maximum((1-l)*sp+l*po,0)) for l in (-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5)],
 'C2 log-extrap away from orig (a, base 0)':[(a,np.exp((1+a)*np.log(np.maximum(sp,1e-12))-a*LO['clean'])) for a in (-0.3,-0.15,0,0.15,0.3,0.5,1.0)],
 'D mix with OOF k7 (lam, base 0)':[(l,(1-l)*sp+l*oof) for l in (0,0.05,0.1,0.2,0.3,0.5)],
 'D2 mix with OOF k8 (lam, base 0)':[(l,(1-l)*sp+l*cf8) for l in (0,0.05,0.1,0.2,0.3,0.5)],
 'F label mass x k, all images (k, base 1)':[(k,lab(sp,k)) for k in (0.3,0.5,0.7,0.85,1,1.2,1.5,2,3)],
 'F2 label mass x k on hardest 20% by sisters':[(k,tailk(sp,0.2,k)) for k in (0.01,0.1,0.3,1,3,10)],
 'F3 label mass x 0.1 on hardest f':[(f,tailk(sp,f,0.1)) for f in (0,0.1,0.2,0.3,0.4,0.6)],
 'G uniform floor (eps, base 0) [control]':[(e,(1-e)*sp+e/100) for e in (0,0.05,0.1,0.2,0.4,0.6)],
}
for lvl in ('sis','law'):
    print(f"\n######## MS level {lvl}")
    b=MSL[lvl]*pred(sp)*RETv; e0,_=loo(b[None]); print('baseline',{k:round(float(v),4) for k,v in stats(b).items()},'LOO rmse %.4f | top6 %.4f'%(np.sqrt((e0**2).mean()),np.sqrt((e0[pub>0.2]**2).mean())))
    for nm,L in FAM.items():
        P=np.stack([MSL[lvl]*pred(G)*RETv for _,G in L]); e,ch=loo(P)
        print(f"-- {nm}: LOO rmse {np.sqrt((e**2).mean()):.4f} (top6 {np.sqrt((e[pub>0.2]**2).mean()):.4f}) | LOO picks {sorted(set(L[c][0] for c in ch))}")
        for (th,G),p in zip(L,P):
            s=stats(p); pp=pred(G); print(f"     {th:>6}: pear {s['pear']:.3f} scale {s['scale']:.2f} ratio {s['ratio']:.2f} rms {s['rms']:.4f} | PRED c13 {pp[names.index('c13')]:.2f} qB {pp[names.index('qB')]:.2f} E18 {pp[names.index('E18')]:.2f} sM {pp[names.index('sM_ep08')]:.2f} | KL(G||base) {(np.maximum(G/G.sum(1,keepdims=True),1e-12)*(np.log(np.maximum(G/G.sum(1,keepdims=True),1e-12))-np.log(np.maximum(sp,1e-12)))).sum(1).mean():.3f}")
