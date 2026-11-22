import json, sys, numpy as np
S=sys.argv[1]
exec(open(S+'/tomo/feat.py').read().split("rows=[]")[0].replace("S=sys.argv[1]",""))
def P2(k,v='clean'): return np.exp(lsm(d[k+'::'+v],2))
def gold(aS,aE,ek='oracle__E18',floor=0.0):
    pe=P2(ek); qe=1-pe[I,y]; s=q['s']
    wS=aS*s; wE=aE*qe; tot=wS+wE; sc=np.minimum(1,0.98/np.maximum(tot,1e-9)); wS*=sc; wE*=sc
    g=wS[:,None]*W['S']+wE[:,None]*nony(pe); g[I,y]=1-wS-wE
    return g
def kl(g,lq): return (g*(np.log(np.maximum(g,1e-12))-lq)).sum(1)
def pred(g,lsub,v='clean'):
    return 1-kl(g,lsub).mean()/kl(g,lsm(o[v])).mean()
def stats(z):
    ce=-lsm(z,1.0)[I,y]; return ce.mean(),ce.std(),(z.argmax(1)==y).mean()
cands={}
for nm,k in (('c13','sis__t_f0.12_c13__ckpt_ep04'),('qB','sis__qB__ckpt_ep03'),('e2','sis__e2__ckpt_ep04'),('sM','sis__sM__ckpt_ep08'),('E18','oracle__E18'),('E14','oracle__E14'),('T07','oracle__T07'),('A4','oracle__A4'),('ep02','cal_lr1e-5__ckpt_ep02'),('hi_A2','hi__A2__submission'),('G1','login__G1__submission')):
    cands[nm]=d[k+'::clean'].astype(np.float64)
def mixlog(ps,ws): 
    p=sum(w*pp for w,pp in zip(ws,ps)); return 2*np.log(np.maximum(p,1e-12))
pc=P2('sis__t_f0.12_c13__ckpt_ep04'); pq=P2('sis__qB__ckpt_ep03')
scr=[P2(k) for k in ('oracle__E18','oracle__E14','oracle__A4','cal_lr1e-5__ckpt_ep02','oracle__T05')]
pscr=sum(scr)/len(scr)
for w in (0.8,0.65,0.5):
    cands[f'mix qB:{w} E18:{1-w:.2f}']=mixlog([pq,P2('oracle__E18')],[w,1-w])
    cands[f'mix qB:{w} scrubEns5:{1-w:.2f}']=mixlog([pq,pscr],[w,1-w])
    cands[f'mix c13:{w} A4:{1-w:.2f}']=mixlog([pc,P2('oracle__A4')],[w,1-w])
cands['mix sisT2:.6 scrubEns5:.4']=mixlog([sp,pscr],[.6,.4])
print('gold variants: (aS,aE,basis) -> D0, gold own CE mean/std if sampled...')
for (aS,aE,ek) in ((0.81,0.32,'oracle__E18'),(1.04,0.31,'oracle__E18'),(0.89,0.55,'oracle__A4'),(0.91,0.66,'cal_lr1e-5__ckpt_ep02'),(1.0,0.0,'oracle__E18')):
    g=gold(aS,aE,ek); D0=kl(g,lsm(o['clean'])).mean()
    print(f'\n== g*: aS {aS} aE {aE} basis {ek}: D0 {D0:.2f}, gold mass on y mean {g[I,y].mean():.3f}, gold argmax!=y {(g.argmax(1)!=y).mean():.3f}')
    for nm,z in cands.items():
        m,s,a=stats(z); print(f'   {nm:28s} PRED* {pred(g,lsm(z)):.3f} | CE {m:.2f}/{s:.2f} acc {a:.2f} | T2 maxprob {np.exp(lsm(z)).max(1).mean():.2f}')
