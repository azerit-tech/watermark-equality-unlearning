import json, sys, numpy as np
from scipy.optimize import nnls
S=sys.argv[1]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(S+'/tomo/feat.py').read().split("rows=[]")[0].replace("S=sys.argv[1]",""))
D=json.load(open(S+'/tomo/feat.json')); rows=D['rows']; OA=D['oa']
B=~A; dose=cnt[y]/130.0; s=q['s']
pE=np.exp(lsm(d['oracle__E18::clean'],2)); W['E']=nony(pE); qE=1-pE[I,y]
Q={'sA':s*A,'sB':s*B,'s':s,'A':A*1.0,'B':B*1.0,'dB':dose*B,'d':dose,'d2':dose**2,'sd':s*dose,'qE':qE,'qEA':qE*A,'qEB':qE*B,'1':np.ones(N)}
def feats_of(zc,za):
    f={}
    for v,z in (('clean',zc),('alt',za)):
        X=lsm(z)-lsm(o[v]); Xy=X[I,y]; f['F0']=f.get('F0',0)+Xy.mean()/2
        ce=-lsm(z,1.0)[I,y]; f[v+'_m']=ce.mean(); f[v+'_s']=ce.std()
        for wn in ('S','E','C','U'):
            g=(W[wn]*X).sum(1)-Xy
            for qn,qq in Q.items(): f[qn+'_'+wn]=f.get(qn+'_'+wn,0)+(qq*g).mean()/2
    return f
feat={r['n']:feats_of(d[key[r['n']]+'::clean'],d[key[r['n']]+'::alt']) for r in rows}
def MSf(f,mg,sg):
    c=max(0.0,1-(abs(f['clean_m']-mg)+abs(f['clean_s']-sg))/(mg+sg))
    a=max(0.0,1-(abs(f['alt_m']-mg)+abs(f['alt_s']-sg))/(abs(OA[0]-mg)+abs(OA[1]-sg)))
    return (c+a)/2
def RET(r,rho): return max(0,(1-rho*r['t2i_dis999']))/(1+rho*r['t2i_kl999'])
rr=[r for r in rows if r['pub']>0.03]
def fit(feats,mg=2.4,sg=4.4,rho=0.35):
    P=np.array([r['pub']/(MSf(feat[r['n']],mg,sg)*RET(r,rho)) for r in rr]); w=1/(0.02+0.1*P)
    Am=np.array([[feat[r['n']][k] for k in feats]+[-p] for r,p in zip(rr,P)]); b=np.array([-feat[r['n']]['F0'] for r in rr])
    x,_=nnls(Am,b)
    for it in range(4):
        D0=max(x[-1],1e-2); x,_=nnls(Am*(w/D0)[:,None],b*(w/D0))
    D0=max(x[-1],1e-6); pp=np.array([(feat[r['n']]['F0']+sum(a*feat[r['n']][k] for a,k in zip(x[:-1],feats)))/D0 for r in rr])
    print(f"{'+'.join(feats):28s} D0 {D0:5.2f} a {np.round(x[:-1],2)} rmsPRED {np.sqrt(np.mean((pp-P)**2)):.3f}")
    return feats,x
print('E18/T07/c13 forget accuracy by class-count bin (generic methods: dose effect?)')
for n in ('E18','T07','hi_A2','c13','qB'):
    z=d[key[n]+'::clean']; ok=z.argmax(1)==y
    print(' ',n,' '.join(f"{lo}-{hi}:{ok[(cnt[y]>=lo)&(cnt[y]<hi)].mean():.2f}" for lo,hi in ((40,56),(56,70),(70,90),(90,110),(125,135))))
print(' sister flip by bin', ' '.join(f"{lo}-{hi}:{(sp1.argmax(1)!=y)[(cnt[y]>=lo)&(cnt[y]<hi)].mean():.2f}" for lo,hi in ((40,56),(56,70),(70,90),(90,110),(125,135))))
laws=[]
for nu in ((2.4,4.4,0.35),(2.77,4.72,0.37)):
    print('== nuisance',nu)
    for fs in (['s_S'],['sA_S','sB_S'],['A_S','sB_S'],['d_S'],['d2_S'],['sd_S'],['d_S','s_S'],['d2_S','s_S'],['A_S','dB_S'],['A_S','dB_S','sB_S'],['s_S','qE_E'],['A_E','sB_S'],['d_E','s_S'],['d2_E','s_S'],['A_S','sB_S','qE_E']):
        laws.append((nu,)+fit(fs,*nu))
# candidates: start from qB logits (bulk + tail), modify A images
zq={v:d['sis__qB__ckpt_ep03::'+v].astype(np.float64) for v in ('clean','alt')}
zs={v:d['sis__sM__ckpt_ep08::'+v].astype(np.float64) for v in ('clean','alt')}
def targ(mA, wname='S', base=zq, capA=None):
    out={}
    for v in ('clean','alt'):
        z=base[v].copy()
        g=np.zeros((N,100)); g[I,y]=1-mA; g+=mA*W[wname]; g=np.maximum(g,1e-9)
        zz=2*np.log(g)
        if capA is not None:  # impose CE=capA on A by lowering y logit
            for i in np.where(A)[0]:
                oth=np.delete(zz[i],y[i]); zz[i,y[i]]=np.log(np.exp(oth-oth.max()).sum())+oth.max()-capA
        z[A]=zz[A]; out[v]=z
    return out
C={'qB actual':zq,'sM actual':zs}
for wn in ('S','E','C'):
    for mA in (0.7,0.9,0.97):
        C[f'qB, A->(1-{mA})y+{mA}*{wn}']=targ(mA,wn)
        C[f'sM, A->(1-{mA})y+{mA}*{wn}']=targ(mA,wn,zs)
    C[f'sM, A-> {wn} only, CE_A=8']=targ(0.99,wn,zs,8.0)
print(f"\n{'candidate':34s} CE clean    | PRED / MS / forget under each law")
sel=[l for l in laws if '+'.join(l[1]) in ('s_S','A_S+sB_S','s_S+qE_E','A_E+sB_S','A_S+sB_S+qE_E','d_S+s_S')]
print(' '*48+' | '.join(f"{'+'.join(l[1])[:12]:12s}{l[0][0]}" for l in sel))
for nm,zz in C.items():
    f=feats_of(zz['clean'],zz['alt']); line=f"{nm:34s} {f['clean_m']:.2f}/{f['clean_s']:.2f} |"
    for nu,fs,x in sel:
        pr=(f['F0']+sum(a*f[k] for a,k in zip(x[:-1],fs)))/max(x[-1],1e-6); ms=MSf(f,nu[0],nu[1])
        line+=f" {pr:.2f}/{ms:.2f}/{pr*ms:.3f} |"
    print(line)
