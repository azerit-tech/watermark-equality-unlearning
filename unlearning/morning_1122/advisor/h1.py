import json, sys, numpy as np
from scipy.optimize import nnls
S=sys.argv[1]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(S+'/tomo/feat.py').read().split("rows=[]")[0].replace("S=sys.argv[1]",""))
D=json.load(open(S+'/tomo/feat.json')); rows=D['rows']; OA=D['oa']
B=~A
pE=np.exp(lsm(d['oracle__E18::clean'],2)); W['E']=nony(pE); qE=1-pE[I,y]
s=q['s']
Q={'A':A.astype(float),'B':B.astype(float),'sA':s*A,'sB':s*B,'s':s,'1':np.ones(N),'qE':qE,'qEA':qE*A,'qEB':qE*B}
feat={}
for r in rows:
    k=key[r['n']]; f={}
    for v in ('clean','alt'):
        X=lsm(d[k+'::'+v])-lsm(o[v]); Xy=X[I,y]
        f.setdefault('F0',0); f['F0']+=Xy.mean()/2
        f.setdefault('F0A',0); f['F0A']+=(Xy*A).mean()/2
        for wn in ('S','R2','U','C','E'):
            g=(W[wn]*X).sum(1)-Xy
            for qn,qq in Q.items():
                f[qn+'_'+wn]=f.get(qn+'_'+wn,0)+(qq*g).mean()/2
    feat[r['n']]=f
def MS(r,mg,sg):
    c=max(0.0,1-(abs(r['clean_m']-mg)+abs(r['clean_s']-sg))/(mg+sg))
    a=max(0.0,1-(abs(r['alt_m']-mg)+abs(r['alt_s']-sg))/(abs(OA[0]-mg)+abs(OA[1]-sg)))
    return (c+a)/2
def RET(r,rho): return max(0,(1-rho*r['t2i_dis999']))/(1+rho*r['t2i_kl999'])
rr=[r for r in rows if r['pub']>0.03]
def fit(feats,mg=2.4,sg=4.4,rho=0.35,excl=(),show=False):
    tr=[r for r in rr if r['n'] not in excl]
    def solve(tr):
        P=np.array([r['pub']/(MS(r,mg,sg)*RET(r,rho)) for r in tr]); w=1/(0.02+0.1*P)
        Am=np.array([[feat[r['n']][k] for k in feats]+[-p] for r,p in zip(tr,P)]); b=np.array([-feat[r['n']]['F0'] for r in tr])
        x,_=nnls(Am,b)
        for it in range(4):
            D0=max(x[-1],1e-2); x,_=nnls(Am*(w/D0)[:,None],b*(w/D0))
        return x
    x=solve(tr); D0=max(x[-1],1e-6)
    P=np.array([r['pub']/(MS(r,mg,sg)*RET(r,rho)) for r in rr])
    pp=np.array([(feat[r['n']]['F0']+sum(a*feat[r['n']][k] for a,k in zip(x[:-1],feats)))/D0 for r in rr])
    m=np.array([r['n'] not in excl for r in rr])
    rms=np.sqrt(np.mean((pp[m]-P[m])**2)); out=f"{'+'.join(feats):30s} D0 {D0:5.2f} a {np.round(x[:-1],2)} rmsPRED {rms:.3f}"
    if excl: out+=f" | heldout rms {np.sqrt(np.mean((pp[~m]-P[~m])**2)):.3f} bias {np.mean(pp[~m]-P[~m]):+.3f}"
    print(out)
    if show: print('      '+' '.join(f"{r['n']}:{a:.2f}/{b:.2f}" for r,a,b in zip(rr,pp,P)))
    return x
print('PRED_obs / features for key models (A-group share of F0):')
for n in ('hi_A2','hi_B2','E18','c13','qB','sM_ep08','G1'):
    f=feat[n]; print(f"  {n:8s} F0 {f['F0']:.2f} (A part {f['F0A']:.2f}) A_S {f['A_S']:.2f} A_R2 {f['A_R2']:.2f} A_C {f['A_C']:.2f} A_U {f['A_U']:.2f} B_S {f['B_S']:.2f} sA_S {f['sA_S']:.2f} sB_S {f['sB_S']:.2f} qEA_E {f['qEA_E']:.2f} qEB_E {f['qEB_E']:.2f}")
sets=[['s_S'],['sA_S','sB_S'],['A_S','sB_S'],['A_C','sB_S'],['A_R2','sB_S'],['A_U','sB_S'],['A_E','sB_S'],['s_S','qE_E'],['sA_S','sB_S','qEA_E','qEB_E'],['A_S','sB_S','qEB_E'],['A_C','sB_S','qEB_E'],['A_E','sB_S','qEB_E'],['A_S','A_C','A_R2','A_U','A_E','sA_S','sB_S','qEA_E','qEB_E'],['B_S','B_C','B_R2','B_U','B_E','sA_S','sB_S','qEA_E','qEB_E'],['1_S','1_C','1_R2','1_U','1_E','s_S','qE_E']]
SIS={"sM_ep08","c11","c13","c16","e2","qB","wA"}
for nu in ((2.4,4.4,0.35),(2.77,4.72,0.37)):
    print('=== nuisance',nu)
    for f in sets: fit(f,*nu)
    print(' -- sister family held out')
    for f in sets[:12]: fit(f,*nu,excl=SIS)
    print(' -- hi_A2, hi_B2 held out')
    for f in sets[:12]: fit(f,*nu,excl={'hi_A2','hi_B2'},show=False)
