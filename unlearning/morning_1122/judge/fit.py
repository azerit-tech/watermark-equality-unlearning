import json, sys, numpy as np, itertools
from scipy.optimize import least_squares
S=sys.argv[1]
D=json.load(open(S+'/tomo/feat.json')); rows=D['rows']; OA=D['oa']
y=np.array([r['pub'] for r in rows])
def MS(r,mg,sg):
    c=max(0.0,1-(abs(r['clean_m']-mg)+abs(r['clean_s']-sg))/(mg+sg))
    a=max(0.0,1-(abs(r['alt_m']-mg)+abs(r['alt_s']-sg))/(abs(OA[0]-mg)+abs(OA[1]-sg)))
    return (c+a)/2
def RET(r,rho): return max(0,(1-rho*r['t2i_dis999']))/(1+rho*r['t2i_kl999'])
def F(r,k): return (r['clean_'+k]+r['alt_'+k])/2
def run(feats, free_nuis=True, verbose=False, excl=()):
    rr=[r for r in rows if r['n'] not in excl]; yy=np.array([r['pub'] for r in rr])
    def model(p,r):
        mg,sg,rho,b0=p[:4]; a=p[4:]
        pr=b0*(F(r,'F0')+sum(ai*F(r,k) for ai,k in zip(a,feats)))
        return MS(r,mg,sg)*RET(r,rho)*min(1,max(0,pr)), pr
    res=lambda p:(np.array([model(p,r)[0] for r in rr])-yy)/(0.006+0.04*yy)
    best=None
    for st in itertools.product([2.4,3.0],[4.4],[0.1,0.5],[0.1,0.3],[0.05,0.5]):
        p0=list(st[:4])+[st[4]]*len(feats)
        lo=[1.5,3.0,0,0.001]+[0]*len(feats); hi=[4.5,6.5,1.5,5]+[1.0]*len(feats)
        if not free_nuis: lo[:3]=[2.39,4.39,0.29]; hi[:3]=[2.41,4.41,0.31]; p0[:3]=[2.4,4.4,0.3]
        o=least_squares(res,p0,bounds=(lo,hi))
        if best is None or o.cost<best.cost: best=o
    p=best.x; pred=np.array([model(p,r)[0] for r in rr])
    print(f"{'+'.join(feats):40s} chi2 {2*best.cost:7.1f} rms {np.sqrt(np.mean((pred-yy)**2)):.4f} | mg {p[0]:.2f} sg {p[1]:.2f} rho {p[2]:.2f} 1/D0 {p[3]:.3f} | a "+' '.join(f'{k}={a:.3f}' for k,a in zip(feats,p[4:])))
    if verbose:
        for r,pp in zip(rr,pred):
            print(f"     {r['n']:13s} pub {r['pub']:.3f} fit {pp:.3f} MS {MS(r,p[0],p[1]):.2f} RET {RET(r,p[2]):.2f} PRED {model(p,r)[1]:.2f}")
    return p,best.cost
sets=[['1_R2'],['1_S'],['1_U'],['1_C'],['s_S'],['flip_S'],['s_R2'],['A_R2'],['A_S'],['1_R1'],['1_S1'],
      ['1_R2','s_S'],['1_R2','1_S'],['1_R2','flip_S'],['1_U','s_S'],['1_C','s_S'],['1_R2','A_R2'],['1_S','A_S'],['1_R2','1_U'],['A_R2','B_R2'],['A_S','B_S'],
      ['1_R2','s_S','A_R2'],['1_R2','s_S','1_U'],['1_R2','s_S','1_C'],['1_R2','flip_S','A_S'],['B_R2','A_R2','s_S'],['1_R2','1_S','1_C','1_U'],['1_R2','s_S','s1_S1','flip_S']]
for fn in (False,True):
    print('=== nuisance', 'free' if fn else 'fixed 2.4/4.4/0.3')
    for f in sets: run(f,fn)
run(['1_R2','s_S'],True,True)
run(['1_R2','1_S','1_C','1_U'],True,True)
