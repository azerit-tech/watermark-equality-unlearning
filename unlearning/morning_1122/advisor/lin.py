import json, sys, numpy as np
from scipy.optimize import nnls
S=sys.argv[1]
D=json.load(open(S+'/tomo/feat.json')); rows=D['rows']; OA=D['oa']
def MS(r,mg,sg):
    c=max(0.0,1-(abs(r['clean_m']-mg)+abs(r['clean_s']-sg))/(mg+sg))
    a=max(0.0,1-(abs(r['alt_m']-mg)+abs(r['alt_s']-sg))/(abs(OA[0]-mg)+abs(OA[1]-sg)))
    return (c+a)/2
def RET(r,rho): return max(0,(1-rho*r['t2i_dis999']))/(1+rho*r['t2i_kl999'])
def F(r,k): return (r['clean_'+k]+r['alt_'+k])/2
rr=[r for r in rows if r['pub']>0.03]
print('n models',len(rr))
print('F0 and features per model:')
for r in rr: print(f"  {r['n']:13s} pub {r['pub']:.3f} F0 {F(r,'F0'):6.2f} s_S {F(r,'s_S'):5.2f} 1_S {F(r,'1_S'):5.2f} 1_U {F(r,'1_U'):5.2f} 1_R2 {F(r,'1_R2'):5.2f} 1_C {F(r,'1_C'):5.2f} qE18 {F(r,'qE18_E18'):5.2f} flip_S {F(r,'flip_S'):5.2f}")
for (mg,sg,rho) in ((2.4,4.4,0.35),(2.77,4.72,0.37),(2.4,4.4,0.6),(2.0,5.0,0.35),(3.0,5.0,0.35)):
    P=np.array([r['pub']/(MS(r,mg,sg)*RET(r,rho)) for r in rr]); w=1/(0.02+0.1*P)
    print(f'--- nuisance {mg}/{sg}/{rho}: PRED_obs min {P.min():.2f} max {P.max():.2f}')
    for feats in (['s_S'],['1_U'],['1_S'],['s_S','1_U'],['s_S','qE18_E18'],['s_S','qE18_E18','1_U'],['1_S','1_U'],['1_C','1_U'],['1_R2','1_U'],['s_S','1_R2']):
        # F0 = D0*P - sum a f  ->  -F0 = -D0*P + sum a f ; solve nnls on [ -P?]. Use: sum a f - D0*P = -F0 ; D0>=0 -> columns [f..., -P]
        Am=np.array([[F(r,k) for k in feats]+[-p] for r,p in zip(rr,P)]); b=np.array([-F(r,'F0') for r in rr])
        # weight in PRED units: residual/D0 ; iterate
        x,_=nnls(Am,b); 
        for it in range(3):
            D0=max(x[-1],1e-3); x,_=nnls(Am*(w/D0)[:,None],b*(w/D0))
        D0=x[-1]; predP=(np.array([F(r,'F0')+sum(a*F(r,k) for a,k in zip(x[:-1],feats)) for r in rr]))/D0
        rms=np.sqrt(np.mean(((predP-P))**2))
        print(f"   {'+'.join(feats):22s} D0 {D0:6.2f} a {np.round(x[:-1],3)} rms(PRED) {rms:.3f}")
