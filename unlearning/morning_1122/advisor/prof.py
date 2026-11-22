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
best={'clean_m':2.58,'clean_s':4.56,'alt_m':2.40,'alt_s':4.25}
def fit(feats,mg,sg,rho):
    P=np.array([r['pub']/max(MS(r,mg,sg)*RET(r,rho),1e-3) for r in rr]); 
    Am=np.array([[F(r,k) for k in feats]+[-p] for r,p in zip(rr,P)]); b=np.array([-F(r,'F0') for r in rr])
    x,_=nnls(Am,b)
    D0=max(x[-1],1e-6); pp=np.array([(F(r,'F0')+sum(a*F(r,k) for a,k in zip(x[:-1],feats)))/D0 for r in rr])
    sc=pp*np.array([MS(r,mg,sg)*RET(r,rho) for r in rr]); y=np.array([r['pub'] for r in rr])
    return np.sqrt(np.mean(((sc-y)/(0.006+0.04*y))**2)), D0, x[:-1]
for feats in (['s_S','qE18_E18'],['s_S']):
    print('law',feats,': chi-rms of score fit (lower=better) | D0 | MS of our best model, by gold (mean,std), rho=0.35')
    for mg in (1.6,2.0,2.4,2.8,3.2,3.6,4.0):
        print(f' mean {mg}: '+'  '.join(f"sd{sg}: {fit(feats,mg,sg,0.35)[0]:.2f}|{fit(feats,mg,sg,0.35)[1]:.1f}|{MS(best,mg,sg):.2f}" for sg in (3.2,3.6,4.0,4.4,4.8,5.2,5.8)))
print('rho profile (law s_S+qE18, gold 2.4/4.4):',' '.join(f"rho{rho}:{fit(['s_S','qE18_E18'],2.4,4.4,rho)[0]:.2f}" for rho in (0.1,0.2,0.35,0.5,0.7,1.0,1.4)))
print('rho profile (gold 2.8/4.8):',' '.join(f"rho{rho}:{fit(['s_S','qE18_E18'],2.8,4.8,rho)[0]:.2f}" for rho in (0.1,0.2,0.35,0.5,0.7,1.0,1.4)))
