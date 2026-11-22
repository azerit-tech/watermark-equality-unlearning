import json, sys, numpy as np, itertools
from scipy.optimize import least_squares
S=sys.argv[1]
exec(open(S+'/tomo/feat.py').read().split("rows=[]")[0].replace("S=sys.argv[1]",""))
D=json.load(open(S+'/tomo/feat.json')); rows=D['rows']; OA=D['oa']
def MSv(cm,cs,am,as_,mg,sg):
    c=max(0.0,1-(abs(cm-mg)+abs(cs-sg))/(mg+sg)); a=max(0.0,1-(abs(am-mg)+abs(as_-sg))/(abs(OA[0]-mg)+abs(OA[1]-sg))); return (c+a)/2
def RET(r,rho): return max(0,(1-rho*r['t2i_dis999']))/(1+rho*r['t2i_kl999'])
def F(r,k): return (r['clean_'+k]+r['alt_'+k])/2
def fit(feats,excl=(),free=True):
    rr=[r for r in rows if r['n'] not in excl]; yy=np.array([r['pub'] for r in rr])
    def model(p,r):
        mg,sg,rho,b0=p[:4]; pr=b0*(F(r,'F0')+sum(ai*F(r,k) for ai,k in zip(p[4:],feats)))
        return MSv(r['clean_m'],r['clean_s'],r['alt_m'],r['alt_s'],mg,sg)*RET(r,rho)*min(1,max(0,pr))
    res=lambda p:(np.array([model(p,r) for r in rr])-yy)/(0.006+0.04*yy)
    best=None
    for st in itertools.product([2.4,3.0],[4.4],[0.1,0.5],[0.1,0.3],[0.05,0.5]):
        p0=list(st[:4])+[st[4]]*len(feats); lo=[1.5,3.0,0,0.001]+[0]*len(feats); hi=[4.5,6.5,1.5,5]+[50.0]*len(feats)
        if not free: lo[:3]=[2.39,4.39,0.29]; hi[:3]=[2.41,4.41,0.31]; p0[:3]=[2.4,4.4,0.3]
        o=least_squares(res,p0,bounds=(lo,hi))
        if best is None or o.cost<best.cost: best=o
    return best.x, model
SIS={"sM_ep08","c11","c13","c16","e2","qB","wA"}; SCR={"ep05","ep02","E12","E13","E14","E16","E18","B1x","T05","T07","l5e3_a3","surgery","E13_tau1.3"}
print("--- cross-validation of the 2-basis law (s_S + qE18_E18), free nuisance")
for nm,ex in (("leave sister family out",SIS),("leave SCRUB family out",SCR),("leave c13,qB,wA,e2 out",{"c13","qB","wA","e2"})):
    p,model=fit(['s_S','qE18_E18'],ex)
    print(nm,'params',np.round(p,3)); print('   held-out: '+' '.join(f"{r['n']}:{model(p,r):.3f}/{r['pub']:.3f}" for r in rows if r['n'] in ex))
    p1,model1=fit(['s_S'],ex)
    print('   s_S only held-out: '+' '.join(f"{r['n']}:{model1(p1,r):.3f}/{r['pub']:.3f}" for r in rows if r['n'] in ex))
# candidate evaluation
def P2(k,v='clean'): return np.exp(lsm(d[k+'::'+v],2))
pE={v:P2('oracle__E18',v) for v in ('clean','alt')}
def feats_of(zc,za):
    out={}
    for v,z in (('clean',zc),('alt',za)):
        X=lsm(z)-lsm(o[v]); Xy=X[I,y]; ce=-lsm(z,1.0)[I,y]
        out[v+'_m']=ce.mean(); out[v+'_s']=ce.std(); out[v+'_F0']=Xy.mean()
        out[v+'_s_S']=(q['s']*((W['S']*X).sum(1)-Xy)).mean()
        w=nony(pE['clean']); qq=1-pE['clean'][I,y]; out[v+'_qE18_E18']=(qq*((w*X).sum(1)-Xy)).mean()
    return out
def tail(z,frac=0.12,cap=13.0,score=None):
    z=z.astype(np.float64).copy(); ce=-lsm(z,1.0)[I,y]; sc=ce if score is None else score
    idx=np.argsort(-sc)[:int(frac*N)]
    for i in idx:
        if ce[i]<cap:
            # lower logit y until CE=cap
            oth=np.delete(z[i],y[i]); lse=np.log(np.exp(oth-oth.max()).sum())+oth.max(); z[i,y[i]]=lse-cap
    return z
def mixlog(ps,ws): return 2*np.log(np.maximum(sum(w*pp for w,pp in zip(ws,ps)),1e-12))
L={k:{v:d[k+'::'+v].astype(np.float64) for v in ('clean','alt')} for k in ('sis__sM__ckpt_ep08','sis__t_f0.12_c13__ckpt_ep04','sis__qB__ckpt_ep03','sis__e2__ckpt_ep04','oracle__E18','oracle__E14','oracle__A4','cal_lr1e-5__ckpt_ep02','oracle__T05','oracle__E16')}
def PP(k,v): return np.exp(lsm(L[k][v],2))
scrk=('oracle__E18','oracle__E14','oracle__A4','cal_lr1e-5__ckpt_ep02','oracle__T05')
C={}
C['c13 (actual)']=(L['sis__t_f0.12_c13__ckpt_ep04']['clean'],L['sis__t_f0.12_c13__ckpt_ep04']['alt'])
C['qB (actual)']=(L['sis__qB__ckpt_ep03']['clean'],L['sis__qB__ckpt_ep03']['alt'])
C['e2 (actual)']=(L['sis__e2__ckpt_ep04']['clean'],L['sis__e2__ckpt_ep04']['alt'])
C['sM (actual, no tail)']=(L['sis__sM__ckpt_ep08']['clean'],L['sis__sM__ckpt_ep08']['alt'])
C['sM + emulated tail 12% cap13']=tuple(tail(L['sis__sM__ckpt_ep08'][v]) for v in ('clean','alt'))
for w in (0.85,0.7,0.55):
    for nm,ks in (('E18',('oracle__E18',)),('E16',('oracle__E16',)),('scrub5',scrk)):
        m={v:mixlog([PP('sis__sM__ckpt_ep08',v),sum(PP(k,v) for k in ks)/len(ks)],[w,1-w]) for v in ('clean','alt')}
        C[f'mix sM:{w}+{nm}:{1-w:.2f}, no tail']=(m['clean'],m['alt'])
        for fr,cap in ((0.12,13),(0.16,13),(0.2,11)):
            sc=-lsm(m['clean'],1.0)[I,y]
            C[f'mix sM:{w}+{nm}:{1-w:.2f} + tail {fr}/{cap}']=(tail(m['clean'],fr,cap,sc),tail(m['alt'],fr,cap,sc))
laws=[]
for feats,free in ((['s_S','qE18_E18'],True),(['s_S','qE18_E18'],False),(['s_S'],True),(['s_S'],False)):
    p,_=fit(feats,(),free); laws.append((feats,free,p))
    print('law',feats,'free' if free else 'fixed',np.round(p,3))
print(f"{'candidate':42s} CE clean      | "+" | ".join(f"{'+'.join(f)[:14]:14s}{'fr' if fr else 'fx'} PRED  MS  F*" for f,fr,p in laws))
for nm,(zc,za) in C.items():
    f=feats_of(zc,za); line=f"{nm:42s} {f['clean_m']:.2f}/{f['clean_s']:.2f} |"
    for feats,free,p in laws:
        pr=p[3]*((f['clean_F0']+f['alt_F0'])/2+sum(a*(f['clean_'+k]+f['alt_'+k])/2 for a,k in zip(p[4:],feats)))
        ms=MSv(f['clean_m'],f['clean_s'],f['alt_m'],f['alt_s'],p[0],p[1])
        line+=f"            {pr:.3f} {ms:.2f} {pr*ms:.3f} |"
    print(line)
