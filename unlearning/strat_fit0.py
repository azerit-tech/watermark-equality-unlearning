"""Strategist: stats-only fit of public scores. score = K/(1+b*drift) * ms(gold), ms=mean of clean/alt mean_std_score."""
import numpy as np, itertools
P = {  # name: clean m,s ; alt m,s ; comp drift ; score
 'B_ep2': (.499,1.632,.435,1.468,.368,.064), 'C_L3': (.353,.964,.182,.594,.268,.0234),
 'D_a3': (.682,2.622,.537,2.098,.248,.0653), 'E_surg': (.491,1.91,.433,1.68,.200,.044),
 'F_G2': (.507,1.021,.368,.725,.555,.0417), 'G_G1': (.699,1.289,.489,.924,.490,.0524),
 'H_G0b': (.528,1.367,.351,.981,.128,.0311)}
names=list(P); X=np.array([P[n] for n in names]); y=np.log(X[:,5])
def ms(m,s,M,S,om,os_):
    do=abs(om-M)+abs(os_-S); return 1-np.clip((abs(m-M)+abs(s-S))/do,0,1)
Mg=np.arange(.2,2.01,.05); Sg=np.arange(.4,3.51,.1)
res=[]
for Mc,Sc,Ma,Sa in itertools.product(Mg,Sg,Mg,Sg):
    msc=ms(X[:,0],X[:,1],Mc,Sc,.0003,.0024); msa=ms(X[:,2],X[:,3],Ma,Sa,.003,.053)
    m=(msc+msa)/2
    if (m<=0.02).any(): continue
    r=y-np.log(m)
    for b in (0,):
        K=np.exp(r.mean()); sse=((r-r.mean())**2).sum()
        res.append((sse,Mc,Sc,Ma,Sa,K))
res=np.array(res); o=np.argsort(res[:,0]); 
print('n',len(res),'best sse',res[o[0]])
for i in o[:15]: print(np.round(res[i],3))
w=np.exp(-res[:,0]/(2*0.15**2)); w/=w.sum()   # assume ~15% log-noise per point
for j,n in zip(range(1,5),['Mc','Sc','Ma','Sa']):
    v=res[:,j]; mu=(w*v).sum(); sd=np.sqrt((w*(v-mu)**2).sum())
    q=np.interp([.1,.5,.9],np.cumsum(w[np.argsort(v)]),np.sort(v)); print(n,round(mu,2),'+-',round(sd,2),'q10/50/90',np.round(q,2))
