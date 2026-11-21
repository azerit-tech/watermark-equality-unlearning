import pandas as pd, numpy as np, sys
df=pd.read_pickle('resp.pkl')
df['start']=df.ts-pd.to_timedelta(df.el,unit='s')
df['slot']=df.start.dt.floor('3min').astype(str)
fam=pd.read_csv('../families_ep.csv').set_index('endpoint').family.to_dict()
for i in range(1,10): fam[f'ref-0{i}']='ref'
df['family']=df.ep.map(fam)

def huber(X,y,k=1.345,it=50):
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    for _ in range(it):
        r=y-X@b; s=np.median(np.abs(r-np.median(r)))/0.6745+1e-9
        w=np.minimum(1,k*s/np.maximum(np.abs(r),1e-12)); sw=np.sqrt(w)
        b2=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
        if np.max(np.abs(b2-b))<1e-8: b=b2;break
        b=b2
    return b

def design(g, eps, refep):
    slots=sorted(g.slot.unique()); S=(g.slot.values[:,None]==np.array(slots)[None,:]).astype(float)
    E=np.stack([(g.ep==e).astype(float).values for e in eps if e!=refep],1)
    m=(g.mt.values-150)/100.0
    EM=E*m[:,None]
    X=np.hstack([S,m[:,None],E,EM]); return X,len(slots)

def fit(g, refep, nboot=200, seed=0):
    eps=sorted(g.ep.unique()); others=[e for e in eps if e!=refep]; k=len(others)
    out={}
    for name,y in [('log',np.log(g.el.values)),('lin',g.el.values)]:
        X,ns=design(g,eps,refep); b=huber(X,y)
        main=b[ns+1:ns+1+k]; slope=b[ns+1+k:]
        # bootstrap over slots (cluster)
        rng=np.random.default_rng(seed); slots=g.slot.unique(); BM=[];BS=[]
        for _ in range(nboot):
            pick=rng.choice(slots,len(slots),replace=True)
            gb=pd.concat([g[g.slot==s].assign(slot=f'{s}_{j}') for j,s in enumerate(pick)])
            if set(gb.ep.unique())!=set(eps): continue
            Xb,nsb=design(gb,eps,refep); bb=huber(Xb,np.log(gb.el.values) if name=='log' else gb.el.values,it=20)
            BM.append(bb[nsb+1:nsb+1+k]); BS.append(bb[nsb+1+k:])
        BM=np.array(BM);BS=np.array(BS)
        out[name]=pd.DataFrame({'ep':others,'main':main,'lo':np.percentile(BM,2.5,0),'hi':np.percentile(BM,97.5,0),
            'slope':slope,'slo':np.percentile(BS,2.5,0),'shi':np.percentile(BS,97.5,0)})
    return out
