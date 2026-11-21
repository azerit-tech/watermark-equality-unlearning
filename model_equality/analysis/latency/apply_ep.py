from fit import *
g=df[df.ep.str.startswith('ep-')].copy()
eps=sorted(g.ep.unique()); ref='ep-01'; others=[e for e in eps if e!=ref]; k=len(others)
def est(gg):
    X,ns=design(gg,eps,ref); bl=huber(X,np.log(gg.el.values)); bn=huber(X,gg.el.values)
    main=np.r_[0,bl[ns+1:ns+1+k]]; sl=np.r_[0,bl[ns+1+k:]]
    # linear model: effect at mt=0 (s) = main_lin - 1.5*slope_lin ; slope in ms/token = slope_lin*10
    ml=np.r_[0,bn[ns+1:ns+1+k]]; sll=np.r_[0,bn[ns+1+k:]]
    ft=ml-1.5*sll; dt=sll*10
    R=pd.DataFrame({'ep':eps,'main':main,'slope':sl,'ft':ft,'dt':dt}); R['family']=R.ep.map(fam)
    for c in ['main','slope','ft','dt']: R[c]=R[c]-R.groupby('family')[c].transform('median')
    return R.set_index('ep')
P=est(g)
rng=np.random.default_rng(1); slots=g.slot.unique(); B=[]
for _ in range(200):
    pick=rng.choice(slots,len(slots),replace=True)
    gb=pd.concat([g[g.slot==s].assign(slot=f'{s}_{j}') for j,s in enumerate(pick)])
    if set(gb.ep.unique())!=set(eps): continue
    B.append(est(gb))
print('boot reps',len(B))
A={c:np.stack([b[c].values for b in B]) for c in ['main','slope','ft','dt']}
q=g[g.nch>300].assign(mspc=lambda d:1000*d.el/d.nch).groupby('ep').mspc.median()
o=pd.DataFrame({'endpoint':eps,'family':[fam[e] for e in eps],'n':g.groupby('ep').size().reindex(eps).values,
 'throughput_effect_pct':(np.exp(P.main.values)-1)*100,
 'throughput_ci_lo_pct':(np.exp(np.percentile(A['main'],2.5,0))-1)*100,
 'throughput_ci_hi_pct':(np.exp(np.percentile(A['main'],97.5,0))-1)*100,
 'per_token_ms_diff':P.dt.values,'per_token_ci_lo':np.percentile(A['dt'],2.5,0),'per_token_ci_hi':np.percentile(A['dt'],97.5,0),
 'first_token_effect_s':P.ft.values,'first_token_ci_lo':np.percentile(A['ft'],2.5,0),'first_token_ci_hi':np.percentile(A['ft'],97.5,0),
 'quicklook_ms_per_char':q.reindex(eps).values,'median_text_chars':g.groupby('ep').nch.median().reindex(eps).values})
o['anomaly']=(o.throughput_effect_pct.abs()>3)&((o.throughput_ci_lo_pct>1)|(o.throughput_ci_hi_pct<-1))
o['proposed_class']='UNSURE'
o=o.sort_values(['family','endpoint'])
o.round(3).to_csv('latency_ep.csv',index=False)
pd.set_option('display.width',250)
print(o.round(2).to_string(index=False))
# quicklook vs text length correlation
print('corr quicklook ms/char vs 1/median chars(>300 subset)', np.corrcoef(o.quicklook_ms_per_char, 1/g[g.nch>300].groupby('ep').nch.median().reindex(o.endpoint).values)[0,1])
