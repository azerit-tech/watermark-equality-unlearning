from fit import *
r=df[df.family=='ref'].copy()
r['regime']=np.where(r.run.isin(['4b06777d820d','8fe29aab46db','d6f7321281c7','a170d321c13b']),'loaded','clean')
for reg in ['clean','all']:
    g=r if reg=='all' else r[r.regime=='clean']
    o=fit(g,'ref-01')
    L=o['log'].copy(); 
    for c in ['main','lo','hi']: L[c]=(np.exp(L[c])-1)*100
    for c in ['slope','slo','shi']: L[c]*=100
    print('=== bench',reg,len(g),'log model: main=% diff vs ref-01 at mt=150; slope=extra %-points per +100 max_tokens')
    print(L.round(2).to_string(index=False))
    Lin=o['lin']; Lin[['slope','slo','shi']]*=10  # s per 100 tok -> ms/token
    print('lin model: main=s at mt=150; slope=ms/token diff'); print(Lin.round(3).to_string(index=False))
# quick-look metric on refs
q=r[r.nch>300].assign(mspc=lambda d:1000*d.el/d.nch)
print(q.groupby('ep').agg(n=('el','size'),mspc=('mspc','median'),nch=('nch','median'),el=('el','median')).round(1).to_string())
