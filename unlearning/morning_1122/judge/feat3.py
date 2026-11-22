import json, sys, numpy as np
S=sys.argv[1]
exec(open(S+'/tomo/feat.py').read().split("rows=[]")[0].replace("S=sys.argv[1]",""))
def basis(k,v='clean'):
    p=np.exp(lsm(d[k+'::'+v],2)); return nony(p), 1-p[I,y]
B={}
for nm,k in (('E18','oracle__E18'),('E14','oracle__E14'),('T07','oracle__T07'),('ep02','cal_lr1e-5__ckpt_ep02'),('G1','login__G1__submission'),('c13','sis__t_f0.12_c13__ckpt_ep04'),('A4','oracle__A4'),('hA','sis__hA__ckpt_ep40')):
    B[nm]=basis(k)
# agreement stats between estimators of "wrong class"
sflip=sp1.argmax(1)!=y; sw=W['S1'].argmax(1)
for nm in ('E18','T07','A4','c13','hA'):
    z=d[{'E18':'oracle__E18','T07':'oracle__T07','A4':'oracle__A4','c13':'sis__t_f0.12_c13__ckpt_ep04','hA':'sis__hA__ckpt_ep40'}[nm]+'::clean']
    fl=z.argmax(1)!=y; w=B[nm][0].argmax(1)
    both=fl&sflip
    print(nm,'flip %.3f | sister flip %.3f | both %.3f (indep would be %.3f) | same wrong class when both flip %.3f | top-nony class agree overall %.3f | orig runner-up agree: %s %.3f, sisters %.3f'%(fl.mean(),sflip.mean(),both.mean(),fl.mean()*sflip.mean(),(w[both]==sw[both]).mean(),(w==sw).mean(),nm,(w==W['R2'].argmax(1)).mean(),(sw==W['R2'].argmax(1)).mean()))
D=json.load(open(S+'/tomo/feat.json'))
for r in D['rows']:
    k=key[r['n']]
    for v in ('clean','alt'):
        X=lsm(d[k+'::'+v])-lsm(o[v]); Xy=X[I,y]
        for nm,(w,qq) in B.items():
            g=(w*X).sum(1); r[f'{v}_q{nm}_{nm}']=(qq*(g-Xy)).mean()
            r[f'{v}_s_{nm}']=(q['s']*(g-Xy)).mean()
        # mixture basis: average of sister T2 and E18 T2 probs
        for nm in ('E18','T07','A4'):
            pm=0.5*sp+0.5*np.exp(lsm(d[{'E18':'oracle__E18','T07':'oracle__T07','A4':'oracle__A4'}[nm]+'::clean'],2))
            w=nony(pm); qq=1-pm[I,y]; r[f'{v}_mix{nm}']=(qq*((w*X).sum(1)-Xy)).mean()
json.dump(D,open(S+'/tomo/feat.json','w'))
