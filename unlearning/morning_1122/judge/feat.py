import json, sys, numpy as np
S=sys.argv[1]
d=np.load(S+'/an2/logits.npz'); sis=np.load(S+'/tomo/sis.npz')
R={r['n']:r for r in json.load(open(S+'/an2/R.json'))}
key = {"ep05": "cal_lr1e-5__ckpt_ep05", "ep02": "cal_lr1e-5__ckpt_ep02", "L3": "sweep1__L3__submission", "l5e3_a3": "oracle__l5e3_a3", "surgery": "surgery__surg_ep2_g_lateMLPhead",
       "real_G2": "login__real_G2__submission", "G1": "login__G1__submission", "G0_bank": "login__G0_bank__submission", "real_G2_bank": "login__real_G2_bank__submission", "G1_bank": "login__G1_bank__submission",
       "E12": "oracle__E12", "E13": "oracle__E13", "E14": "oracle__E14", "E13_tau1.3": "oracle__E13_tau1.3", "E16": "oracle__E16", "E18": "oracle__E18", "B1x": "ens__B1x", "T05": "oracle__T05", "T07": "oracle__T07",
       "hi_A2": "hi__A2__submission", "hi_B2": "hi__B2__submission", "sM_ep08": "sis__sM__ckpt_ep08", "c11": "sis__t_f0.12_c11__ckpt_ep04", "c13": "sis__t_f0.12_c13__ckpt_ep04", "c16": "sis__t_f0.12_c16__ckpt_ep04", "e2": "sis__e2__ckpt_ep04",
       "qB":"sis__qB__ckpt_ep03","wA":"sis__wA__ckpt_ep03"}
pub={n:R[n]['pub'] for n in R}; pub['qB']=0.2703; pub['wA']=0.268
def lsm(z,T=2.0):
    z=z.astype(np.float64)/T; z=z-z.max(1,keepdims=True); return z-np.log(np.exp(z).sum(1,keepdims=True))
o={v:d['..__model::'+v] for v in ('clean','alt','t2i','ext')}
y=o['clean'].argmax(1); N=len(y); I=np.arange(N)
cnt=np.bincount(y,minlength=100); A=np.isin(y,np.where(cnt>=129)[0])
print('A images',A.sum(),'classes',np.where(cnt>=129)[0])
def nony(p):
    p=p.astype(np.float64).copy(); p[I,y]=0; p=np.maximum(p,1e-12); p[I,y]=0; return p/p.sum(1,keepdims=True)
sp=sis['probs_T2'].astype(np.float64); sp1=sis['probs'].astype(np.float64)
print('sister labels agree', (sis['labels']==y).mean(), 'sis T1 p_y mean',sp1[I,y].mean(),'sis T2 p_y mean',sp[I,y].mean(), 'sis argmax!=y', (sp1.argmax(1)!=y).mean())
W={}
W['R2']=nony(np.exp(lsm(o['clean'],2))); W['R1']=nony(np.exp(lsm(o['clean'],1))); W['S']=nony(sp); W['S1']=nony(sp1)
U=np.ones((N,100)); W['U']=nony(U)
C=np.zeros((100,100))
for c in range(100):
    if cnt[c]: C[c]=sp[y==c].mean(0)
W['C']=nony(C[y])
q={'1':np.ones(N),'s':1-sp[I,y],'s1':1-sp1[I,y],'A':A.astype(float),'B':(~A).astype(float),'flip':(sp1.argmax(1)!=y).astype(float)}
rows=[]
for n,k in key.items():
    r=dict(n=n,pub=pub[n])
    for v in ('clean','alt'):
        z=d[k+'::'+v]; X=lsm(z)-lsm(o[v])
        ce=-lsm(z,1.0)[I,y]; r[v+'_m']=ce.mean(); r[v+'_s']=ce.std(); r[v+'_acc']=(z.argmax(1)==y).mean()
        r[v+'_mA']=ce[A].mean(); r[v+'_mB']=ce[~A].mean(); r[v+'_accA']=(z.argmax(1)==y)[A].mean(); r[v+'_accB']=(z.argmax(1)==y)[~A].mean()
        Xy=X[I,y]; r[v+'_F0']=Xy.mean(); r[v+'_F0A']=(Xy*A).mean(); r[v+'_F0B']=(Xy*~A).mean()
        for wn,w in W.items():
            g=(w*X).sum(1)
            for qn,qq in q.items(): r[f'{v}_{qn}_{wn}']=(qq*(g-Xy)).mean()
    for sub in ('t2i','ext'):
        z=d[k+'::'+sub]; lo=lsm(o[sub]); ls=lsm(z); po=np.exp(lo); kl=(po*(lo-ls)).sum(1)
        conf=np.exp(lsm(o[sub],1.0)).max(1)
        r[sub+'_kl']=kl.mean(); r[sub+'_dis']=(z.argmax(1)!=o[sub].argmax(1)).mean()
        m=conf>=0.999; r[sub+'_kl999']=kl[m].mean(); r[sub+'_dis999']=(z.argmax(1)!=o[sub].argmax(1))[m].mean()
    rows.append(r)
oa=-lsm(o['alt'],1.0)[I,y]; print('orig alt', oa.mean(), oa.std())
json.dump(dict(rows=rows,oa=[oa.mean(),oa.std()]),open(S+'/tomo/feat.json','w'))
for r in rows: print(r['n'].ljust(13),r['pub'],'cl %.2f/%.2f acc %.2f | A %.2f acc %.2f B %.2f acc %.2f | F0 %.3f 1_R2 %.3f 1_S %.3f s_S %.3f flip_S %.3f A_R2 %.3f A_S %.3f 1_U %.3f'%(r['clean_m'],r['clean_s'],r['clean_acc'],r['clean_mA'],r['clean_accA'],r['clean_mB'],r['clean_accB'],r['clean_F0'],r['clean_1_R2'],r['clean_1_S'],r['clean_s_S'],r['clean_flip_S'],r['clean_A_R2'],r['clean_A_S'],r['clean_1_U']))
