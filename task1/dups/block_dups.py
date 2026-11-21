# recherche de contenu partagé à translation près (blocs 64x64, pas 16) + test miroir
import os, json, numpy as np, torch
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
torch.set_num_threads(8)
TASK=os.environ['TASK']; R=f'{TASK}/release'
names=[('train',n[:-4]) for n in sorted(os.listdir(f'{R}/train/images'))]+[('test',n[:-4]) for n in sorted(os.listdir(f'{R}/test/images'))]
def load(a):
    s,n=a; return np.asarray(Image.open(f'{R}/{s}/images/{n}.png').convert('L'),np.float32)
with ProcessPoolExecutor(8) as ex: G=np.stack(list(ex.map(load,names,chunksize=16)))
N=len(G)
P=torch.from_numpy(G).view(N,64,8,64,8).mean((2,4))  # 64x64 grille 8px
def blocks(P,step):  # blocs 8x8 cellules (=64 px), pas en cellules
    u=P.unfold(1,8,step).unfold(2,8,step)  # N,h,w,8,8
    return u.reshape(N,-1,64)
def norm(x):
    x=x-x.mean(-1,keepdim=True); s=x.norm(dim=-1,keepdim=True); return x/(s+1e-6), s.squeeze(-1)
DB,sdb=norm(blocks(P,2)); nb=DB.shape[1]  # pas 16 px
Q,sq=norm(blocks(P,16)); nq=Q.shape[1]     # pas 128 px -> 49? (64-8)/16+1=4 ->16
print('db',DB.shape,'q',Q.shape)
DBf=DB.reshape(-1,64); owner=torch.arange(N).repeat_interleave(nb)
tex=(sq/8>15)  # écart-type pixel approx > 15
best=torch.full((N,N),-1.0)
Qf=Q.reshape(-1,64); qown=torch.arange(N).repeat_interleave(nq); qtex=tex.reshape(-1)
for c in range(0,Qf.shape[0],512):
    S=Qf[c:c+512]@DBf.T  # 512 x 1.29M
    S=S.view(S.shape[0],N,nb).amax(-1)  # max par image
    S[torch.arange(S.shape[0]),qown[c:c+512]]=-1
    S[~qtex[c:c+512]]=-1
    for k in range(S.shape[0]):
        best[qown[c+k]]=torch.maximum(best[qown[c+k]],S[k])
# compte de blocs-requêtes texturés ayant un match >0.97 dans une même autre image
m=best.max(1).values
print('max block corr to other image quantiles',np.quantile(m.numpy(),[0.5,0.9,0.99,1]))
idx=torch.argsort(m,descending=True)[:15]
for i in idx: print(names[i],names[int(best[i].argmax())],float(m[i]))
# miroir global
F=torch.from_numpy(G[:,:,::-1].copy()).view(N,64,8,64,8).mean((2,4)).reshape(N,-1)
A,_=norm(P.reshape(N,-1)); B,_=norm(F)
C=A@B.T; C.fill_diagonal_(-1); print('flip max off-diag',float(C.max()))
# vérification pixel : pour chaque image, 3 meilleurs partenaires, matchTemplate pleine résolution des blocs requêtes
import cv2
top3=torch.topk(best,3,dim=1).indices.numpy()
def verify(i):
    out=[]
    for j in top3[i]:
        bm=(-1,0,0)
        for by in range(4):
            for bx in range(4):
                y,x=32+by*128,32+bx*128
                t=G[i,y:y+64,x:x+64]
                if t.std()<15: continue
                r=cv2.matchTemplate(G[j],t,cv2.TM_CCOEFF_NORMED)
                _,mx,_,loc=cv2.minMaxLoc(r)
                if mx>bm[0]:
                    u=G[j,loc[1]:loc[1]+64,loc[0]:loc[0]+64]
                    bm=(mx,float(np.abs(u-t).mean()),(y-loc[1],x-loc[0]))
        out.append((i,int(j))+bm)
    return out
with ProcessPoolExecutor(8) as ex: V=[r for o in ex.map(verify,range(N),chunksize=8) for r in o]
V.sort(key=lambda r:-r[2])
nc=np.array([r[2] for r in V]); print('pixel NCC best per pair quantiles',np.quantile(nc,[0.5,0.9,0.99,1]))
print('pairs NCC>0.98:',int((nc>0.98).sum()),' MAD<3:',sum(r[3]<3 for r in V))
for r in V[:15]: print(names[r[0]],names[r[1]],'ncc%.4f mad%.2f shift%s'%(r[2],r[3],r[4]))
print('--- tri par MAD (NCC>0.9) ---')
for r in sorted([r for r in V if r[2]>0.9],key=lambda r:r[3])[:12]: print(names[r[0]],names[r[1]],'ncc%.4f mad%.2f shift%s'%(r[2],r[3],r[4]))
import pickle; pickle.dump((V,names),open(os.environ['TASK']+'/t1/out/dups/block_verify.pkl','wb'))
