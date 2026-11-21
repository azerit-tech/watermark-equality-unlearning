import os, json, numpy as np
from PIL import Image
from scipy.fft import dctn
from concurrent.futures import ProcessPoolExecutor
TASK=os.environ['TASK']; R=f'{TASK}/release'
OUT=f'{TASK}/t1/out/dups'
tr=json.load(open(f'{TASK}/t1/common/train_index.json'))
te=json.load(open(f'{R}/test/metadata.json')); tem={d['image_name']:d for d in te}
names=[('train',n[:-4]) for n in sorted(os.listdir(f'{R}/train/images'))]+[('test',n[:-4]) for n in sorted(os.listdir(f'{R}/test/images'))]
def load(a):
    s,n=a; im=Image.open(f'{R}/{s}/images/{n}.png').convert('L')
    return np.asarray(im.resize((32,32),Image.BILINEAR),np.float32), np.asarray(im.resize((64,64),Image.BILINEAR),np.float32)
with ProcessPoolExecutor(8) as ex: res=list(ex.map(load,names,chunksize=16))
T=np.stack([r[0] for r in res]); T64=np.stack([r[1] for r in res])
N=len(names)
ah=(T.reshape(N,-1)>T.reshape(N,-1).mean(1,keepdims=True))
D=np.stack([dctn(t,norm='ortho')[:8,:8].ravel()[1:] for t in T]); ph=D>np.median(D,1,keepdims=True)
ham_a=(ah[:,None,:]!=ah[None]).sum(-1); ham_p=(ph[:,None,:]!=ph[None]).sum(-1)
X=T64.reshape(N,-1); Xc=X-X.mean(1,keepdims=True); Xc/=np.linalg.norm(Xc,axis=1,keepdims=True)+1e-6
corr=Xc@Xc.T
# also flipped
Xf=T64[:,:,::-1].reshape(N,-1); Xf=Xf-Xf.mean(1,keepdims=True); Xf/=np.linalg.norm(Xf,axis=1,keepdims=True)+1e-6
corrf=Xc@Xf.T
np.fill_diagonal(corr,-1); np.fill_diagonal(ham_p,999); np.fill_diagonal(ham_a,999)
print('nn corr quantiles',np.quantile(corr.max(1),[0.5,0.9,0.99,1]))
print('nn corr flip max',corrf.max())
print('min ham p',np.sort(ham_p.min(1))[:20]); print('min ham a',np.sort(ham_a.min(1))[:20])
print('top corr',np.sort(corr.max(1))[::-1][:20])
iu=np.triu_indices(N,1)
cand=set()
for i,j in zip(*iu):
    if ham_p[i,j]<=10 or ham_a[i,j]<=100 or corr[i,j]>0.9 or corrf[i,j]>0.95: cand.add((i,j))
print('candidates',len(cand))
def info(k):
    s,n=names[k]
    if s=='train': d=tr[n]; return f'{s}:{n}:{d["method"]}:{d["edit_type"]}'
    return f'{s}:{n}:{tem[n]["method"]}'
rows=[]
for i,j in sorted(cand,key=lambda p:-corr[p]):
    a=np.asarray(Image.open(f'{R}/{names[i][0]}/images/{names[i][1]}.png').convert('RGB'),np.int16)
    b=np.asarray(Image.open(f'{R}/{names[j][0]}/images/{names[j][1]}.png').convert('RGB'),np.int16)
    if a.shape!=b.shape: print('shape',a.shape,b.shape); continue
    d=np.abs(a-b).max(-1)
    rows.append((i,j,float(corr[i,j]),int(ham_p[i,j]),int(ham_a[i,j]),float(d.mean()),float((d<=8).mean()),float((d==0).mean())))
    print(info(i),info(j),'corr%.3f hp%d ha%d mad%.2f le8 %.3f eq0 %.3f'%rows[-1][2:])
np.savez(f'{OUT}/cand.npz',rows=np.array(rows),names=np.array(['%s/%s'%x for x in names]))
