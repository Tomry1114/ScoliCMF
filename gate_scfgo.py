"""SC-FGO Gate (NO training, CPU). SC-FGO's premise: there is a PREDICTABLE long-range cranio-caudal
coupling (coronal balance / upper-thoracic<->lumbar relative adjustment) that full self-attention
cannot LEARN from 432 cases, but a restricted fractional-Green operator would capture. This premise is
directly testable on the low-dim signal the user named: the per-row midline curve c(y) (coronal-plane
horizontal centroid). Target = Dc(y)=c_post(y)-c_pre(y). Question: is Dc(y_i) predictable from DISTANT
pre-op rows (GLOBAL) beyond what LOCAL rows give?

Ridge probes (train 432 -> held-out val 54), per latent row i (n_y=24):
  POINT   : Dc_i ~ c_pre[i]                (same row only)
  LOCAL   : Dc_i ~ c_pre[i-w..i+w]         (w=2 band)
  GLOBAL  : Dc_i ~ c_pre[all 24 rows]      (full column = long-range)
  SHUFFLE : GLOBAL but distant rows permuted across cases (control: kills real long-range info)
Report mean held-out R^2. PASS (SC-FGO worth building): R2_global > 0.15 AND R2_global-R2_local >= 0.05
(real predictable LONG-RANGE signal, currently exploitable). FAIL: global~=local (no long-range gain)
OR global~=0 (coronal deformation is plan-determined => ceiling, already in the warp)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
HOME=os.path.expanduser("~/ScoliCMF"); H,W=480,240; NY=24
def cfg():
    c=load_config(os.path.join(HOME,"configs/s2_base.yaml")); return c
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg()["data"]["root"]),size=(H,W),
                          split_file=os.path.join(HOME,"splits/%s.txt"%split))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP).numpy(), torch.cat(XQ).numpy()
def midline(imgs):  # imgs (N,1,H,W) in [0,1] -> per-row centroid curve, pooled to NY rows, x normalized [0,1]
    N=imgs.shape[0]; xs=np.arange(W)[None,:]  # (1,W)
    out=np.zeros((N,NY),dtype=np.float64)
    for n in range(N):
        im=imgs[n,0]                                   # (H,W)
        w=np.clip(im-im.mean(),0,None)                 # bright-structure weight
        num=(w*xs).sum(1); den=w.sum(1)+1e-8
        c=num/den/(W-1)                                # (H,) in [0,1]
        c[den<1e-6]=np.nan
        # pool H->NY by block mean (nan-aware)
        blk=np.array_split(c,NY)
        out[n]=[np.nanmean(b) if np.any(~np.isnan(b)) else np.nan for b in blk]
    col=np.nanmean(out,0); out=np.where(np.isnan(out),col[None,:],out)   # fill empty with column mean
    return out                                          # (N,NY)
def ridge_fit(Xtr,ytr,lam=1.0):
    Xtr=np.concatenate([Xtr,np.ones((Xtr.shape[0],1))],1)
    A=Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A,Xtr.T@ytr)
def ridge_pred(X,w): return np.concatenate([X,np.ones((X.shape[0],1))],1)@w
def r2(ytr_mean,yte,pred):
    ss_res=((yte-pred)**2).sum(); ss_tot=((yte-ytr_mean)**2).sum()+1e-8; return 1-ss_res/ss_tot

print("loading...",flush=True)
xptr,xqtr=load("train"); xpva,xqva=load("val")
Ctr_pre=midline(xptr); Ctr_post=midline(xqtr); Dtr=Ctr_post-Ctr_pre
Cva_pre=midline(xpva); Cva_post=midline(xpva*0+xqva); Dva=Cva_post-Cva_pre
print("train %d val %d  NY=%d"%(xptr.shape[0],xpva.shape[0],NY),flush=True)
print("  mean|Dc| train=%.4f val=%.4f  (magnitude of coronal shift, [0,1] units)"%(np.abs(Dtr).mean(),np.abs(Dva).mean()),flush=True)

def band(C,i,w):
    lo=max(0,i-w); hi=min(NY,i+w+1); B=C[:,lo:hi]
    if B.shape[1]<2*w+1: B=np.pad(B,((0,0),(0,2*w+1-B.shape[1])),mode="edge")
    return B
rng=np.random.RandomState(0)
def probe(mode,w=2):
    R=[]
    for i in range(NY):
        ytr=Dtr[:,i]; yte=Dva[:,i]; ym=ytr.mean()
        if mode=="point": Xtr=Ctr_pre[:,i:i+1]; Xte=Cva_pre[:,i:i+1]
        elif mode=="local": Xtr=band(Ctr_pre,i,w); Xte=band(Cva_pre,i,w)
        elif mode=="global": Xtr=Ctr_pre; Xte=Cva_pre
        elif mode=="shuffle":
            Xtr=Ctr_pre.copy(); Xte=Cva_pre.copy()
            dist=[j for j in range(NY) if abs(j-i)>w]
            for j in dist:
                Xtr[:,j]=Xtr[rng.permutation(Xtr.shape[0]),j]
        wv=ridge_fit(Xtr,ytr); R.append(r2(ym,yte,ridge_pred(Xte,wv)))
    return np.array(R)
print("==== held-out R^2 (mean over %d rows) ===="%NY,flush=True)
res={}
for m in ["point","local","global","shuffle"]:
    R=probe(m); res[m]=R; print("  %-8s R2=%.4f  (upper %.3f / mid %.3f / lower %.3f)"%(
        m,R.mean(),R[:8].mean(),R[8:16].mean(),R[16:].mean()),flush=True)
gain=res["global"].mean()-res["local"].mean()
print("  LONG-RANGE gain (global-local) = %.4f"%gain,flush=True)
print("  global-vs-shuffle (real long-range info) = %.4f"%(res["global"].mean()-res["shuffle"].mean()),flush=True)
passed = (res["global"].mean()>0.15) and (gain>=0.05)
print("VERDICT: %s  (need global>0.15 AND gain>=0.05)"%("PASS - SC-FGO has headroom" if passed else "FAIL - no exploitable long-range signal"),flush=True)
print("SCFGO_GATE_DONE",flush=True)
