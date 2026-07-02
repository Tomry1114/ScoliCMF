"""Phenotype-routing headroom gate (CPU, no big training). Tests the framework's core thesis:
'different deformity states -> different pre->post transition priors'.
Part1 (thesis, zero-train): per-phenotype-cell mean transition delta vs GLOBAL mean delta, held-out val.
Part2 (adds over x_pre?): ridge x_pre_pca -> delta_pca, WITH vs WITHOUT phenotype one-hot, held-out val.
Decision: P1 & P2 positive => build adapters. P1 positive P2 null => phenotype is an x_pre proxy.
P1 null => thesis fails."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from metrics_img import ssim as ssim_fn
HOME=os.path.expanduser("~/ScoliCMF"); H,W=480,240; dh,dw=120,60; AGREE=5/7-1e-6
def cfg(): return load_config(os.path.join(HOME,"configs/s2_base.yaml"))
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg()["data"]["root"]),size=(H,W),return_stem=True,
                          split_file=os.path.join(HOME,"splits/%s.txt"%split))
    P=[];Q=[];S=[]
    for xp,xq,st in DataLoader(ds,batch_size=32,shuffle=False):
        P.append(xp);Q.append(xq);S+=list(st)
    return torch.cat(P),torch.cat(Q),S
# labels
lab={}
for l in open(os.path.join(HOME,"labels.json")):
    if l.strip():
        r=json.loads(l)
        cell=(f'{r["dominant_location"]}|{r["dominant_direction"]}'
              if r["agree_loc"]>=AGREE and r["agree_dir"]>=AGREE and r["dominant_location"]!="uncertain"
              and r["dominant_direction"]!="uncertain" else "uncertain")
        lab[r["stem"]]=cell
def cells_of(stems): return np.array([lab.get(s,"uncertain") for s in stems])
def ds_delta(P,Q):  # downsample -> flatten delta and pre
    import torch.nn.functional as F
    p=F.interpolate(P,size=(dh,dw),mode="area").flatten(1).numpy()
    q=F.interpolate(Q,size=(dh,dw),mode="area").flatten(1).numpy()
    return p, q-p
Ptr,Qtr,Str=load("train"); Pva,Qva,Sva=load("val")
pre_tr,dl_tr=ds_delta(Ptr,Qtr); pre_va,dl_va=ds_delta(Pva,Qva)
ctr=cells_of(Str); cva=cells_of(Sva)
from collections import Counter
print("=== label coverage ===",flush=True)
print("  train cells:",dict(Counter(ctr)),flush=True)
print("  val   cells:",dict(Counter(cva)),flush=True)
def ssim_ds(predflat,postP):
    import torch.nn.functional as F
    pr=torch.tensor(predflat.reshape(-1,1,dh,dw),dtype=torch.float32).clamp(0,1)
    po=F.interpolate(postP,size=(dh,dw),mode="area")
    return float(ssim_fn(pr,po).mean())
# ---- Part 1: cell-mean vs global-mean transition prior ----
gmean=dl_tr.mean(0,keepdims=True)
cellmean={c:dl_tr[ctr==c].mean(0) for c in set(ctr) if (ctr==c).sum()>=5}
pred_g=pre_va+gmean
pred_p=np.stack([pre_va[i]+(cellmean[cva[i]] if cva[i] in cellmean and cva[i]!="uncertain" else gmean[0]) for i in range(len(Sva))])
print("=== PART 1: transition prior (held-out val) ===",flush=True)
print("  delta-MSE  global-mean=%.5f  phenotype-mean=%.5f  (lower=better)"%(((dl_va-gmean)**2).mean(),((dl_va-(pred_p-pre_va))**2).mean()),flush=True)
print("  SSIM(x_pre+prior, x_post)  global=%.4f  phenotype=%.4f"%(ssim_ds(pred_g,Qva),ssim_ds(pred_p,Qva)),flush=True)
frac_routed=np.mean([cva[i] in cellmean and cva[i]!="uncertain" for i in range(len(Sva))])
print("  (val cases actually routed to a cell prior: %.0f%%)"%(100*frac_routed),flush=True)
# ---- Part 2: does phenotype add over x_pre? ridge on PCA ----
def pca_fit(X,k):
    mu=X.mean(0); U,S,Vt=np.linalg.svd(X-mu,full_matrices=False); return mu,Vt[:k]
def pca(X,mu,Vt): return (X-mu)@Vt.T
Kp,Kd=40,40
mup,Vp=pca_fit(pre_tr,Kp); mud,Vd=pca_fit(dl_tr,Kd)
Xtr=pca(pre_tr,mup,Vp); Xva=pca(pre_va,mup,Vp); Ytr=pca(dl_tr,mud,Vd); Yva=pca(dl_va,mud,Vd)
allcells=sorted(set(c for c in ctr if c!="uncertain"))
def onehot(cs): return np.stack([[1.0 if c==k else 0.0 for k in allcells] for c in cs])
def ridge(X,Y,lam=1.0):
    A=X.T@X+lam*np.eye(X.shape[1]); return np.linalg.solve(A,X.T@Y)
def r2(Yte,pred): return 1-((Yte-pred)**2).sum()/(((Yte-Yte.mean(0))**2).sum()+1e-9)
XA_tr=np.c_[Xtr,np.ones(len(Xtr))]; XA_va=np.c_[Xva,np.ones(len(Xva))]
WA=ridge(XA_tr,Ytr); r2A=r2(Yva,XA_va@WA)
XB_tr=np.c_[Xtr,onehot(ctr),np.ones(len(Xtr))]; XB_va=np.c_[Xva,onehot(cva),np.ones(len(Xva))]
WB=ridge(XB_tr,Ytr); r2B=r2(Yva,XB_va@WB)
# phenotype-only (no x_pre) as sanity
XC_tr=np.c_[onehot(ctr),np.ones(len(Xtr))]; XC_va=np.c_[onehot(cva),np.ones(len(Xva))]
WC=ridge(XC_tr,Ytr); r2C=r2(Yva,XC_va@WC)
print("=== PART 2: held-out delta-PCA R^2 (higher=better) ===",flush=True)
print("  pooled x_pre only        R2=%.4f"%r2A,flush=True)
print("  x_pre + phenotype onehot R2=%.4f   (delta vs pooled = %+.4f)"%(r2B,r2B-r2A),flush=True)
print("  phenotype onehot ONLY    R2=%.4f   (proxy strength)"%r2C,flush=True)
print("ROUTE_GATE_DONE",flush=True)
