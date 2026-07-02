"""Routing gate v3 — fixes all 6 flaws Rui raised.
 (1) centered-deviation soft state (s - s_bar), drop last category => no exact collinearity with X.
 (3) SOFT vote probabilities q_loc(3)/q_dir(2) from the 7 votes, FACTORIZED, uses ALL cases (no 24-case
     hard-threshold discard).
 (4) residual fit: shared W0 first, then route on residual R=Y-XW0, separate lambda, intercept unpenalized.
 (2) DECISIVE control: matched vs SHUFFLED routing at identical capacity/lambda/W. Permutation null
     (inference-time shuffle, N=300) + training-time shuffle. Only matched < shuffled => state carries signal.
 (5) low-rank route W_route truncated to rank in {1,2,4,8,full}.
 (6) conclusion scoped to: hard? no -> soft state, centered linear interaction, PCA-pixel repr.
Reports held-out delta-PCA R^2. Positive gap (matched better than shuffle-null) = state has usable signal."""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
HOME=os.path.expanduser("~/ScoliCMF"); H,W=480,240; dh,dw=120,60; Kx,Kd=30,40
LAMBDAS=[1e-3,1e-2,1e-1,1,10,100,1000]; LOCN=["thoracic","thoracolumbar","lumbar"]; DIRN=["image_left","image_right"]
def cfg(): return load_config(os.path.join(HOME,"configs/s2_base.yaml"))
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg()["data"]["root"]),size=(H,W),return_stem=True,
                          split_file=os.path.join(HOME,"splits/%s.txt"%split))
    P=[];Q=[];S=[]
    for xp,xq,st in DataLoader(ds,batch_size=32,shuffle=False): P.append(xp);Q.append(xq);S+=list(st)
    return torch.cat(P),torch.cat(Q),S
# soft factorized state from the 7 votes (uses ALL cases)
votes={}
for l in open(os.path.join(HOME,"labels.json")):
    if l.strip(): r=json.loads(l); votes[r["stem"]]=[v for v in r["votes"] if v and v[0]!="ERR"]
def soft(stems):
    QL=np.zeros((len(stems),3)); QD=np.zeros((len(stems),2))
    for i,s in enumerate(stems):
        vs=votes.get(s,[])
        if not vs: QL[i]=1/3; QD[i]=1/2; continue
        for ln,dn in vs:
            if ln in LOCN: QL[i,LOCN.index(ln)]+=1
            if dn in DIRN: QD[i,DIRN.index(dn)]+=1
        QL[i]=QL[i]/max(QL[i].sum(),1); QD[i]=QD[i]/max(QD[i].sum(),1)
    return QL,QD
def ds_delta(P,Q):
    p=F.interpolate(P,size=(dh,dw),mode="area").flatten(1).numpy()
    q=F.interpolate(Q,size=(dh,dw),mode="area").flatten(1).numpy(); return p,q-p
def pca_fit(Xm,k): mu=Xm.mean(0); U,S,Vt=np.linalg.svd(Xm-mu,full_matrices=False); return mu,Vt[:k]
def pca(Xm,mu,Vt): return (Xm-mu)@Vt.T
def ridge_ic(X,Y,lam):  # intercept unpenalized
    Xa=np.c_[X,np.ones(len(X))]; A=Xa.T@Xa+lam*np.eye(Xa.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A,Xa.T@Y)
def pred_ic(X,W): return np.c_[X,np.ones(len(X))]@W
def cv_lam(X,Y,folds=5):
    n=len(X); idx=np.arange(n); np.random.RandomState(0).shuffle(idx); best=(1e18,1.0)
    for lam in LAMBDAS:
        e=[]
        for f in range(folds):
            te=idx[f::folds]; tr=np.setdiff1d(idx,te); Wt=ridge_ic(X[tr],Y[tr],lam); e.append(((Y[te]-pred_ic(X[te],Wt))**2).mean())
        if np.mean(e)<best[0]: best=(np.mean(e),lam)
    return best[1]
def r2(Y,P): return 1-((Y-P)**2).sum()/(((Y-Y.mean(0))**2).sum()+1e-9)
def route_feats(X,ql,qd,mL,mD):  # centered deviation, drop last category
    dl=(ql-mL)[:,:-1]; dd=(qd-mD)[:,:-1]
    lx=(dl[:,:,None]*X[:,None,:]).reshape(len(X),-1); dx=(dd[:,:,None]*X[:,None,:]).reshape(len(X),-1)
    return np.concatenate([dl,dd,lx,dx],1)

Ptr,Qtr,Str=load("train"); Pva,Qva,Sva=load("val")
pre_tr,dl_tr=ds_delta(Ptr,Qtr); pre_va,dl_va=ds_delta(Pva,Qva)
mup,Vp=pca_fit(pre_tr,Kx); mud,Vd=pca_fit(dl_tr,Kd)
Xtr=pca(pre_tr,mup,Vp); Xva=pca(pre_va,mup,Vp); Ytr=pca(dl_tr,mud,Vd); Yva=pca(dl_va,mud,Vd)
QLtr,QDtr=soft(Str); QLva,QDva=soft(Sva); mL=QLtr.mean(0); mD=QDtr.mean(0)
Ztr=route_feats(Xtr,QLtr,QDtr,mL,mD); Zva=route_feats(Xva,QLva,QDva,mL,mD)
# shared (residual base)
lamS=cv_lam(Xtr,Ytr); WS=ridge_ic(Xtr,Ytr,lamS); PSva=pred_ic(Xva,WS); r2S=r2(Yva,PSva)
Rtr=Ytr-pred_ic(Xtr,WS)
# route on residual, own lambda
lamR=cv_lam(Ztr,Rtr); WR_full=ridge_ic(Ztr,Rtr,lamR)
print(f"=== v3: soft state, centered, residual-fit | shared R2={r2S:.4f} (lamS={lamS}) | route dim={Ztr.shape[1]} lamR={lamR} ===",flush=True)
def route_pred(Zv,Wr): return PSva+pred_ic(Zv,Wr)
def lowrank(W,r):
    Wb=W[:-1]; b=W[-1:]; U,s,Vt=np.linalg.svd(Wb,full_matrices=False); s2=s.copy(); s2[r:]=0
    return np.vstack([U@np.diag(s2)@Vt, b])
rng=np.random.RandomState(0)
print("  rank |  matched R2 | shuffled R2 (mean±sd, N=300) | matched>shuffled? | p(matched beats)",flush=True)
for rk in [1,2,4,8,"full"]:
    Wr=WR_full if rk=="full" else lowrank(WR_full,rk)
    r2m=r2(Yva,route_pred(Zva,Wr))
    sh=[]
    for _ in range(300):
        perm=rng.permutation(len(Sva)); Zs=route_feats(Xva,QLva[perm],QDva[perm],mL,mD)
        sh.append(r2(Yva,route_pred(Zs,Wr)))
    sh=np.array(sh); pbeat=np.mean(r2m>sh)
    print(f"  {str(rk):>4} |   {r2m:+.4f}  |  {sh.mean():+.4f} ± {sh.std():.4f}      |  {r2m-sh.mean():+.4f}        | {pbeat:.2f}",flush=True)
# training-time shuffle control (full rank): shuffle train state, refit, apply matched val
perm=rng.permutation(len(Str)); Ztr_s=route_feats(Xtr,QLtr[perm],QDtr[perm],mL,mD)
WR_s=ridge_ic(Ztr_s,Rtr,lamR); r2_trshuf=r2(Yva,route_pred(Zva,WR_s))
print(f"  [train-shuffle control, full rank] matched-state val R2 with route trained on SHUFFLED train state = {r2_trshuf:+.4f}  (vs matched-train {r2(Yva,route_pred(Zva,WR_full)):+.4f})",flush=True)
print("SCOPE: negative here = no signal under SOFT state + centered linear interaction + PCA-pixel repr; does NOT rule out nonlinear deep-feature low-rank routing.",flush=True)
print("ROUTE3_GATE_DONE",flush=True)
