"""Routing gate v4 — fixes all 7 implementation issues (Rui round 2).
 (1) confidence-GATED soft state: g=known/7, p=dist over known, c=g*Helmert(p-p_bar); all-uncertain=>0.
 (4) Helmert orthogonal contrast (not treatment/drop-last) + per-column z-score of route features Z.
 (2) route has NO intercept (shared already has one).
 (3) JOINT block-ridge: min ||Y-[X|Z]beta-b||^2 + lamS||W||^2 + lamR||V||^2, nested CV over (lamS,lamR);
     no residual leakage.
 (5) full-rank linear gate ONLY (no post-hoc SVD 'low-rank' claim).
 (6) train-shuffle control 300x (not 1).
 (7) real one-sided permutation p=(1+#{null>=matched})/(N+1); + patient-level bootstrap 95% CI.
Metric: held-out delta-PCA R^2. Effect = matched vs shuffled(same capacity) and matched vs shared."""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
HOME=os.path.expanduser("~/ScoliCMF"); H,W=480,240; dh,dw=120,60; Kx,Kd=30,40
LAMS=[1e-2,1e-1,1,10,100,1000]; LOCN=["thoracic","thoracolumbar","lumbar"]; DIRN=["image_left","image_right"]
HL=np.array([[1,-1,0],[1,1,-2]],float); HL=HL/np.linalg.norm(HL,axis=1,keepdims=True)  # 2x3 Helmert
HD=np.array([[1,-1]],float)/np.sqrt(2)                                                   # 1x2
def cfg(): return load_config(os.path.join(HOME,"configs/s2_base.yaml"))
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg()["data"]["root"]),size=(H,W),return_stem=True,
                          split_file=os.path.join(HOME,"splits/%s.txt"%split))
    P=[];Q=[];S=[]
    for xp,xq,st in DataLoader(ds,batch_size=32,shuffle=False): P.append(xp);Q.append(xq);S+=list(st)
    return torch.cat(P),torch.cat(Q),S
votes={}
for l in open(os.path.join(HOME,"labels.json")):
    if l.strip(): r=json.loads(l); votes[r["stem"]]=[v for v in r["votes"] if v and v[0]!="ERR"]
def raw_pg(stems):  # per-case p (dist over known) and g (confidence), for loc and dir
    PL=np.zeros((len(stems),3)); GL=np.zeros(len(stems)); PD=np.zeros((len(stems),2)); GD=np.zeros(len(stems))
    for i,s in enumerate(stems):
        vs=votes.get(s,[]); cl=np.zeros(3); cd=np.zeros(2); nl=nd=0
        for ln,dn in vs:
            if ln in LOCN: cl[LOCN.index(ln)]+=1; nl+=1
            if dn in DIRN: cd[DIRN.index(dn)]+=1; nd+=1
        GL[i]=nl/7.0; GD[i]=nd/7.0
        PL[i]=cl/nl if nl>0 else 0; PD[i]=cd/nd if nd>0 else 0
    return PL,GL,PD,GD
def contrasts(PL,GL,PD,GD,pbarL,pbarD):
    cL=GL[:,None]*((PL-pbarL)@HL.T)   # (N,2)
    cD=GD[:,None]*((PD-pbarD)@HD.T)   # (N,1)
    return cL,cD
def ds_delta(P,Q):
    p=F.interpolate(P,size=(dh,dw),mode="area").flatten(1).numpy()
    q=F.interpolate(Q,size=(dh,dw),mode="area").flatten(1).numpy(); return p,q-p
def pca_fit(Xm,k): mu=Xm.mean(0); U,S,Vt=np.linalg.svd(Xm-mu,full_matrices=False); return mu,Vt[:k]
def pca(Xm,mu,Vt): return (Xm-mu)@Vt.T
def buildZ(cL,cD,X):  # raw route features [cL, cD, cL@X, cD@X]
    lx=(cL[:,:,None]*X[:,None,:]).reshape(len(X),-1); dx=(cD[:,:,None]*X[:,None,:]).reshape(len(X),-1)
    return np.concatenate([cL,cD,lx,dx],1)
def r2(Y,P): return 1-((Y-P)**2).sum()/(((Y-Y.mean(0))**2).sum()+1e-9)
def joint_fit(X,Z,Y,lamS,lamR):  # [X|Z|1], penalty lamS on X, lamR on Z, 0 on intercept
    Xf=np.concatenate([X,Z,np.ones((len(X),1))],1)
    pen=np.concatenate([np.full(X.shape[1],lamS),np.full(Z.shape[1],lamR),[0.0]])
    A=Xf.T@Xf+np.diag(pen); return np.linalg.solve(A,Xf.T@Y)
def joint_pred(X,Z,W): return np.concatenate([X,Z,np.ones((len(X),1))],1)@W
def nested_cv(X,Z,Y,folds=5):
    n=len(X); idx=np.arange(n); np.random.RandomState(0).shuffle(idx); best=(1e18,1,1)
    for lS in LAMS:
        for lR in LAMS:
            e=[]
            for f in range(folds):
                te=idx[f::folds]; tr=np.setdiff1d(idx,te)
                Wt=joint_fit(X[tr],Z[tr],Y[tr],lS,lR); e.append(((Y[te]-joint_pred(X[te],Z[te],Wt))**2).mean())
            if np.mean(e)<best[0]: best=(np.mean(e),lS,lR)
    return best[1],best[2]

Ptr,Qtr,Str=load("train"); Pva,Qva,Sva=load("val")
pre_tr,dl_tr=ds_delta(Ptr,Qtr); pre_va,dl_va=ds_delta(Pva,Qva)
mup,Vp=pca_fit(pre_tr,Kx); mud,Vd=pca_fit(dl_tr,Kd)
Xtr=pca(pre_tr,mup,Vp); Xva=pca(pre_va,mup,Vp); Ytr=pca(dl_tr,mud,Vd); Yva=pca(dl_va,mud,Vd)
PLt,GLt,PDt,GDt=raw_pg(Str); PLv,GLv,PDv,GDv=raw_pg(Sva)
pbarL=(PLt*GLt[:,None]).sum(0)/GLt.sum(); pbarD=(PDt*GDt[:,None]).sum(0)/GDt.sum()  # confidence-weighted train mean
cLt,cDt=contrasts(PLt,GLt,PDt,GDt,pbarL,pbarD); cLv,cDv=contrasts(PLv,GLv,PDv,GDv,pbarL,pbarD)
Ztr_raw=buildZ(cLt,cDt,Xtr); Zva_raw=buildZ(cLv,cDv,Xva)
zmu=Ztr_raw.mean(0,keepdims=True); zsd=Ztr_raw.std(0,keepdims=True).clip(1e-6)
def std(Zr): return (Zr-zmu)/zsd
Ztr=std(Ztr_raw); Zva=std(Zva_raw)
# shared-only baseline
def cvS(X,Y,folds=5):
    n=len(X);idx=np.arange(n);np.random.RandomState(0).shuffle(idx);best=(1e18,1)
    for l in LAMS:
        e=[]
        for f in range(folds):
            te=idx[f::folds];tr=np.setdiff1d(idx,te)
            Xf=np.c_[X[tr],np.ones(len(tr))];A=Xf.T@Xf+l*np.eye(Xf.shape[1]);A[-1,-1]-=l;Wt=np.linalg.solve(A,Xf.T@Y[tr])
            e.append(((Y[te]-np.c_[X[te],np.ones(len(te))]@Wt)**2).mean())
        if np.mean(e)<best[0]:best=(np.mean(e),l)
    return best[1]
lS0=cvS(Xtr,Ytr); Xf=np.c_[Xtr,np.ones(len(Xtr))];A=Xf.T@Xf+lS0*np.eye(Xf.shape[1]);A[-1,-1]-=lS0;WS=np.linalg.solve(A,Xf.T@Ytr)
r2_shared=r2(Yva,np.c_[Xva,np.ones(len(Xva))]@WS)
lamS,lamR=nested_cv(Xtr,Ztr,Ytr); Wj=joint_fit(Xtr,Ztr,Ytr,lamS,lamR)
r2_matched=r2(Yva,joint_pred(Xva,Zva,Wj))
print(f"=== v4 joint block-ridge (lamS={lamS} lamR={lamR}), route dim={Ztr.shape[1]} ===",flush=True)
print(f"  shared-only R2 = {r2_shared:+.4f}   matched R2 = {r2_matched:+.4f}   net = {r2_matched-r2_shared:+.4f}",flush=True)
rng=np.random.RandomState(0); N=300
# inference-shuffle: permute val state contrasts, rebuild Z_va
inf=[]
for _ in range(N):
    p=rng.permutation(len(Sva)); Zs=std(buildZ(cLv[p],cDv[p],Xva)); inf.append(r2(Yva,joint_pred(Xva,Zs,Wj)))
inf=np.array(inf)
# train-shuffle: permute train state, refit joint, eval matched val
trs=[]
for _ in range(N):
    p=rng.permutation(len(Str)); Zs=std(buildZ(cLt[p],cDt[p],Xtr)); Wt=joint_fit(Xtr,Zs,Ytr,lamS,lamR); trs.append(r2(Yva,joint_pred(Xva,Zva,Wt)))
trs=np.array(trs)
def pval(null,obs): return (1+np.sum(null>=obs))/(len(null)+1)
print(f"  INFERENCE-shuffle null R2 = {inf.mean():+.4f} ± {inf.std():.4f}  | matched-null = {r2_matched-inf.mean():+.4f}  perm p = {pval(inf,r2_matched):.3f}",flush=True)
print(f"  TRAIN-shuffle null    R2 = {trs.mean():+.4f} ± {trs.std():.4f}  | matched-null = {r2_matched-trs.mean():+.4f}  perm p = {pval(trs,r2_matched):.3f}",flush=True)
# patient-level bootstrap of (matched - shared) and (matched - inf-shuffle-per-boot)
B=1000; d_ms=[]
for _ in range(B):
    bi=rng.randint(0,len(Sva),len(Sva))
    rm=r2(Yva[bi],joint_pred(Xva[bi],Zva[bi],Wj)); rs=r2(Yva[bi],np.c_[Xva[bi],np.ones(len(bi))]@WS); d_ms.append(rm-rs)
d_ms=np.array(d_ms)
print(f"  patient bootstrap (matched-shared): mean {d_ms.mean():+.4f}  95%CI [{np.percentile(d_ms,2.5):+.4f}, {np.percentile(d_ms,97.5):+.4f}]",flush=True)
print("SCOPE: applies to SOFT confidence-gated state + Helmert-contrast linear interaction + PCA-pixel repr; does NOT test nonlinear deep-feature low-rank adapter.",flush=True)
print("ROUTE4_GATE_DONE",flush=True)
