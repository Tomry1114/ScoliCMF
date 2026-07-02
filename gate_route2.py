"""Routing gate v2 — fixes #2,#3,#4,#5. Tests state-CONDITIONED TRANSFORMATION (interaction s@X),
FACTORIZED (loc adapter + dir adapter), with lambda-CV, REGIONAL metrics, and ROUTED-SUBSET separate.
Model compared:  shared  Y=W0 X   vs   routed  Y=W0 X + sum_l s_l W_l X + sum_d s_d W_d X + biases."""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
HOME=os.path.expanduser("~/ScoliCMF"); H,W=480,240; dh,dw=120,60; AGREE=5/7-1e-6
Kx,Kd=30,40; LAMBDAS=[1e-4,1e-3,1e-2,1e-1,1,10,100,1000]
def cfg(): return load_config(os.path.join(HOME,"configs/s2_base.yaml"))
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg()["data"]["root"]),size=(H,W),return_stem=True,
                          split_file=os.path.join(HOME,"splits/%s.txt"%split))
    P=[];Q=[];S=[]
    for xp,xq,st in DataLoader(ds,batch_size=32,shuffle=False): P.append(xp);Q.append(xq);S+=list(st)
    return torch.cat(P),torch.cat(Q),S
lab={}
for l in open(os.path.join(HOME,"labels_pair.json")):
    if l.strip():
        r=json.loads(l)
        ok=r["agree_pair"]>=AGREE and r["dominant_location"]!="uncertain" and r["dominant_direction"]!="uncertain"
        lab[r["stem"]]=(r["dominant_location"],r["dominant_direction"]) if ok else ("uncertain","uncertain")
def cells_of(stems): return [lab.get(s,("uncertain","uncertain")) for s in stems]
def ds_delta(P,Q):
    p=F.interpolate(P,size=(dh,dw),mode="area").flatten(1).numpy()
    q=F.interpolate(Q,size=(dh,dw),mode="area").flatten(1).numpy(); return p,q-p
# --- their factorized routing features ---
LOCN=["thoracic","thoracolumbar","lumbar"]; DIRN=["image_left","image_right"]
def factorized_onehot(cells):
    loc=np.zeros((len(cells),3),np.float32); dr=np.zeros((len(cells),2),np.float32)
    for i,(ln,dn) in enumerate(cells):
        if ln in LOCN: loc[i,LOCN.index(ln)]=1.0
        if dn in DIRN: dr[i,DIRN.index(dn)]=1.0
    return loc,dr
def routed_features(X,loc,dr):
    loc_x=(loc[:,:,None]*X[:,None,:]).reshape(len(X),-1)
    dir_x=(dr[:,:,None]*X[:,None,:]).reshape(len(X),-1)
    return np.concatenate([X,loc,dr,loc_x,dir_x,np.ones((len(X),1))],1)
def pca_fit(Xm,k): mu=Xm.mean(0); U,S,Vt=np.linalg.svd(Xm-mu,full_matrices=False); return mu,Vt[:k]
def pca(Xm,mu,Vt): return (Xm-mu)@Vt.T
def ridge(X,Y,lam): A=X.T@X+lam*np.eye(X.shape[1]); return np.linalg.solve(A,X.T@Y)
def cv_lambda(X,Y,folds=5):
    n=len(X); idx=np.arange(n); rng=np.random.RandomState(0); rng.shuffle(idx); best=(1e9,1.0)
    for lam in LAMBDAS:
        errs=[]
        for f in range(folds):
            te=idx[f::folds]; tr=np.setdiff1d(idx,te)
            Wt=ridge(X[tr],Y[tr],lam); errs.append(((Y[te]-X[te]@Wt)**2).mean())
        m=np.mean(errs);
        if m<best[0]: best=(m,lam)
    return best[1]
def r2(Y,P): return 1-((Y-P)**2).sum()/(((Y-Y.mean(0))**2).sum()+1e-9)

Ptr,Qtr,Str=load("train"); Pva,Qva,Sva=load("val")
pre_tr,dl_tr=ds_delta(Ptr,Qtr); pre_va,dl_va=ds_delta(Pva,Qva)
ctr=cells_of(Str); cva=cells_of(Sva)
mup,Vp=pca_fit(pre_tr,Kx); mud,Vd=pca_fit(dl_tr,Kd)
Xtr=pca(pre_tr,mup,Vp); Xva=pca(pre_va,mup,Vp); Ytr=pca(dl_tr,mud,Vd); Yva=pca(dl_va,mud,Vd)
loc_tr,dir_tr=factorized_onehot(ctr); loc_va,dir_va=factorized_onehot(cva)
routed_mask=np.array([c!=("uncertain","uncertain") for c in cva])
print(f"=== coverage: train routed {sum(c!=('uncertain','uncertain') for c in ctr)}/{len(ctr)}  val routed {routed_mask.sum()}/{len(cva)} ===",flush=True)
# shared
XA_tr=np.c_[Xtr,np.ones(len(Xtr))]; XA_va=np.c_[Xva,np.ones(len(Xva))]
lamA=cv_lambda(XA_tr,Ytr); WA=ridge(XA_tr,Ytr,lamA); PA=XA_va@WA
# routed (factorized interactions)
XB_tr=routed_features(Xtr,loc_tr,dir_tr); XB_va=routed_features(Xva,loc_va,dir_va)
lamB=cv_lambda(XB_tr,Ytr); WB=ridge(XB_tr,Ytr,lamB); PB=XB_va@WB
print(f"  lambda: shared={lamA}  routed={lamB}  (routed feat dim={XB_tr.shape[1]})",flush=True)
print("=== held-out delta-PCA R^2 (higher=better) ===",flush=True)
print(f"  ALL val     shared={r2(Yva,PA):.4f}  routed={r2(Yva,PB):.4f}  delta={r2(Yva,PB)-r2(Yva,PA):+.4f}",flush=True)
m=routed_mask
if m.sum()>3:
    print(f"  ROUTED sub  shared={r2(Yva[m],PA[m]):.4f}  routed={r2(Yva[m],PB[m]):.4f}  delta={r2(Yva[m],PB[m])-r2(Yva[m],PA[m]):+.4f}  (n={m.sum()})",flush=True)
# regional MSE in pixel space (reconstruct delta), fix #4
def recon(P): return P@Vd+mud
dhat_A=recon(PA).reshape(-1,dh,dw); dhat_B=recon(PB).reshape(-1,dh,dw); dtrue=dl_va.reshape(-1,dh,dw)
def rmse(a,b,sel=None):
    d=(a-b)**2
    if sel is not None: d=d[sel]
    return d
regions={"full":(slice(None),slice(None)),"corridor":(slice(None),slice(dw//3,2*dw//3)),
         "upper":(slice(0,dh//3),slice(None)),"mid":(slice(dh//3,2*dh//3),slice(None)),"lower":(slice(2*dh//3,dh),slice(None))}
print("=== regional MSE (shared -> routed), ALL val | ROUTED subset ===",flush=True)
for rn,(rs,cs) in regions.items():
    A_all=((dhat_A[:,rs,cs]-dtrue[:,rs,cs])**2).mean(); B_all=((dhat_B[:,rs,cs]-dtrue[:,rs,cs])**2).mean()
    line=f"  {rn:8s} ALL {A_all:.5f}->{B_all:.5f} ({100*(B_all-A_all)/A_all:+.1f}%)"
    if m.sum()>3:
        A_r=((dhat_A[m][:,rs,cs]-dtrue[m][:,rs,cs])**2).mean(); B_r=((dhat_B[m][:,rs,cs]-dtrue[m][:,rs,cs])**2).mean()
        line+=f"  | ROUTED {A_r:.5f}->{B_r:.5f} ({100*(B_r-A_r)/A_r:+.1f}%)"
    print(line,flush=True)
print("ROUTE2_GATE_DONE",flush=True)
