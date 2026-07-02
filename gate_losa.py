"""LOSA Gate A(+7)+B, NO training. Frozen AlexNet features + linear CCA between pre/post.
A: does a longitudinal shared subspace exist? matched-vs-shuffled AUC, retrieval Recall@k, CCA spectrum (train-fit, val-eval).
7: occlusion -- spine-only vs periphery-only; if periphery alone re-identifies -> acquisition shortcut.
B: does APTD preserve the shared info? d_real=|c_pre - Sig^-1 c_post|, d_gen=|c_pre - Sig^-1 c_gen|; need d_gen>d_real + corr with LPIPS."""
import os, sys, torch
import torch.nn.functional as F
import torchvision
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from metrics_img import lpips_fn
from aptd_model import APTDNet
dev="cuda"; H,W=480,240; HOME=os.path.expanduser("~/ScoliCMF")
def cfgfull():
    c=load_config(os.path.join(HOME,"configs/s2_base.yaml")); c["model"]["xpre_mode"]="full"; return c
cfg=cfgfull(); mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"], sigma_m=cfg["meanflow"]["sigma_m"]); path=mf.path
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits",split))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP),torch.cat(XQ)
xptr,xqtr=load("train.txt"); xpva,xqva=load("val.txt")
alex=torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).features.to(dev).eval()
xcol=torch.linspace(0,1,W,device=dev); cmask=((xcol-0.5).abs()<0.15).view(1,1,1,W).float()
@torch.no_grad()
def feats(x,mask=None):  # x [N,1,H,W] -> [N,256]
    out=[]
    for i in range(0,x.shape[0],32):
        b=x[i:i+32].to(dev)
        if mask=="spine": b=b*cmask
        elif mask=="peri": b=b*(1-cmask)
        f=alex(b.repeat(1,3,1,1)); out.append(F.adaptive_avg_pool2d(f,1).flatten(1).cpu())
    return torch.cat(out)
def pca_fit(X,k=60):
    mu=X.mean(0); Xc=X-mu; U,S,V=torch.linalg.svd(Xc,full_matrices=False); return mu,V[:k].T
def cca_fit(A,B,lam=1e-2,k=8):
    A=A-A.mean(0); B=B-B.mean(0); n=A.shape[0]
    Caa=A.T@A/n+lam*torch.eye(A.shape[1]); Cbb=B.T@B/n+lam*torch.eye(B.shape[1]); Cab=A.T@B/n
    def invh(C): e,V=torch.linalg.eigh(C); return V@torch.diag(e.clamp_min(1e-6)**-0.5)@V.T
    Ra=invh(Caa); Rb=invh(Cbb); K=Ra@Cab@Rb; U,S,Vt=torch.linalg.svd(K)
    return Ra@U[:,:k], Rb@Vt[:k].T, S[:k]
def auc(matched,mism):
    return float((matched.view(-1,1)>mism.view(1,-1)).float().mean())
def evalset(maskp=None,maskq=None,tag=""):
    Fp=feats(xptr,maskp); Fq=feats(xqtr,maskq); Fpv=feats(xpva,maskp); Fqv=feats(xqva,maskq)
    mup,Pp=pca_fit(Fp); muq,Pq=pca_fit(Fq)
    Ap=(Fp-mup)@Pp; Aq=(Fq-muq)@Pq; Apv=(Fpv-mup)@Pp; Aqv=(Fqv-muq)@Pq
    Wa,Wb,S=cca_fit(Ap,Aq)
    cu=Apv@Wa; cv=Aqv@Wb                    # [Nv,k] canonical coords on val
    cu=cu/(cu.norm(dim=1,keepdim=True)+1e-8); cv=cv/(cv.norm(dim=1,keepdim=True)+1e-8)
    Sm=cu@cv.T                               # sim matrix
    Nv=Sm.shape[0]; diag=Sm.diag(); off=Sm[~torch.eye(Nv,dtype=torch.bool)]
    A_=auc(diag,off)
    rank=(Sm>=diag.view(-1,1)).sum(1)        # rank of true (1=best)
    r1=float((rank<=1).float().mean()); r5=float((rank<=5).float().mean())
    print("  %-14s val matched-vs-shuffled AUC=%.3f  Recall@1=%.2f @5=%.2f  top-CCA sigma(train)=%s"
          %(tag,A_,r1,r5,",".join("%.2f"%s for s in S[:5])),flush=True)
    return dict(Wa=Wa,Wb=Wb,S=S,mup=mup,Pp=Pp,muq=muq,Pq=Pq)

print("==== LOSA GATE A: longitudinal shared subspace ====",flush=True)
full=evalset(None,None,"full-image")
print("==== GATE 7: shortcut check (occlusion) ====",flush=True)
evalset("spine","spine","spine-only")
evalset("peri","peri","periphery-only")
# shuffled null
Fpv=feats(xpva); Fqv=feats(xqva)
cu=((Fpv-full["mup"])@full["Pp"])@full["Wa"]; 
perm=torch.randperm(xqva.shape[0])
cv=((Fqv[perm]-full["muq"])@full["Pq"])@full["Wb"]
cu=cu/(cu.norm(dim=1,keepdim=True)+1e-8); cv=cv/(cv.norm(dim=1,keepdim=True)+1e-8)
Sm=cu@cv.T; Nv=Sm.shape[0]
print("  shuffled-null AUC=%.3f (should be ~0.5)"%auc(Sm.diag(),Sm[~torch.eye(Nv,dtype=torch.bool)]),flush=True)

print("==== GATE B: does APTD preserve the shared subspace? ====",flush=True)
bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
st=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_5000.pt"),map_location=dev)
for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev))
m.eval()
@torch.no_grad()
def apt(xp):
    B=xp.shape[0]; z=xp.clone(); t=torch.ones(B,device=dev); r=torch.zeros(B,device=dev)
    return m(z,r,t,xp)["xhat"].clamp(0,1)
GEN=torch.cat([apt(xpva[i:i+6].to(dev)).cpu() for i in range(0,xpva.shape[0],6)])
def proj(X,mu,P,Wm): return ((feats(X)-mu)@P)@Wm
cu=proj(xpva,full["mup"],full["Pp"],full["Wa"]); cvR=proj(xqva,full["muq"],full["Pq"],full["Wb"]); cvG=proj(GEN,full["muq"],full["Pq"],full["Wb"])
Sig=full["S"].clamp_min(1e-3)
d_real=((cu-cvR/Sig)**2).sum(1).sqrt(); d_gen=((cu-cvG/Sig)**2).sum(1).sqrt()
lp=torch.cat([lpips_fn(GEN[i:i+6].to(dev),xqva[i:i+6].to(dev)).cpu() for i in range(0,xpva.shape[0],6)])
def spear(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std(); return float((ra*rb).mean())
print("  d_real=%.4f  d_gen=%.4f  (need d_gen>d_real for headroom)  ratio=%.2f"%(d_real.mean(),d_gen.mean(),d_gen.mean()/d_real.mean().clamp_min(1e-6)),flush=True)
print("  Spearman(d_gen, LPIPS)=%.3f  (need >0 for it to matter to generation)"%spear(d_gen,lp),flush=True)
print("LOSA_GATE_DONE")
