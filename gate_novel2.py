"""Two free pre-registration gates for the SECOND-novelty decision (no training).
A) FOLDING: Jacobian det of APTD warp phi on val. If phi already fold-free -> diffeomorphic-geodesic
   novelty adds nothing (kill). If phi folds (esp. at curve apex) -> the guarantee is real & free.
B) CONFORMAL: turn the training-free defect score d=|1step-2step| into a GUARANTEED reliability
   certificate. Report (i) marginal split-CP coverage vs nominal (validity), (ii) Mondrian d-conditional
   error bounds (low-d tighter than high-d = efficiency), (iii) selective acceptance. Many random splits."""
import os, sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from metrics_img import ssim as ssim_fn
from aptd_model import APTDNet
dev="cuda" if __import__("torch").cuda.is_available() else "cpu"; H,W=480,240; HOME=os.path.expanduser("~/ScoliCMF")
def cfgfull():
    c=load_config(os.path.join(HOME,"configs/s2_base.yaml")); c["model"]["xpre_mode"]="full"; return c
cfg=cfgfull()
ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits/val.txt"))
XP=[];XQ=[]
for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
xpva=torch.cat(XP); xqva=torch.cat(XQ); Nv=xpva.shape[0]
path=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"], sigma_m=cfg["meanflow"]["sigma_m"]).path
def load_m(stp):
    bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
    st=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_%d.pt"%stp),map_location=dev)
    for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev))
    m.eval(); return m
def _v4(x): return x.view(-1,1,1,1)

# ---------- A) FOLDING ----------
print("==== GATE A: folding rate of APTD warp phi (det J<=0), source-only 1-step ====",flush=True)
for stp in [2000,5000]:
    m=load_m(stp); fr=[]; mind=[]; anyf=0
    with torch.no_grad():
        for i in range(0,Nv,6):
            xp=xpva[i:i+6].to(dev); B=xp.shape[0]
            out=m(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)
            theta=m.head.__dict__.get("theta",None)
            base=F.affine_grid(m.theta.expand(B,2,3),(B,1,H,W),align_corners=False) if hasattr(m,"theta") else None
            g=base+out["flow"].permute(0,2,3,1)          # (B,H,W,2) normalized (x,y)
            gx=g[...,0]; gy=g[...,1]
            dgx=torch.gradient(gx,dim=(1,2)); dgy=torch.gradient(gy,dim=(1,2))  # d/di(H), d/dj(W)
            # J=[[dgx/dj,dgx/di],[dgy/dj,dgy/di]]; det=dgx_dj*dgy_di - dgx_di*dgy_dj
            det=dgx[1]*dgy[0]-dgx[0]*dgy[1]
            fr.append((det<=0).float().mean(dim=(1,2)).cpu())
            mind.append(det.amin(dim=(1,2)).cpu()); anyf+=int(((det<=0).any(dim=(1,2))).sum())
    fr=torch.cat(fr); mind=torch.cat(mind)
    print("  step%-5d  mean fold-frac=%.4f%%  median=%.4f%%  max-case=%.3f%%  min detJ=%.3f  imgs-with-any-fold=%d/%d"%(
        stp, 100*fr.mean(),100*fr.median(),100*fr.max(),mind.min(),anyf,Nv),flush=True)

# ---------- B) CONFORMAL ----------
print("==== GATE B: conformalize defect d=|1step-2step| into guaranteed reliability ====",flush=True)
m=load_m(5000)
def sample(xp,nfe):
    B=xp.shape[0]; z=xp.clone(); tv=torch.linspace(1,0,nfe+1,device=dev); xhat=None
    for k in range(nfe):
        t=tv[k].expand(B); r=tv[k+1].expand(B); xhat=m(z,r,t,xp)["xhat"]; z=xp+_v4(path.alpha(r))*(xhat-xp)
    return xhat.clamp(0,1)
e=[]; d=[]
with torch.no_grad():
    for i in range(0,Nv,6):
        xp=xpva[i:i+6].to(dev); q=xqva[i:i+6].to(dev)
        s1=sample(xp,1); s2=sample(xp,2)
        e.append((1-ssim_fn(s1,q)).cpu())                 # true error of the 1-NFE prediction
        d.append((s1-s2).abs().flatten(1).mean(1).cpu())  # training-free, GT-free defect score
e=torch.cat(e).numpy(); d=torch.cat(d).numpy()
from scipy.stats import spearmanr
print("  n=%d  Spearman(d,e)=%.3f  mean err=%.4f"%(Nv,spearmanr(d,e).correlation,e.mean()),flush=True)
rng=np.random.RandomState(0)
def qhat(cal_e,alpha):
    n=len(cal_e); k=int(math.ceil((n+1)*(1-alpha)))-1; k=min(max(k,0),n-1); return np.sort(cal_e)[k]
for alpha in [0.1,0.2]:
    cov=[]; covL=[];covH=[];qL=[];qH=[]; acc=[];accE=[];fullE=[]
    for _ in range(1000):
        idx=rng.permutation(Nv); cal=idx[:Nv//2]; te=idx[Nv//2:]
        qh=qhat(e[cal],alpha); cov.append(np.mean(e[te]<=qh))          # marginal validity
        med=np.median(d[cal]); loC=cal[d[cal]<=med]; hiC=cal[d[cal]>med]
        if len(loC)>2 and len(hiC)>2:
            qlo=qhat(e[loC],alpha); qhi=qhat(e[hiC],alpha); qL.append(qlo);qH.append(qhi)
            loT=te[d[te]<=med]; hiT=te[d[te]>med]
            if len(loT): covL.append(np.mean(e[loT]<=qlo))
            if len(hiT): covH.append(np.mean(e[hiT]<=qhi))
        # selective: accept test cases with d<=cal median; report their mean error vs all
        tau=med; acT=te[d[te]<=tau]
        if len(acT): acc.append(len(acT)/len(te)); accE.append(e[acT].mean()); fullE.append(e[te].mean())
    print("  alpha=%.2f | MARGINAL coverage=%.3f (target %.2f)  | Mondrian: qLOW-d=%.4f qHIGH-d=%.4f (efficiency: low bound %.0f%% of high) covLOW=%.3f covHIGH=%.3f"%(
        alpha, np.mean(cov),1-alpha, np.mean(qL),np.mean(qH),100*np.mean(qL)/max(np.mean(qH),1e-9),np.mean(covL),np.mean(covH)),flush=True)
    print("          | SELECTIVE: accept low-d cases -> accept-frac=%.2f  mean-err(accepted)=%.4f vs mean-err(all)=%.4f"%(
        np.mean(acc),np.mean(accE),np.mean(fullE)),flush=True)
print("NOVEL2_GATE_DONE",flush=True)
