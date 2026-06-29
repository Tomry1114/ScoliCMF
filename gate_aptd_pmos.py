"""Two cheap pre-gates (no module training):
 PMOS gate  = clusterability of the Bridge residual dB_res=B_post-B_base (oracle best-of-K
              headroom: does K-means on train residuals explain val residual variance?).
 APTD gate  = oracle per-pair dense-warp recoverability of x_post from x_pre (how much of the
              change a deformation explains; LPIPS of warp vs Bridge vs no-op; hi/lo-freq residual)."""
import os,sys,torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt
from metrics_img import lpips_fn

dev="cuda"; H,W=480,240
tcfg=load_config(os.path.expanduser("~/ScoliCMF/configs/shmm_v2.yaml"))
bcfg=load_config(os.path.expanduser("~/ScoliCMF/configs/s2_base.yaml"))
tok=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/shmm_v2/ckpts/step_5000.pt"),tcfg,H,W,None,dev,True)
cond=getattr(tok,"module",tok).cond; cond.eval()
bridge=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/s2_base/ckpts/step_5000.pt"),bcfg,H,W,None,dev,True); bridge.eval()
mf=SourceAnchoredMeanFlow(gamma=bcfg["meanflow"]["gamma"])
J=cond.J; D=tcfg["model"]["dim"]; mu=cond.mu
torch.manual_seed(0)

@torch.no_grad()
def load_split(split):
    ds=PairedSpineDataset(root=os.path.join(os.path.expanduser("~/ScoliCMF"),tcfg["data"]["root"]),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/%s"%split))
    xp=[];xq=[]
    for a,b in DataLoader(ds,batch_size=16,num_workers=2): xp.append(a);xq.append(b)
    return torch.cat(xp),torch.cat(xq)
@torch.no_grad()
def toks(xp):
    Fm=cond.stem(xp); _,Dd,Hf,Wf=Fm.shape; Ff=Fm.flatten(2).transpose(1,2)
    ygr=torch.linspace(0,1,Hf,device=dev).view(Hf,1).expand(Hf,Wf).reshape(-1)
    xgr=torch.linspace(0,1,Wf,device=dev).view(1,Wf).expand(Hf,Wf).reshape(-1)
    xc=cond._xc_cubic(xp,ygr); qn=F.normalize(cond.q,dim=-1); fn=F.normalize(cond.Wf(Ff),dim=-1)
    content=torch.einsum("jd,bnd->bjn",qn,fn)
    spatial=(-cond.beta*(ygr[None,None,:]-mu[None,:,None])**2 - cond.eta*(xgr[None,None,:]-xc[:,None,:])**2)
    pi=torch.softmax(content+spatial,dim=-1); return torch.einsum("bjn,bnd->bjd",pi,Ff),pi

# ---------- PMOS gate: residual clusterability ----------
@torch.no_grad()
def residuals(xp,xq):
    R=[]
    for i in range(0,xp.shape[0],16):
        a=xp[i:i+16].to(dev); b=xq[i:i+16].to(dev)
        Bp,pi=toks(a); xhat=mf.sample(bridge,a,steps=4)
        Bq=torch.einsum("bjn,bnd->bjd",pi,cond.stem(b).flatten(2).transpose(1,2))
        Bb=torch.einsum("bjn,bnd->bjd",pi,cond.stem(xhat).flatten(2).transpose(1,2))
        R.append((Bq-Bb).reshape(a.shape[0],-1).cpu())
    return torch.cat(R)
def kmeans(X,K,iters=80):
    C=X[torch.randperm(X.shape[0])[:K]].clone()
    for _ in range(iters):
        a=torch.cdist(X,C).argmin(1)
        for k in range(K):
            m=a==k
            if m.any(): C[k]=X[m].mean(0)
    return C
trxp,trxq=load_split("train.txt"); vaxp,vaxq=load_split("val.txt")
Rtr=residuals(trxp,trxq).to(dev); Rva=residuals(vaxp,vaxq).to(dev)
rbar=Rtr.mean(0,keepdim=True); denom=((Rva-rbar)**2).sum().clamp_min(1e-9)
print("===== PMOS gate: Bridge-residual clusterability (val) =====")
for K in [4,8]:
    C=kmeans(Rtr,K)
    a=torch.cdist(Rva,C).argmin(1)                  # ORACLE best-of-K assignment (pick nearest prototype)
    ev=float(1-((Rva-C[a])**2).sum()/denom)
    ar=torch.randint(0,K,(Rva.shape[0],),device=dev) # random assignment null
    evr=float(1-((Rva-C[ar])**2).sum()/denom)
    print("K=%d  oracle best-of-K EV(residual var explained)=%.4f   random-assign EV=%.4f"%(K,ev,evr))

# ---------- APTD gate: oracle per-pair warp recoverability (val) ----------
print("===== APTD gate: oracle dense-warp recoverability (val) =====")
xp=vaxp.to(dev); xq=vaxq.to(dev); B=xp.shape[0]
hf,wf=H//8,W//8
phi=torch.zeros(B,2,hf,wf,device=dev,requires_grad=True)
theta=torch.tensor([[1.,0,0],[0,1.,0]],device=dev).unsqueeze(0).expand(B,2,3)
base=F.affine_grid(theta,(B,1,H,W),align_corners=False)
opt=torch.optim.Adam([phi],lr=0.05)
for it in range(250):
    flow=F.interpolate(phi,size=(H,W),mode="bilinear",align_corners=False).permute(0,2,3,1)
    warp=F.grid_sample(xp,base+flow,align_corners=False,padding_mode="border")
    tv=(phi[:,:,1:]-phi[:,:,:-1]).abs().mean()+(phi[:,:,:,1:]-phi[:,:,:,:-1]).abs().mean()
    loss=((warp-xq)**2).mean()+0.02*tv
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    flow=F.interpolate(phi,size=(H,W),mode="bilinear",align_corners=False).permute(0,2,3,1)
    warp=F.grid_sample(xp,base+flow,align_corners=False,padding_mode="border").clamp(0,1)
    R=xq-warp
    chg=((xq-xp)**2).sum(); res_frac=float((R**2).sum()/chg.clamp_min(1e-9))
    # hi/lo freq split of residual
    k=11; pad=k//2; w1d=torch.ones(1,1,k,k,device=dev)/(k*k)
    Rlo=F.conv2d(F.pad(R,[pad]*4,mode="reflect"),w1d); Rhi=R-Rlo
    lo=float((Rlo**2).sum()/(R**2).sum().clamp_min(1e-9)); hi=float((Rhi**2).sum()/(R**2).sum().clamp_min(1e-9))
    xhat=mf.sample(bridge,xp,steps=4)
    lp_warp=float(lpips_fn(warp,xq).mean()); lp_pre=float(lpips_fn(xp.clamp(0,1),xq).mean())
    lp_bridge=float(lpips_fn(xhat,xq).mean())
print("warp residual fraction ||x_post-warp||^2/||x_post-x_pre||^2 = %.4f  (low=%.3f hi=%.3f of residual)"%(res_frac,lo,hi))
print("LPIPS  no-op(x_pre)=%.4f   Bridge(4NFE)=%.4f   oracle-warp=%.4f"%(lp_pre,lp_bridge,lp_warp))
print("GATES_DONE")
