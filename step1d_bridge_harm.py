"""Does the single-head Bridge already reproduce the predictable harmonic transport?
  y_true = Pool_pi(x_post - x_pre);  y_base = Pool_pi(x_hat_base - x_pre)   (token transport)
  c*     = U_low^T y_true;           c_base = U_low^T y_base                 (Kg harmonic coeffs)
Report: (1) harmonic error ||c*-c_base||^2/||c*||^2  (how much Bridge MISSES),
        (2) transport magnitude ratio ||y_base||/||y_true|| (does Bridge move enough),
        (3) EV of the Bridge harmonic MISS (c*-c_base) from B_pre (is the miss predictable?)."""
import os,sys,torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt
from sc_pga import path_laplacian, _topk_eigvecs

dev="cuda"; H,W=480,240; Kg=4
tcfg=load_config(os.path.expanduser("~/ScoliCMF/configs/shmm_v2.yaml"))
bcfg=load_config(os.path.expanduser("~/ScoliCMF/configs/s2_base.yaml"))
tok=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/shmm_v2/ckpts/step_5000.pt"),tcfg,H,W,None,dev,True)
cond=getattr(tok,"module",tok).cond; cond.eval()
bridge=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/s2_base/ckpts/step_5000.pt"),bcfg,H,W,None,dev,True); bridge.eval()
mf=SourceAnchoredMeanFlow(gamma=bcfg["meanflow"]["gamma"])
J=cond.J; D=tcfg["model"]["dim"]; mu=cond.mu
Ulow=_topk_eigvecs(path_laplacian(J).to(dev), Kg, low=True)

@torch.no_grad()
def extract(split):
    ds=PairedSpineDataset(root=os.path.join(os.path.expanduser("~/ScoliCMF"),tcfg["data"]["root"]),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/%s"%split))
    ld=DataLoader(ds,batch_size=8,num_workers=2); BP=[];YT=[];YB=[]
    for xp,xq in ld:
        xp,xq=xp.to(dev),xq.to(dev)
        Fm=cond.stem(xp); _,Dd,Hf,Wf=Fm.shape; Ff=Fm.flatten(2).transpose(1,2)
        ygr=torch.linspace(0,1,Hf,device=dev).view(Hf,1).expand(Hf,Wf).reshape(-1)
        xgr=torch.linspace(0,1,Wf,device=dev).view(1,Wf).expand(Hf,Wf).reshape(-1)
        xc=cond._xc_cubic(xp,ygr); qn=F.normalize(cond.q,dim=-1); fn=F.normalize(cond.Wf(Ff),dim=-1)
        content=torch.einsum("jd,bnd->bjn",qn,fn)
        spatial=(-cond.beta*(ygr[None,None,:]-mu[None,:,None])**2 - cond.eta*(xgr[None,None,:]-xc[:,None,:])**2)
        pi=torch.softmax(content+spatial,dim=-1)
        B_pre=torch.einsum("bjn,bnd->bjd",pi,Ff)
        xhat=mf.sample(bridge,xp,steps=4)
        dt=F.interpolate(xq-xp,size=(Hf,Wf),mode="bilinear",align_corners=False).flatten(1)
        db=F.interpolate(xhat-xp,size=(Hf,Wf),mode="bilinear",align_corners=False).flatten(1)
        BP.append(B_pre.cpu()); YT.append(torch.einsum("bjn,bn->bj",pi,dt).cpu()); YB.append(torch.einsum("bjn,bn->bj",pi,db).cpu())
    return torch.cat(BP),torch.cat(YT),torch.cat(YB)

trp,trYt,trYb=extract("train.txt"); vap,vaYt,vaYb=extract("val.txt")
ct=(trYt.to(dev)@Ulow); cb=(trYb.to(dev)@Ulow); ctv=(vaYt.to(dev)@Ulow); cbv=(vaYb.to(dev)@Ulow)
print("N train=%d val=%d  Kg=%d"%(trp.shape[0],vap.shape[0],Kg))
print("harmonic transport magnitude  ||c_base||/||c*|| (val) = %.4f  (Bridge moves this fraction of harmonic transport)"%float(cbv.norm()/ctv.norm().clamp_min(1e-9)))
print("Bridge harmonic ERROR  ||c*-c_base||^2/||c*||^2 (val) = %.4f  (fraction of harmonic transport Bridge MISSES)"%float(((ctv-cbv)**2).sum()/(ctv**2).sum().clamp_min(1e-9)))
# is the Bridge harmonic MISS predictable from pre-op?
miss_tr=(ct-cb).cpu(); miss_va=(ctv-cbv).cpu()
mu_in=trp.mean((0,1),keepdim=True); sd_in=trp.std((0,1),keepdim=True).clamp_min(1e-4)
Xtr=((trp-mu_in)/sd_in).to(dev); Xva=((vap-mu_in)/sd_in).to(dev)
class MLP(nn.Module):
    def __init__(s,o,h=512): super().__init__(); s.net=nn.Sequential(nn.Linear(J*D,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,o))
    def forward(s,x): b=x.shape[0]; return s.net(x.reshape(b,-1))
def run(Ytr_,Yva_,tag):
    Ytr=Ytr_.to(dev); Yva=Yva_.to(dev); Ybar=Ytr.mean(0,keepdim=True); o=Ytr.shape[1]
    fps=float(((Yva-Ybar)**2).sum()/(Yva**2).sum().clamp_min(1e-9))
    net=MLP(o).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2); best=-9;bt=0
    def EV(p,Y): return float(1-((Y-p)**2).sum()/((Y-Ybar)**2).sum().clamp_min(1e-9))
    for e in range(1000):
        net.train(); opt.zero_grad(); loss=((net(Xtr)-Ytr)**2).mean(); loss.backward(); opt.step()
        if e%100==0 or e==999:
            net.eval()
            with torch.no_grad(): et=EV(net(Xtr),Ytr); ev=EV(net(Xva),Yva)
            if ev>best: best=ev;bt=et
    print("[%-14s] frac_patient_specific(val)=%.4f  BEST EV_val=%.4f (train@best=%.4f)"%(tag,fps,best,bt))
run(miss_tr,miss_va,"BridgeHarmMiss")   # EV of (c* - c_base) from pre-op
print("STEP1D_DONE")
