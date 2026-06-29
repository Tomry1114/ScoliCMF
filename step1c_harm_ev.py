"""EV_harm gate: can pre-op predict the LOW-FREQ SPINAL HARMONIC coefficients of the
TOTAL transport (NOT the unpredictable Bridge residual)?
  y*  = Pool_pi(x_post - x_pre)         (image delta pooled to J spinal tokens)
  c*  = U_low^T y*  in R^Kg             (fixed path-Laplacian low-freq harmonics)
Predict c* from B_pre; report EV_harm + harmonic energy share + full-y* EV (context)."""
import os,sys,torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from eval_gates import load_ckpt
from sc_pga import path_laplacian, _topk_eigvecs

dev="cuda"; H,W=480,240; Kg=4
tcfg=load_config(os.path.expanduser("~/ScoliCMF/configs/shmm_v2.yaml"))
tok=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/shmm_v2/ckpts/step_5000.pt"),tcfg,H,W,None,dev,True)
cond=getattr(tok,"module",tok).cond; cond.eval()
J=cond.J; D=tcfg["model"]["dim"]; mu=cond.mu
Ulow=_topk_eigvecs(path_laplacian(J).to(dev), Kg, low=True)        # (J,Kg) orthonormal

@torch.no_grad()
def extract(split):
    ds=PairedSpineDataset(root=os.path.join(os.path.expanduser("~/ScoliCMF"),tcfg["data"]["root"]),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/%s"%split))
    ld=DataLoader(ds,batch_size=8,num_workers=2); BP=[];YS=[]
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
        dg=F.interpolate(xq-xp,size=(Hf,Wf),mode="bilinear",align_corners=False).flatten(1)   # (B,N)
        ystar=torch.einsum("bjn,bn->bj",pi,dg)                       # (B,J) pooled token transport
        BP.append(B_pre.cpu()); YS.append(ystar.cpu())
    return torch.cat(BP),torch.cat(YS)

trp,trY=extract("train.txt"); vap,vaY=extract("val.txt")
print("N train=%d val=%d  Kg=%d"%(trp.shape[0],vap.shape[0],Kg))
cstar_tr=(trY.to(dev)@Ulow).cpu(); cstar_va=(vaY.to(dev)@Ulow).cpu()
# harmonic energy share of pooled token transport
hs=float(((vaY.to(dev)@Ulow)**2).sum()/(vaY.to(dev)**2).sum().clamp_min(1e-9))
print("harmonic energy share (val) ||U_low^T y*||^2/||y*||^2 = %.4f"%hs)
mu_in=trp.mean((0,1),keepdim=True); sd_in=trp.std((0,1),keepdim=True).clamp_min(1e-4)
Xtr=((trp-mu_in)/sd_in).to(dev); Xva=((vap-mu_in)/sd_in).to(dev)
class MLP(nn.Module):
    def __init__(s,o,h=512): super().__init__(); s.net=nn.Sequential(nn.Linear(J*D,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,o)); s.o=o
    def forward(s,x): b=x.shape[0]; return s.net(x.reshape(b,-1))
def run(Ytr_,Yva_,tag):
    Ytr=Ytr_.reshape(Ytr_.shape[0],-1).to(dev); Yva=Yva_.reshape(Yva_.shape[0],-1).to(dev)
    Ybar=Ytr.mean(0,keepdim=True); o=Ytr.shape[1]
    fps=float(((Yva-Ybar)**2).sum()/(Yva**2).sum().clamp_min(1e-9))
    net=MLP(o).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2); best=-9;bt=0
    def EV(p,Y): return float(1-((Y-p)**2).sum()/((Y-Ybar)**2).sum().clamp_min(1e-9))
    for e in range(1000):
        net.train(); opt.zero_grad(); loss=((net(Xtr)-Ytr)**2).mean(); loss.backward(); opt.step()
        if e%100==0 or e==999:
            net.eval()
            with torch.no_grad(): et=EV(net(Xtr),Ytr); ev=EV(net(Xva),Yva)
            if ev>best: best=ev;bt=et
    # per-dim val EV at best is omitted; report aggregate
    print("[%-10s o=%d] frac_patient_specific(val)=%.4f  BEST EV_val=%.4f (EV_train@best=%.4f)"%(tag,o,fps,best,bt))
run(cstar_tr,cstar_va,"c_harm(Kg)")   # THE gate
run(trY,vaY,"y_full(J)")              # context: full pooled token transport
print("STEP1C_DONE")
