import os,sys,torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from eval_gates import load_ckpt

dev="cuda"; H,W=480,240
cfg=load_config(os.path.expanduser("~/ScoliCMF/configs/shmm_v2.yaml"))
m=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/shmm_v2/ckpts/step_5000.pt"),cfg,H,W,None,dev,use_ema=True)
core=getattr(m,"module",m); cond=core.cond; cond.eval()
J=cond.J; D=cfg["model"]["dim"]; mu=cond.mu
print("J=%d D=%d"%(J,D))

def extract(split):
    ds=PairedSpineDataset(root=os.path.join(os.path.expanduser("~/ScoliCMF"),cfg["data"]["root"]),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/%s"%split))
    ld=DataLoader(ds,batch_size=8,num_workers=2)
    Bp=[];Bq=[]
    with torch.no_grad():
        for xp,xq in ld:
            xp,xq=xp.to(dev),xq.to(dev)
            Fmp=cond.stem(xp); _,Dd,Hf,Wf=Fmp.shape
            Ffp=Fmp.flatten(2).transpose(1,2); Ffq=cond.stem(xq).flatten(2).transpose(1,2)
            ygr=torch.linspace(0,1,Hf,device=dev).view(Hf,1).expand(Hf,Wf).reshape(-1)
            xgr=torch.linspace(0,1,Wf,device=dev).view(1,Wf).expand(Hf,Wf).reshape(-1)
            xc=cond._xc_cubic(xp,ygr)
            qn=F.normalize(cond.q,dim=-1); fn=F.normalize(cond.Wf(Ffp),dim=-1)
            content=torch.einsum("jd,bnd->bjn",qn,fn)
            spatial=(-cond.beta*(ygr[None,None,:]-mu[None,:,None])**2 - cond.eta*(xgr[None,None,:]-xc[:,None,:])**2)
            pi=torch.softmax(content+spatial,dim=-1)
            Bp.append(torch.einsum("bjn,bnd->bjd",pi,Ffp).cpu()); Bq.append(torch.einsum("bjn,bnd->bjd",pi,Ffq).cpu())
    return torch.cat(Bp),torch.cat(Bq)

trp,trq=extract("train.txt"); vap,vaq=extract("val.txt")
dBtr=(trq-trp); dBva=(vaq-vap)
print("N train=%d val=%d"%(dBtr.shape[0],dBva.shape[0]))
mu_in=trp.mean((0,1),keepdim=True); sd_in=trp.std((0,1),keepdim=True).clamp_min(1e-4)
Xtr=((trp-mu_in)/sd_in).to(dev); Ytr=dBtr.to(dev)
Xva=((vap-mu_in)/sd_in).to(dev); Yva=dBva.to(dev)
dBbar=dBtr.mean(0,keepdim=True).to(dev)   # population-mean change (from TRAIN)

E_tot=float((Yva**2).sum()); E_mc=float(((Yva-dBbar)**2).sum())
print("frac patient-specific (val) ||dB-dBbar||^2 / ||dB||^2 = %.4f"%(E_mc/max(E_tot,1e-9)))
def EV(pred,Y): return float(1-((Y-pred)**2).sum()/((Y-dBbar)**2).sum().clamp_min(1e-9))

# ---- (a) regularized LINEAR probe (flatten) ----
class Lin(nn.Module):
    def __init__(s): super().__init__(); s.w=nn.Linear(J*D,J*D)
    def forward(s,x): b=x.shape[0]; return s.w(x.reshape(b,-1)).reshape(b,J,D)
# ---- (b) small MLP ----
class MLP(nn.Module):
    def __init__(s,h=512): super().__init__(); s.net=nn.Sequential(nn.Linear(J*D,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,J*D))
    def forward(s,x): b=x.shape[0]; return s.net(x.reshape(b,-1)).reshape(b,J,D)

for name,model,wd,ep in [("linear",Lin(),1e-1,800),("mlp",MLP(),1e-2,800)]:
    model=model.to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=wd)
    best=-9; bt=0
    for e in range(ep):
        model.train(); opt.zero_grad()
        loss=((model(Xtr)-Ytr)**2).mean(); loss.backward(); opt.step()
        if e%50==0 or e==ep-1:
            model.eval()
            with torch.no_grad(): et=EV(model(Xtr),Ytr); ev=EV(model(Xva),Yva)
            if ev>best: best=ev; bt=et
            print("[%-6s] ep%4d EV_train=%.4f EV_val=%.4f"%(name,e,et,ev))
    print("[%-6s] BEST EV_val=%.4f (EV_train@best=%.4f)"%(name,best,bt))
print("STEP1_DONE")
