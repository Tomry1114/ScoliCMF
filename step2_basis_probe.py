import os,sys,torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from eval_gates import load_ckpt
from sc_pga import build_static_projector, build_v2_projector

dev="cuda"; H,W=480,240
cfg=load_config(os.path.expanduser("~/ScoliCMF/configs/shmm_v2.yaml"))
m=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/shmm_v2/ckpts/step_5000.pt"),cfg,H,W,None,dev,use_ema=True)
core=getattr(m,"module",m); cond=core.cond; cond.eval()
J=cond.J; D=cfg["model"]["dim"]; mu=cond.mu

def extract(split):
    ds=PairedSpineDataset(root=os.path.join(os.path.expanduser("~/ScoliCMF"),cfg["data"]["root"]),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/%s"%split))
    ld=DataLoader(ds,batch_size=8,num_workers=2)
    Bp=[];Bq=[];RS=[];VR=[]
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
            grid=torch.stack([ygr,xgr],-1); pos=torch.einsum("bjn,nc->bjc",pi,grid)
            var=(torch.einsum("bjn,nc->bjc",pi,grid**2)-pos**2).clamp_min(0)
            RS.append(torch.stack([pos[...,0]-mu[None,:],pos[...,1]-0.5],-1).cpu()); VR.append(var.sqrt().cpu())
    return torch.cat(Bp),torch.cat(Bq),torch.cat(RS),torch.cat(VR)

trp,trq,trr,trv=extract("train.txt"); vap,vaq,var_,vav=extract("val.txt")
dBtr=(trq-trp).to(dev); dBva=(vaq-vap).to(dev)
mu_in=trp.mean((0,1),keepdim=True); sd_in=trp.std((0,1),keepdim=True).clamp_min(1e-4)
Xtr=((trp-mu_in)/sd_in).to(dev); Xva=((vap-mu_in)/sd_in).to(dev)
print("J=%d D=%d  N train=%d val=%d"%(J,D,dBtr.shape[0],dBva.shape[0]))

def cov(Pi,Y):
    PY=torch.einsum("jk,bkd->bjd",Pi,Y) if Pi.dim()==2 else torch.einsum("bjk,bkd->bjd",Pi,Y)
    return float((PY**2).sum()/(Y**2).sum().clamp_min(1e-9))
def oracle(Y,K):  # per-sample SVD top-K coverage
    ev=torch.linalg.svdvals(Y).pow(2); return float((ev[:,:K].sum(1)/ev.sum(1).clamp_min(1e-9)).mean())

class BasisNet(nn.Module):
    def __init__(s,J,D,K,h=256):
        super().__init__(); s.tok=nn.Sequential(nn.Linear(D,h),nn.GELU()); s.mix=nn.Linear(J*h,J*h); s.out=nn.Linear(h,K); s.J,s.h,s.K=J,h,K
    def forward(s,B):
        b=B.shape[0]; t=s.tok(B); t=F.gelu(s.mix(t.reshape(b,-1)).reshape(b,s.J,s.h)); lg=s.out(t)
        Q,_=torch.linalg.qr(lg,mode="reduced"); return Q@Q.transpose(-1,-2)

print("%-4s | DCT    v1     v2     random | learnedQ(val) [train] | oracle(SVD topK)"%"K")
for K in [4,6,8]:
    Pdct=build_static_projector("dct",J,K).to(dev); Pv1=build_static_projector("v1",J,K).to(dev)
    g=torch.Generator().manual_seed(0); Prnd=(lambda Q:(Q@Q.T))(torch.linalg.qr(torch.randn(J,K,generator=g))[0]).to(dev)
    Pv2tr=build_v2_projector(trr.to(dev),trv.to(dev),K,cond.tau,cond.w_min,cond.lam_sigma)
    Pv2va=build_v2_projector(var_.to(dev),vav.to(dev),K,cond.tau,cond.w_min,cond.lam_sigma)
    c_dct=cov(Pdct,dBva); c_v1=cov(Pv1,dBva); c_v2=cov(Pv2va,dBva); c_rnd=cov(Prnd,dBva)
    # learned Q_phi
    net=BasisNet(J,D,K).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
    best=-9; bt=0
    for e in range(600):
        net.train(); opt.zero_grad()
        Pi=net(Xtr); PY=torch.einsum("bjk,bkd->bjd",Pi,dBtr)
        loss=(((dBtr-PY)**2).sum())/((dBtr**2).sum()); loss.backward(); opt.step()
        if e%50==0 or e==599:
            net.eval()
            with torch.no_grad(): cv=cov(net(Xva),dBva); ct=cov(net(Xtr),dBtr)
            if cv>best: best=cv; bt=ct
    print("K=%-2d | %.4f %.4f %.4f %.4f | %.4f        [%.4f] | %.4f"%(K,c_dct,c_v1,c_v2,c_rnd,best,bt,oracle(dBva,K)))
print("STEP2_DONE")
