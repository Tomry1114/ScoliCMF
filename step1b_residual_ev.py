"""Bridge-RESIDUAL predictability gate (the real identifiability test).
Step1 measured EV over total change B_post-B_pre (0.358). The Bridge already
removed the most predictable part. Here: can pre-op predict the BRIDGE residual
dB_res = E(x_post) - E(x_hat_base), x_hat_base = frozen-Bridge 4-NFE sample?"""
import os,sys,torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt

dev="cuda"; H,W=480,240
tcfg=load_config(os.path.expanduser("~/ScoliCMF/configs/shmm_v2.yaml"))
bcfg=load_config(os.path.expanduser("~/ScoliCMF/configs/s2_base.yaml"))
tok=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/shmm_v2/ckpts/step_5000.pt"),tcfg,H,W,None,dev,True)
cond=getattr(tok,"module",tok).cond; cond.eval()
bridge=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/s2_base/ckpts/step_5000.pt"),bcfg,H,W,None,dev,True); bridge.eval()
mf=SourceAnchoredMeanFlow(gamma=bcfg["meanflow"]["gamma"])
J=cond.J; D=tcfg["model"]["dim"]; mu=cond.mu

@torch.no_grad()
def tok_pi(xp):
    Fm=cond.stem(xp); _,Dd,Hf,Wf=Fm.shape; Ff=Fm.flatten(2).transpose(1,2)
    ygr=torch.linspace(0,1,Hf,device=dev).view(Hf,1).expand(Hf,Wf).reshape(-1)
    xgr=torch.linspace(0,1,Wf,device=dev).view(1,Wf).expand(Hf,Wf).reshape(-1)
    xc=cond._xc_cubic(xp,ygr); qn=F.normalize(cond.q,dim=-1); fn=F.normalize(cond.Wf(Ff),dim=-1)
    content=torch.einsum("jd,bnd->bjn",qn,fn)
    spatial=(-cond.beta*(ygr[None,None,:]-mu[None,:,None])**2 - cond.eta*(xgr[None,None,:]-xc[:,None,:])**2)
    return torch.softmax(content+spatial,dim=-1), Ff
@torch.no_grad()
def extract(split):
    ds=PairedSpineDataset(root=os.path.join(os.path.expanduser("~/ScoliCMF"),tcfg["data"]["root"]),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/%s"%split))
    ld=DataLoader(ds,batch_size=8,num_workers=2); P=[];Q=[];Bs=[]
    for xp,xq in ld:
        xp,xq=xp.to(dev),xq.to(dev); pi,Ffp=tok_pi(xp)
        xhat=mf.sample(bridge,xp,steps=4)
        Ffq=cond.stem(xq).flatten(2).transpose(1,2); Ffb=cond.stem(xhat).flatten(2).transpose(1,2)
        P.append(torch.einsum("bjn,bnd->bjd",pi,Ffp).cpu()); Q.append(torch.einsum("bjn,bnd->bjd",pi,Ffq).cpu()); Bs.append(torch.einsum("bjn,bnd->bjd",pi,Ffb).cpu())
    return torch.cat(P),torch.cat(Q),torch.cat(Bs)

trp,trq,trb=extract("train.txt"); vap,vaq,vab=extract("val.txt")
res_tr=(trq-trb); res_va=(vaq-vab); tot_tr=(trq-trp); tot_va=(vaq-vap)
print("N train=%d val=%d"%(trp.shape[0],vap.shape[0]))
print("||dB_res||^2 / ||dB_tot||^2 (val) = %.4f  (frac of total change STILL unexplained by Bridge)"%float((res_va**2).sum()/(tot_va**2).sum().clamp_min(1e-9)))
mu_in=trp.mean((0,1),keepdim=True); sd_in=trp.std((0,1),keepdim=True).clamp_min(1e-4)
Xtr=((trp-mu_in)/sd_in).to(dev); Xva=((vap-mu_in)/sd_in).to(dev)
class MLP(nn.Module):
    def __init__(s,h=512): super().__init__(); s.net=nn.Sequential(nn.Linear(J*D,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,J*D))
    def forward(s,x): b=x.shape[0]; return s.net(x.reshape(b,-1)).reshape(b,J,D)
def run(Ytr_,Yva_,tag):
    Ytr=Ytr_.to(dev); Yva=Yva_.to(dev); Ybar=Ytr_.mean(0,keepdim=True).to(dev)
    fpsv=float(((Yva-Ybar)**2).sum()/(Yva**2).sum().clamp_min(1e-9))
    net=MLP().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2); best=-9;bt=0
    def EV(p,Y): return float(1-((Y-p)**2).sum()/((Y-Ybar)**2).sum().clamp_min(1e-9))
    for e in range(800):
        net.train(); opt.zero_grad(); loss=((net(Xtr)-Ytr)**2).mean(); loss.backward(); opt.step()
        if e%100==0 or e==799:
            net.eval()
            with torch.no_grad(): et=EV(net(Xtr),Ytr); ev=EV(net(Xva),Yva)
            if ev>best: best=ev;bt=et
    print("[%-8s] frac_patient_specific(val)=%.4f  BEST EV_val=%.4f (EV_train@best=%.4f)"%(tag,fpsv,best,bt))
run(tot_tr,tot_va,"dB_total")   # sanity: should reproduce Step1 ~0.358
run(res_tr,res_va,"dB_RESID")   # the real identifiability gate
print("STEP1B_DONE")
