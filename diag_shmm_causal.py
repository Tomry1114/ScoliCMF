import os,sys,torch
import torch.nn.functional as F
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt
from metrics_img import ssim, lpips_fn
from sc_pga import build_static_projector, build_v2_projector

dev="cuda"; H,W=480,240
cfg=load_config(os.path.expanduser("~/ScoliCMF/configs/shmm_v2.yaml"))
mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])   # sigma_m=0 -> deterministic z_t
ds=PairedSpineDataset(root=os.path.join(os.path.expanduser("~/ScoliCMF"),cfg["data"]["root"]),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
ld=DataLoader(ds,batch_size=6,num_workers=2)
m=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/shmm_v2/ckpts/step_5000.pt"),cfg,H,W,None,dev,use_ema=True)
core=getattr(m,"module",m); cond=core.cond; cond.eval()
J,Kg=cond.J,cond.Kg

def endpoint():
    ss,lp,outs=[],[],[]
    with torch.no_grad():
        for xp,xq in ld:
            xp,xq=xp.to(dev),xq.to(dev)
            z=mf.sample(m,xp,steps=4)
            ss.append(ssim(z,xq).detach().cpu()); lp.append(lpips_fn(z,xq).detach().cpu()); outs.append(z.detach().cpu())
    return float(torch.cat(ss).mean()), float(torch.cat(lp).mean()), torch.cat(outs)

print("======== D1: same-checkpoint projector swap (NO retrain; tests projector ON v2-trained feature) ========")
b_ss,b_lp,b_out=endpoint()
print("v2(baseline)   SSIM4=%.4f LPIPS4=%.4f  dOut=0.0000"%(b_ss,b_lp))
o_proj=cond.proj; o_pis=getattr(cond,"Pi_static",None); o_perm=cond.perm
for name in ["dct","v1","random","identity","permuted"]:
    cond.perm=None
    if name=="permuted":
        cond.proj=o_proj
        if o_pis is not None: cond.Pi_static=o_pis
        g=torch.Generator().manual_seed(0); cond.perm=torch.randperm(J,generator=g)
    else:
        cond.proj=name; cond.Pi_static=build_static_projector(name,J,Kg).to(dev)
    ss,lp,out=endpoint()
    dout=float((out-b_out).norm()/b_out.norm().clamp_min(1e-6))
    print("swap->%-9s SSIM4=%.4f LPIPS4=%.4f  dOut=%.4f"%(name,ss,lp,dout))
cond.proj=o_proj; cond.perm=o_perm
if o_pis is not None: cond.Pi_static=o_pis

print("======== D2: dynamic-branch causal intervention (same checkpoint) ========")
class ZeroMod(torch.nn.Module):
    def forward(self,x): return torch.zeros_like(x)
Cdyn=[]; Cstat=[]
with torch.no_grad():
    for xp,xq in ld:
        xp,xq=xp.to(dev),xq.to(dev); Bn=xp.shape[0]
        t=torch.full((Bn,),0.5,device=dev); r=torch.full((Bn,),0.25,device=dev)
        zt=mf.path.z_t(xp,xq,torch.full((Bn,1,1,1),0.5,device=dev))
        uf=core(zt,r,t,xp)
        cond.dyn_off=True; ud=core(zt,r,t,xp); cond.dyn_off=False
        Ms=cond.M_static; cond.M_static=ZeroMod(); us=core(zt,r,t,xp); cond.M_static=Ms
        nf=uf.flatten(1).norm(dim=1).clamp_min(1e-9)
        Cdyn.append(((uf-ud).flatten(1).norm(dim=1)/nf).cpu())
        Cstat.append(((uf-us).flatten(1).norm(dim=1)/nf).cpu())
ss_f,_,_=endpoint()
cond.dyn_off=True; ss_d,_,_=endpoint(); cond.dyn_off=False
print("C_dyn  (||u_full-u_dynoff ||/||u_full||) = %.4f"%float(torch.cat(Cdyn).mean()))
print("C_stat (||u_full-u_statoff||/||u_full||) = %.4f  [note: time-emb e still leaks to c_patch]"%float(torch.cat(Cstat).mean()))
print("endpoint SSIM4  full=%.4f  dyn_off=%.4f  (delta=%.4f)"%(ss_f,ss_d,ss_f-ss_d))

print("======== D3: subspace representability of TRUE post-op change dB=B_post-B_pre (energy ratio) ========")
mu=cond.mu
num={"dct":0.,"v1":0.,"v2":0.}; den=0.; etop=0.; nb=0
with torch.no_grad():
    for xp,xq in ld:
        xp,xq=xp.to(dev),xq.to(dev); Bn=xp.shape[0]
        Fm=cond.stem(xp); _,D,Hf,Wf=Fm.shape
        Ffp=Fm.flatten(2).transpose(1,2); Ffq=cond.stem(xq).flatten(2).transpose(1,2)
        ygr=torch.linspace(0,1,Hf,device=dev).view(Hf,1).expand(Hf,Wf).reshape(-1)
        xgr=torch.linspace(0,1,Wf,device=dev).view(1,Wf).expand(Hf,Wf).reshape(-1)
        xc=cond._xc_cubic(xp,ygr)
        qn=F.normalize(cond.q,dim=-1); fn=F.normalize(cond.Wf(Ffp),dim=-1)
        content=torch.einsum("jd,bnd->bjn",qn,fn)
        spatial=(-cond.beta*(ygr[None,None,:]-mu[None,:,None])**2 - cond.eta*(xgr[None,None,:]-xc[:,None,:])**2)
        pi=torch.softmax(content+spatial,dim=-1)
        Bpre=torch.einsum("bjn,bnd->bjd",pi,Ffp); Bpost=torch.einsum("bjn,bnd->bjd",pi,Ffq)
        dB=Bpost-Bpre
        grid=torch.stack([ygr,xgr],-1)
        pos=torch.einsum("bjn,nc->bjc",pi,grid)
        var=(torch.einsum("bjn,nc->bjc",pi,grid**2)-pos**2).clamp_min(0)
        res=torch.stack([pos[...,0]-mu[None,:],pos[...,1]-0.5],-1)
        Pi={"dct":build_static_projector("dct",J,Kg).to(dev),
            "v1":build_static_projector("v1",J,Kg).to(dev),
            "v2":build_v2_projector(res,var.sqrt(),Kg,cond.tau,cond.w_min,cond.lam_sigma)}
        den+=float(dB.pow(2).sum())
        for k in Pi: num[k]+=float(cond._proj_apply(Pi[k],dB).pow(2).sum())
        ev=torch.linalg.svdvals(dB).pow(2); etop+=float((ev[:,:4].sum(1)/ev.sum(1).clamp_min(1e-9)).sum()); nb+=Bn
print("E_top4(dB) = %.4f  (frac of TRUE change energy already in top-4 SVD modes)"%(etop/nb))
for k in ["dct","v1","v2"]:
    print("E_%-3s(captured frac of true post-op change) = %.4f"%(k,num[k]/den))
print("VERDICT: E_v2 %s E_dct  => patient graph %s represent true change better than fixed basis"%(
    (">" if num["v2"]>num["dct"] else "<="), ("DOES" if num["v2"]>num["dct"] else "does NOT")))
print("DIAG_CAUSAL_DONE")
