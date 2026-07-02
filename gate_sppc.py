"""SPPC Gate (NO training). Screened-Poisson coupling of two APTD checkpoints on the perception-
distortion tradeoff. y*(w)=(y_D + rho|w|^2 y_P)/(1+rho|w|^2): low-freq/structure from y_D (high-SSIM),
high-freq/edges from y_P (good-LPIPS). Sweep rho; compare vs single ckpts, linear blend, Laplacian
pyramid. PASS: SPPC strictly Pareto-dominates the step2000 point (SSIM>0.2554 & LPIPS<0.4429) AND
beats linear/pyramid; correction not photometry."""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from metrics_img import ssim, lpips_fn
from aptd_model import APTDNet
dev="cuda"; H,W=480,240; HOME=os.path.expanduser("~/ScoliCMF")
def cfgfull():
    c=load_config(os.path.join(HOME,"configs/s2_base.yaml")); c["model"]["xpre_mode"]="full"; return c
cfg=cfgfull()
ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits/val.txt"))
XP=[];XQ=[]
for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
xpva=torch.cat(XP); xqva=torch.cat(XQ); Nv=xpva.shape[0]
@torch.no_grad()
def predict(stp):
    bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
    st=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_%d.pt"%stp),map_location=dev)
    for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev))
    m.eval(); out=[]
    for i in range(0,Nv,6):
        xp=xpva[i:i+6].to(dev); B=xp.shape[0]
        o=m(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)["xhat"].clamp(0,1)
        out.append(o.cpu())
    return torch.cat(out)
Y={s:predict(s) for s in [1000,2000,3000,4000,5000]}
def metrics(Yp):
    S=[];L=[]
    for i in range(0,Nv,6):
        o=Yp[i:i+6].to(dev); q=xqva[i:i+6].to(dev); S.append(ssim(o,q).cpu()); L.append(lpips_fn(o,q).cpu())
    return float(torch.cat(S).mean()), float(torch.cat(L).mean())
print("==== single checkpoints (raw val, 1-NFE) ====",flush=True)
for s in [1000,2000,3000,4000,5000]:
    a,b=metrics(Y[s]); print("  step%-5d SSIM=%.4f LPIPS=%.4f"%(s,a,b),flush=True)
# laplacian eigenvalues
u=torch.arange(H,device=dev).float(); v=torch.arange(W,device=dev).float()
lam=(2-2*torch.cos(2*3.141592653589793*u/H)).view(H,1)+(2-2*torch.cos(2*3.141592653589793*v/W)).view(1,W)  # [H,W]
def sppc(yD,yP,rho):
    YD=torch.fft.fft2(yD.to(dev)); YP=torch.fft.fft2(yP.to(dev))
    Yc=(YD+rho*lam*YP)/(1+rho*lam)
    return torch.fft.ifft2(Yc).real.clamp(0,1).cpu()
def lap_pyr(yD,yP,sig):  # low from yD, high from yP (hard cutoff via gaussian blur)
    k=int(2*sig)*2+1
    def blur(x):
        g=torch.arange(k,device=dev).float()-k//2; g=torch.exp(-(g**2)/(2*sig*sig)); g=g/g.sum()
        x=x.to(dev); x=F.conv2d(x,g.view(1,1,1,k),padding=(0,k//2)); x=F.conv2d(x,g.view(1,1,k,1),padding=(k//2,0)); return x
    return (blur(yD)+ (yP.to(dev)-blur(yP))).clamp(0,1).cpu()
def photo(ystar,yD):
    d=ystar-yD; return d.mean().abs().item()/d.abs().mean().item()

print("==== SPPC sweep (yD=high-SSIM, yP=good-LPIPS) ; ref step2000 SSIM>0.2554 & LPIPS<0.4429 ====",flush=True)
for (D,P) in [(5000,1000),(5000,2000),(4000,1000),(4000,2000)]:
    print(" -- yD=step%d  yP=step%d --"%(D,P),flush=True)
    for rho in [0.01,0.03,0.1,0.3,1,3,10]:
        yc=sppc(Y[D],Y[P],rho); a,b=metrics(yc)
        dom="  <== dominates step2000" if (a>0.2554 and b<0.4429) else ""
        print("   rho=%-5s SSIM=%.4f LPIPS=%.4f  photo=%.2f%s"%(rho,a,b,photo(yc,Y[D]),dom),flush=True)
print("==== baselines: linear blend & Laplacian pyramid (yD=5000,yP=1000) ====",flush=True)
for al in [0.25,0.5,0.75]:
    yl=(al*Y[5000]+(1-al)*Y[1000]); a,b=metrics(yl); print("  linear a=%.2f SSIM=%.4f LPIPS=%.4f"%(al,a,b),flush=True)
for sg in [2,4,8]:
    yp=lap_pyr(Y[5000],Y[1000],sg); a,b=metrics(yp); print("  pyramid sig=%d SSIM=%.4f LPIPS=%.4f"%(sg,a,b),flush=True)
print("SPPC_GATE_DONE")
