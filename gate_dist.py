"""Preflight for Plan-Marginalized MeanFlow. NO training.
Gate A (prize existence): among patients with SIMILAR pre-op (kNN in coarse feature space),
  how divergent are their POST-OP images? Compare that irreducible conditional spread to the
  deterministic APTD model residual. If conditional spread ~ model residual -> error is mostly
  irreducible plan variance -> distribution modeling justified.
Gate B (capturability): draw K stochastic samples from the existing APTD (path noise sigma),
  raw-frame best-of-K vs single-sample. Headroom = how much an oracle ensemble could cover."""
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
cfg=cfgfull(); mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"], sigma_m=cfg["meanflow"]["sigma_m"]); path=mf.path
def _v4(x): return x.view(-1,1,1,1)
def psnr(a,b): return -10*torch.log10(((a-b)**2).mean(dim=(1,2,3)).clamp_min(1e-10))
def load_split(s):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits",s))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP),torch.cat(XQ)
xptr,xqtr=load_split("train.txt"); xpva,xqva=load_split("val.txt")
XPa=torch.cat([xptr,xpva]); XQa=torch.cat([xqtr,xqva]); N=XPa.shape[0]
print("loaded N=%d (train+val)"%N, flush=True)

# ---------- Gate A ----------
# coarse pre-op feature = avgpool to 20x10 (anatomy-dominant on canonicalized imgs)
feat=F.adaptive_avg_pool2d(XPa,(20,10)).flatten(1)            # [N,200]
fd=torch.cdist(feat,feat)                                     # [N,N]
fd.fill_diagonal_(1e9)
K=8
nn=fd.topk(K,largest=False).indices                          # [N,K] pre-op neighbors
def dssim_pair(a,b):  # a,b [M,1,H,W] -> 1-SSIM per pair
    return (1-ssim(a.to(dev),b.to(dev))).cpu()
def dlpips_pair(a,b):
    return lpips_fn(a.to(dev),b.to(dev)).cpu()
Scs=[];Scl=[];Spre=[]
for i in range(N):
    js=nn[i]
    ai=XQa[i:i+1].repeat(K,1,1,1); bj=XQa[js]
    Scs.append(dssim_pair(ai,bj).mean()); Scl.append(dlpips_pair(ai,bj).mean())
    Spre.append(dssim_pair(XPa[i:i+1].repeat(K,1,1,1),XPa[js]).mean())
Scond_ssim=torch.stack(Scs).mean().item(); Scond_lpips=torch.stack(Scl).mean().item(); Spre_ssim=torch.stack(Spre).mean().item()
# global: random pairing
g=torch.Generator().manual_seed(0); rp=torch.randint(0,N,(N,K),generator=g)
Sgs=[];Sgl=[]
for i in range(N):
    js=rp[i]; ai=XQa[i:i+1].repeat(K,1,1,1); bj=XQa[js]
    Sgs.append(dssim_pair(ai,bj).mean()); Sgl.append(dlpips_pair(ai,bj).mean())
Sglob_ssim=torch.stack(Sgs).mean().item(); Sglob_lpips=torch.stack(Sgl).mean().item()

# deterministic APTD residual on val (raw frame)
bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
st=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_5000.pt"),map_location=dev)
for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev))
m.eval()
@torch.no_grad()
def apt_pred(xp, eps_sigma=0.0, nfe=4, seed=0):
    B=xp.shape[0]; z=xp.clone()
    gz=torch.Generator(device=dev).manual_seed(seed)
    tv=torch.linspace(1,0,nfe+1,device=dev); xhat=None
    for k in range(nfe):
        t=torch.full((B,),tv[k].item(),device=dev); r=torch.full((B,),tv[k+1].item(),device=dev)
        xhat=m(z,r,t,xp)["xhat"]; z=xp+_v4(path.alpha(r))*(xhat-xp)
        if eps_sigma>0:
            eps=torch.randn(z.shape,generator=gz,device=dev)
            z=z+eps_sigma*(torch.sin(3.14159*r)**2).view(-1,1,1,1)*eps
    return xhat.clamp(0,1)
Rs=[];Rl=[]
for i in range(0,xpva.shape[0],6):
    xp=xpva[i:i+6].to(dev); xq=xqva[i:i+6].to(dev); o=apt_pred(xp)
    Rs.append((1-ssim(o,xq)).cpu()); Rl.append(lpips_fn(o,xq).cpu())
Rmodel_ssim=torch.cat(Rs).mean().item(); Rmodel_lpips=torch.cat(Rl).mean().item()

print("==== GATE A: prize existence ====", flush=True)
print("  pre-op kNN sanity  (1-SSIM between neighbor PRE-OPs)   = %.4f"%Spre_ssim)
print("  conditional spread (1-SSIM between neighbor POST-OPs)  = %.4f   | LPIPS %.4f"%(Scond_ssim,Scond_lpips))
print("  global spread      (1-SSIM between random   POST-OPs)  = %.4f   | LPIPS %.4f"%(Sglob_ssim,Sglob_lpips))
print("  pre-op constrains post-op? cond/global = %.2f (SSIM) / %.2f (LPIPS)  (<1 => yes)"%(Scond_ssim/Sglob_ssim, Scond_lpips/Sglob_lpips))
print("  deterministic APTD residual (raw val)  = %.4f   | LPIPS %.4f"%(Rmodel_ssim,Rmodel_lpips))
print("  >> irreducible cond spread vs model residual: SSIM %.2fx  LPIPS %.2fx  (~1 => error mostly irreducible => model distribution)"%(Scond_ssim/Rmodel_ssim, Scond_lpips/Rmodel_lpips))

# ---------- Gate B: best-of-K ----------
print("==== GATE B: best-of-K headroom (existing APTD + path noise) ====", flush=True)
Kk=10
for sig in [0.1,0.3,0.5]:
    perS=[];perL=[]  # per-case best
    sinS=[];sinL=[]  # per-sample mean
    for i in range(0,xpva.shape[0],6):
        xp=xpva[i:i+6].to(dev); xq=xqva[i:i+6].to(dev); B=xp.shape[0]
        ss=torch.zeros(B,Kk); ll=torch.zeros(B,Kk)
        for k in range(Kk):
            o=apt_pred(xp,eps_sigma=sig,seed=k)
            ss[:,k]=ssim(o,xq).cpu(); ll[:,k]=lpips_fn(o,xq).cpu()
        perS.append(ss.max(1).values); perL.append(ll.min(1).values)
        sinS.append(ss.mean(1)); sinL.append(ll.mean(1))
    bS=torch.cat(perS).mean().item(); bL=torch.cat(perL).mean().item()
    mS=torch.cat(sinS).mean().item(); mL=torch.cat(sinL).mean().item()
    print("  sigma=%.1f  single SSIM=%.4f -> best-of-%d SSIM=%.4f (+%.4f) | single LPIPS=%.4f -> best %.4f (%.4f)"
          %(sig,mS,Kk,bS,bS-mS,mL,bL,bL-mL), flush=True)
print("GATE_DIST_DONE")
