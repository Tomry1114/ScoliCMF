"""RC-MF Gate 1 (image-space commutation proxy, NO training). Tests whether the finite-interval
flow map T commutes with coarse-graining C: E_comm = ||C.T(x_pre) - T.C(x_pre)||. Image-space C
(downsample->upsample) avoids the OOD confound of running the fine-trained DiT on coarse tokens.
Positive (RC-MF alive): E_comm rises as LPIPS worsens across checkpoints (rho>0.5) & per-case
correlates with generation error. Flat / late-more-consistent => dead."""
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
def C(x,k):  # coarse-grain: avgpool k then upsample
    return F.interpolate(F.avg_pool2d(x,k),size=(H,W),mode="bilinear",align_corners=False)
def spear(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-8); rb=(rb-rb.mean())/rb.std().clamp_min(1e-8); return float((ra*rb).mean())
def pearson(a,b):
    a=a-a.mean(); b=b-b.mean(); return float((a*b).mean()/(a.std().clamp_min(1e-8)*b.std().clamp_min(1e-8)))
rows=[]
for stp in [1000,2000,3000,4000,5000]:
    bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
    st=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_%d.pt"%stp),map_location=dev)
    for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev))
    m.eval()
    @torch.no_grad()
    def T(xp):
        B=xp.shape[0]; return m(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)["xhat"].clamp(0,1)
    Ec={4:[],8:[]}; LP=[]; ERR=[]
    with torch.no_grad():
        for i in range(0,Nv,6):
            xp=xpva[i:i+6].to(dev); q=xqva[i:i+6].to(dev)
            y=T(xp)
            for k in (4,8):
                p1=C(y,k); p2=T(C(xp,k))
                e=((p1-p2).pow(2).mean(dim=(1,2,3))/p1.pow(2).mean(dim=(1,2,3)).clamp_min(1e-9))
                Ec[k].append(e.cpu())
            LP.append(lpips_fn(y,q).cpu()); ERR.append((y-q).abs().mean(dim=(1,2,3)).cpu())
    Ec4=torch.cat(Ec[4]); Ec8=torch.cat(Ec[8]); LP=torch.cat(LP); ERR=torch.cat(ERR)
    S=[];
    with torch.no_grad():
        for i in range(0,Nv,6): S.append(ssim(T(xpva[i:i+6].to(dev)),xqva[i:i+6].to(dev)).cpu())
    S=torch.cat(S)
    rows.append((stp,float(S.mean()),float(LP.mean()),float(Ec4.mean()),float(Ec8.mean()),spear(Ec8,LP),spear(Ec8,ERR)))
    print("  step%-5d SSIM=%.4f LPIPS=%.4f  E_comm(k4)=%.4f E_comm(k8)=%.4f  percase rho(Ecomm,LPIPS)=%.2f rho(Ecomm,err)=%.2f"
          %rows[-1],flush=True)
import statistics
stps=[r[0] for r in rows]; lps=[r[2] for r in rows]; e8=[r[4] for r in rows]
print("==== ACROSS-CHECKPOINT ====",flush=True)
print("  Pearson(E_comm_k8, LPIPS) across 5 ckpts = %.3f  (need >0.5 for RC-MF alive)"%pearson(torch.tensor(e8),torch.tensor(lps)),flush=True)
print("  E_comm k8: step1000=%.4f -> step5000=%.4f  (%s)"%(e8[0],e8[-1],"RISES with blur" if e8[-1]>e8[0] else "FLAT/falls => dead"),flush=True)
print("RCMF_GATE_DONE")
