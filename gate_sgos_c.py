"""SGOS Gate C (DECISIVE): sparse recoverability of the optimal deformation.
Gate B showed best-case dense 2D warp reaches SSIM 0.4262 (>> x_pre 0.1996, >> APTD 0.30) = real
geometric headroom. SGOS's premise: K sparse 2D control anchors are enough to specify it. Test: take
the per-case OPTIMAL dense flow (oracle), subsample to Ky x Kx anchors, smoothly re-interpolate, warp,
measure SSIM/LPIPS vs K. If K~5 recovers most of 0.4262 -> deformation is low-bandwidth -> SGOS GREEN
LIGHT (first positive gate). If sparse collapses toward 0.20 -> deformation too high-dim for clicks.
Refs: x_pre 0.1996/0.4283 ; dense-oracle 0.4262/0.4119 ; APTD 0.2554-0.3018."""
import os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from metrics_img import ssim as ssim_fn, lpips_fn
HOME=os.path.expanduser("~/ScoliCMF"); H,W=480,240; dev="cuda" if torch.cuda.is_available() else "cpu"
GY,GX=24,12
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,load_config(os.path.join(HOME,"configs/s2_base.yaml"))["data"]["root"]),
                          size=(H,W),split_file=os.path.join(HOME,"splits/%s.txt"%split))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP),torch.cat(XQ)
xpva,xqva=load("val"); N=xpva.shape[0]
base=torch.stack(torch.meshgrid(torch.linspace(-1,1,H,device=dev),torch.linspace(-1,1,W,device=dev),indexing="ij"),-1)[...,[1,0]].unsqueeze(0)
def apply_flow(xp,fhw):  # fhw (1,2,H,W)
    g=(base+fhw.permute(0,2,3,1)).clamp(-1,1); return F.grid_sample(xp,g,align_corners=True,padding_mode="border")
def opt_flow(xp,q,iters=300):
    flow=torch.zeros(1,2,GY,GX,device=dev,requires_grad=True); opt=torch.optim.Adam([flow],lr=0.02)
    for _ in range(iters):
        opt.zero_grad(); fhw=F.interpolate(flow,size=(H,W),mode="bilinear",align_corners=True)
        o=apply_flow(xp,fhw)
        loss=((o-q)**2).mean()+0.5*((flow[:,:,1:]-flow[:,:,:-1])**2).mean()+0.5*((flow[:,:,:,1:]-flow[:,:,:,:-1])**2).mean()
        loss.backward(); opt.step()
    with torch.no_grad(): return F.interpolate(flow.detach(),size=(H,W),mode="bilinear",align_corners=True)
def sparsify(fhw,Ky,Kx):  # subsample dense flow to Ky x Kx anchors then smoothly re-interp
    a=F.adaptive_avg_pool2d(fhw,(Ky,Kx)); return F.interpolate(a,size=(H,W),mode="bilinear",align_corners=True)
Kcfg=[("K=1",1,1),("K=3",3,1),("K=5",5,1),("K=6",3,2),("K=10",5,2),("K=24",8,3),("dense",GY,GX)]
acc={k[0]:([],[]) for k in Kcfg}
for n in range(N):
    xp=xpva[n:n+1].to(dev); q=xqva[n:n+1].to(dev); fhw=opt_flow(xp,q)
    for name,Ky,Kx in Kcfg:
        o=apply_flow(xp, fhw if name=="dense" else sparsify(fhw,Ky,Kx)).clamp(0,1)
        acc[name][0].append(float(ssim_fn(o,q))); acc[name][1].append(float(lpips_fn(o,q)))
print("== SGOS Gate C: sparse recovery of optimal deformation, N=%d =="%N,flush=True)
print("  x_pre 0.1996/0.4283 | APTD 0.2554-0.3018 | dense-oracle target ~0.4262/0.4119",flush=True)
for name,_,_ in Kcfg:
    S,L=acc[name]; print("  %-7s SSIM=%.4f LPIPS=%.4f"%(name,np.mean(S),np.mean(L)),flush=True)
print("SGOS_GATEC_DONE",flush=True)
