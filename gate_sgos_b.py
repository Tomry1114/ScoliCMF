"""SGOS Gate B: GEOMETRIC CEILING. Best-case dense 2D smooth warp of x_pre->x_post (per case, optimized).
This is the upper bound of ANY warp/control-based method (far richer than sparse clicks). If even the
optimal dense 2D deformation cannot bring x_pre close to x_post, then the pre->post gap is dominated by
APPEARANCE/CONTENT, not geometry -> sparse spatial control (SGOS) has no channel to help.
Compare vs x_pre(0.1996/0.4283) and APTD-no-control (step2000 0.2554/0.4429, step5000 0.3018/0.6650)."""
import os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from metrics_img import ssim as ssim_fn, lpips_fn
HOME=os.path.expanduser("~/ScoliCMF"); H,W=480,240; dev="cuda" if torch.cuda.is_available() else "cpu"
GY,GX=24,12  # smooth control-grid resolution (upsampled to full flow)
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,load_config(os.path.join(HOME,"configs/s2_base.yaml"))["data"]["root"]),
                          size=(H,W),split_file=os.path.join(HOME,"splits/%s.txt"%split))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP),torch.cat(XQ)
xpva,xqva=load("val"); N=xpva.shape[0]
base=torch.stack(torch.meshgrid(torch.linspace(-1,1,H,device=dev),torch.linspace(-1,1,W,device=dev),indexing="ij"),-1)  # (H,W,2) (y,x)
base=base[...,[1,0]].unsqueeze(0)  # (1,H,W,2) as (x,y)
def warp_grid(flow):  # flow (1,2,GY,GX) normalized-units -> upsample -> apply
    f=F.interpolate(flow,size=(H,W),mode="bilinear",align_corners=True).permute(0,2,3,1)  # (1,H,W,2)
    return (base+f).clamp(-1,1)
def best_warp(xp,q,iters=300):
    flow=torch.zeros(1,2,GY,GX,device=dev,requires_grad=True)
    opt=torch.optim.Adam([flow],lr=0.02)
    for _ in range(iters):
        opt.zero_grad(); g=warp_grid(flow); o=F.grid_sample(xp,g,align_corners=True,padding_mode="border")
        loss=((o-q)**2).mean()+0.5*((flow[:,:,1:]-flow[:,:,:-1])**2).mean()+0.5*((flow[:,:,:,1:]-flow[:,:,:,:-1])**2).mean()
        loss.backward(); opt.step()
    with torch.no_grad(): return F.grid_sample(xp,warp_grid(flow),align_corners=True,padding_mode="border").clamp(0,1)
print("== SGOS Gate B: best-case dense 2D warp (geometric ceiling), N=%d =="%N,flush=True)
S=[];L=[]
for n in range(N):
    xp=xpva[n:n+1].to(dev); q=xqva[n:n+1].to(dev); o=best_warp(xp,q)
    S.append(float(ssim_fn(o,q))); L.append(float(lpips_fn(o,q)))
print("  x_pre(no warp)        SSIM 0.1996 / LPIPS 0.4283   (ref)",flush=True)
print("  BEST dense-2D warp    SSIM=%.4f / LPIPS=%.4f"%(np.mean(S),np.mean(L)),flush=True)
print("  APTD no-control       SSIM 0.2554-0.3018            (ref)",flush=True)
print("SGOS_GATEB_DONE",flush=True)
