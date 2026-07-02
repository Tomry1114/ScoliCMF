"""ONOP Gate 1 (NO training, oracle). Does a NORMAL (non-smooth-geometric) error component exist,
and does correcting ONLY it help? e = post - yhat. Build low-dim smooth deformation tangent space
T = grad(yhat) . coarse-grid displacement basis. e_T = P_T e (geometry-explainable), e_N = e - e_T.
rho_N = |e_N|^2/|e|^2. Oracle yhat+e_N vs yhat. e_N low-freq fraction = geometry-leak check.
On the RAW frame, acquisition misalignment is smooth-geometric -> absorbed by e_T -> e_N should be
acquisition-free appearance/texture/artifact error (the thing ONOP targets)."""
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
ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits/val.txt"))
XP=[];XQ=[]
for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
xpva=torch.cat(XP); xqva=torch.cat(XQ); Nv=xpva.shape[0]
bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
st=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_5000.pt"),map_location=dev)
for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev))
m.eval()
@torch.no_grad()
def apt(xp):
    B=xp.shape[0]; z=xp.clone(); t=torch.ones(B,device=dev); r=torch.zeros(B,device=dev)
    return m(z,r,t,xp)["xhat"].clamp(0,1)
def grads(y):  # y [B,1,H,W]
    gx=torch.zeros_like(y); gy=torch.zeros_like(y)
    gx[:,:,:,1:-1]=(y[:,:,:,2:]-y[:,:,:,:-2])*0.5
    gy[:,:,1:-1,:]=(y[:,:,2:,:]-y[:,:,:-2,:])*0.5
    return gx,gy
def make_bumps(Gh,Gw):  # [Ng,H,W] smooth bilinear control-grid bumps
    e=torch.eye(Gh*Gw,device=dev).view(Gh*Gw,1,Gh,Gw)
    return F.interpolate(e,size=(H,W),mode="bilinear",align_corners=False).squeeze(1)  # [Ng,H,W]
def lowfreq_frac(r):  # fraction of energy in coarse (downsample x8) component
    c=F.avg_pool2d(r,8); cu=F.interpolate(c,size=(H,W),mode="bilinear",align_corners=False)
    return (cu.pow(2).sum(dim=(1,2,3))/r.pow(2).sum(dim=(1,2,3)).clamp_min(1e-9))

@torch.no_grad()
def run(Gh,Gw,eta=1.0):
    bumps=make_bumps(Gh,Gw)                      # [Ng,H,W]
    Ng=bumps.shape[0]
    rhoN=[]; S0=[];Sn=[];St=[];L0=[];Ln=[];lfN=[]
    for i in range(Nv):
        xp=xpva[i:i+1].to(dev); post=xqva[i:i+1].to(dev); yh=apt(xp)
        gx,gy=grads(yh)
        # tangent columns: [gx*bump, gy*bump] flattened -> T [P, 2Ng]
        Tx=(gx.squeeze(0)*bumps).reshape(Ng,-1)   # [Ng,P]
        Ty=(gy.squeeze(0)*bumps).reshape(Ng,-1)
        T=torch.cat([Tx,Ty],0).T                  # [P,2Ng]
        T=T/ (T.norm(dim=0,keepdim=True)+1e-8)
        e=(post-yh).reshape(-1,1)                 # [P,1]
        G=T.T@T + 1e-3*torch.eye(2*Ng,device=dev)
        coef=torch.linalg.solve(G, T.T@e)         # [2Ng,1]
        eT=(T@coef).reshape(1,1,H,W); eN=(post-yh)-eT
        rhoN.append((eN.pow(2).sum()/(post-yh).pow(2).sum().clamp_min(1e-9)).item())
        yn=(yh+eta*eN).clamp(0,1); yt=(yh+eta*eT).clamp(0,1)
        S0.append(ssim(yh,post).item()); Sn.append(ssim(yn,post).item()); St.append(ssim(yt,post).item())
        L0.append(lpips_fn(yh,post).item()); Ln.append(lpips_fn(yn,post).item())
        lfN.append(lowfreq_frac(eN).item())
    import statistics as sta
    print("  [basis %dx%d=%d dof] rho_N=%.3f | SSIM base=%.4f +e_N(oracle)=%.4f (%+.4f)  +e_T(geom)=%.4f | LPIPS base=%.4f +e_N=%.4f (%+.4f) | e_N lowfreq=%.2f"
          %(Gh,Gw,2*Ng, sta.mean(rhoN), sta.mean(S0),sta.mean(Sn),sta.mean(Sn)-sta.mean(S0),sta.mean(St),
            sta.mean(L0),sta.mean(Ln),sta.mean(Ln)-sta.mean(L0), sta.mean(lfN)), flush=True)

print("==== ONOP GATE 1: normal-error existence (oracle) ====", flush=True)
print("  (rho_N>=0.30 & SSIM +>=0.01 or LPIPS -<=-0.03 & e_N not low-freq => PASS)", flush=True)
run(8,4); run(12,6); run(6,3)
print("ONOP_GATE1_DONE")
