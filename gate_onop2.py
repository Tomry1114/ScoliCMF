"""ONOP Gate 2 (+3): train a small patch post-op score prior, then test APTD vs unprojected update
vs ONOP (tangent-projected). Decompose the ONOP correction (Gate 3) to check it is NOT just photometry."""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
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
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits",split))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP),torch.cat(XQ)
xptr,xqtr=load("train.txt"); xpva,xqva=load("val.txt"); Nv=xpva.shape[0]

# ---- small DnCNN noise predictor ----
class DnCNN(nn.Module):
    def __init__(s,ch=48,d=8):
        super().__init__(); L=[nn.Conv2d(1,ch,3,1,1),nn.ReLU(True)]
        for _ in range(d-2): L+=[nn.Conv2d(ch,ch,3,1,1),nn.BatchNorm2d(ch),nn.ReLU(True)]
        L+=[nn.Conv2d(ch,1,3,1,1)]; s.net=nn.Sequential(*L)
    def forward(s,x): return s.net(x)
net=DnCNN().to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
XQtr=xqtr.to(dev); NT=XQtr.shape[0]
g=torch.Generator(device=dev).manual_seed(0)
print("training patch post-op score prior...", flush=True); net.train()
for step in range(1,3001):
    idx=torch.randint(0,NT,(32,),generator=g,device=dev)
    ys=[]
    for j in idx:
        i0=torch.randint(0,H-96,(1,),generator=g,device=dev).item(); j0=torch.randint(0,W-96,(1,),generator=g,device=dev).item()
        ys.append(XQtr[j:j+1,:,i0:i0+96,j0:j0+96])
    y=torch.cat(ys,0)
    sig=(0.01+0.07*torch.rand(y.shape[0],1,1,1,generator=g,device=dev))
    noise=sig*torch.randn(y.shape,generator=g,device=dev); yn=(y+noise).clamp(0,1)
    pred=net(yn); loss=F.mse_loss(pred,noise)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print("  step %d mse %.5f"%(step,loss.item()),flush=True)
net.eval()

@torch.no_grad()
def apt(xp):
    B=xp.shape[0]; z=xp.clone(); t=torch.ones(B,device=dev); r=torch.zeros(B,device=dev)
    return m(z,r,t,xp)["xhat"].clamp(0,1)
bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
st=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_5000.pt"),map_location=dev)
for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev))
m.eval()
def grads(y):
    gx=torch.zeros_like(y); gy=torch.zeros_like(y)
    gx[:,:,:,1:-1]=(y[:,:,:,2:]-y[:,:,:,:-2])*0.5; gy[:,:,1:-1,:]=(y[:,:,2:,:]-y[:,:,:-2,:])*0.5
    return gx,gy
bumps=F.interpolate(torch.eye(32,device=dev).view(32,1,8,4),size=(H,W),mode="bilinear",align_corners=False).squeeze(1)
@torch.no_grad()
def tang_proj(y,s):  # return P_T s  (s,y: [1,1,H,W])
    gx,gy=grads(y); Ng=bumps.shape[0]
    Tx=(gx.squeeze(0)*bumps).reshape(Ng,-1); Ty=(gy.squeeze(0)*bumps).reshape(Ng,-1)
    T=torch.cat([Tx,Ty],0).T; T=T/(T.norm(dim=0,keepdim=True)+1e-8)
    sv=s.reshape(-1,1); G=T.T@T+1e-3*torch.eye(2*Ng,device=dev)
    return (T@torch.linalg.solve(G,T.T@sv)).reshape(1,1,H,W)
def metr(y,q): return ssim(y,q).item(), lpips_fn(y,q).item()

@torch.no_grad()
def evalall(eta):
    S={"base":[],"unproj":[],"onop":[]}; L={"base":[],"unproj":[],"onop":[]}
    # gate3 accumulators for onop correction
    dcAbs=0.; dTot=0.; lf=0.; cen=0.; per=0.; tangfrac_un=0.; tangfrac_on=0.
    xcol=torch.linspace(0,1,W,device=dev); cmask=((xcol-0.5).abs()<0.15).view(1,1,1,W).float()
    for i in range(Nv):
        xp=xpva[i:i+1].to(dev); q=xqva[i:i+1].to(dev); yh=apt(xp)
        s=-net(yh)                       # score toward manifold = denoised - yh
        PTs=tang_proj(yh,s); sN=s-PTs
        yu=(yh+eta*s).clamp(0,1); yo=(yh+eta*sN).clamp(0,1)
        for k,y in [("base",yh),("unproj",yu),("onop",yo)]:
            a,b=metr(y,q); S[k].append(a); L[k].append(b)
        d=yo-yh
        dcAbs+=d.mean().abs().item(); dTot+=d.abs().mean().item()
        c=F.avg_pool2d(d,8); cu=F.interpolate(c,size=(H,W),mode="bilinear",align_corners=False)
        lf+=(cu.pow(2).sum()/d.pow(2).sum().clamp_min(1e-9)).item()
        cen+=(d.abs()*cmask).sum().item()/cmask.sum().item(); per+=(d.abs()*(1-cmask)).sum().item()/(1-cmask).sum().item()
        tangfrac_un+=(PTs.pow(2).sum()/(eta*s).pow(2).sum().clamp_min(1e-9)).item()
        tangfrac_on+=((tang_proj(yh,sN)).pow(2).sum()/sN.pow(2).sum().clamp_min(1e-9)).item()
    mean=lambda a: sum(a)/len(a)
    print("  eta=%.1f | SSIM base=%.4f unproj=%.4f onop=%.4f | LPIPS base=%.4f unproj=%.4f onop=%.4f"
          %(eta,mean(S["base"]),mean(S["unproj"]),mean(S["onop"]),mean(L["base"]),mean(L["unproj"]),mean(L["onop"])),flush=True)
    print("     [G3 onop-corr] |mean(d)|/|d|=%.3f (photometry) lowfreq=%.2f central/periph=%.2f tang-frac unproj=%.2f onop=%.2f"
          %(dcAbs/dTot, lf/Nv, (cen/Nv)/max(per/Nv,1e-9), tangfrac_un/Nv, tangfrac_on/Nv),flush=True)

print("==== ONOP GATE 2 (+G3): APTD vs unprojected vs ONOP (raw val) ====", flush=True)
for eta in [0.5,1.0,2.0]: evalall(eta)
print("ONOP_GATE2_DONE")
