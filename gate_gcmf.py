"""GC-MF Gate A+B (zero training). Small acquisition gauge group G=Sim+(2): +-3deg rot, +-0.03 trans, 0.95-1.05 scale (NO photometry).
A: gauge defect of the interval operator. E_G=|rho(g^-1) T(rho(g)x) - T(x)| vs resampling floor E_interp=|rho(g^-1)rho(g)x - x|.
   Per-case correlate E_G with SSIM/LPIPS/err; train vs val. If model ~equivariant (E_G ~ E_interp) or E_G uncorrelated with error -> dead.
B: zero-train Reynolds projection yhat_G = mean_k rho(g_k^-1) T(rho(g_k)x). Compare single / noise-avg(same compute) / Reynolds.
   PASS: dSSIM>=0.005 or dLPIPS<=-0.01 AND Reynolds clearly beats noise-avg AND not mostly global photometry."""
import os, sys, math, torch
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
def loadsplit(s):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits",s))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP),torch.cat(XQ)
xpva,xqva=loadsplit("val.txt"); xptr,_=loadsplit("train.txt"); Nv=xpva.shape[0]
g_seed=torch.Generator(device=dev).manual_seed(0)
def rand_g():
    th=(torch.rand(1,generator=g_seed,device=dev)*2-1)*(3*math.pi/180)
    s=0.95+torch.rand(1,generator=g_seed,device=dev)*0.10
    tx=(torch.rand(1,generator=g_seed,device=dev)*2-1)*0.03; ty=(torch.rand(1,generator=g_seed,device=dev)*2-1)*0.03
    c,si=torch.cos(th),torch.sin(th)
    M=torch.tensor([[ (s*c).item(),(-s*si).item(),tx.item()],[ (s*si).item(),(s*c).item(),ty.item()]],device=dev)
    A=torch.eye(3,device=dev); A[:2]=M; Ai=torch.linalg.inv(A)
    return M, Ai[:2]
def warp(x,M):
    grid=F.affine_grid(M[None].expand(x.shape[0],-1,-1),x.shape,align_corners=False)
    return F.grid_sample(x,grid,align_corners=False,padding_mode="border")
def loadm(stp):
    bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
    st=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_%d.pt"%stp),map_location=dev)
    for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev))
    return m.eval()
def T(m,x):
    B=x.shape[0]; return m(x,torch.zeros(B,device=dev),torch.ones(B,device=dev),x)["xhat"].clamp(0,1)
def spear(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-8); rb=(rb-rb.mean())/rb.std().clamp_min(1e-8); return float((ra*rb).mean())
K=8; gs=[rand_g() for _ in range(K)]
@torch.no_grad()
def gauge_defect(m,X):  # returns per-case E_G(rel), E_interp(rel)
    EG=[];EI=[]
    for i in range(0,X.shape[0],6):
        x=X[i:i+6].to(dev); Tx=T(m,x); chg=(Tx-x).abs().mean(dim=(1,2,3)).clamp_min(1e-6)
        dg=torch.zeros(x.shape[0],device=dev); di=torch.zeros(x.shape[0],device=dev)
        for M,Mi in gs:
            gx=warp(x,M); back=warp(T(m,gx),Mi); dg+=(back-Tx).abs().mean(dim=(1,2,3))
            di+=(warp(gx,Mi)-x).abs().mean(dim=(1,2,3))
        EG.append((dg/K/chg).cpu()); EI.append((di/K/chg).cpu())
    return torch.cat(EG),torch.cat(EI)
print("==== GATE A: gauge defect vs resampling floor (rel to |T(x)-x|) ====",flush=True)
res={}
for stp in [1000,2000,3000,4000,5000]:
    m=loadm(stp); EG,EI=gauge_defect(m,xpva)
    res[stp]=(m,EG,EI); print("  step%-5d  E_G=%.3f  E_interp=%.3f  ratio=%.2f"%(stp,EG.mean(),EI.mean(),EG.mean()/EI.mean().clamp_min(1e-6)),flush=True)
m5=res[5000][0]; EG5=res[5000][1]
# per-case corr with SSIM/LPIPS/err on step5000
S=[];L=[];ER=[]
with torch.no_grad():
    for i in range(0,Nv,6):
        x=xpva[i:i+6].to(dev); q=xqva[i:i+6].to(dev); y=T(m5,x)
        S.append(ssim(y,q).cpu()); L.append(lpips_fn(y,q).cpu()); ER.append((y-q).abs().mean(dim=(1,2,3)).cpu())
S=torch.cat(S);L=torch.cat(L);ER=torch.cat(ER)
print("  [step5000] per-case rho(E_G, LPIPS)=%.2f  rho(E_G, err)=%.2f  rho(E_G, -SSIM)=%.2f"%(spear(EG5,L),spear(EG5,ER),spear(EG5,-S)),flush=True)
EGtr,_=gauge_defect(m5,xptr[:54]); print("  [step5000] E_G train=%.3f  val=%.3f  (train<val => finite-sample overfit)"%(EGtr.mean(),EG5.mean()),flush=True)
print("==== GATE B: zero-train Reynolds projection (step5000) ====",flush=True)
@torch.no_grad()
def variants(X,Q):
    outs={"single":[],"reyn":[],"noise":[]}; SS={k:[] for k in outs}; LL={k:[] for k in outs}; photo=0.;pn=0.
    for i in range(0,X.shape[0],6):
        x=X[i:i+6].to(dev); q=Q[i:i+6].to(dev)
        y=T(m5,x)
        ry=torch.zeros_like(y)
        for M,Mi in gs: ry+=warp(T(m5,warp(x,M)),Mi)
        ry=(ry/K).clamp(0,1)
        ny=torch.zeros_like(y)
        for k in range(K): ny+=T(m5,(x+0.03*torch.randn(x.shape,generator=g_seed,device=dev)).clamp(0,1))
        ny=(ny/K).clamp(0,1)
        for k,v in [("single",y),("reyn",ry),("noise",ny)]:
            SS[k].append(ssim(v,q).cpu()); LL[k].append(lpips_fn(v,q).cpu())
        d=ry-y; photo+=d.mean().abs().item()*x.shape[0]; pn+=d.abs().mean().item()*x.shape[0]
    for k in SS: print("  %-8s SSIM=%.4f LPIPS=%.4f"%(k,torch.cat(SS[k]).mean(),torch.cat(LL[k]).mean()),flush=True)
    print("  [Reynolds corr] |mean(d)|/|d|=%.2f (photometry; want small)"%(photo/pn),flush=True)
variants(xpva,xqva)
print("GCMF_GATE_DONE")
