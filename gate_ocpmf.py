"""OC-PMF Gate (miniature, direct mechanism test). OC-PMF claims: at the perception-distortion
constraint boundary D=eps, an epsilon-constraint natural-gradient flow moves DOWN in P (LPIPS)
while holding D (distortion) fixed to first order -> reaching a point OFF the frontier that fixed-lambda
training cannot. We test the ACTUAL mechanism, not a proxy: start from a mid-frontier checkpoint
(step2000), take ~80 real projected-gradient steps optimizing on TRAIN, evaluate (SSIM,LPIPS) on VAL.

Arms:
  A  baseline (frozen step2000)                          reference point on the frontier
  B  OC-PMF (G=I): v = -(g_P orthogonalized to g_D) + kappa*(D-eps) restoration   the core claim
  C  naive weighted L = D + lam*P (plain SGD)            does the projection beat just LPIPS-in-loss?

Known achievable frontier (from gate_sppc, raw val 1-NFE):
  step2000  SSIM 0.2554 / LPIPS 0.4429     linear-blend a=0.5  SSIM 0.2651 / LPIPS 0.4370
PASS(B): reaches SSIM>=0.2554 AND LPIPS<0.4370 (strictly below the blend envelope = breaks frontier).
FAIL: slides along/above the frontier, or the fixed-D constraint leaks (SSIM drifts down with LPIPS)."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from eval_gates import build_model
from metrics_img import ssim as ssim_fn, lpips_fn, lpips_loss
from aptd_model import APTDNet
dev="cuda"; H,W=480,240; HOME=os.path.expanduser("~/ScoliCMF"); BETA=0.5; EPS_SSIM=0.2554
def cfgfull():
    c=load_config(os.path.join(HOME,"configs/s2_base.yaml")); c["model"]["xpre_mode"]="full"; return c
def loaddata(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfgfull()["data"]["root"]),size=(H,W),
                          split_file=os.path.join(HOME,"splits/%s.txt"%split))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP),torch.cat(XQ)
xptr,xqtr=loaddata("train"); xpva,xqva=loaddata("val"); Ntr=xptr.shape[0]; Nv=xpva.shape[0]
print("train %d  val %d"%(Ntr,Nv),flush=True)

def build_from_step(stp,train_mode):
    bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
    st=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_%d.pt"%stp),map_location=dev)
    for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev))
    m.train(train_mode); return m
def fwd(m,xp):  # source-only 1-NFE output
    B=xp.shape[0]; return m(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)["xhat"].clamp(0,1)
@torch.no_grad()
def evalval(m):
    m.eval(); S=[];L=[]
    for i in range(0,Nv,6):
        xp=xpva[i:i+6].to(dev); q=xqva[i:i+6].to(dev); o=fwd(m,xp)
        S.append(ssim_fn(o,q).cpu()); L.append(lpips_fn(o,q).cpu())
    m.train(); return float(torch.cat(S).mean()), float(torch.cat(L).mean())

def D_P(m,xp,q):  # differentiable distortion & perception on a minibatch
    o=fwd(m,xp)
    Dd=(1-ssim_fn(o,q)).mean()+BETA*(o-q).abs().flatten(1).mean(1).mean()
    Pp=lpips_loss(o,q).mean()
    return Dd,Pp
def flatgrad(loss,params):
    g=torch.autograd.grad(loss,params,retain_graph=False,create_graph=False,allow_unused=True)
    return [ (gi if gi is not None else torch.zeros_like(p)) for gi,p in zip(g,params) ]
def dot(a,b): return sum((x*y).sum() for x,y in zip(a,b))

def minibatch(step,bs=16):
    i=(step*bs)%(Ntr-bs); return xptr[i:i+bs].to(dev), xqtr[i:i+bs].to(dev)

def run_ocpmf(lr,kappa,steps=80):
    m=build_from_step(2000,True); params=[p for p in m.parameters() if p.requires_grad]
    traj=[]
    s0,l0=evalval(m); traj.append((0,s0,l0)); print("  [B lr=%.3g k=%.2g] step0 SSIM=%.4f LPIPS=%.4f"%(lr,kappa,s0,l0),flush=True)
    for t in range(1,steps+1):
        xp,q=minibatch(t)
        Dd,_=D_P(m,xp,q); gD=flatgrad(Dd,params)
        _,Pp=D_P(m,xp,q); gP=flatgrad(Pp,params)
        nD=dot(gD,gD)+1e-12
        coef=dot(gP,gD)/nD                     # orthogonalize gP wrt gD
        Dval=float(Dd.detach())
        with torch.no_grad():
            for p,a,b in zip(params,gP,gD):
                vperp=-(a-coef*b)               # LPIPS descent, D-neutral (1st order)
                vrest=-kappa*(Dval-(1-EPS_SSIM))*b/nD   # restore toward boundary D=eps
                p.add_(lr*(vperp+vrest))
        if t%20==0:
            s,l=evalval(m); traj.append((t,s,l)); print("  [B lr=%.3g k=%.2g] step%d SSIM=%.4f LPIPS=%.4f"%(lr,kappa,t,s,l),flush=True)
    return traj
def run_weighted(lr,lam,steps=80):
    m=build_from_step(2000,True); params=[p for p in m.parameters() if p.requires_grad]
    s0,l0=evalval(m); print("  [C lr=%.3g lam=%.2g] step0 SSIM=%.4f LPIPS=%.4f"%(lr,lam,s0,l0),flush=True)
    for t in range(1,steps+1):
        xp,q=minibatch(t); Dd,Pp=D_P(m,xp,q); loss=Dd+lam*Pp; g=flatgrad(loss,params)
        with torch.no_grad():
            for p,gi in zip(params,g): p.add_(-lr*gi)
        if t%20==0:
            s,l=evalval(m); print("  [C lr=%.3g lam=%.2g] step%d SSIM=%.4f LPIPS=%.4f"%(lr,lam,t,s,l),flush=True)

print("==== reference frontier ====",flush=True)
print("  step2000 SSIM 0.2554 LPIPS 0.4429 | blend a=0.5 SSIM 0.2651 LPIPS 0.4370  (PASS(B)=SSIM>=0.2554 & LPIPS<0.4370)",flush=True)
print("==== ARM B: OC-PMF (G=I, distortion-constrained LPIPS descent) ====",flush=True)
for lr in [2e-4,1e-3,5e-3]:
    run_ocpmf(lr,kappa=1.0)
print("==== ARM C: naive weighted D+lam*P (reference) ====",flush=True)
for lam in [1.0,4.0]:
    run_weighted(1e-3,lam)
print("OCPMF_GATE_DONE",flush=True)
