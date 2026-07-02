"""IC-MF Gate A (near-source target-leakage probe). Tests whether the CURRENT bridge leaks the
post-op residual delta=x_post-x_pre into z_t near the source (t->1), and whether an information-causal
IC schedule (sigma_IC=kappa*sqrt(alpha(1-alpha)), energy-matched) removes it.
Analytical: effective delta-noise sigma/alpha (exact, zero-training). Empirical: tiny probe R^2 recovering delta from [z_t,x_pre]."""
import os, sys, math, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
dev="cuda"; H,W=480,240; HOME=os.path.expanduser("~/ScoliCMF")
cfg=load_config(os.path.join(HOME,"configs/s2_base.yaml")); gamma=cfg["meanflow"]["gamma"]; sm=cfg["meanflow"]["sigma_m"]
def load(s):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits",s))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP),torch.cat(XQ)
xptr,xqtr=load("train.txt"); xpva,xqva=load("val.txt")
dtr=(xqtr-xptr); dva=(xqva-xpva)
def alpha(t): return (math.exp(gamma*(1-t))-1)/(math.exp(gamma)-1)
def sig_old(t): return sm*math.sin(math.pi*t)**2
# energy-match kappa: int sig_old^2 = int kappa^2 alpha(1-alpha)
ts=[i/2000 for i in range(1,2000)]
E_old=sum(sig_old(t)**2 for t in ts)/len(ts)
E_ai=sum(alpha(t)*(1-alpha(t)) for t in ts)/len(ts)
kappa=math.sqrt(E_old/E_ai)
def sig_ic(t): return kappa*math.sqrt(max(alpha(t)*(1-alpha(t)),1e-12))
print("gamma=%.1f sigma_m=%.2f kappa=%.4f (energy-matched)"%(gamma,sm,kappa),flush=True)
print("==== ANALYTICAL: effective delta-noise sigma/alpha (lower => MORE leakage) ====",flush=True)
print("  %-8s %-8s %-10s %-10s %-12s %-12s"%("t","alpha","sig_old","sig_IC","old sig/a","IC sig/a"),flush=True)
for t in [0.90,0.95,0.98,0.995,0.999]:
    a=alpha(t); print("  %-8.3f %-8.4f %-10.5f %-10.5f %-12.4f %-12.4f"%(t,a,sig_old(t),sig_ic(t),sig_old(t)/a,sig_ic(t)/a),flush=True)

class Probe(nn.Module):
    def __init__(s,ch=32):
        super().__init__(); s.n=nn.Sequential(nn.Conv2d(2,ch,3,1,1),nn.ReLU(True),nn.Conv2d(ch,ch,3,1,1),nn.ReLU(True),nn.Conv2d(ch,ch,3,1,1),nn.ReLU(True),nn.Conv2d(ch,1,3,1,1))
    def forward(s,x): return s.n(x)
def train_probe(sigfn, t, steps=400):
    p=Probe().to(dev); opt=torch.optim.Adam(p.parameters(),1e-3)
    XP=xptr.to(dev); D=dtr.to(dev); a=alpha(t); s=sigfn(t); NT=XP.shape[0]
    gg=torch.Generator(device=dev).manual_seed(0)
    p.train()
    for it in range(steps):
        idx=torch.randint(0,NT,(16,),generator=gg,device=dev); xp=XP[idx]; d=D[idx]
        z=xp+a*d+s*torch.randn(d.shape,generator=gg,device=dev)
        pred=p(torch.cat([z,xp],1)); loss=F.mse_loss(pred,d)
        opt.zero_grad(); loss.backward(); opt.step()
    p.eval()
    with torch.no_grad():
        XPv=xpva.to(dev); Dv=dva.to(dev); a_=a
        z=XPv+a*Dv+s*torch.randn(Dv.shape,generator=gg,device=dev)
        pr=p(torch.cat([z,XPv],1))
        ssres=((pr-Dv)**2).sum(); sstot=((Dv-Dv.mean())**2).sum()
        r2=1-(ssres/sstot).item()
        cos=F.cosine_similarity(pr.flatten(1),Dv.flatten(1)).mean().item()
    return r2,cos
print("==== EMPIRICAL: tiny probe R^2 recovering delta from [z_t, x_pre] (val) ====",flush=True)
print("  %-8s | current-path R2  cos | IC-path R2  cos"%("t"),flush=True)
for t in [0.90,0.95,0.98,0.995,0.999]:
    ro,co=train_probe(sig_old,t); ri,ci=train_probe(sig_ic,t)
    print("  %-8.3f | %7.3f %7.3f | %7.3f %7.3f"%(t,ro,co,ri,ci),flush=True)
print("ICMF_GATEA_DONE")
