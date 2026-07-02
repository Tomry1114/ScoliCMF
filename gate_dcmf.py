"""Preflight for Defect-Calibrated MeanFlow (DC-MF). NO training. Uses existing APTD ckpt.
Gate 1: is the direct-vs-split DEFECT predictive of which cases BENEFIT from more NFE?
  d_i = |x1step - x2step|; benefit_i = SSIM(x2)-SSIM(x1). Key: high defect => larger benefit.
Gate 2: oracle adaptive-NFE upper bound. Control NFE by defect (and by GT-oracle) and compare
  to fixed 1/2/4 at matched average NFE. Prize exists if adaptive @ avg~1.3-1.5 >= fixed-2."""
import os, sys, torch
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
for p,e in zip(m.parameters(),st["ema"]): p.data.copy_(e.to(dev)); 
m.eval()
@torch.no_grad()
def sample(xp,nfe):
    B=xp.shape[0]; z=xp.clone(); tv=torch.linspace(1,0,nfe+1,device=dev); xhat=None
    for k in range(nfe):
        t=torch.full((B,),tv[k].item(),device=dev); r=torch.full((B,),tv[k+1].item(),device=dev)
        xhat=m(z,r,t,xp)["xhat"]; z=xp+_v4(path.alpha(r))*(xhat-xp)
    return xhat.clamp(0,1)
# per-case predictions for nfe 1/2/4
S={1:[],2:[],4:[]}; L={1:[],2:[],4:[]}; preds={1:[],2:[],4:[]}; POST=[]
for i in range(0,Nv,6):
    xp=xpva[i:i+6].to(dev); xq=xqva[i:i+6].to(dev); POST.append(xq.cpu())
    for nf in (1,2,4):
        o=sample(xp,nf); preds[nf].append(o.cpu())
        S[nf].append(ssim(o,xq).cpu()); L[nf].append(lpips_fn(o,xq).cpu())
for nf in (1,2,4): S[nf]=torch.cat(S[nf]); L[nf]=torch.cat(L[nf]); preds[nf]=torch.cat(preds[nf])
POST=torch.cat(POST)
# defect & error (per case, L1)
d12=(preds[1]-preds[2]).abs().mean(dim=(1,2,3)); d14=(preds[1]-preds[4]).abs().mean(dim=(1,2,3))
e1=(preds[1]-POST).abs().mean(dim=(1,2,3))
ben2=S[2]-S[1]; ben4=S[4]-S[1]
def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std(); return float((ra*rb).mean())
print("==== GATE 1: defect vs benefit ====", flush=True)
print("  mean SSIM  1step=%.4f 2step=%.4f 4step=%.4f"%(S[1].mean(),S[2].mean(),S[4].mean()))
print("  mean LPIPS 1step=%.4f 2step=%.4f 4step=%.4f"%(L[1].mean(),L[2].mean(),L[4].mean()))
print("  frac cases 2step>1step (SSIM)=%.2f   4step>1step=%.2f"%((ben2>0).float().mean(),(ben4>0).float().mean()))
print("  Spearman(defect_d12, true_err_e1)      = %.3f"%spearman(d12,e1))
print("  Spearman(defect_d12, benefit2=S2-S1)   = %.3f"%spearman(d12,ben2))
print("  Spearman(defect_d14, benefit4=S4-S1)   = %.3f"%spearman(d14,ben4))
# terciles by defect d12
order=d12.argsort(); g=Nv//3; lo=order[:g]; mid=order[g:2*g]; hi=order[2*g:]
for nm,idx in [("LOW-defect",lo),("MID-defect",mid),("HIGH-defect",hi)]:
    print("  %s: mean d12=%.4f  benefit2(S2-S1)=%+.4f  benefit4(S4-S1)=%+.4f"%(nm,d12[idx].mean(),ben2[idx].mean(),ben4[idx].mean()))
print("==== GATE 2: oracle adaptive NFE ====", flush=True)
print("  Fixed-1: SSIM=%.4f LPIPS=%.4f  (avgNFE 1.0)"%(S[1].mean(),L[1].mean()))
print("  Fixed-2: SSIM=%.4f LPIPS=%.4f  (avgNFE 2.0)"%(S[2].mean(),L[2].mean()))
print("  Fixed-4: SSIM=%.4f LPIPS=%.4f  (avgNFE 4.0)"%(S[4].mean(),L[4].mean()))
# defect-controlled: top-f by defect get 2step, rest 1step
for f in (0.35,0.5):
    k=int(Nv*f); hi_idx=set(d12.argsort(descending=True)[:k].tolist())
    sel=torch.tensor([2 if j in hi_idx else 1 for j in range(Nv)])
    ss=torch.where(sel==2,S[2],S[1]).mean(); ll=torch.where(sel==2,L[2],L[1]).mean(); avg=sel.float().mean()
    print("  Defect-adaptive(top %.0f%%->2step): SSIM=%.4f LPIPS=%.4f  (avgNFE %.2f)"%(f*100,ss,ll,avg))
# 3-level defect: top 20%->4, next 30%->2, rest->1
o=d12.argsort(descending=True); sel=torch.ones(Nv,dtype=torch.long)
sel[o[:int(Nv*0.2)]]=4; sel[o[int(Nv*0.2):int(Nv*0.5)]]=2
ssf=torch.stack([S[int(s.item())][j] for j,s in enumerate(sel)]).mean(); avg=sel.float().mean()
print("  Defect-adaptive(3-level 20/30/50): SSIM=%.4f  (avgNFE %.2f)"%(ssf,avg))
# GT oracle upper bound: pick best NFE per case by SSIM
stack=torch.stack([S[1],S[2],S[4]],1); bestnf=stack.argmax(1); nfval=torch.tensor([1,2,4])[bestnf]
print("  GT-ORACLE best-NFE/case: SSIM=%.4f  (avgNFE %.2f)  -- upper bound"%(stack.max(1).values.mean(),nfval.float().mean()))
print("GATE_DCMF_DONE")
