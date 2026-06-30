"""Exp2 cheap pre-gate: decompose the canonical-frame eval gain into geometry vs photometric.
Fixed RAW-trained warpres model; evaluate on val under different cleaned reference frames.
If raw->photo captures most of raw->full, the gain is photometric (narrative=acquisition norm);
if raw->geo dominates, it is geometric source-frame canonicalization."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from metrics_img import ssim, lpips_fn
from aptd_model import APTDNet

def _v4(x): return x.view(-1, 1, 1, 1)
HOME = os.path.expanduser("~/ScoliCMF"); dev = "cuda"
cfg = load_config(os.path.join(HOME, "configs/s2_base.yaml")); H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
cfg["model"]["xpre_mode"] = "full"
mf = SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"], sigma_m=cfg["meanflow"]["sigma_m"]); path = mf.path

def inorder(split):
    ds = PairedSpineDataset(root=os.path.join(HOME, cfg["data"]["root"]), size=(H, W), split_file=os.path.join(HOME, "splits", split))
    XP=[];XQ=[]
    for x,y in DataLoader(ds,batch_size=64,shuffle=False): XP.append(x);XQ.append(y)
    return torch.cat(XP), torch.cat(XQ)
def psnr(a,b): return -10*torch.log10(((a-b)**2).mean(dim=(1,2,3)).clamp_min(1e-10))

XPva, XQraw = inorder("val.txt")
bb = build_model(cfg,H,W).to(dev); m = APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
st = torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_5000.pt"), map_location=dev)
for p,e in zip(m.parameters(), st["ema"]): p.data.copy_(e.to(dev))
m.eval()

@torch.no_grad()
def ev(tgt, nfe=1):
    S=[];P=[];L=[]
    for i in range(0, XPva.shape[0], 6):
        xp=XPva[i:i+6].to(dev); xq=tgt[i:i+6].to(dev); B=xp.shape[0]
        z=xp; tv=torch.linspace(1,0,nfe+1,device=dev); xhat=None
        for k in range(nfe):
            t=torch.full((B,),tv[k].item(),device=dev); r=torch.full((B,),tv[k+1].item(),device=dev)
            xhat=m(z,r,t,xp)["xhat"]; z=xp+_v4(path.alpha(r))*(xhat-xp)
        o=xhat.clamp(0,1); S.append(ssim(o,xq).cpu());P.append(psnr(o,xq).cpu());L.append(lpips_fn(o,xq).cpu())
    return float(torch.cat(S).mean()),float(torch.cat(P).mean()),float(torch.cat(L).mean())

frames = {
 "raw       ": XQraw,
 "geo-only  ": torch.load(os.path.join(HOME,"runs/adoc/clean_geo_val.pt"))["clean"],
 "photo-only": torch.load(os.path.join(HOME,"runs/adoc/clean_photo_val.pt"))["clean"],
 "full      ": torch.load(os.path.join(HOME,"runs/adoc/clean_val.pt"))["clean"],
}
print("== Exp2 gate: fixed RAW-warpres model, eval on different reference frames ==", flush=True)
res={}
for k,t in frames.items():
    s,p,l = ev(t); res[k.strip()]=s
    print("  frame=%s  SSIM=%.4f PSNR=%.3f LPIPS=%.4f" % (k,s,p,l), flush=True)
g=res["geo-only"]-res["raw"]; ph=res["photo-only"]-res["raw"]; f=res["full"]-res["raw"]
print("  dSSIM raw->geo=%.4f  raw->photo=%.4f  raw->full=%.4f  (geo share=%.0f%%, photo share=%.0f%%)"
      % (g, ph, f, 100*g/max(f,1e-6), 100*ph/max(f,1e-6)), flush=True)
print("GEOPHOTO_GATE_DONE")
