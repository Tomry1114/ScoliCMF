"""Exp1 independence 2x2, evaluated ENTIRELY on the canonical (full-clean) val frame so all four
cells share one fixed reference. Paired-diff bootstrap CIs for the ADOC effect within each
generator row. Independence proven if the DIRECT (plain x0 Bridge) row also shows ADOC>raw."""
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
cfg=cfgfull()
mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"], sigma_m=cfg["meanflow"]["sigma_m"]); path=mf.path
def psnr(a,b): return -10*torch.log10(((a-b)**2).mean(dim=(1,2,3)).clamp_min(1e-10))
dsv=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]), size=(H,W), split_file=os.path.join(HOME,"splits/val.txt"))
XP=[]
for a,b in DataLoader(dsv,batch_size=64,shuffle=False): XP.append(a)
xpva=torch.cat(XP); clva=torch.load(os.path.join(HOME,"runs/adoc/clean_val.pt"))["clean"]

@torch.no_grad()
def percase(ckpt, mode, fs=0.15, use_ema=True):
    bb=build_model(cfgfull(),H,W).to(dev); m=APTDNet(bb,mode,flow_scale=fs).to(dev)
    st=torch.load(ckpt,map_location=dev)
    if use_ema and "ema" in st:
        for p,e in zip(m.parameters(), st["ema"]): p.data.copy_(e.to(dev))
    else:
        m.load_state_dict(st["model"])
    m.eval(); S=[];P=[];L=[]
    for i in range(0,xpva.shape[0],6):
        xp=xpva[i:i+6].to(dev); xq=clva[i:i+6].to(dev); B=xp.shape[0]
        r0=torch.zeros(B,device=dev); t1=torch.ones(B,device=dev)
        o=m(xp,r0,t1,xp)["xhat"].clamp(0,1)
        S.append(ssim(o,xq).cpu()); P.append(psnr(o,xq).cpu()); L.append(lpips_fn(o,xq).cpu())
    return torch.cat(S),torch.cat(P),torch.cat(L)

g=torch.Generator().manual_seed(0); n=xpva.shape[0]; idx=torch.randint(0,n,(2000,n),generator=g)
def ci(x): bs=x[idx].mean(1); return float(x.mean()),float(bs.quantile(0.025)),float(bs.quantile(0.975))
def cidiff(a,b): bs=(a[idx]-b[idx]).mean(1); return float((a-b).mean()),float(bs.quantile(0.025)),float(bs.quantile(0.975))
def line(t,x): m,lo,hi=ci(x); print("   %-30s %.4f [%.4f, %.4f]"%(t,m,lo,hi))
def dline(t,a,b): m,lo,hi=cidiff(a,b); s="EXCLUDES 0 (sig)" if (lo>0 or hi<0) else "includes 0"; print("   %-30s d=%+.4f [%+.4f, %+.4f]  %s"%(t,m,lo,hi,s))

CK={"dr":(os.path.join(HOME,"runs/ind_direct_raw/ckpts/step_5000.pt"),"direct",True),
    "da":(os.path.join(HOME,"runs/ind_direct_adoc/ckpts/step_5000.pt"),"direct",True),
    "wr":(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_5000.pt"),"warpres",True),
    "wa":(os.path.join(HOME,"runs/aptd_adoc/ckpts/step_5000.pt"),"warpres",False)}
R={}
for k,(c,mo,ue) in CK.items(): R[k]=percase(c,mo,use_ema=ue)
print("======== Exp1 INDEPENDENCE 2x2  (ALL on canonical full-clean val) ========")
print("-- generator=DIRECT (plain x0 Bridge) --")
line("direct+raw  SSIM",R["dr"][0]); line("direct+ADOC SSIM",R["da"][0])
line("direct+raw  PSNR",R["dr"][1]); line("direct+ADOC PSNR",R["da"][1])
line("direct+raw  LPIPS",R["dr"][2]); line("direct+ADOC LPIPS",R["da"][2])
dline("DIRECT dSSIM (ADOC-raw)",R["da"][0],R["dr"][0])
dline("DIRECT dPSNR (ADOC-raw)",R["da"][1],R["dr"][1])
dline("DIRECT dLPIPS(ADOC-raw)",R["da"][2],R["dr"][2])
print("-- generator=APTD (warpres) --")
line("APTD+raw    SSIM",R["wr"][0]); line("APTD+ADOC   SSIM",R["wa"][0])
dline("APTD   dSSIM (ADOC-raw)",R["wa"][0],R["wr"][0])
dline("APTD   dPSNR (ADOC-raw)",R["wa"][1],R["wr"][1])
dline("APTD   dLPIPS(ADOC-raw)",R["wa"][2],R["wr"][2])
print("INDEPENDENCE_DONE")
