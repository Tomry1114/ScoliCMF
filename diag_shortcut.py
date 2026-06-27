import os, sys, numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt
from metrics_img import ssim, psnr, lpips_fn
dev="cuda"
def run(stem, step):
    cfg=load_config(os.path.expanduser(f"~/ScoliCMF/configs/{stem}.yaml"))
    H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]; ROOT=os.path.expanduser(cfg["data"]["root"])
    mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])
    m=load_ckpt(os.path.expanduser(f"~/ScoliCMF/runs/{stem}/ckpts/step_{step}.pt"),cfg,H,W,None,dev,use_ema=True)
    ds=PairedSpineDataset(root=ROOT,size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
    ld=DataLoader(ds,batch_size=4,num_workers=2)
    a={k:[] for k in("pp_ssim","pre_ssim","id_ssim","pp_lpips","id_lpips","gain")}
    with torch.no_grad():
        for xp,xq in ld:
            xp,xq=xp.to(dev),xq.to(dev)
            z=mf.sample(m,xp,steps=4)
            a["pp_ssim"].append(ssim(z,xq).cpu().numpy())        # pred vs POST (target)
            a["pre_ssim"].append(ssim(z,xp).cpu().numpy())       # pred vs PRE  (copy?)
            a["id_ssim"].append(ssim(xp,xq).cpu().numpy())       # PRE vs POST  (identity baseline)
            a["pp_lpips"].append(lpips_fn(z,xq).cpu().numpy())
            a["id_lpips"].append(lpips_fn(xp,xq).cpu().numpy())
            a["gain"].append((ssim(z,xq)-ssim(xp,xq)).cpu().numpy())  # improvement over identity
    return {k:float(np.concatenate(v).mean()) for k,v in a.items()}
for stem in ["s2_base","s5b_scpga_v2"]:
    r=run(stem,5000)
    print("%-16s pred-vs-POST=%.3f  pred-vs-PRE=%.3f  identity(PRE-POST)=%.3f  GAIN=%.4f | LPIPS pred=%.3f id=%.3f"%(
        stem,r["pp_ssim"],r["pre_ssim"],r["id_ssim"],r["gain"],r["pp_lpips"],r["id_lpips"]))
print("DIAG_DONE")
