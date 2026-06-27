import os, sys, glob, numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt
from metrics_img import ssim
dev="cuda"; stem="s5b_scpga_v2"
cfg=load_config(os.path.expanduser(f"~/ScoliCMF/configs/{stem}.yaml"))
H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]; ROOT=os.path.expanduser(cfg["data"]["root"])
mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])
ds=PairedSpineDataset(root=ROOT,size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
ld=DataLoader(ds,batch_size=4,num_workers=2)
print("step | val_SSIM4(vs post) | pred-vs-pre")
for st in range(500,5001,500):
    m=load_ckpt(os.path.expanduser(f"~/ScoliCMF/runs/{stem}/ckpts/step_{st}.pt"),cfg,H,W,None,dev,use_ema=True)
    pp,pr=[],[]
    with torch.no_grad():
        for xp,xq in ld:
            xp,xq=xp.to(dev),xq.to(dev); z=mf.sample(m,xp,steps=4)
            pp.append(ssim(z,xq).cpu().numpy()); pr.append(ssim(z,xp).cpu().numpy())
    print("%5d | %.4f | %.4f"%(st,float(np.concatenate(pp).mean()),float(np.concatenate(pr).mean())))
print("TRAJ_DONE")
