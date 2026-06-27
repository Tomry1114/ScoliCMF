import os,sys,numpy as np,torch
from torch.utils.data import DataLoader
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt
from metrics_img import ssim,lpips_fn
dev="cuda"; stem=sys.argv[1]; steps=[int(x) for x in sys.argv[2].split(",")]
cfg=load_config(os.path.expanduser(f"~/ScoliCMF/configs/{stem}.yaml"))
H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]; ROOT=os.path.expanduser(cfg["data"]["root"])
mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])
ds=PairedSpineDataset(root=ROOT,size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
ld=DataLoader(ds,batch_size=4,num_workers=2)
print("step | SSIM4 | LPIPS4 | pred-vs-pre")
for st in steps:
    p=os.path.expanduser(f"~/ScoliCMF/runs/{stem}/ckpts/step_{st}.pt")
    if not os.path.exists(p): continue
    m=load_ckpt(p,cfg,H,W,None,dev,use_ema=True); ss,lp,pr=[],[],[]
    with torch.no_grad():
        for xp,xq in ld:
            xp,xq=xp.to(dev),xq.to(dev); z=mf.sample(m,xp,steps=4)
            ss.append(ssim(z,xq).cpu().numpy()); lp.append(lpips_fn(z,xq).cpu().numpy()); pr.append(ssim(z,xp).cpu().numpy())
    print("%5d | %.4f | %.4f | %.4f"%(st,np.concatenate(ss).mean(),np.concatenate(lp).mean(),np.concatenate(pr).mean()),flush=True)
print("MULTI_DONE")
