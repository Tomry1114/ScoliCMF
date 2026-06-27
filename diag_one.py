import os,sys,numpy as np,torch
from torch.utils.data import DataLoader
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt
from metrics_img import ssim,lpips_fn
dev="cuda"; stem=sys.argv[1]; step=int(sys.argv[2])
cfg=load_config(os.path.expanduser(f"~/ScoliCMF/configs/{stem}.yaml"))
H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]; ROOT=os.path.expanduser(cfg["data"]["root"])
mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])
m=load_ckpt(os.path.expanduser(f"~/ScoliCMF/runs/{stem}/ckpts/step_{step}.pt"),cfg,H,W,None,dev,use_ema=True)
ds=PairedSpineDataset(root=ROOT,size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
ld=DataLoader(ds,batch_size=4,num_workers=2)
ss,lp,pr=[],[],[]
with torch.no_grad():
    for xp,xq in ld:
        xp,xq=xp.to(dev),xq.to(dev); z=mf.sample(m,xp,steps=4)
        ss.append(ssim(z,xq).cpu().numpy()); lp.append(lpips_fn(z,xq).cpu().numpy()); pr.append(ssim(z,xp).cpu().numpy())
print("%s step%d: val SSIM4=%.4f  LPIPS4=%.4f  pred-vs-pre=%.4f"%(stem,step,np.concatenate(ss).mean(),np.concatenate(lp).mean(),np.concatenate(pr).mean()))
print("ONE_DONE")
