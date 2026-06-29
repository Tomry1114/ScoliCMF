import os,sys,torch
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from eval_gates import load_ckpt
dev="cuda"; H,W=480,240
ds=PairedSpineDataset(root=os.path.expanduser("~/ScoliCMF/data/Spine生成_Miccai数据集"),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
xp,_=next(iter(DataLoader(ds,batch_size=8))); xp=xp.to(dev)
print("run        | tok_cos E_top4 | R_removed_amp R_removed_eng")
for stem in ["shmm_dct","shmm_v1","shmm_v2"]:
    cfg=load_config(os.path.expanduser(f"~/ScoliCMF/configs/{stem}.yaml"))
    m=load_ckpt(os.path.expanduser(f"~/ScoliCMF/runs/{stem}/ckpts/step_5000.pt"),cfg,H,W,None,dev,use_ema=True)
    core=getattr(m,"module",m); cond=core.cond; cond.diag=True
    r=torch.full((8,),0.25,device=dev); t=torch.full((8,),0.5,device=dev)
    with torch.no_grad():
        _=core(xp,r,t,xp)
        d=cond._diag; raw=d["raw_dyn"]; Pi=d["Pi"]
        resid=raw-cond._proj_apply(Pi,raw)
        e_amp=float(resid.norm()/raw.norm().clamp_min(1e-6))
        e_eng=float(resid.square().sum()/raw.square().sum().clamp_min(1e-9))
    cond.diag=False
    _,aux=cond(xp,r,t,torch.rand(8,cfg["model"]["dim"],device=dev),torch.rand(8,cfg["model"]["dim"],device=dev))
    print("%-10s | %.4f  %.4f | %.4f        %.4f"%(stem,float(aux["tok_cos"]),float(aux["E_top4"]),e_amp,e_eng),flush=True)
print("DIAG_NEW_DONE")
