import os, sys, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from metrics_img import ssim, psnr, lpips_fn
import sc_pga_ce0b042 as oldpga
from models.sc_dit import SCDiT
dev="cuda"; H,W=480,240
def boot(x,B=2000,seed=0):
    r=np.random.default_rng(seed); s=np.array([x[r.integers(0,len(x),len(x))].mean() for _ in range(B)])
    return float(x.mean()),float(np.quantile(s,.025)),float(np.quantile(s,.975))
def build_old(cfg):
    mc=cfg["model"]
    cond=oldpga.SCPGA(img_size=(H,W),dim=mc["dim"],patch_size=mc["patch_size"],J=mc.get("J",12),Kg=mc.get("Kg",4),
        Kt=mc.get("Kt",2),beta=mc.get("beta",40.0),eta=mc.get("eta",4.0),tau=mc.get("tau",1.0),w_min=mc.get("w_min",0.1),
        proj=mc.get("proj","v2"),dyn_off=mc.get("dyn_off",False),cond_mode=mc.get("cond_mode","secant_full"))
    return SCDiT(img_size=(H,W),patch_size=mc["patch_size"],data_channels=1,cond_channels=1,dim=mc["dim"],
        depth=mc["depth"],num_heads=mc["num_heads"],mlp_ratio=mc["mlp_ratio"],cond_module=cond,
        decode_head=mc.get("decode_head","conv"),xpre_mode=mc.get("xpre_mode","full")).to(dev)
for stem in ["scm_static","scm_point","scm_secant"]:
    cfg=load_config(os.path.expanduser(f"~/ScoliCMF/configs/{stem}.yaml"))
    mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])
    m=build_old(cfg)
    sd=torch.load(os.path.expanduser(f"~/ScoliCMF/runs/{stem}_pre_tokdiv/ckpts/step_5000.pt"),map_location="cpu")
    m.load_state_dict(sd["ema"]); m.eval()
    ds=PairedSpineDataset(root=os.path.expanduser(cfg["data"]["root"]),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
    ld=DataLoader(ds,batch_size=4,num_workers=2)
    acc={k:[] for k in ("ssim4","psnr4","lpips4")}
    with torch.no_grad():
        for xp,xq in ld:
            xp,xq=xp.to(dev),xq.to(dev); z=mf.sample(m,xp,steps=4)
            acc["ssim4"].append(ssim(z,xq).cpu().numpy()); acc["psnr4"].append(psnr(z,xq).cpu().numpy()); acc["lpips4"].append(lpips_fn(z,xq).cpu().numpy())
    r={k:np.concatenate(v) for k,v in acc.items()}
    print("%-11s cond_mode=%-11s | SSIM4=%.4f [%.4f,%.4f]  PSNR4=%.3f  LPIPS4=%.4f"%(
        stem,cfg["model"]["cond_mode"],*boot(r["ssim4"]),float(r["psnr4"].mean()),float(r["lpips4"].mean())))
print("SCM_REEVAL_DONE")
