"""Unified ablation eval: for each config, sweep ckpts -> best-val (by val SSIM@4),
then full PSNR/SSIM/LPIPS + L1/cos @ NFE 1/4 on val+train with bootstrap CI."""
import os, sys, glob, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt
from metrics_img import ssim, psnr, lpips_fn

ROOT = None
def boot(x, B=2000, seed=0):
    r = np.random.default_rng(seed); s = np.array([x[r.integers(0,len(x),len(x))].mean() for _ in range(B)])
    return float(x.mean()), float(np.quantile(s,0.025)), float(np.quantile(s,0.975))

@torch.no_grad()
def evalset(model, mf, sf, dev, nfes, full=False):
    ds = PairedSpineDataset(root=ROOT, size=(H,W), split_file=sf)
    ld = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    acc = {}
    for s in nfes:
        for m in ("ssim","psnr","l1","cos") + (("lpips",) if full else ()):
            acc[f"{m}{s}"] = []
    for xp, xq in ld:
        xp, xq = xp.to(dev), xq.to(dev)
        for s in nfes:
            z = mf.sample(model, xp, steps=s)
            acc[f"ssim{s}"].append(ssim(z,xq).cpu().numpy())
            acc[f"psnr{s}"].append(psnr(z,xq).cpu().numpy())
            acc[f"l1{s}"].append((z-xq).abs().flatten(1).mean(1).cpu().numpy())
            dp=(z-xp).flatten(1); dt=(xq-xp).flatten(1)
            acc[f"cos{s}"].append(((dp*dt).sum(1)/(dp.norm(dim=1)*dt.norm(dim=1)+1e-8)).cpu().numpy())
            if full:
                acc[f"lpips{s}"].append(lpips_fn(z,xq).cpu().numpy())
    return {k: np.concatenate(v) for k,v in acc.items()}

def main():
    global H, W, ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="comma list of config stems")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for stem in a.configs.split(","):
        cfg = load_config(os.path.expanduser(f"~/ScoliCMF/configs/{stem}.yaml"))
        H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
        ROOT = os.path.expanduser(cfg["data"]["root"])
        mf = SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])
        ckdir = os.path.expanduser(f"~/ScoliCMF/runs/{stem}/ckpts")
        cks = sorted(glob.glob(os.path.join(ckdir,"step_*.pt")), key=lambda p:int(p.split("_")[-1].split(".")[0]))
        if not cks:
            print(f"== {stem}: NO CKPTS =="); continue
        # sweep best-val by val SSIM@4
        valf = os.path.expanduser("~/ScoliCMF/splits/val.txt")
        best=(-1,None)
        for ck in cks:
            m = load_ckpt(ck, cfg, H, W, None, dev, use_ema=True)
            r = evalset(m, mf, valf, dev, [4], full=False)
            sv = float(r["ssim4"].mean())
            if sv>best[0]: best=(sv,ck)
        bk = best[1]; st = int(bk.split("_")[-1].split(".")[0])
        m = load_ckpt(bk, cfg, H, W, None, dev, use_ema=True)
        print(f"\n######## {stem}  best-val=step_{st}  (val SSIM@4={best[0]:.4f}) ########")
        for sp,name in [("val.txt","val"),("train.txt","train")]:
            r = evalset(m, mf, os.path.expanduser(f"~/ScoliCMF/splits/{sp}"), dev, [1,4], full=True)
            print(f"[{name}]")
            for s in (1,4):
                for k in (f"ssim{s}",f"psnr{s}",f"lpips{s}",f"l1{s}",f"cos{s}"):
                    mu,lo,hi=boot(r[k]); print(f"   {k:8s} {mu:8.4f}  [{lo:.4f},{hi:.4f}]")
    print("\nEVAL_ABLATION_DONE")

if __name__ == "__main__":
    main()
