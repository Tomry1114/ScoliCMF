"""Eval original baseline with the SAME metrics_img on the SAME val/train split as s5b.
Reports SSIM/PSNR/L1 at NFE 1/4/20 (few-NFE = my methods selling point; 20 = original default)."""
import os, sys, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
sys.path.insert(0, os.path.dirname(__file__))
from dataset_sa import PairedSpineDataset
from metrics_img import ssim, psnr
from baseline_orig import MFDiT_orig, MeanFlowOrig

ROOT = os.path.expanduser("~/ScoliCMF/data/Spine生成_Miccai数据集")

def boot(x, B=2000, seed=0):
    r = np.random.default_rng(seed); s = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return float(x.mean()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))

@torch.no_grad()
def run(model, mf, loader, dev, n, nfes):
    acc = {}
    for s in nfes:
        acc[f"ssim{s}"] = []; acc[f"psnr{s}"] = []; acc[f"l1_{s}"] = []
    seen = 0
    for x_pre, x_post in loader:
        x_pre, x_post = x_pre.to(dev), x_post.to(dev)
        for s in nfes:
            z = mf.sample_given_cond(model, x_pre, sample_steps=s)
            acc[f"ssim{s}"].append(ssim(z, x_post).cpu().numpy())
            acc[f"psnr{s}"].append(psnr(z, x_post).cpu().numpy())
            acc[f"l1_{s}"].append((z - x_post).abs().flatten(1).mean(1).cpu().numpy())
        seen += x_pre.shape[0]
        if seen >= n: break
    return {k: np.concatenate(v) for k, v in acc.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--splits", default="train,val")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--nfes", default="1,4,20")
    ap.add_argument("--use_ema", type=int, default=1)
    a = ap.parse_args()
    nfes = [int(x) for x in a.nfes.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = 480, 240
    model = MFDiT_orig(img_size=(H, W), patch_size=8, data_channels=1, cond_channels=1,
                       dim=384, depth=12, num_heads=6).to(dev)
    sd = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(sd["ema" if a.use_ema else "model"]); model.eval()
    mf = MeanFlowOrig(channels=1, flow_ratio=0.75)
    print(f"ckpt={os.path.basename(a.ckpt)} use_ema={a.use_ema}  (higher SSIM/PSNR better; lower L1 better)")
    for sp in a.splits.split(","):
        sf = os.path.expanduser(f"~/ScoliCMF/splits/{sp}.txt")
        ds = PairedSpineDataset(root=ROOT, size=(H, W), split_file=sf)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
        r = run(model, mf, loader, dev, a.n, nfes)
        print(f"\n[{sp}] n={min(a.n, len(ds))}")
        for s in nfes:
            for k in (f"ssim{s}", f"psnr{s}", f"l1_{s}"):
                mu, lo, hi = boot(r[k]); print(f"   {k:8s} {mu:8.4f}  95%CI=[{lo:.4f},{hi:.4f}]")

if __name__ == "__main__":
    main()
