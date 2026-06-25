"""Image-quality eval: SSIM / PSNR / L1 of 1-NFE & 4-NFE predictions vs GT, per split.
Reports mean + 95% bootstrap CI. Diagnoses overfitting honestly (train vs val)."""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model, load_ckpt
from metrics_img import ssim, psnr


def boot(x, B=2000, seed=0):
    r = np.random.default_rng(seed); s = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return float(x.mean()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


@torch.no_grad()
def run(model, mf, loader, dev, n):
    acc = {k: [] for k in ("ssim1", "ssim4", "psnr1", "psnr4", "l1_1", "l1_4", "cos1")}
    seen = 0
    for x_pre, x_post in loader:
        x_pre, x_post = x_pre.to(dev), x_post.to(dev)
        z1 = mf.sample(model, x_pre, steps=1); z4 = mf.sample(model, x_pre, steps=4)
        acc["ssim1"].append(ssim(z1, x_post).cpu().numpy()); acc["ssim4"].append(ssim(z4, x_post).cpu().numpy())
        acc["psnr1"].append(psnr(z1, x_post).cpu().numpy()); acc["psnr4"].append(psnr(z4, x_post).cpu().numpy())
        acc["l1_1"].append((z1 - x_post).abs().flatten(1).mean(1).cpu().numpy())
        acc["l1_4"].append((z4 - x_post).abs().flatten(1).mean(1).cpu().numpy())
        dp = (z1 - x_pre).flatten(1); dt = (x_post - x_pre).flatten(1)
        cos = (dp * dt).sum(1) / (dp.norm(dim=1) * dt.norm(dim=1) + 1e-8)
        acc["cos1"].append(cos.cpu().numpy())   # direction of 1-step toward true delta (no-leak)
        seen += x_pre.shape[0]
        if seen >= n:
            break
    return {k: np.concatenate(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", default="configs/s5b_scpga_v2.yaml")
    ap.add_argument("--splits", default="train,val"); ap.add_argument("--n", type=int, default=80)
    a = ap.parse_args()
    cfg = load_config(a.config); H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_ckpt(a.ckpt, cfg, H, W, None, dev, use_ema=True)
    mf = SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])
    print(f"ckpt={os.path.basename(a.ckpt)}  (higher SSIM/PSNR better; lower L1 better)")
    for sp in a.splits.split(","):
        sf = os.path.expanduser(f"~/ScoliCMF/splits/{sp}.txt")
        ds = PairedSpineDataset(root=os.path.expanduser(cfg["data"]["root"]), size=(H, W), split_file=sf)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
        r = run(m, mf, loader, dev, a.n)
        print(f"\n[{sp}] n={min(a.n, len(ds))}")
        for k in ("ssim1", "ssim4", "psnr1", "psnr4", "l1_1", "l1_4", "cos1"):
            mu, lo, hi = boot(r[k]); print(f"   {k:7s} {mu:8.4f}  95%CI=[{lo:.4f},{hi:.4f}]")


if __name__ == "__main__":
    main()
