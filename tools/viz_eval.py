#!/usr/bin/env python3
"""Batch-visualize a trained checkpoint on a split: per-patient
[preop | postop-GT | pred 1-NFE | pred 4-NFE] PNGs into an output folder."""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model, load_ckpt


def to_img(t):  # (1,H,W) [0,1] -> uint8 HxW
    return (t.squeeze(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/s5b_scpga_v2.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=os.path.expanduser("~/ScoliCMF/eval_viz"))
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    cfg = load_config(args.config)
    H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    m = load_ckpt(args.ckpt, cfg, H, W, None, dev, use_ema=True)
    mf = SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])
    sf = os.path.expanduser(f"~/ScoliCMF/splits/{args.split}.txt")
    ds = PairedSpineDataset(root=os.path.expanduser(cfg["data"]["root"]), size=(H, W),
                            split_file=sf, return_stem=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    sep = np.full((H, 4), 255, np.uint8)
    done = 0
    for x_pre, x_post, stem in loader:
        x_pre, x_post = x_pre.to(dev), x_post.to(dev)
        p1 = mf.sample(m, x_pre, steps=1)
        p4 = mf.sample(m, x_pre, steps=4)
        cols = [to_img(x_pre[0]), sep, to_img(x_post[0]), sep, to_img(p1[0]), sep, to_img(p4[0])]
        Image.fromarray(np.concatenate(cols, 1)).save(os.path.join(args.out, f"{stem[0]}.png"))
        done += 1
        if done >= args.n:
            break
    print(f"wrote {done} montages -> {args.out}  (cols: preop | postop-GT | 1-NFE | 4-NFE)")


if __name__ == "__main__":
    main()
