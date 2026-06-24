#!/usr/bin/env python3
"""S1 prepare + QA for source-anchored pixel pipeline (numpy+PIL only).

Finding (R-Stage1, 2026-06-24): the *_standardized images are already
centroid-aligned (preop/postop |Δcx|~0.013, both ~centered, near-full vertical
span), but the bony thorax is NOT registered (global pose/scale differs; postop
carries surgical hardware). The Stage-0 CASE2 crop concern was about the CURVES,
not these images. Non-rigid registration is out of scope (doc forbids it) and the
curve-based canonicalization needs curves we dropped -> geometric registration is
DEFERRED (revisit only if S2 shows global pose dominates). This tool therefore does
light intensity normalization (optional) + a QA montage, not a geometric warp.
"""
import os
import glob
import argparse
import numpy as np
from PIL import Image

ROOT = os.path.expanduser("~/ScoliCMF/data/Spine生成_Miccai数据集")
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
H, W = 480, 240


def stem_map(d):
    return {os.path.splitext(os.path.basename(f))[0]: f
            for f in sorted(glob.glob(os.path.join(d, "*")))
            if os.path.splitext(f)[1].lower() in EXTS}


def load(f):
    return np.asarray(Image.open(f).convert("L").resize((W, H)), dtype=np.float32)


def percentile_norm(a, lo=1.0, hi=99.0):
    plo, phi = np.percentile(a, lo), np.percentile(a, hi)
    if phi - plo < 1e-6:
        return np.clip(a, 0, 255)
    return np.clip((a - plo) / (phi - plo), 0, 1) * 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None, help="if set, write normalized preop/postop png cache here")
    ap.add_argument("--norm", choices=["none", "percentile"], default="none")
    ap.add_argument("--montage", default=os.path.expanduser("~/ScoliCMF/doc/stage1_montage.png"))
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    pre = stem_map(os.path.join(args.root, args.split, "preop_standardized"))
    post = stem_map(os.path.join(args.root, args.split, "postop_standardized"))
    stems = sorted(set(pre) & set(post))
    n_jpg = sum(pre[s].lower().endswith(".jpg") for s in stems)
    print(f"[contract] paired stems={len(stems)} (preop .jpg={n_jpg}); "
          f"preop-only={len(set(pre)-set(post))} postop-only={len(set(post)-set(pre))}")

    norm = (lambda a: percentile_norm(a)) if args.norm == "percentile" else (lambda a: a)

    if args.out:
        for sub, mp in (("preop", pre), ("postop", post)):
            od = os.path.join(args.out, sub)
            os.makedirs(od, exist_ok=True)
            for s in stems:
                Image.fromarray(norm(load(mp[s])).astype(np.uint8)).save(os.path.join(od, f"{s}.png"))
        print(f"[cache] wrote normalized pairs -> {args.out}/{{preop,postop}} (norm={args.norm})")

    # QA montage: rows of [preop | postop | absdiff], mix of png + jpg-preop stems
    jpg = [s for s in stems if pre[s].lower().endswith(".jpg")]
    pick = ([stems[0], stems[len(stems)//3], stems[2*len(stems)//3]]
            + ([jpg[0], jpg[len(jpg)//2], jpg[-1]] if jpg else []))[:args.n]
    rows = []
    for s in pick:
        a, b = norm(load(pre[s])), norm(load(post[s]))
        rows.append(np.concatenate([a, b, np.abs(b - a)], axis=1))
    grid = np.clip(np.concatenate(rows, axis=0), 0, 255).astype(np.uint8)
    Image.fromarray(grid).save(args.montage)
    print(f"[montage] {args.montage} shape={grid.shape} cols=preop|postop|absdiff rows={pick}")


if __name__ == "__main__":
    main()
