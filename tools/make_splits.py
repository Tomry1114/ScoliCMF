#!/usr/bin/env python3
"""Deterministic patient-level train/val/test split of paired stems.

Each stem = one patient pair (one preop+postop), so stem-level == patient-level (no
intra-patient leakage). Writes splits/{train,val,test}.txt. Seeded & reproducible.
"""
import os
import glob
import random
import argparse

ROOT = os.path.expanduser("~/ScoliCMF/data/Spine生成_Miccai数据集")
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def stems(d):
    return {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(d, "*"))
            if os.path.splitext(f)[1].lower() in EXTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=os.path.expanduser("~/ScoliCMF/splits"))
    ap.add_argument("--seed", type=int, default=1114)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    a = ap.parse_args()
    common = sorted(stems(os.path.join(a.root, "train", "preop_standardized"))
                    & stems(os.path.join(a.root, "train", "postop_standardized")))
    random.Random(a.seed).shuffle(common)
    n = len(common)
    nv, nt = int(n * a.val), int(n * a.test)
    parts = {"test": common[:nt], "val": common[nt:nt + nv], "train": common[nt + nv:]}
    os.makedirs(a.out, exist_ok=True)
    for k, v in parts.items():
        with open(os.path.join(a.out, k + ".txt"), "w") as f:
            f.write("\n".join(sorted(v)) + "\n")
        print(f"{k}: {len(v)}")
    print(f"total={n} seed={a.seed} -> {a.out}/{{train,val,test}}.txt")


if __name__ == "__main__":
    main()
