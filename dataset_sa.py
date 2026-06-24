"""Source-anchored spine pair dataset (S1).

Extension-agnostic stem matching (fixes the 76 jpg/png mismatched pairs that the
original mydataset.py drops via exact-filename matching). Loads ONLY
preop_standardized + postop_standardized (no curves). Pixel-space for S1/S2;
swap `canon_dir` to a canonicalized cache once tools/canonicalize.py is run.
"""
import os
import glob
from typing import Optional, Tuple

import torch  # noqa: F401  (kept for downstream type/use)
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

DEFAULT_ROOT = os.path.expanduser("~/ScoliCMF/data/Spine生成_Miccai数据集")
_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _stem_map(d: str):
    """stem -> filepath, extension-agnostic (last one wins if dup stem)."""
    return {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in sorted(glob.glob(os.path.join(d, "*")))
        if os.path.splitext(f)[1].lower() in _EXTS
    }


class PairedSpineDataset(Dataset):
    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        split: str = "train",
        size: Tuple[int, int] = (480, 240),   # (H, W) -> 240x480 portrait
        canon_dir: Optional[str] = None,
        return_stem: bool = False,
    ):
        super().__init__()
        self.return_stem = return_stem
        if canon_dir is not None:
            pre_d = os.path.join(canon_dir, "preop")
            post_d = os.path.join(canon_dir, "postop")
        else:
            pre_d = os.path.join(root, split, "preop_standardized")
            post_d = os.path.join(root, split, "postop_standardized")
        self.pre = _stem_map(pre_d)
        self.post = _stem_map(post_d)
        self.stems = sorted(set(self.pre) & set(self.post))
        if not self.stems:
            raise RuntimeError(f"No paired stems found in {pre_d} / {post_d}")
        self.tf = T.Compose([T.Resize(size), T.ToTensor()])  # -> [1,H,W] in [0,1]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        x_pre = self.tf(Image.open(self.pre[s]).convert("L"))
        x_post = self.tf(Image.open(self.post[s]).convert("L"))
        if self.return_stem:
            return x_pre, x_post, s
        return x_pre, x_post


if __name__ == "__main__":
    ds = PairedSpineDataset(return_stem=True)
    print("paired stems:", len(ds))
    xp, xq, s = ds[0]
    print(f"sample stem={s} x_pre={tuple(xp.shape)} range=[{xp.min():.3f},{xp.max():.3f}]"
          f" x_post={tuple(xq.shape)}")
    exts = {os.path.splitext(ds.pre[k])[1].lower() for k in ds.stems}
    n_jpg = sum(os.path.splitext(ds.pre[k])[1].lower() == ".jpg" for k in ds.stems)
    print("preop exts present:", exts, "| preop .jpg count:", n_jpg)
