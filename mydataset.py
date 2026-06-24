import os
from typing import List
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

class PairedImageDataset(Dataset):
    def __init__(self,
                 cond_root: str,
                 target_root: str,
                 transform_cond=None,
                 transform_target=None,
                 exts: List[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        super().__init__()
        self.cond_root = cond_root
        self.target_root = target_root
        self.transform_cond = transform_cond
        self.transform_target = transform_target
        
        # Filter files that exist in both directories
        self.filenames = [
            f for f in sorted(os.listdir(cond_root))
            if any(f.lower().endswith(e) for e in exts) and 
            os.path.exists(os.path.join(target_root, f))
        ]
        
        if not self.filenames:
            raise RuntimeError(f"No paired images found in {cond_root} and {target_root}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        # Open as Grayscale (L)
        cond_img = Image.open(os.path.join(self.cond_root, fname)).convert("L")
        target_img = Image.open(os.path.join(self.target_root, fname)).convert("L")

        if self.transform_cond:
            cond_img = self.transform_cond(cond_img)
        if self.transform_target:
            target_img = self.transform_target(target_img)

        return cond_img, target_img