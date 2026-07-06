"""Sample I2SB val predictions: for each val preop (x1), run reverse bridge -> postop (x0 est).
Saves grayscale preds/<stem>.png. Reuses Runner (loads results/scoli/latest.pt via --ckpt scoli)."""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.backends.cudnn.enabled = False
from PIL import Image
from torch.utils.data import DataLoader
from logger import Logger
from i2sb import Runner, download_ckpt
from train import create_training_options
from dataset.scoli_paired import PairedScoli

opt = create_training_options()
opt.corrupt = "mixture"
download_ckpt("data/")
torch.cuda.set_device(0)
opt.device = "cuda"; opt.global_rank = 0; opt.local_rank = 0; opt.global_size = 1; opt.distributed = False
log = Logger(0, opt.log_dir)
run = Runner(opt, log)               # loads finetuned net from results/scoli/latest.pt (--ckpt scoli)
run.net.eval()
OUT = os.path.expanduser("~/ScoliCMF/compare_exp/02_i2sb/preds"); os.makedirs(OUT, exist_ok=True)
ds = PairedScoli("val", opt.image_size)
loader = DataLoader(ds, batch_size=6, shuffle=False)
idx = 0; n = 0
with torch.no_grad():
    for x0, x1, y in loader:
        x1 = x1.to(opt.device); cond = x1 if opt.cond_x1 else None
        xs, pred_x0 = run.ddpm_sampling(opt, x1, mask=None, cond=cond, clip_denoise=True, nfe=100, verbose=False)
        final = ((pred_x0[:, -1].clamp(-1, 1) + 1) / 2)          # (b,3,256,256) -> [0,1]
        for j in range(final.shape[0]):
            stem = ds.stems[idx]; idx += 1
            g = final[j].mean(0).cpu().numpy()                   # 3ch -> gray
            Image.fromarray((g * 255).astype(np.uint8), mode="L").save(os.path.join(OUT, stem + ".png")); n += 1
        log.info(f"sampled {n}/{len(ds)}")
print(f"I2SB_SAMPLED {n} preds -> {OUT}", flush=True)
