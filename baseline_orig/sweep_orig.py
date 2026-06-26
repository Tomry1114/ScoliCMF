import os, sys, numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
sys.path.insert(0, os.path.dirname(__file__))
from dataset_sa import PairedSpineDataset
from metrics_img import ssim, psnr
from baseline_orig import MFDiT_orig, MeanFlowOrig
ROOT = os.path.expanduser("~/ScoliCMF/data/Spine生成_Miccai数据集")
dev, H, W = "cuda", 480, 240
mf = MeanFlowOrig(channels=1, flow_ratio=0.75)
def evalset(model, sf, nfes):
    ds = PairedSpineDataset(root=ROOT, size=(H, W), split_file=sf)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    out = {s: {"ssim": [], "psnr": [], "l1": []} for s in nfes}
    with torch.no_grad():
        for xp, xq in ld:
            xp, xq = xp.to(dev), xq.to(dev)
            for s in nfes:
                z = mf.sample_given_cond(model, xp, sample_steps=s)
                out[s]["ssim"].append(ssim(z, xq).cpu().numpy())
                out[s]["psnr"].append(psnr(z, xq).cpu().numpy())
                out[s]["l1"].append((z - xq).abs().flatten(1).mean(1).cpu().numpy())
    return {s: {k: float(np.concatenate(v).mean()) for k, v in d.items()} for s, d in out.items()}
steps = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 16000]
valf = os.path.expanduser("~/ScoliCMF/splits/val.txt")
print("  step |  vSSIM@4  vSSIM@20  vPSNR@20  vL1@20")
best = (-1.0, None)
for st in steps:
    ck = os.path.expanduser("~/ScoliCMF/runs/orig_baseline/ckpts/step_%d.pt" % st)
    if not os.path.exists(ck):
        continue
    m = MFDiT_orig(img_size=(H, W), patch_size=8, dim=384, depth=12, num_heads=6).to(dev)
    m.load_state_dict(torch.load(ck, map_location="cpu")["ema"]); m.eval()
    r = evalset(m, valf, [4, 20])
    s4, s20, p20, l20 = r[4]["ssim"], r[20]["ssim"], r[20]["psnr"], r[20]["l1"]
    print("%6d | %8.4f %9.4f %9.3f %7.4f" % (st, s4, s20, p20, l20), flush=True)
    if s20 > best[0]:
        best = (s20, st)
print("BEST_VAL_SSIM@20 = %.4f at step_%s" % best)
