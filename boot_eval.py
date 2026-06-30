"""Bootstrap CIs for the two main comparisons (per-case, paired). Inference only.
Table1 (RAW frame): APTD@2000 vs Bridge(s2_base).  Table2 (CANONICAL frame): ADOC-APTD@5000 vs raw-APTD@5000."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model, load_ckpt
from metrics_img import ssim, lpips_fn
from aptd_model import APTDNet
dev = "cuda"; H, W = 480, 240; HOME = os.path.expanduser("~/ScoliCMF")
def cfgfull():
    c = load_config(os.path.join(HOME, "configs/s2_base.yaml")); c["model"]["xpre_mode"] = "full"; return c
def cfgbridge(): return load_config(os.path.join(HOME, "configs/s2_base.yaml"))
mf = SourceAnchoredMeanFlow(gamma=cfgbridge()["meanflow"]["gamma"], sigma_m=cfgbridge()["meanflow"]["sigma_m"]); path = mf.path
def psnr(a, b): return -10 * torch.log10(((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10))
# data in order
dsv = PairedSpineDataset(root=os.path.join(HOME, cfgbridge()["data"]["root"]), size=(H, W), split_file=os.path.join(HOME, "splits/val.txt"))
XP = []; XQ = []
for a, b in DataLoader(dsv, batch_size=64, shuffle=False): XP.append(a); XQ.append(b)
xpva = torch.cat(XP); rawva = torch.cat(XQ); clva = torch.load(os.path.join(HOME, "runs/adoc/clean_val.pt"))["clean"]

@torch.no_grad()
def percase_aptd(ckpt, target, fs=0.15, use_ema=True):
    bb = build_model(cfgfull(), H, W).to(dev); m = APTDNet(bb, "warpres", flow_scale=fs).to(dev)
    st = torch.load(ckpt, map_location=dev)
    if use_ema and "ema" in st:
        for p, e in zip(m.parameters(), st["ema"]): p.data.copy_(e.to(dev))
    else:
        m.load_state_dict(st["model"])
    m.eval(); S = []; P = []; L = []
    for i in range(0, xpva.shape[0], 6):
        xp = xpva[i:i+6].to(dev); xq = target[i:i+6].to(dev); B = xp.shape[0]
        r0 = torch.zeros(B, device=dev); t1 = torch.ones(B, device=dev)
        o = m(xp, r0, t1, xp)["xhat"].clamp(0, 1)
        S.append(ssim(o, xq).cpu()); P.append(psnr(o, xq).cpu()); L.append(lpips_fn(o, xq).cpu())
    return torch.cat(S), torch.cat(P), torch.cat(L)

@torch.no_grad()
def percase_bridge(ckpt, target):
    m = load_ckpt(ckpt, cfgbridge(), H, W, None, dev, use_ema=True)
    S = []; P = []; L = []
    for i in range(0, xpva.shape[0], 6):
        xp = xpva[i:i+6].to(dev); xq = target[i:i+6].to(dev)
        o = mf.sample(m, xp, steps=4)
        S.append(ssim(o, xq).cpu()); P.append(psnr(o, xq).cpu()); L.append(lpips_fn(o, xq).cpu())
    return torch.cat(S), torch.cat(P), torch.cat(L)

g = torch.Generator().manual_seed(0); n = xpva.shape[0]; idx = torch.randint(0, n, (2000, n), generator=g)
def ci(x):
    bs = x[idx].mean(1); return float(x.mean()), float(bs.quantile(0.025)), float(bs.quantile(0.975))
def cidiff(a, b):
    bs = (a[idx] - b[idx]).mean(1); return float((a - b).mean()), float(bs.quantile(0.025)), float(bs.quantile(0.975))
def line(tag, x): m, lo, hi = ci(x); print("   %-26s %.4f [%.4f, %.4f]" % (tag, m, lo, hi))
def dline(tag, a, b): m, lo, hi = cidiff(a, b); sig = "EXCLUDES 0 (sig)" if (lo > 0 or hi < 0) else "includes 0"; print("   %-26s d=%+.4f [%+.4f, %+.4f]  %s" % (tag, m, lo, hi, sig))

print("======== TABLE 1: APTD@2000 vs Bridge (RAW val frame) ========")
bS, bP, bL = percase_bridge(os.path.join(HOME, "runs/s2_base/ckpts/step_5000.pt"), rawva)
aS, aP, aL = percase_aptd(os.path.join(HOME, "runs/aptd_long_fs015/ckpts/step_2000.pt"), rawva)
for nm, x in [("Bridge SSIM", bS), ("APTD SSIM", aS), ("Bridge PSNR", bP), ("APTD PSNR", aP), ("Bridge LPIPS", bL), ("APTD LPIPS", aL)]: line(nm, x)
dline("dSSIM (APTD-Bridge)", aS, bS); dline("dPSNR", aP, bP); dline("dLPIPS (APTD-Bridge)", aL, bL)

print("======== TABLE 2: ADOC-APTD@5000 vs raw-APTD@5000 (CANONICAL/cleaned val) ========")
rS, rP, rL = percase_aptd(os.path.join(HOME, "runs/aptd_long_fs015/ckpts/step_5000.pt"), clva, use_ema=False)
cS, cP, cL = percase_aptd(os.path.join(HOME, "runs/aptd_adoc/ckpts/step_5000.pt"), clva, use_ema=False)
for nm, x in [("raw-APTD SSIM", rS), ("ADOC SSIM", cS), ("raw-APTD PSNR", rP), ("ADOC PSNR", cP), ("raw-APTD LPIPS", rL), ("ADOC LPIPS", cL)]: line(nm, x)
dline("dSSIM (ADOC-raw)", cS, rS); dline("dPSNR", cP, rP); dline("dLPIPS (ADOC-raw)", cL, rL)
print("BOOT_DONE")
