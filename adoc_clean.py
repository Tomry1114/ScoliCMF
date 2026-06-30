"""ADOC corrector (per-pair restricted optimization form): align observed x_post into the x_pre
SOURCE frame using ONLY restricted acquisition transforms (small affine + photometric), with a
CENTER-SUPPRESSED Huber loss so the central spine / surgical correction is NOT fit (alignment
relies on periphery: torso/ribs/pelvis/borders). Output = cleaned source-frame target x_tilde_post.
Saves cleaned targets in dataset (sorted-stem) order for train+val."""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset

dev = "cuda"; H, W = 480, 240
cfg = load_config(os.path.expanduser("~/ScoliCMF/configs/s2_base.yaml"))
ROOT = os.path.join(os.path.expanduser("~/ScoliCMF"), cfg["data"]["root"])
outdir = os.path.expanduser("~/ScoliCMF/runs/adoc"); os.makedirs(outdir, exist_ok=True)

# center-suppression weight over columns x in [0,1]: spine is vertical near x=0.5
xcol = torch.linspace(0, 1, W, device=dev)
Wacq = (1 - 0.8 * torch.exp(-((xcol - 0.5) ** 2) / (2 * 0.15 ** 2))).view(1, 1, 1, W)   # (1,1,1,W)
def huber(r, d=0.05): a = r.abs(); return torch.where(a < d, 0.5 * r * r / d, a - 0.5 * d)

def clean_chunk(xp, xq, iters=250):
    B = xp.shape[0]
    tp = torch.zeros(B, 5, device=dev, requires_grad=True)   # dx,dy,lsx,lsy,theta
    pp = torch.zeros(B, 3, device=dev, requires_grad=True)   # la,lg,b
    opt = torch.optim.Adam([tp, pp], lr=0.02)
    def apply(tp, pp):
        dx = 0.12 * torch.tanh(tp[:, 0]); dy = 0.12 * torch.tanh(tp[:, 1])
        sx = torch.exp(0.12 * torch.tanh(tp[:, 2])); sy = torch.exp(0.12 * torch.tanh(tp[:, 3])); th = 0.20 * torch.tanh(tp[:, 4])
        theta = torch.stack([torch.stack([sx * torch.cos(th), -sx * torch.sin(th), dx], 1),
                             torch.stack([sy * torch.sin(th), sy * torch.cos(th), dy], 1)], 1)
        grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
        xw = F.grid_sample(xq, grid, align_corners=False, padding_mode="border")
        a = torch.exp(0.4 * torch.tanh(pp[:, 0])).view(B, 1, 1, 1); g = torch.exp(0.4 * torch.tanh(pp[:, 1])).view(B, 1, 1, 1); b = (0.12 * torch.tanh(pp[:, 2])).view(B, 1, 1, 1)
        return (a * xw.clamp_min(1e-4) ** g + b).clamp(0, 1)
    for it in range(iters):
        xw = apply(tp, pp)
        loss = (Wacq * huber(xp - xw)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return apply(tp, pp).cpu(), tp.detach().cpu(), pp.detach().cpu()

for split in ["train.txt", "val.txt"]:
    ds = PairedSpineDataset(root=ROOT, size=(H, W), split_file=os.path.expanduser("~/ScoliCMF/splits/%s" % split))
    XP = []; XQ = []
    for a, b in DataLoader(ds, batch_size=64, shuffle=False): XP.append(a); XQ.append(b)
    XP = torch.cat(XP); XQ = torch.cat(XQ); N = XP.shape[0]
    clean = []; tps = []; pps = []
    for i in range(0, N, 64):
        c, tp, pp = clean_chunk(XP[i:i + 64].to(dev), XQ[i:i + 64].to(dev))
        clean.append(c); tps.append(tp); pps.append(pp)
    clean = torch.cat(clean); tps = torch.cat(tps); pps = torch.cat(pps)
    name = split.replace(".txt", "")
    torch.save({"clean": clean, "tp": tps, "pp": pps}, os.path.join(outdir, "clean_%s.pt" % name))
    moved = (clean - XQ).abs().mean().item()
    dorig = (XP - XQ).abs().mean().item(); dclean = (XP - clean).abs().mean().item()
    print("[%s] N=%d  mean|x_tilde-x_post|=%.4f  L1(pre,post)=%.4f -> L1(pre,clean)=%.4f  (center-suppressed)" % (name, N, moved, dorig, dclean), flush=True)
print("ADOC_CLEAN_DONE")
