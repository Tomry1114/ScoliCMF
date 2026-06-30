"""Parametrized ADOC cleaner for ablations. Toggles geometry / photometric, and center-suppression
strength. Produces cleaned source-frame targets + reports acquisition-alignment vs central-change
preservation diagnostics. Used by Exp2 (geo/photo split) and Exp3 (center protection)."""
import os, sys, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset

dev = "cuda"; H, W = 480, 240
cfg = load_config(os.path.expanduser("~/ScoliCMF/configs/s2_base.yaml"))
ROOT = os.path.join(os.path.expanduser("~/ScoliCMF"), cfg["data"]["root"])
outdir = os.path.expanduser("~/ScoliCMF/runs/adoc"); os.makedirs(outdir, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--geo", type=int, default=1)
ap.add_argument("--photo", type=int, default=1)
ap.add_argument("--center", default="gauss")   # none|gauss|strong
ap.add_argument("--tag", required=True)
ap.add_argument("--iters", type=int, default=250)
a = ap.parse_args()

xcol = torch.linspace(0, 1, W, device=dev)
if a.center == "none":
    Wacq = torch.ones(1, 1, 1, W, device=dev)
elif a.center == "strong":
    Wacq = (1 - 0.95 * torch.exp(-((xcol - 0.5) ** 2) / (2 * 0.22 ** 2))).view(1, 1, 1, W)
else:  # gauss (default, == adoc_clean.py)
    Wacq = (1 - 0.8 * torch.exp(-((xcol - 0.5) ** 2) / (2 * 0.15 ** 2))).view(1, 1, 1, W)

# central region mask (where the spine / surgical change lives), inverse of periphery
cen = (xcol - 0.5).abs() < 0.15            # (W,)
cen_m = cen.view(1, 1, 1, W).float(); per_m = (~cen).view(1, 1, 1, W).float()

def huber(r, d=0.05):
    x = r.abs(); return torch.where(x < d, 0.5 * r * r / d, x - 0.5 * d)

def clean_chunk(xp, xq, iters):
    B = xp.shape[0]
    tp = torch.zeros(B, 5, device=dev, requires_grad=True)
    pp = torch.zeros(B, 3, device=dev, requires_grad=True)
    params = ([tp] if a.geo else []) + ([pp] if a.photo else [])
    opt = torch.optim.Adam(params, lr=0.02) if params else None
    def apply(tp, pp):
        if a.geo:
            dx = 0.12 * torch.tanh(tp[:, 0]); dy = 0.12 * torch.tanh(tp[:, 1])
            sx = torch.exp(0.12 * torch.tanh(tp[:, 2])); sy = torch.exp(0.12 * torch.tanh(tp[:, 3])); th = 0.20 * torch.tanh(tp[:, 4])
            theta = torch.stack([torch.stack([sx * torch.cos(th), -sx * torch.sin(th), dx], 1),
                                 torch.stack([sy * torch.sin(th), sy * torch.cos(th), dy], 1)], 1)
            grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
            xw = F.grid_sample(xq, grid, align_corners=False, padding_mode="border")
        else:
            xw = xq
        if a.photo:
            av = torch.exp(0.4 * torch.tanh(pp[:, 0])).view(B, 1, 1, 1); g = torch.exp(0.4 * torch.tanh(pp[:, 1])).view(B, 1, 1, 1); b = (0.12 * torch.tanh(pp[:, 2])).view(B, 1, 1, 1)
            xw = (av * xw.clamp_min(1e-4) ** g + b).clamp(0, 1)
        return xw
    if opt is not None:
        for it in range(iters):
            xw = apply(tp, pp); loss = (Wacq * huber(xp - xw)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return apply(tp, pp).cpu(), tp.detach().cpu(), pp.detach().cpu()

for split in ["train.txt", "val.txt"]:
    ds = PairedSpineDataset(root=ROOT, size=(H, W), split_file=os.path.expanduser("~/ScoliCMF/splits/%s" % split))
    XP = []; XQ = []
    for x, y in DataLoader(ds, batch_size=64, shuffle=False): XP.append(x); XQ.append(y)
    XP = torch.cat(XP); XQ = torch.cat(XQ); N = XP.shape[0]
    clean = []; tps = []; pps = []
    for i in range(0, N, 64):
        c, tp, pp = clean_chunk(XP[i:i + 64].to(dev), XQ[i:i + 64].to(dev), a.iters)
        clean.append(c); tps.append(tp); pps.append(pp)
    clean = torch.cat(clean); tps = torch.cat(tps); pps = torch.cat(pps)
    name = split.replace(".txt", "")
    torch.save({"clean": clean, "tp": tps, "pp": pps}, os.path.join(outdir, "clean_%s_%s.pt" % (a.tag, name)))
    # diagnostics (val is the one we report)
    per_c = per_m.cpu(); cen_c = cen_m.cpu()
    per_L1 = ((XP - clean).abs() * per_c).sum() / (per_c.sum() * N * H)
    cen_true = ((XP - XQ).abs() * cen_c).sum() / (cen_c.sum() * N * H)
    cen_kept = ((XP - clean).abs() * cen_c).sum() / (cen_c.sum() * N * H)
    ratio = (cen_kept / cen_true).item()
    print("[%s %s] geo=%d photo=%d center=%s | periphery_L1(clean,pre)=%.4f  central_change_true=%.4f kept=%.4f  PRESERVED=%.3f"
          % (a.tag, name, a.geo, a.photo, a.center, per_L1.item(), cen_true.item(), cen_kept.item(), ratio), flush=True)
print("VARIANTS_DONE tag=%s" % a.tag)
