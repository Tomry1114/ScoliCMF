"""ASW pre-gate: can a K-segment ARTICULATED (piecewise-affine, axially-blended) warp of x_pre
approximate x_post as well as a FREE dense warp? If yes -> the spine is well-represented by a
rigid chain -> ASW (structured low-dim warp) has material without an accuracy cost."""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from metrics_img import lpips_fn

dev = "cuda"; H, W = 480, 240
cfg = load_config(os.path.expanduser("~/ScoliCMF/configs/s2_base.yaml"))
ds = PairedSpineDataset(root=os.path.join(os.path.expanduser("~/ScoliCMF"), cfg["data"]["root"]),
                        size=(H, W), split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
xp = []; xq = []
for a, b in DataLoader(ds, batch_size=64): xp.append(a); xq.append(b)
xp = torch.cat(xp).to(dev); xq = torch.cat(xq).to(dev); B = xp.shape[0]
print("val pairs=%d" % B)

theta = torch.tensor([[1., 0, 0], [0, 1., 0]], device=dev).unsqueeze(0).expand(B, 2, 3)
base = F.affine_grid(theta, (B, 1, H, W), align_corners=False)             # (B,H,W,2) [x,y]
gxg = base[..., 0]; gyg = base[..., 1]                                     # (B,H,W)
yrow = torch.linspace(-1, 1, H, device=dev)
lp_pre = float(lpips_fn(xp.clamp(0, 1), xq).mean())

def free_dense():
    hf, wf = H // 8, W // 8
    phi = torch.zeros(B, 2, hf, wf, device=dev, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=0.05)
    for it in range(250):
        flow = F.interpolate(phi, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        warp = F.grid_sample(xp, base + flow, align_corners=False, padding_mode="border")
        tv = (phi[:, :, 1:] - phi[:, :, :-1]).abs().mean() + (phi[:, :, :, 1:] - phi[:, :, :, :-1]).abs().mean()
        loss = ((warp - xq) ** 2).mean() + 0.02 * tv
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        flow = F.interpolate(phi, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        warp = F.grid_sample(xp, base + flow, align_corners=False, padding_mode="border").clamp(0, 1)
    return warp

def articulated(K):
    cy = torch.linspace(-1, 1, K, device=dev)
    sigma = 2.0 / K
    w = torch.softmax(-((yrow[:, None] - cy[None, :]) ** 2) / (2 * sigma ** 2), dim=1)   # (H,K) per-row seg weight
    aff = torch.zeros(B, K, 2, 3, device=dev)
    aff[:, :, 0, 0] = 1; aff[:, :, 1, 1] = 1                                  # identity init
    aff.requires_grad_(True)
    opt = torch.optim.Adam([aff], lr=0.02)
    for it in range(400):
        nx = torch.zeros(B, H, W, device=dev); ny = torch.zeros(B, H, W, device=dev)
        for k in range(K):
            A = aff[:, k]
            xpr = A[:, 0, 0].view(B, 1, 1) * gxg + A[:, 0, 1].view(B, 1, 1) * gyg + A[:, 0, 2].view(B, 1, 1)
            ypr = A[:, 1, 0].view(B, 1, 1) * gxg + A[:, 1, 1].view(B, 1, 1) * gyg + A[:, 1, 2].view(B, 1, 1)
            wk = w[:, k].view(1, H, 1)
            nx = nx + wk * xpr; ny = ny + wk * ypr
        grid = torch.stack([nx, ny], -1)
        warp = F.grid_sample(xp, grid, align_corners=False, padding_mode="border")
        smooth = ((aff[:, 1:] - aff[:, :-1]) ** 2).mean()
        loss = ((warp - xq) ** 2).mean() + 0.01 * smooth
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        nx = torch.zeros(B, H, W, device=dev); ny = torch.zeros(B, H, W, device=dev)
        for k in range(K):
            A = aff[:, k]
            xpr = A[:, 0, 0].view(B, 1, 1) * gxg + A[:, 0, 1].view(B, 1, 1) * gyg + A[:, 0, 2].view(B, 1, 1)
            ypr = A[:, 1, 0].view(B, 1, 1) * gxg + A[:, 1, 1].view(B, 1, 1) * gyg + A[:, 1, 2].view(B, 1, 1)
            nx = nx + w[:, k].view(1, H, 1) * xpr; ny = ny + w[:, k].view(1, H, 1) * ypr
        warp = F.grid_sample(xp, torch.stack([nx, ny], -1), align_corners=False, padding_mode="border").clamp(0, 1)
    return warp

def report(name, warp):
    R = xq - warp
    rf = float((R ** 2).sum() / ((xq - xp) ** 2).sum().clamp_min(1e-9))
    lp = float(lpips_fn(warp, xq).mean())
    print("%-16s residual_frac=%.4f  LPIPS=%.4f" % (name, rf, lp))

print("no-op (x_pre)    residual_frac=1.0000  LPIPS=%.4f" % lp_pre)
report("free-dense-warp", free_dense())
for K in [4, 8, 12, 17]:
    report("articulated K=%d" % K, articulated(K))
print("ASW_GATE_DONE")
