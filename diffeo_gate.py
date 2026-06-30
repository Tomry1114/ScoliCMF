"""Diffeomorphic transport pre-gate. Does a DIFFEOMORPHIC warp (stationary velocity field +
scaling-and-squaring -> guaranteed invertible / no folding) fit x_post as well as a FREE
displacement warp? And is its LPIPS cleaner (no fold artifacts)? If diffeo ~ free on residual
AND >= free on LPIPS -> deformation-space (diffeomorphic) transport is the right sharp+regular
model -> worth building the Diffeomorphic Mean-Flow."""
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
theta = torch.tensor([[1., 0, 0], [0, 1., 0]], device=dev).unsqueeze(0).expand(B, 2, 3)
base = F.affine_grid(theta, (B, 1, H, W), align_corners=False)             # (B,H,W,2)
print("val pairs=%d" % B)

def report(name, warp):
    R = xq - warp
    rf = float((R ** 2).sum() / ((xq - xp) ** 2).sum().clamp_min(1e-9))
    print("%-18s residual_frac=%.4f  LPIPS=%.4f" % (name, rf, float(lpips_fn(warp, xq).mean())))

# ---- free displacement warp (reference) ----
def free_dense():
    hf, wf = H // 8, W // 8
    phi = torch.zeros(B, 2, hf, wf, device=dev, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=0.05)
    for it in range(300):
        flow = F.interpolate(phi, (H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        warp = F.grid_sample(xp, base + flow, align_corners=False, padding_mode="border")
        tv = (phi[:, :, 1:] - phi[:, :, :-1]).abs().mean() + (phi[:, :, :, 1:] - phi[:, :, :, :-1]).abs().mean()
        loss = ((warp - xq) ** 2).mean() + 0.02 * tv
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        flow = F.interpolate(phi, (H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        return F.grid_sample(xp, base + flow, align_corners=False, padding_mode="border").clamp(0, 1), phi.detach()

# ---- diffeomorphic warp: SVF + scaling-and-squaring ----
def compose(phi):                                              # phi:(B,2,H,W) displacement in grid coords
    grid = base + phi.permute(0, 2, 3, 1)
    warped = F.grid_sample(phi, grid, align_corners=False, padding_mode="border")
    return phi + warped
def integrate(v, nsq=6):
    phi = v / (2 ** nsq)
    for _ in range(nsq): phi = compose(phi)
    return phi
def diffeo(nsq=6):
    hf, wf = H // 8, W // 8
    vlow = torch.zeros(B, 2, hf, wf, device=dev, requires_grad=True)
    opt = torch.optim.Adam([vlow], lr=0.05)
    for it in range(300):
        v = F.interpolate(vlow, (H, W), mode="bilinear", align_corners=False)
        phi = integrate(v, nsq)
        warp = F.grid_sample(xp, base + phi.permute(0, 2, 3, 1), align_corners=False, padding_mode="border")
        tv = (vlow[:, :, 1:] - vlow[:, :, :-1]).abs().mean() + (vlow[:, :, :, 1:] - vlow[:, :, :, :-1]).abs().mean()
        loss = ((warp - xq) ** 2).mean() + 0.02 * tv
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        v = F.interpolate(vlow, (H, W), mode="bilinear", align_corners=False)
        phi = integrate(v, nsq)
        warp = F.grid_sample(xp, base + phi.permute(0, 2, 3, 1), align_corners=False, padding_mode="border").clamp(0, 1)
        # folding check: Jacobian determinant of (id+phi); count negatives
        jx = phi[:, 0]; jy = phi[:, 1]
        dxx = (jx[:, :, 2:] - jx[:, :, :-2]); dyy = (jy[:, 2:] - jy[:, :-2])
        return warp, float((dxx[:, 1:-1] < -1.0).float().mean())   # crude fold proxy
fw, fphi = free_dense()
with torch.no_grad():
    jx = fphi; # free fold proxy on low-res
    fold_free = float(((fphi[:, 0, :, 2:] - fphi[:, 0, :, :-2]) < -0.5).float().mean())
report("free-displacement", fw)
print("   free-warp fold-ish frac (low-res, crude) = %.4f" % fold_free)
dw, fold_d = diffeo(6)
report("diffeomorphic SVF", dw)
print("   diffeo fold frac (should be ~0) = %.4f" % fold_d)
print("DIFFEO_GATE_DONE")
