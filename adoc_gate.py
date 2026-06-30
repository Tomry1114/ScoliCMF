"""ADOC pre-gate (honest, RESTRICTED only -- NO free dense registration).
Does aligning the OBSERVED post-op x_post to the APTD prediction x_hat via ONLY restricted
acquisition transforms (small affine + photometric) reduce the error? If a small restricted
alignment improves LPIPS>~0.02 or SSIM>~0.01 (majority consistent) -> acquisition factors
occupy real error -> cleaning the supervision target (ADOC) is worth building."""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from eval_gates import build_model
from metrics_img import ssim, lpips_fn
from aptd_model import APTDNet

dev = "cuda"; H, W = 480, 240
cfg = load_config(os.path.expanduser("~/ScoliCMF/configs/s2_base.yaml"))
cfg["model"]["xpre_mode"] = "full"
bb = build_model(cfg, H, W).to(dev)
model = APTDNet(bb, "warpres", flow_scale=0.15).to(dev)
sd = torch.load(os.path.expanduser("~/ScoliCMF/runs/aptd_long_fs015/ckpts/step_2000.pt"), map_location=dev)["model"]
model.load_state_dict(sd); model.eval()
ds = PairedSpineDataset(root=os.path.join(os.path.expanduser("~/ScoliCMF"), cfg["data"]["root"]),
                        size=(H, W), split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
xp = []; xq = []
for a, b in DataLoader(ds, batch_size=64): xp.append(a); xq.append(b)
xp = torch.cat(xp).to(dev); xq = torch.cat(xq).to(dev); B = xp.shape[0]
with torch.no_grad():
    r0 = torch.zeros(B, device=dev); t1 = torch.ones(B, device=dev)
    xhat = model(xp, r0, t1, xp)["xhat"].clamp(0, 1)
print("val pairs=%d" % B)

def metrics(a, b): return float(ssim(a, b).mean()), float(lpips_fn(a, b).mean())

def align(use_affine, use_photo, iters=200):
    # params: translation(2), log-scale x/y(2), rotation(1); photometric a,gamma,b(3)
    tp = torch.zeros(B, 5, device=dev, requires_grad=True)      # dx,dy,lsx,lsy,theta
    pp = torch.zeros(B, 3, device=dev, requires_grad=True)      # la, lg, b
    opt = torch.optim.Adam([tp, pp], lr=0.02)
    for it in range(iters):
        dx = 0.1 * torch.tanh(tp[:, 0]); dy = 0.1 * torch.tanh(tp[:, 1])
        if use_affine:
            sx = torch.exp(0.1 * torch.tanh(tp[:, 2])); sy = torch.exp(0.1 * torch.tanh(tp[:, 3])); th = 0.17 * torch.tanh(tp[:, 4])
        else:
            s = torch.exp(0.1 * torch.tanh(tp[:, 2])); sx = s; sy = s; th = torch.zeros(B, device=dev)
        theta = torch.stack([torch.stack([sx * torch.cos(th), -sx * torch.sin(th), dx], 1),
                             torch.stack([sy * torch.sin(th), sy * torch.cos(th), dy], 1)], 1)   # (B,2,3)
        grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
        xw = F.grid_sample(xq, grid, align_corners=False, padding_mode="border")
        if use_photo:
            a = torch.exp(0.3 * torch.tanh(pp[:, 0])).view(B, 1, 1, 1); g = torch.exp(0.3 * torch.tanh(pp[:, 1])).view(B, 1, 1, 1); bb_ = (0.1 * torch.tanh(pp[:, 2])).view(B, 1, 1, 1)
            xw = (a * xw.clamp_min(1e-4) ** g + bb_).clamp(0, 1)
        loss = ((xhat - xw) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        dx = 0.1 * torch.tanh(tp[:, 0]); dy = 0.1 * torch.tanh(tp[:, 1])
        if use_affine:
            sx = torch.exp(0.1 * torch.tanh(tp[:, 2])); sy = torch.exp(0.1 * torch.tanh(tp[:, 3])); th = 0.17 * torch.tanh(tp[:, 4])
        else:
            s = torch.exp(0.1 * torch.tanh(tp[:, 2])); sx = s; sy = s; th = torch.zeros(B, device=dev)
        theta = torch.stack([torch.stack([sx * torch.cos(th), -sx * torch.sin(th), dx], 1),
                             torch.stack([sy * torch.sin(th), sy * torch.cos(th), dy], 1)], 1)
        xw = F.grid_sample(xq, F.affine_grid(theta, (B, 1, H, W), align_corners=False), align_corners=False, padding_mode="border")
        if use_photo:
            a = torch.exp(0.3 * torch.tanh(pp[:, 0])).view(B, 1, 1, 1); g = torch.exp(0.3 * torch.tanh(pp[:, 1])).view(B, 1, 1, 1); bb_ = (0.1 * torch.tanh(pp[:, 2])).view(B, 1, 1, 1)
            xw = (a * xw.clamp_min(1e-4) ** g + bb_).clamp(0, 1)
        sw = ssim(xhat, xw); lw = lpips_fn(xhat, xw)
        # per-case consistency vs baseline
        return float(sw.mean()), float(lw.mean()), sw.cpu(), lw.cpu(), float(dx.abs().mean()), float(dy.abs().mean()), float(th.abs().mean())

s0, l0 = metrics(xhat, xq)
sb = ssim(xhat, xq).cpu(); lb = lpips_fn(xhat, xq).cpu()
print("[0 no-align]        SSIM=%.4f LPIPS=%.4f" % (s0, l0))
s1, l1, s1v, l1v, *_ = align(False, False)
print("[1 transl+scale]    SSIM=%.4f LPIPS=%.4f | dSSIM=%.4f dLPIPS=%.4f | frac SSIM-up=%.2f LPIPS-down=%.2f" % (s1, l1, s1 - s0, l1 - l0, float((s1v > sb).float().mean()), float((l1v < lb).float().mean())))
s2, l2, s2v, l2v, mdx, mdy, mth = align(True, True)
print("[2 small-affine+photo] SSIM=%.4f LPIPS=%.4f | dSSIM=%.4f dLPIPS=%.4f | frac SSIM-up=%.2f LPIPS-down=%.2f" % (s2, l2, s2 - s0, l2 - l0, float((s2v > sb).float().mean()), float((l2v < lb).float().mean())))
print("   typical |dx|=%.4f |dy|=%.4f |theta|=%.4f rad (%.1f deg)" % (mdx, mdy, mth, mth * 180 / math.pi))
print("ADOC_GATE_DONE")
