"""Image generation metrics (torch, no external deps): SSIM, PSNR. x,y: (B,1,H,W) in [0,1]."""
import torch
import torch.nn.functional as F


def _gauss_window(win, sigma, device):
    c = torch.arange(win, device=device).float() - win // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]            # (1,1,win,win)


def ssim(x, y, win=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    w = _gauss_window(win, sigma, x.device)
    p = win // 2
    def _c(t):                                          # reflect-pad conv (issue-8: avoid zero-pad edge bias)
        return F.conv2d(F.pad(t, [p, p, p, p], mode="reflect"), w)
    mu_x = _c(x); mu_y = _c(y)
    mx2, my2, mxy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx = _c(x * x) - mx2
    sy = _c(y * y) - my2
    sxy = _c(x * y) - mxy
    smap = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mx2 + my2 + C1) * (sx + sy + C2))
    return smap.flatten(1).mean(1)                          # (B,)


def psnr(x, y):
    mse = (x - y).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))     # (B,)


# ---------- LPIPS (AlexNet) perceptual distance, lower=better ----------
_LPIPS = None

def lpips_fn(x, y):
    """LPIPS-Alex perceptual distance per sample (lower better). x,y:(B,1,H,W) in [0,1].
    Grayscale -> 3ch, [0,1] -> [-1,1]. AlexNet weights cached on shared home (offline-safe)."""
    global _LPIPS
    import lpips as _lp
    if _LPIPS is None:
        _LPIPS = _lp.LPIPS(net="alex", verbose=False).to(x.device).eval()
        for p in _LPIPS.parameters():
            p.requires_grad_(False)
    def prep(t):
        t = t.clamp(0, 1)
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        return t * 2 - 1
    with torch.no_grad():
        return _LPIPS(prep(x), prep(y)).flatten()           # (B,)


def lpips_loss(x, y):
    """DIFFERENTIABLE LPIPS-Alex (for use as a training loss). x,y:(B,1,H,W) in [0,1]."""
    global _LPIPS
    import lpips as _lp
    if _LPIPS is None:
        _LPIPS = _lp.LPIPS(net="alex", verbose=False).to(x.device).eval()
        for p in _LPIPS.parameters():
            p.requires_grad_(False)
    def prep(t):
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        return t * 2 - 1
    return _LPIPS(prep(x), prep(y)).mean()
