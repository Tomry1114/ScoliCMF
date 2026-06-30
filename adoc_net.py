"""ADOC self-supervised acquisition corrector C_psi (paper version).
C_psi(moving, ref) -> 8 restricted acquisition params (5 geometric + 3 photometric). apply_A
applies them (forward). Because C_psi is trained ONLY on synthetic acquisition perturbations of
single images and can ONLY emit affine+photometric, it cannot absorb the (local, non-affine)
surgical correction -> built-in surgical-change protection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_A(x, p):
    """x:(B,1,H,W), p:(B,8) raw -> acquisition-transformed image (bounded, restricted)."""
    B, _, H, W = x.shape
    dx = 0.12 * torch.tanh(p[:, 0]); dy = 0.12 * torch.tanh(p[:, 1])
    sx = torch.exp(0.12 * torch.tanh(p[:, 2])); sy = torch.exp(0.12 * torch.tanh(p[:, 3])); th = 0.20 * torch.tanh(p[:, 4])
    theta = torch.stack([torch.stack([sx * torch.cos(th), -sx * torch.sin(th), dx], 1),
                         torch.stack([sy * torch.sin(th), sy * torch.cos(th), dy], 1)], 1)
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    xw = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    a = torch.exp(0.4 * torch.tanh(p[:, 5])).view(B, 1, 1, 1); g = torch.exp(0.4 * torch.tanh(p[:, 6])).view(B, 1, 1, 1); b = (0.12 * torch.tanh(p[:, 7])).view(B, 1, 1, 1)
    return (a * xw.clamp_min(1e-4) ** g + b).clamp(0, 1)


class AcquisitionCorrector(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(blk(2, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4), blk(ch * 4, ch * 4), blk(ch * 4, ch * 4))
        self.head = nn.Sequential(nn.Linear(ch * 4, ch * 2), nn.GELU(), nn.Linear(ch * 2, 8))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)   # start = identity correction

    def forward(self, moving, ref):
        f = self.net(torch.cat([moving, ref], 1)).mean(dim=(2, 3))
        return self.head(f)                                                        # (B,8) raw params
