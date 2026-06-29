"""APTD: Anatomy-Preserving Transport Decomposition (x0-parameterized).
Model predicts x_hat_post from (z_t,r,t,x_pre) via a structured decode head:
  direct   : x_hat = D(h)                      (free image regen; no content reuse -- baseline)
  residual : x_hat = x_pre + R(h)              (no-warp sharpen control)
  warp     : x_hat = W(x_pre, phi(h))          (warp-only)
  warpres  : x_hat = W(x_pre, phi(h)) + R(h)   (FULL APTD: warp-carried anatomy + new-content residual)
Backbone reuses SCDiT (its own decode head is unused). Zero-init last conv => identity warp / zero
residual at start, so training departs smoothly from x_pre."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sc_dit import RMSNorm


class WarpResidualHead(nn.Module):
    def __init__(self, dim, patch, gh, gw, mode="warpres", flow_scale=0.3):
        super().__init__()
        self.mode, self.gh, self.gw, self.flow_scale = mode, gh, gw, flow_scale
        oc = {"direct": 1, "residual": 1, "warp": 2, "warpres": 3}[mode]
        hid = max(32, dim // 2)
        self.norm = RMSNorm(dim)
        self.proj = nn.Conv2d(dim, hid * patch * patch, 1)
        self.ps = nn.PixelShuffle(patch)
        self.refine = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(hid, oc, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)

    def forward(self, h, x_pre, base_grid):
        B = h.shape[0]
        x = self.norm(h).transpose(1, 2).reshape(B, -1, self.gh, self.gw)
        o = self.refine(self.ps(self.proj(x)))                       # (B,oc,H,W)
        flow = res = None
        if self.mode == "direct":
            xhat = x_pre + o                                          # anchored free residual (no structure)
            res = o
        elif self.mode == "residual":
            xhat = x_pre + o; res = o
        elif self.mode == "warp":
            flow = torch.tanh(o) * self.flow_scale
            xhat = F.grid_sample(x_pre, base_grid + flow.permute(0, 2, 3, 1), align_corners=False, padding_mode="border")
        else:                                                        # warpres (full APTD)
            flow = torch.tanh(o[:, :2]) * self.flow_scale; res = o[:, 2:3]
            warp = F.grid_sample(x_pre, base_grid + flow.permute(0, 2, 3, 1), align_corners=False, padding_mode="border")
            xhat = warp + res
        return {"xhat": xhat, "flow": flow, "res": res}


class APTDNet(nn.Module):
    """x0-parameterized: forward(z_t,r,t,x_pre) -> dict(xhat,flow,res). Reuses SCDiT backbone."""
    def __init__(self, backbone, mode="warpres", flow_scale=0.3):
        super().__init__()
        self.bb = backbone
        self.head = WarpResidualHead(backbone.x_embedder.proj.out_channels if False else self._dim(backbone),
                                     backbone.patch_size, backbone.gh, backbone.gw, mode, flow_scale)
        self.register_buffer("theta", torch.tensor([[1., 0, 0], [0, 1., 0]]).unsqueeze(0))

    @staticmethod
    def _dim(bb):
        return bb.pos_embed.shape[-1]

    def forward(self, z_t, r, t, x_pre):
        h, c, _ = self.bb.forward_features(z_t, r, t, x_pre)
        B, _, H, W = x_pre.shape
        base = F.affine_grid(self.theta.expand(B, 2, 3), (B, 1, H, W), align_corners=False)
        return self.head(h, x_pre, base)
