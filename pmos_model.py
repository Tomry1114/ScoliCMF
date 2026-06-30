"""PMOS: Plan-Marginalized Outcome Set (module 2, built on APTD).
Unobserved surgical plan -> K discrete empirical prototypes. Shared APTD backbone;
K prototype embeddings each modulate the warp+residual head -> K candidate post-op images.
Inference returns the SET {x_hat^k}; trained by soft-min set loss (each case explained by
its best prototype) + usage balance. NOT a claim of predicting the plan from pre-op."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from aptd_model import WarpResidualHead


class PMOSNet(nn.Module):
    def __init__(self, backbone, K=4, mode="warpres", flow_scale=0.3):
        super().__init__()
        self.bb = backbone; self.K = K
        dim = backbone.pos_embed.shape[-1]
        self.head = WarpResidualHead(dim, backbone.patch_size, backbone.gh, backbone.gw, mode, flow_scale)
        self.proto = nn.Embedding(K, dim)
        nn.init.normal_(self.proto.weight, 0.0, 0.1)             # diverse init (break symmetry)
        with torch.no_grad(): self.proto.weight[0].zero_()      # prototype 0 == base APTD
        self.register_buffer("theta", torch.tensor([[1., 0, 0], [0, 1., 0]]).unsqueeze(0))

    def forward_all(self, z_t, r, t, x_pre):
        h, c, _ = self.bb.forward_features(z_t, r, t, x_pre)     # (B,T,D) shared
        B, _, H, W = x_pre.shape
        base = F.affine_grid(self.theta.expand(B, 2, 3), (B, 1, H, W), align_corners=False)
        xh, fl, rs = [], [], []
        for k in range(self.K):
            ek = self.proto.weight[k].view(1, 1, -1)             # (1,1,D)
            out = self.head(h + ek, x_pre, base)
            xh.append(out["xhat"]); fl.append(out["flow"]); rs.append(out["res"])
        xhat = torch.stack(xh, 1)                                # (B,K,1,H,W)
        flow = torch.stack(fl, 1) if fl[0] is not None else None
        res = torch.stack(rs, 1) if rs[0] is not None else None
        return {"xhat": xhat, "flow": flow, "res": res}
