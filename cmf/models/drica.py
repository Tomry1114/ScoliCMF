"""DRICA — Diagnosis-Routed Interval Cross-Attention.

Replaces the FGA global image-conditioning path. The current generation state (z_t tokens) is the QUERY;
the pre-op image (source tokens) is KEY/VALUE. Diagnosis (region/direction/joint) routes WHERE to
retrieve along the two spinal axes; the MeanFlow interval (t,r) modulates retrieval strength.

JVP-safe by construction (MeanFlow uses torch.autograd.functional.jvp with create_graph=True, i.e.
forward-over-reverse double-backward): only Linear / SiLU / sigmoid / softmax / einsum / matmul /
elementwise-mul / LayerNorm(eps>0). No SDPA, no in-place, no sqrt-at-zero.
"""
import torch, torch.nn as nn
from einops import rearrange

NUM_REGIONS, NUM_DIRECTIONS, NUM_JOINT = 3, 2, 6


class IntervalEncoder(nn.Module):
    """MeanFlow interval (t,r) -> context vector + per-branch route strengths in (0,1)."""
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.scale_head = nn.Linear(dim, 4)   # [s_vertical, s_horizontal, s_joint, s_residual]
    def forward(self, p, q):                   # p = t_emb+r_emb, q = t_emb-r_emb  (each [B,dim])
        ctx = self.enc(torch.cat([p, q], dim=-1))
        scales = torch.sigmoid(self.scale_head(ctx))   # [B,4]
        return ctx, scales


class DiagnosisRouter(nn.Module):
    """Diagnosis -> vertical/horizontal attention-LOGIT biases (retrieval routing, not position vectors)
    + per-head branch weights (how vertical/horizontal/joint evidence fuse)."""
    def __init__(self, dim, num_heads, grid_h, grid_w):
        super().__init__()
        self.num_heads = num_heads
        self.region_vertical_bias      = nn.Parameter(torch.zeros(NUM_REGIONS, num_heads, grid_h))
        self.direction_horizontal_bias = nn.Parameter(torch.zeros(NUM_DIRECTIONS, num_heads, grid_w))
        self.joint_encoder = nn.Sequential(nn.Linear(NUM_JOINT, dim), nn.SiLU())
        self.branch_router = nn.Linear(2 * dim, num_heads * 3)
        # small (not zero) so the 3 diagnosis classes don't receive identical gradients at t=0
        nn.init.normal_(self.region_vertical_bias, std=0.01)
        nn.init.normal_(self.direction_horizontal_bias, std=0.01)
    def forward(self, region_prob, direction_prob, joint_prob, interval_ctx):
        vbias = torch.einsum("br,rhk->bhk", region_prob, self.region_vertical_bias)        # [B,heads,grid_h]
        hbias = torch.einsum("bd,dhw->bhw", direction_prob, self.direction_horizontal_bias)  # [B,heads,grid_w]
        joint_ctx = self.joint_encoder(joint_prob)                                          # [B,dim]
        routing = torch.cat([joint_ctx, interval_ctx], dim=-1)                              # [B,2dim]
        branch = self.branch_router(routing).reshape(-1, self.num_heads, 3).softmax(dim=-1) # [B,heads,3]
        return vbias, hbias, branch


class AxialCrossAttention(nn.Module):
    """Cross-attention along ONE axis. axis='v': for each column, query current over source ROWS (H);
    axis='h': for each row, query current over source COLS (W). Diagnosis bias is added to the key logits.
    Returns [B,H,W,heads,head_dim] (heads kept separate for per-head branch fusion)."""
    def __init__(self, dim, num_heads, axis):
        super().__init__()
        assert axis in ("v", "h")
        self.nh, self.hd, self.axis = num_heads, dim // num_heads, axis
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.qn = nn.LayerNorm(self.hd); self.kn = nn.LayerNorm(self.hd)   # qk-norm -> smoother JVP
        self.scale = self.hd ** -0.5
    def forward(self, current, source, bias):     # current/source [B,H,W,D]; bias [B,heads,L]
        B, H, W, D = current.shape
        if self.axis == "v":
            cur = rearrange(current, "b h w d -> (b w) h d"); src = rearrange(source, "b h w d -> (b w) h d"); L, rep = H, W
        else:
            cur = rearrange(current, "b h w d -> (b h) w d"); src = rearrange(source, "b h w d -> (b h) w d"); L, rep = W, H
        M = cur.shape[0]                                                   # B*rep
        q = self.qn(self.q(cur).reshape(M, L, self.nh, self.hd)).transpose(1, 2)   # [M,heads,L,hd]
        k = self.kn(self.k(src).reshape(M, L, self.nh, self.hd)).transpose(1, 2)
        v = self.v(src).reshape(M, L, self.nh, self.hd).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) * self.scale                    # [M,heads,L,L]
        bias_rep = bias.repeat_interleave(rep, dim=0).unsqueeze(2)         # [M,heads,1,L] over KEY dim ((b w)/(b h): rep inner)
        logits = logits + bias_rep
        out = logits.softmax(dim=-1) @ v                                   # [M,heads,L,hd]
        out = out.transpose(1, 2).reshape(M, L, self.nh, self.hd)          # [M,L,heads,hd]
        if self.axis == "v":
            out = rearrange(out, "(b w) h nh hd -> b h w nh hd", b=B, w=W)
        else:
            out = rearrange(out, "(b h) w nh hd -> b h w nh hd", b=B, h=H)
        return out                                                        # [B,H,W,heads,hd]


class DRICABlock(nn.Module):
    """One DRICA update: diagnosis-routed axial retrieval from source into current, interval-scaled,
    per-head-fused, zero-init residual (starts ~identity so it doesn't wreck the DiT backbone)."""
    def __init__(self, dim, num_heads, grid_h, grid_w):
        super().__init__()
        self.nh, self.hd, self.gh, self.gw = num_heads, dim // num_heads, grid_h, grid_w
        self.norm_c = nn.LayerNorm(dim); self.norm_s = nn.LayerNorm(dim)
        self.router = DiagnosisRouter(dim, num_heads, grid_h, grid_w)
        self.vca = AxialCrossAttention(dim, num_heads, "v")
        self.hca = AxialCrossAttention(dim, num_heads, "h")
        self.joint_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.output_proj.weight); nn.init.zeros_(self.output_proj.bias)   # start = identity
    def forward(self, current_tokens, source_tokens, interval_ctx, scales,
                region_prob, direction_prob, joint_prob, return_aux=False):
        B, N, D = current_tokens.shape; H, W = self.gh, self.gw
        cur = self.norm_c(current_tokens).reshape(B, H, W, D)
        src = self.norm_s(source_tokens).reshape(B, H, W, D)
        vbias, hbias, branch = self.router(region_prob, direction_prob, joint_prob, interval_ctx)
        out_v = self.vca(cur, src, vbias)                     # [B,H,W,heads,hd]
        out_h = self.hca(cur, src, hbias)
        out_j = self.joint_proj((out_v * out_h).reshape(B, H, W, D)).reshape(B, H, W, self.nh, self.hd)  # region-direction coupling
        av = branch[..., 0].reshape(B, 1, 1, self.nh, 1)      # per-head branch weights
        ah = branch[..., 1].reshape(B, 1, 1, self.nh, 1)
        aj = branch[..., 2].reshape(B, 1, 1, self.nh, 1)
        sv = scales[:, 0].reshape(B, 1, 1, 1, 1); sh = scales[:, 1].reshape(B, 1, 1, 1, 1)
        sj = scales[:, 2].reshape(B, 1, 1, 1, 1); sr = scales[:, 3].reshape(B, 1, 1)
        upd = sv * av * out_v + sh * ah * out_h + sj * aj * out_j          # [B,H,W,heads,hd]
        upd = self.output_proj(upd.reshape(B, H, W, D)).reshape(B, N, D)
        out = current_tokens + sr * upd
        if return_aux:
            return out, {"branch": branch, "scales": scales, "vbias": vbias, "hbias": hbias}
        return out
