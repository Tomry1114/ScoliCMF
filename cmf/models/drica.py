"""DRICA — Diagnosis-Routed Interval Cross-Attention.

Current generation state z_t (Query) retrieves evidence from the pre-op image (Key/Value). Diagnosis
routes WHERE to retrieve along the two spinal axes; the MeanFlow interval (t,r) modulates, per attention
head, WHERE (retrieval-bias strength) and how strongly the retrieved evidence is written back. The joint
phenotype modulates HOW the vertical and horizontal evidence couple (FiLM), not merely their mix weight.

JVP-safe (MeanFlow uses autograd.functional.jvp, create_graph=True = forward-over-reverse double-backward):
only Linear / SiLU / sigmoid / softmax / einsum / matmul / elementwise-mul / LayerNorm(eps>0).
No SDPA, no in-place, no sqrt-at-zero.
"""
import torch, torch.nn as nn
from einops import rearrange

NUM_REGIONS, NUM_DIRECTIONS, NUM_JOINT = 3, 2, 6


class IntervalEncoder(nn.Module):
    """MeanFlow interval (t,r) -> context vector + PER-HEAD route strengths in (0,1):
    [s_v (vertical retrieval-bias), s_h (horizontal retrieval-bias), s_j (joint coupling), s_res (residual),
     s_g (ungated GLOBAL retrieval = copy unchanged anatomy e.g. ribs from the aligned pre-op image)]."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.enc = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.scale_head = nn.Linear(dim, num_heads * 5)
    def forward(self, p, q):                   # p = t_emb+r_emb, q = t_emb-r_emb  (each [B,dim])
        ctx = self.enc(torch.cat([p, q], dim=-1))
        scales = torch.sigmoid(self.scale_head(ctx)).reshape(-1, self.num_heads, 5)   # [B,heads,5]
        return ctx, scales


class DiagnosisRouter(nn.Module):
    """Diagnosis -> vertical/horizontal attention-LOGIT biases (retrieval routing, not position vectors)
    + per-head branch weights + joint context (for the coupling FiLM)."""
    def __init__(self, dim, num_heads, grid_h, grid_w):
        super().__init__()
        self.num_heads = num_heads
        self.region_vertical_bias      = nn.Parameter(torch.zeros(NUM_REGIONS, num_heads, grid_h))
        self.direction_horizontal_bias = nn.Parameter(torch.zeros(NUM_DIRECTIONS, num_heads, grid_w))
        self.joint_encoder = nn.Sequential(nn.Linear(NUM_JOINT, dim), nn.SiLU())
        self.branch_router = nn.Linear(2 * dim, num_heads * 3)
        nn.init.normal_(self.region_vertical_bias, std=0.01)       # small (not 0): break class symmetry at t=0
        nn.init.normal_(self.direction_horizontal_bias, std=0.01)
    def forward(self, region_prob, direction_prob, joint_prob, interval_ctx):
        vbias = torch.einsum("br,rhk->bhk", region_prob, self.region_vertical_bias)         # [B,heads,grid_h]
        hbias = torch.einsum("bd,dhw->bhw", direction_prob, self.direction_horizontal_bias)   # [B,heads,grid_w]
        joint_ctx = self.joint_encoder(joint_prob)                                           # [B,dim]
        routing = torch.cat([joint_ctx, interval_ctx], dim=-1)                                # [B,2dim]
        branch = self.branch_router(routing).reshape(-1, self.num_heads, 3).softmax(dim=-1)   # [B,heads,3]
        return vbias, hbias, branch, joint_ctx


class AxialCrossAttention(nn.Module):
    """Cross-attention along ONE axis. axis='v': for each column, query current over source ROWS (H);
    axis='h': for each row, query current over source COLS (W). Diagnosis bias added to key logits.
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
        bias_rep = bias.repeat_interleave(rep, dim=0).unsqueeze(2)         # [M,heads,1,L] over KEY dim (rep inner)
        logits = logits + bias_rep
        out = logits.softmax(dim=-1) @ v                                   # [M,heads,L,hd]
        out = out.transpose(1, 2).reshape(M, L, self.nh, self.hd)          # [M,L,heads,hd]
        if self.axis == "v":
            out = rearrange(out, "(b w) h nh hd -> b h w nh hd", b=B, w=W)
        else:
            out = rearrange(out, "(b h) w nh hd -> b h w nh hd", b=B, h=H)
        return out                                                        # [B,H,W,heads,hd]


class GlobalCrossAttention(nn.Module):
    """Plain (un-routed, no diagnosis bias) multi-head cross-attention: current queries ALL source tokens.
    Lets any post-op position copy evidence from any pre-op position -> preserves anatomy that barely changes
    surgically (ribs / soft tissue), which the spine-routed axial branches never retrieve.
    Returns [B,H,W,heads,head_dim]."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.nh, self.hd = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.qn = nn.LayerNorm(self.hd); self.kn = nn.LayerNorm(self.hd)
        self.scale = self.hd ** -0.5
    def forward(self, current, source):        # [B,H,W,D]
        B, H, W, D = current.shape; N = H * W
        cur = current.reshape(B, N, D); src = source.reshape(B, N, D)
        q = self.qn(self.q(cur).reshape(B, N, self.nh, self.hd)).transpose(1, 2)   # [B,heads,N,hd]
        k = self.kn(self.k(src).reshape(B, N, self.nh, self.hd)).transpose(1, 2)
        v = self.v(src).reshape(B, N, self.nh, self.hd).transpose(1, 2)
        out = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1) @ v          # [B,heads,N,hd]
        return out.transpose(1, 2).reshape(B, H, W, self.nh, self.hd)


class DRICABlock(nn.Module):
    """Diagnosis-routed axial retrieval from source into current; interval per-head modulates retrieval
    location + coupling + residual; joint FiLM modulates the vertical*horizontal coupling; zero-init
    residual so it starts ~identity (doesn't wreck the DiT backbone)."""
    def __init__(self, dim, num_heads, grid_h, grid_w):
        super().__init__()
        self.nh, self.hd, self.gh, self.gw = num_heads, dim // num_heads, grid_h, grid_w
        self.norm_c = nn.LayerNorm(dim); self.norm_s = nn.LayerNorm(dim)
        self.router = DiagnosisRouter(dim, num_heads, grid_h, grid_w)
        self.vca = AxialCrossAttention(dim, num_heads, "v")
        self.hca = AxialCrossAttention(dim, num_heads, "h")
        self.gca = GlobalCrossAttention(dim, num_heads)  # ungated global retrieval (copy unchanged anatomy: ribs)
        self.joint_modulator = nn.Linear(dim, 2 * dim)   # joint -> FiLM (gamma,beta) on the coupling
        self.joint_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.joint_modulator.weight); nn.init.zeros_(self.joint_modulator.bias)  # start: coupling = out_v*out_h
        nn.init.zeros_(self.output_proj.weight); nn.init.zeros_(self.output_proj.bias)           # start: identity
    def forward(self, current_tokens, source_tokens, interval_ctx, scales,
                region_prob, direction_prob, joint_prob, return_aux=False):
        B, N, D = current_tokens.shape; H, W = self.gh, self.gw
        cur = self.norm_c(current_tokens).reshape(B, H, W, D)
        src = self.norm_s(source_tokens).reshape(B, H, W, D)
        vbias, hbias, branch, joint_ctx = self.router(region_prob, direction_prob, joint_prob, interval_ctx)
        sv = scales[:, :, 0]; sh = scales[:, :, 1]                          # per-head [B,heads]
        sj = scales[:, :, 2].reshape(B, 1, 1, self.nh, 1); sr = scales[:, :, 3].reshape(B, 1, 1, self.nh, 1)
        sg = scales[:, :, 4].reshape(B, 1, 1, self.nh, 1)                  # global-retrieval (copy) strength
        # interval per-head scales the diagnosis retrieval-bias BEFORE attention -> "where to retrieve"
        out_v = self.vca(cur, src, vbias * sv[:, :, None])                 # [B,H,W,heads,hd]
        out_h = self.hca(cur, src, hbias * sh[:, :, None])
        out_g = self.gca(cur, src)                                         # ungated global copy (ribs/soft tissue)
        # joint FiLM modulates HOW vertical/horizontal couple (gamma,beta start 0 -> plain out_v*out_h)
        gamma, beta = self.joint_modulator(joint_ctx).chunk(2, dim=-1)     # [B,D] each
        coupled = (1 + gamma[:, None, None, :]) * (out_v.reshape(B, H, W, D) * out_h.reshape(B, H, W, D)) + beta[:, None, None, :]
        out_j = self.joint_proj(coupled).reshape(B, H, W, self.nh, self.hd)
        av = branch[..., 0].reshape(B, 1, 1, self.nh, 1)                   # per-head branch weights
        ah = branch[..., 1].reshape(B, 1, 1, self.nh, 1)
        aj = branch[..., 2].reshape(B, 1, 1, self.nh, 1)
        upd = av * out_v + ah * out_h + aj * sj * out_j + sg * out_g       # + ungated global copy branch
        upd = (sr * upd).reshape(B, H, W, D)                              # per-head interval residual strength
        upd = self.output_proj(upd).reshape(B, N, D)
        out = current_tokens + upd
        if return_aux:
            return out, {"branch": branch, "scales": scales, "gamma": gamma, "beta": beta, "vbias": vbias, "hbias": hbias}
        return out
