"""Source-anchored DiT (S2), timm-free, single u-head, FGA removed.

forward(z_t, r, t, x_pre) -> velocity u (B, data_ch, H, W).
- input = concat([z_t, x_pre]) so x_pre conditions spatially (source anchor).
- conditioning vector c (B,D) from a pluggable `cond` module (Base PGA now; SC-PGA at S5)
  combined with time embedding, injected via AdaLN in every block (reused DiT design).
PatchEmbed/Mlp/Attention reimplemented to drop the timm dependency.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- timm-free primitives ----------
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, dim):
        super().__init__()
        self.ih, self.iw = (img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size))
        self.p = patch_size
        self.gh, self.gw = self.ih // self.p, self.iw // self.p
        self.num_patches = self.gh * self.gw
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=self.p, stride=self.p)

    def forward(self, x):
        x = self.proj(x)                      # (B, D, gh, gw)
        return x.flatten(2).transpose(1, 2)   # (B, T, D)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True):
        super().__init__()
        assert dim % num_heads == 0
        self.nh = num_heads
        self.hd = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.qn = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.kn = RMSNorm(self.hd) if qk_norm else nn.Identity()

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # (B, nh, T, hd)
        q, k = self.qn(q), self.kn(k)
        attn = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        attn = attn.softmax(dim=-1)
        x = attn @ v  # (B, nh, T, hd) manual eager (twice-differentiable for L_ST JVP)
        x = x.transpose(1, 2).reshape(B, T, D)
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t * 1000, self.nfreq))


def modulate(x, scale, shift):
    if scale.dim() == 2:                      # global (B,D) -> broadcast over tokens
        scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
    return x * (1 + scale) + shift            # per-token (B,T,D) passes through (axial AdaLN)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, qk_norm=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.adaLN(c).chunk(6, dim=-1)
        ga = g_a.unsqueeze(1) if g_a.dim() == 2 else g_a
        gm = g_m.unsqueeze(1) if g_m.dim() == 2 else g_m
        x = x + ga * self.attn(modulate(self.norm1(x), sc_a, sh_a))
        x = x + gm * self.mlp(modulate(self.norm2(x), sc_m, sh_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_ch):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_ch)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        sh, sc = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), sc, sh))   # (scale, shift) order fixed


class ConvHead(nn.Module):
    """Tokens (B,T,D) -> image via PixelShuffle + conv refine. Cross-patch receptive field
    kills the linear-unpatchify 16x16 checkerboard and lifts endpoint fidelity (P0 fix)."""
    def __init__(self, dim, patch_size, gh, gw, out_ch):
        super().__init__()
        self.gh, self.gw = gh, gw
        self.norm = RMSNorm(dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        hid = max(32, dim // 2)
        self.proj = nn.Conv2d(dim, hid * patch_size * patch_size, 1)
        self.ps = nn.PixelShuffle(patch_size)
        self.refine = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1), nn.GELU(),
            nn.Conv2d(hid, out_ch, 3, padding=1),
        )

    def forward(self, x, c):
        sh, sc = self.adaLN(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), sc, sh)                                  # (B,T,D)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.gh, self.gw)     # (B,D,gh,gw)
        return self.refine(self.ps(self.proj(x)))                           # (B,out,gh*p,gw*p)


# ---------- conditioning seam (Base PGA now; SC-PGA replaces at S5) ----------
class BasePGACond(nn.Module):
    """Global x_pre conditioning (Pi = I, no spinal restriction). Returns (B, D)."""
    def __init__(self, img_size, patch_size, cond_ch, dim):
        super().__init__()
        self.embed = PatchEmbed(img_size, patch_size, cond_ch, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_pre, r, t, t_emb, r_emb):
        tok = self.embed(x_pre)               # (B, Tc, D)
        cond = self.proj(tok.mean(dim=1))     # (B, D)
        c = t_emb + r_emb + cond
        z = c.new_zeros(())
        return c, {"l_time": z, "m_dyn_rms": z, "m_static_rms": z}


class SCDiT(nn.Module):
    def __init__(self, img_size=(480, 240), patch_size=8, data_channels=1, cond_channels=1,
                 dim=384, depth=12, num_heads=6, mlp_ratio=4.0, cond_module=None, decode_head="conv",
                 xpre_mode="full"):
        super().__init__()
        self.data_channels = data_channels
        self.out_channels = data_channels
        self.patch_size = patch_size
        ih, iw = img_size
        self.gh, self.gw = ih // patch_size, iw // patch_size

        self.xpre_mode = xpre_mode   # full | blur | none : P1 anti-shortcut on the cat([z_t,x_pre]) path
        _xc = 0 if xpre_mode == "none" else cond_channels
        self.x_embedder = PatchEmbed(img_size, patch_size, data_channels + _xc, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)
        self.cond = cond_module if cond_module is not None else \
            BasePGACond(img_size, patch_size, cond_channels, dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, dim), requires_grad=True)
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.head = (ConvHead(dim, patch_size, self.gh, self.gw, self.out_channels)
                     if decode_head == "conv" else FinalLayer(dim, patch_size, self.out_channels))
        self._init()

    def _init(self):
        def b(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(b)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.gh, self.gw)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN[-1].weight, 0)
            nn.init.constant_(blk.adaLN[-1].bias, 0)
        nn.init.constant_(self.head.adaLN[-1].weight, 0)
        nn.init.constant_(self.head.adaLN[-1].bias, 0)
        if isinstance(self.head, ConvHead):
            nn.init.constant_(self.head.refine[-1].weight, 0)
            nn.init.constant_(self.head.refine[-1].bias, 0)
        else:
            nn.init.constant_(self.head.linear.weight, 0)
            nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x):
        c, p, gh, gw = self.out_channels, self.patch_size, self.gh, self.gw
        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, gh * p, gw * p)

    def forward(self, z_t, r, t, x_pre, return_aux=False):
        if self.xpre_mode == "none":
            x_in = z_t
        elif self.xpre_mode == "blur":
            xb = F.interpolate(F.interpolate(x_pre, scale_factor=0.125, mode="bilinear", align_corners=False),
                               size=x_pre.shape[-2:], mode="bilinear", align_corners=False)
            x_in = torch.cat([z_t, xb], dim=1)   # blurred x_pre: keeps gross pose, removes pixel-copy
        else:
            x_in = torch.cat([z_t, x_pre], dim=1)
        x = self.x_embedder(x_in) + self.pos_embed
        c, aux = self.cond(x_pre, r, t, self.t_embedder(t), self.r_embedder(r))
        for blk in self.blocks:
            x = blk(x, c)
        u = self.head(x, c) if isinstance(self.head, ConvHead) else self.unpatchify(self.head(x, c))
        return (u, aux) if return_aux else u


# ---------- sincos pos embed (numpy-only, rectangular gh x gw) ----------
def get_2d_sincos_pos_embed(embed_dim, gh, gw):
    gh_a = np.arange(gh, dtype=np.float32)
    gw_a = np.arange(gw, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw_a, gh_a), axis=0).reshape([2, 1, gh, gw])
    emb_h = _1d(embed_dim // 2, grid[0])
    emb_w = _1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _1d(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
