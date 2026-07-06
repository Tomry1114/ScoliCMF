"""MFDiT (original ScoliCMF conditional MeanFlow DiT) — self-contained blocks (no timm),
generalized to NON-SQUARE (H,W), + TEXT (phenotype) conditioning added to the FGA cond vector c.
Faithful to the original: c = (t_emb+r_emb) + gate*mean(cond_embedder(cond_img)) [+ text_adapter(s)]."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
NUM_REGIONS = 3     # thoracic / thoracolumbar / lumbar
NUM_DIRECTIONS = 2  # image_left / image_right

def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.scale = dim ** 0.5; self.g = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, dim):
        super().__init__()
        self.gh, self.gw = img_size[0] // patch_size, img_size[1] // patch_size
        self.num_patches = self.gh * self.gw; self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, T, D)

class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc1 = nn.Linear(dim, hidden); self.act = nn.GELU(approximate="tanh"); self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True):
        super().__init__(); self.nh = num_heads; self.hd = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True); self.proj = nn.Linear(dim, dim)
        self.qn = RMSNorm(self.hd) if qk_norm else nn.Identity(); self.kn = RMSNorm(self.hd) if qk_norm else nn.Identity()
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]; q = self.qn(q); k = self.kn(k)
        attn = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)   # manual attn (double-backward for MeanFlow JVP)
        x = attn.softmax(dim=-1) @ v
        return self.proj(x.transpose(1, 2).reshape(B, N, C))

class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim)); self.nfreq = nfreq
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb
    def forward(self, t):
        return self.mlp(self.timestep_embedding(t * 1000, self.nfreq))

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_type="vanilla", gh=0, gw=0, inner_lr=0.25, cpe=False):
        super().__init__()
        self.attn_type = attn_type; self.gh = gh; self.gw = gw; self.cpe_on = cpe
        if cpe:
            self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)   # DiT^3 conditional positional encoding
        self.norm1 = RMSNorm(dim)
        if attn_type == "ttt":
            from ttt_block import TTT
            self.attn = TTT(dim, num_heads, qkv_bias=True, inner_lr=inner_lr)   # ViT^3 TTT mixer (configurable inner lr)
        else:
            self.attn = Attention(dim, num_heads, qk_norm=True)
        self.norm2 = RMSNorm(dim); self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
    def forward(self, x, c):
        if self.cpe_on:                                                            # DiT^3 CPE: x <- x + DWConv3x3(x)
            B, N, C = x.shape
            x = x + self.cpe(x.reshape(B, self.gh, self.gw, C).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.adaLN_modulation(c).chunk(6, dim=-1)
        hh = modulate(self.norm1(x), sc_a, sh_a)
        a = self.attn(hh, self.gh, self.gw) if self.attn_type == "ttt" else self.attn(hh)
        x = x + g_a.unsqueeze(1) * a
        x = x + g_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sc_m, sh_m))
        return x

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__(); self.norm_final = RMSNorm(dim); self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
    def forward(self, x, c):
        sh, sc = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), sc, sh))   # FIX1: modulate(x, scale, shift) was (shift, scale)

class MFDiT(nn.Module):
    def __init__(self, img_size=(480, 240), patch_size=8, data_channels=1, cond_channels=1,
                 dim=384, depth=12, num_heads=6, mlp_ratio=4.0, text=True, attn_type="vanilla", inner_lr=0.25, cpe=False, text_emb="factorized", inject="global"):
        super().__init__()
        self.data_channels = data_channels; self.cond_channels = cond_channels
        self.in_channels = data_channels + cond_channels; self.out_channels = data_channels
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed(img_size, patch_size, self.in_channels, dim)
        self.cond_embedder = PatchEmbed(img_size, patch_size, cond_channels, dim)
        self.cond_proj = nn.Identity()
        self.gh, self.gw = self.x_embedder.gh, self.x_embedder.gw
        self.tr_gate = nn.Sequential(nn.SiLU(), nn.Linear(2 * dim, dim))
        self.t_embedder = TimestepEmbedder(dim); self.r_embedder = TimestepEmbedder(dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, dim), requires_grad=False)  # FIXED sincos (match official DiT3; avoid double-position overfit on 432 cases)
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, mlp_ratio, attn_type, self.gh, self.gw, inner_lr, cpe) for _ in range(depth)])
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)
        self.agent_condition = text; self.text_emb = text_emb; self.inject = inject
        self.region_prob = None; self.direction_prob = None; self.joint_prob = None
        if self.agent_condition:
            D = 0
            if text_emb in ("factorized", "both"):   # marginals: thoracic-right & lumbar-right share the right emb
                self.region_embedding = nn.Embedding(NUM_REGIONS, dim)
                self.direction_embedding = nn.Embedding(NUM_DIRECTIONS, dim); D += 2 * dim
            if text_emb in ("joint", "both"):         # 6-class joint: keeps (region,direction) correlation/uncertainty
                self.joint_embedding = nn.Embedding(NUM_REGIONS * NUM_DIRECTIONS, dim); D += dim
            if inject in ("global", "both"):
                self.agent_adapter = nn.Sequential(nn.Linear(D, dim), nn.SiLU(), nn.Linear(dim, dim))
            if inject in ("spatial", "both"):         # region->vertical rows, direction->horizontal cols; zero-init
                self.region_vprofile = nn.Parameter(torch.zeros(NUM_REGIONS, self.gh, dim))
                self.direction_hprofile = nn.Parameter(torch.zeros(NUM_DIRECTIONS, self.gw, dim))
        self.initialize_weights()
    def _agent_vec(self):   # concat enabled soft-weighted embeddings
        parts = []
        if self.text_emb in ("factorized", "both"):
            parts.append(self.region_prob @ self.region_embedding.weight)
            parts.append(self.direction_prob @ self.direction_embedding.weight)
        if self.text_emb in ("joint", "both"):
            parts.append(self.joint_prob @ self.joint_embedding.weight)
        return torch.cat(parts, dim=-1)
    def encode_agent_condition(self):
        return self.agent_adapter(self._agent_vec())
    def spatial_bias(self, B):   # region->vertical (rows), direction->horizontal (cols); per-token additive
        vbias = torch.einsum("br,rhd->bhd", self.region_prob, self.region_vprofile)
        hbias = torch.einsum("bk,kwd->bwd", self.direction_prob, self.direction_hprofile)
        return (vbias[:, :, None, :] + hbias[:, None, :, :]).reshape(B, self.gh * self.gw, -1)
    def initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        self.apply(_init)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.gh, self.gw)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        for b in self.blocks:
            nn.init.constant_(b.adaLN_modulation[-1].weight, 0); nn.init.constant_(b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0); nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0); nn.init.constant_(self.final_layer.linear.bias, 0)
        for b in self.blocks:                                  # zero-init CPE -> block starts at identity (like AdaLN-Zero)
            if getattr(b, "cpe_on", False):
                nn.init.zeros_(b.cpe.weight); nn.init.zeros_(b.cpe.bias)
        if self.agent_condition:
            if self.text_emb in ("factorized", "both"):
                nn.init.normal_(self.region_embedding.weight, std=0.02); nn.init.normal_(self.direction_embedding.weight, std=0.02)
            if self.text_emb in ("joint", "both"):
                nn.init.normal_(self.joint_embedding.weight, std=0.02)
            if self.inject in ("global", "both"):
                nn.init.zeros_(self.agent_adapter[-1].weight); nn.init.zeros_(self.agent_adapter[-1].bias)  # start = image-only
            # spatial profiles are torch.zeros -> start image-only
    def unpatchify(self, x):
        c, p, gh, gw = self.out_channels, self.patch_size, self.gh, self.gw
        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, gh * p, gw * p)
    def forward(self, x, t, r, cond_img=None):
        x_in = torch.cat([x, cond_img], dim=1) if (self.cond_channels > 0 and cond_img is not None) else x
        x_tokens = self.x_embedder(x_in) + self.pos_embed
        t_emb = self.t_embedder(t); r_emb = self.r_embedder(r)
        p = t_emb + r_emb; q = t_emb - r_emb
        cond_emb = self.cond_proj(self.cond_embedder(cond_img).mean(dim=1))
        gate = torch.sigmoid(self.tr_gate(torch.cat([p, q], dim=-1)))
        c = p + gate * cond_emb
        if self.agent_condition and self.region_prob is not None and self.direction_prob is not None:
            if self.inject in ("global", "both"):
                c = c + self.encode_agent_condition()
            if self.inject in ("spatial", "both"):
                x_tokens = x_tokens + self.spatial_bias(x_tokens.shape[0])
        for blk in self.blocks:
            x_tokens = blk(x_tokens, c)
        return self.unpatchify(self.final_layer(x_tokens, c))

def get_2d_sincos_pos_embed(embed_dim, gh, gw):
    gw_arr = np.arange(gw, dtype=np.float32); gh_arr = np.arange(gh, dtype=np.float32)
    grid = np.meshgrid(gw_arr, gh_arr)               # [w, h], each (gh, gw)
    grid = np.stack(grid, axis=0).reshape([2, 1, gh, gw])
    emb_h = _1d(embed_dim // 2, grid[1]); emb_w = _1d(embed_dim // 2, grid[0])
    return np.concatenate([emb_h, emb_w], axis=1)    # (gh*gw, embed_dim)

def _1d(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64); omega /= embed_dim / 2.0; omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
