import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp, Attention


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t * 1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RMSNorm,
        )
        self.attn.fused_attn = False

        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MFDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        data_channels=1, 
        cond_channels=0, 
        dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
    ):
        super().__init__()

        self.data_channels = data_channels
        self.cond_channels = cond_channels
        self.in_channels = data_channels + cond_channels
        self.out_channels = data_channels 

        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, self.in_channels, dim
        )

        self.cond_embedder = PatchEmbed(input_size, patch_size, cond_channels, dim)
        self.cond_proj = nn.Identity()

        self.tr_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * dim, dim)
        )

        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, dim), requires_grad=True
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Sin-cos positional encoding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Args:
            x: (N, T, patch_size**2 * C)
        Returns:
            imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, r, cond_img=None):
        if self.cond_channels > 0 and cond_img is not None:
            x_in = torch.cat([x, cond_img], dim=1)  # (N, C_data + C_cond, H, W)
        else:
            x_in = x

        x_tokens = self.x_embedder(x_in) + self.pos_embed  # (N, T, D)
        t_emb = self.t_embedder(t)
        r_emb = self.r_embedder(r)
        
        p = t_emb + r_emb
        q = t_emb - r_emb
        
        cond_tokens = self.cond_embedder(cond_img)      # (B, Tc, D)
        cond_emb = cond_tokens.mean(dim=1)              # (B, D)
        cond_emb = self.cond_proj(cond_emb)             # (B, D)

        gate = torch.sigmoid(self.tr_gate(torch.cat([p, q], dim=-1)))  # (B, D)
        c = p + gate * cond_emb

        for blk in self.blocks:
            x_tokens = blk(x_tokens, c)

        x_tokens = self.final_layer(x_tokens, c)
        imgs = self.unpatchify(x_tokens)  # (N, C_data, H, W)
        return imgs


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed],
            axis=0,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb