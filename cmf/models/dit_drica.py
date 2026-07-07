"""MFDiTDRICA — MeanFlow DiT with the FGA image path replaced by DRICA cross-attention.

Current state z_t and pre-op image are encoded by SEPARATE encoders (no early concat, no global mean
pooling — those destroyed the thoracic/thoracolumbar/lumbar + left/right spatial evidence). They interact
ONLY through DRICA blocks inserted at a few depths. Diagnosis is passed explicitly (no residual model state).
"""
import numpy as np, torch, torch.nn as nn
from models.dit import PatchEmbed, TimestepEmbedder, DiTBlock, FinalLayer, get_2d_sincos_pos_embed
from models.drica import IntervalEncoder, DRICABlock


class MFDiTDRICA(nn.Module):
    def __init__(self, img_size=(480, 240), patch_size=8, data_channels=1, cond_channels=1,
                 dim=384, depth=12, num_heads=6, mlp_ratio=4.0, attn_type="vanilla", inner_lr=0.25,
                 cpe=False, rope=True, drica_layer_ids=(2, 6, 10)):
        super().__init__()
        self.data_channels = data_channels; self.cond_channels = cond_channels
        self.out_channels = data_channels; self.patch_size = patch_size
        self.current_embedder = PatchEmbed(img_size, patch_size, data_channels, dim)   # z_t (current post-op state)
        self.source_embedder  = PatchEmbed(img_size, patch_size, cond_channels, dim)   # x_pre (pre-op evidence)
        self.gh, self.gw = self.current_embedder.gh, self.current_embedder.gw
        N = self.current_embedder.num_patches
        self.current_pos_embed = nn.Parameter(torch.zeros(1, N, dim), requires_grad=False)
        self.source_pos_embed  = nn.Parameter(torch.zeros(1, N, dim), requires_grad=False)
        self.t_embedder = TimestepEmbedder(dim); self.r_embedder = TimestepEmbedder(dim)
        self.interval_encoder = IntervalEncoder(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, mlp_ratio, attn_type, self.gh, self.gw,
                                              inner_lr, cpe, rope=rope) for _ in range(depth)])
        self.drica_layer_ids = tuple(sorted(set(drica_layer_ids)))
        self.drica_blocks = nn.ModuleDict({str(i): DRICABlock(dim, num_heads, self.gh, self.gw)
                                           for i in self.drica_layer_ids})
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        pe = torch.from_numpy(get_2d_sincos_pos_embed(self.current_pos_embed.shape[-1], self.gh, self.gw)).float().unsqueeze(0)
        self.current_pos_embed.data.copy_(pe); self.source_pos_embed.data.copy_(pe)
        # re-zero DRICA residual heads (xavier above overwrote them)
        for i in self.drica_layer_ids:
            b = self.drica_blocks[str(i)]
            nn.init.zeros_(b.output_proj.weight); nn.init.zeros_(b.output_proj.bias)
        # AdaLN gates + final layer zero-init (DiT convention)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN_modulation[-1].weight); nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight); nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight); nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        c, p, gh, gw = self.out_channels, self.patch_size, self.gh, self.gw
        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, gh * p, gw * p)

    def forward(self, x, t, r, cond_img, diagnosis, return_aux=False):
        current = self.current_embedder(x) + self.current_pos_embed
        source  = self.source_embedder(cond_img) + self.source_pos_embed
        t_emb = self.t_embedder(t); r_emb = self.r_embedder(r)
        p = t_emb + r_emb; q = t_emb - r_emb
        interval_ctx, scales = self.interval_encoder(p, q)
        c = p                                                       # AdaLN conditioning = interval only
        rp, dp, jp = diagnosis["region"], diagnosis["direction"], diagnosis["joint"]
        for i, blk in enumerate(self.blocks):
            current = blk(current, c)
            if i in self.drica_layer_ids:
                current = self.drica_blocks[str(i)](current, source, interval_ctx, scales, rp, dp, jp)
        return self.unpatchify(self.final_layer(current, c))
