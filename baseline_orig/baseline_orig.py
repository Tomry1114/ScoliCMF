"""Original ScoliCMF baseline, faithfully reconstructed in the matched harness.
Method = conditional MeanFlow (NOISE -> image, cond=x_pre via channel-cat + FGA gate),
exactly as ScoliCMF-main/{meanflow.py,models/dit.py}. Only non-square (480x240) adaptation +
timm-free primitives reused from models.sc_dit (so the BACKBONE primitives match my method;
the only scientific variables are the generative formulation + conditioning + losses + arch hparams)."""
import os, sys, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import Normalizer, stopgrad, adaptive_l2_loss
from models.sc_dit import (PatchEmbed, DiTBlock, FinalLayer, TimestepEmbedder,
                           get_2d_sincos_pos_embed)


class MFDiT_orig(nn.Module):
    """Faithful re-impl of ScoliCMF-main MFDiT: x_in=cat([z,cond]); FGA = cond mean-pool gated by
    sigmoid(linear([p,q])), p=t+r,q=t-r; global adaLN(c). Non-square via reused primitives."""
    def __init__(self, img_size=(480, 240), patch_size=8, data_channels=1, cond_channels=1,
                 dim=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.data_channels, self.cond_channels = data_channels, cond_channels
        self.in_channels = data_channels + cond_channels
        self.out_channels = data_channels
        self.patch_size = patch_size
        ih, iw = img_size
        self.gh, self.gw = ih // patch_size, iw // patch_size
        self.x_embedder = PatchEmbed(img_size, patch_size, self.in_channels, dim)
        self.cond_embedder = PatchEmbed(img_size, patch_size, cond_channels, dim)
        self.cond_proj = nn.Identity()
        self.tr_gate = nn.Sequential(nn.SiLU(), nn.Linear(2 * dim, dim))
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)
        self._init()

    def _init(self):
        def bi(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        self.apply(bi)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.gh, self.gw)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN[-1].weight, 0); nn.init.constant_(blk.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].weight, 0); nn.init.constant_(self.final_layer.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0); nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c, p, gh, gw = self.out_channels, self.patch_size, self.gh, self.gw
        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, gh * p, gw * p)

    def forward(self, x, t, r, cond_img=None):
        x_in = torch.cat([x, cond_img], dim=1) if (self.cond_channels > 0 and cond_img is not None) else x
        x_tok = self.x_embedder(x_in) + self.pos_embed
        t_emb, r_emb = self.t_embedder(t), self.r_embedder(r)
        p, q = t_emb + r_emb, t_emb - r_emb
        cond_emb = self.cond_proj(self.cond_embedder(cond_img).mean(dim=1))
        gate = torch.sigmoid(self.tr_gate(torch.cat([p, q], dim=-1)))
        c = p + gate * cond_emb
        for blk in self.blocks:
            x_tok = blk(x_tok, c)
        return self.unpatchify(self.final_layer(x_tok, c))


class MeanFlowOrig:
    """Faithful re-impl of ScoliCMF-main MeanFlow: z=(1-t)y+t*e (noise->image), u_tgt=v-(t-r)dudt,
    adaptive-l2. Sampling starts from PURE NOISE conditioned on x_pre. Non-square sampling via cond shape."""
    def __init__(self, channels=1, normalizer=("minmax", None, None), flow_ratio=0.75,
                 time_dist=("lognorm", -0.4, 1.0), jvp_api="autograd"):
        self.channels = channels
        self.normer = Normalizer.from_list(list(normalizer))
        self.flow_ratio = flow_ratio
        self.time_dist = list(time_dist)
        self.jvp_fn = torch.autograd.functional.jvp
        self.create_graph = True

    def sample_t_r(self, B, device):
        if self.time_dist[0] == "uniform":
            s = np.random.rand(B, 2).astype(np.float32)
        else:
            mu, sg = self.time_dist[-2], self.time_dist[-1]
            s = 1 / (1 + np.exp(-(np.random.randn(B, 2).astype(np.float32) * sg + mu)))
        t_np, r_np = np.maximum(s[:, 0], s[:, 1]), np.minimum(s[:, 0], s[:, 1])
        k = int(self.flow_ratio * B)
        if k > 0:
            idx = np.random.permutation(B)[:k]; r_np[idx] = t_np[idx]
        return torch.tensor(t_np, device=device), torch.tensor(r_np, device=device)

    def loss(self, model, y, cond_img):
        B, device = y.shape[0], y.device
        t, r = self.sample_t_r(B, device)
        t_ = rearrange(t, "b -> b 1 1 1").detach(); r_ = rearrange(r, "b -> b 1 1 1").detach()
        e = torch.randn_like(y); y_norm = self.normer.norm(y)
        z = (1 - t_) * y_norm + t_ * e
        v = e - y_norm
        def f(z_in, t_in, r_in):
            return model(z_in, t_in, r_in, cond_img)
        u, dudt = self.jvp_fn(f, (z, t, r), (v, torch.ones_like(t), torch.zeros_like(r)),
                              create_graph=self.create_graph)
        u_tgt = v - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        return adaptive_l2_loss(error), (stopgrad(error) ** 2).mean()

    @torch.no_grad()
    def sample_given_cond(self, model, cond_img, sample_steps=20):
        model.eval()
        B, _, H, W = cond_img.shape
        device = cond_img.device
        z = torch.randn(B, self.channels, H, W, device=device)
        tv = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        for i in range(sample_steps):
            t = torch.full((B,), tv[i].item(), device=device)
            r = torch.full((B,), tv[i + 1].item(), device=device)
            t_ = rearrange(t, "b -> b 1 1 1"); r_ = rearrange(r, "b -> b 1 1 1")
            v = model(z, t, r, cond_img)
            z = z - (t_ - r_) * v
        return self.normer.unnorm(z)
