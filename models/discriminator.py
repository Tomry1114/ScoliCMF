"""Conditional PatchGAN discriminator (pix2pix-style) for ScoliCMF.
D(cat([x_pre, img])) -> real/fake patch map. Pushes generator output onto the sharp
real-postop manifold instead of the blurry pixel-wise conditional mean (the determinstic
source-anchored bridge + L1 collapses to E[x_post|x_pre] = mean = blur)."""
import torch
import torch.nn as nn


class PatchGAN(nn.Module):
    def __init__(self, in_ch=2, base=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        c = base
        for i in range(1, n_layers):
            nc = min(base * 2 ** i, 512)
            layers += [nn.Conv2d(c, nc, 4, 2, 1), nn.InstanceNorm2d(nc), nn.LeakyReLU(0.2, True)]
            c = nc
        nc = min(base * 2 ** n_layers, 512)
        layers += [nn.Conv2d(c, nc, 4, 1, 1), nn.InstanceNorm2d(nc), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(nc, 1, 4, 1, 1)]               # patch logits
        self.net = nn.Sequential(*layers)

    def forward(self, x_pre, img):
        return self.net(torch.cat([x_pre, img], dim=1))     # (B,1,h,w) patch logits


def d_hinge_loss(d_real, d_fake):
    return torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()


def g_hinge_loss(d_fake):
    return -d_fake.mean()
