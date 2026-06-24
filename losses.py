"""Interval sampling + loss helpers for source-anchored MeanFlow (R33 §8).

Sampling fixes (R33-6): sample Delta first then r~U(0,1-Delta) (so t<=1); L_span covers
broad spans (40% local / 40% broad / 20% full); triplets enforce min sub-interval.
"""
import torch


def _sample_delta(B, device, dmin=1e-3):
    u = torch.rand(B, device=device)
    delta = torch.empty(B, device=device)
    loc, brd, full = u < 0.4, (u >= 0.4) & (u < 0.8), u >= 0.8
    # local: Gamma(shape=2, scale=0.15) = sum of two Exp(scale) ; truncate <0.3
    g = (-torch.log(torch.rand(B, device=device).clamp_min(1e-12))
         - torch.log(torch.rand(B, device=device).clamp_min(1e-12))) * 0.15
    delta = torch.where(loc, g.clamp(dmin, 0.3), delta)
    delta = torch.where(brd, torch.rand(B, device=device).clamp(dmin, 1.0), delta)
    delta = torch.where(full, torch.ones(B, device=device), delta)
    return delta.clamp(dmin, 1.0)


def sample_rt(B, device, dmin=1e-3, local_only=False):
    """Return ordered (r, t), r<t, t<=1. local_only -> small-Delta only (for L_ST)."""
    if local_only:
        g = (-torch.log(torch.rand(B, device=device).clamp_min(1e-12))
             - torch.log(torch.rand(B, device=device).clamp_min(1e-12))) * 0.15
        delta = g.clamp(dmin, 0.3)
    else:
        delta = _sample_delta(B, device, dmin)
    r = torch.rand(B, device=device) * (1.0 - delta)
    t = (r + delta).clamp(max=1.0)
    return r, t


def sample_triplet(B, device, dmin=1e-3):
    """Ordered r<s<t with t-r>=2*dmin, s in (r+dmin, t-dmin). For L_comp/L_roll (S4)."""
    delta = (_sample_delta(B, device, 2 * dmin)).clamp(2 * dmin, 1.0)
    r = torch.rand(B, device=device) * (1.0 - delta)
    t = (r + delta).clamp(max=1.0)
    s = r + dmin + torch.rand(B, device=device) * (t - r - 2 * dmin).clamp_min(0.0)
    return r, s, t
