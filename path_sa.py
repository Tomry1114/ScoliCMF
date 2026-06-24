"""Source-anchored transport path (S2, pixel-space).

z_t = x_pre + alpha_g(t) * delta + sigma_m * sin^2(pi t) * eps,  delta = x_post - x_pre
  alpha_g(t) = (e^{g(1-t)} - 1) / (e^g - 1)   # alpha(0)=1 -> z_0=x_post ; alpha(1)=0 -> z_1=x_pre
  sigma(t)   = sigma_m * sin^2(pi t)          # sigma(0)=sigma(1)=0, sigma'(0)=sigma'(1)=0
Convention (matches method_v1 §2.2): t=1 is SOURCE (x_pre), t=0 is TARGET (x_post).
Instantaneous sample velocity v*_t = d z_t / dt = alpha_g'(t) delta + sigma_m pi sin(2 pi t) eps.

NOTE the doc writes alpha(0)=1, alpha(1)=0 (t flows 1->0). The plain MeanFlow code uses
z=(1-t)y+t e (t: 0->1 noise). Here we keep the doc convention; meanflow_sa handles signs.
"""
import math
import torch


class SourceAnchoredPath:
    def __init__(self, gamma: float = 2.0, sigma_m: float = 0.0):
        # gamma fixed (v1, not learned); sigma_m=0 first (deterministic, no source noise / UQ).
        self.gamma = float(gamma)
        self.sigma_m = float(sigma_m)
        self._eg = math.exp(self.gamma) - 1.0  # e^gamma - 1

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        if abs(self.gamma) < 1e-6:                       # gamma->0  =>  alpha = 1 - t (linear)
            return 1.0 - t
        return (torch.exp(self.gamma * (1.0 - t)) - 1.0) / self._eg

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        if abs(self.gamma) < 1e-6:
            return -torch.ones_like(t)
        return (-self.gamma * torch.exp(self.gamma * (1.0 - t))) / self._eg

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_m * torch.sin(math.pi * t) ** 2

    def z_t(self, x_pre, x_post, t, eps=None):
        """t broadcastable to image shape; returns z_t (and uses eps if sigma_m>0)."""
        delta = x_post - x_pre
        z = x_pre + self.alpha(t) * delta
        if self.sigma_m > 0.0:
            if eps is None:
                eps = torch.randn_like(x_pre)
            z = z + self.sigma(t) * eps
        return z

    def v_star(self, x_pre, x_post, t, eps=None):
        """Instantaneous sample velocity dz/dt."""
        delta = x_post - x_pre
        v = self.alpha_dot(t) * delta
        if self.sigma_m > 0.0:
            if eps is None:
                eps = torch.zeros_like(x_pre)
            v = v + self.sigma_m * math.pi * torch.sin(2.0 * math.pi * t) * eps
        return v
