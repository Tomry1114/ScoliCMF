"""Shifted-Legendre time basis for SC-PGA (S5).

ell_k = shifted Legendre on [0,1], zero-mean for k>=1 (ell_1=2t-1, ell_2=6t^2-6t+1, ...).
Potential P_k(t)=int_0^t ell_k. Interval-average condition uses the ANALYTIC divided
difference D[P_k](r,t)=(P_k(t)-P_k(r))/(t-r), expanded as a polynomial in (r,t) so the
(t-r) cancels symbolically -> NO catastrophic cancellation / NaN at small Delta (R33-3,
雷区二). Derivative Gram R_kl=int_0^1 ell'_k ell'_l for the temporal Sobolev reg (R33-3).
All math in fp32 by contract.
"""
import numpy as np
import numpy.polynomial.polynomial as npp
import torch


def _shifted_legendre(K):
    polys = [np.array([1.0]), np.array([-1.0, 2.0])]          # P0=1, P1=2x-1
    for n in range(1, K):
        a = npp.polymul([-1.0, 2.0], polys[n]) * (2 * n + 1)  # (2n+1)(2x-1)P_n
        b = polys[n - 1] * n
        polys.append(npp.polysub(a, b) / (n + 1))
    return polys[:K + 1]


class LegendreBasis:
    def __init__(self, K=2, dtype=torch.float32):
        self.K = K
        self.dtype = dtype
        polys = _shifted_legendre(K)                          # ell_0..ell_K, low->high
        self.ell = polys
        self.pot = [npp.polyint(p) for p in polys]            # potentials (const 0), low->high
        # derivative Gram on [0,1]: R_kl = int_0^1 ell'_k ell'_l
        d = [npp.polyder(p) for p in polys]
        R = np.zeros((K + 1, K + 1))
        for i in range(K + 1):
            for j in range(K + 1):
                prod = npp.polymul(d[i], d[j])
                ip = npp.polyint(prod)
                R[i, j] = npp.polyval(1.0, ip) - npp.polyval(0.0, ip)
        # dynamic modes are k=1..K (drop constant k=0)
        self.R = torch.tensor(R[1:, 1:], dtype=dtype)         # (K,K)
        self._pot_t = [torch.tensor(p, dtype=dtype) for p in self.pot]
        self._ell_t = [torch.tensor(p, dtype=dtype) for p in polys]

    def eval(self, t):
        # t: (...,) -> (..., K) values of ell_1..ell_K ; pure torch (differentiable, JVP-safe)
        cols = []
        for k in range(1, self.K + 1):
            c = self._ell_t[k].to(t.device)
            v = torch.zeros_like(t)
            for i in range(c.shape[0]):
                v = v + c[i] * t.pow(i)
            cols.append(v)
        return torch.stack(cols, dim=-1)

    def potential_dd(self, r, t):
        """Analytic D[P_k](r,t) for k=1..K. r,t: (B,) -> (B,K). fp32, no (t-r) division."""
        r = r.to(self.dtype); t = t.to(self.dtype)
        cols = []
        for k in range(1, self.K + 1):
            pc = self._pot_t[k].to(t.device)                  # (deg+1,) low->high
            acc = torch.zeros_like(t)
            for i in range(1, pc.shape[0]):                   # (t^i - r^i)/(t-r) = sum_{j<i} t^j r^{i-1-j}
                s = torch.zeros_like(t)
                for j in range(i):
                    s = s + t.pow(j) * r.pow(i - 1 - j)
                acc = acc + pc[i] * s
            cols.append(acc)
        return torch.stack(cols, dim=-1)                      # (B,K)
