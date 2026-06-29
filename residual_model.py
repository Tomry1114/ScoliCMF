"""Step-4 residual-correction pilot (doc/correction_grounded_v2.md).

Frozen Bridge produces u_base; a NEW correction-grounded branch predicts ONLY the
Bridge residual e* = u* - sg(u_base). The branch = Correction-Potential SCM
(A_phi, full-interval nonzero) + Correction-Aware learned basis Q_phi + soft
harmonic modulation; injected via a multiplicative residual head with the strict
invariant c_dyn=0 => u_corr=0 (clean dyn-off ablation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sc_pga import ConvStem, RMSNormNA, path_laplacian
from legendre import LegendreBasis


class DynamicCorrectionConditioner(nn.Module):
    """Pre-op tokens -> correction potential A_phi -> interval secant a_bar_{r,t}
    -> correction-aware learned basis Q_phi + soft harmonic gamma -> per-patch c_dyn.
    NO m_static, NO time-embedding leak (time enters only via the secant of A_phi)."""
    def __init__(self, img_size, dim, patch_size, J=12, K=4, Kt=2,
                 beta=40.0, eta=4.0, cond_mode="secant"):
        super().__init__()
        self.dim, self.J, self.K, self.Kt = dim, J, K, Kt
        self.beta, self.eta, self.cond_mode = beta, eta, cond_mode
        ih, iw = img_size
        self.gh, self.gw = ih // patch_size, iw // patch_size
        # --- token extractor (independent copy; pre-op only) ---
        self.stem = ConvStem(1, dim)
        self.q = nn.Parameter(torch.randn(J, dim) * 0.02)
        self.Wf = nn.Linear(dim, dim, bias=False)
        self.register_buffer("mu", torch.linspace(0, 1, J))
        # --- correction potential A_phi: a0 + Kt zero-mean modes ---
        self.f_corr = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.g = nn.ModuleList([nn.Linear(dim, dim) for _ in range(Kt)])
        self.leg = LegendreBasis(Kt)
        # --- correction-aware learned orthonormal basis Q_phi(B_pre) ---
        h = max(64, dim // 2)
        self.basis_tok = nn.Sequential(nn.Linear(dim, h), nn.GELU())
        self.basis_mix = nn.Linear(J * h, J * h)
        self.basis_out = nn.Linear(h, K)
        # --- soft harmonic gate gamma_{r,t} in (0,1) ---
        self.hfgate = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
        self.register_buffer("Lpath", path_laplacian(J))
        self.rmsna = RMSNormNA()

    def _xc_cubic(self, x_pre, ygrid):
        B = x_pre.shape[0]
        f = F.adaptive_avg_pool2d(x_pre, (self.gh, self.gw))[:, 0]
        xcoord = torch.linspace(0, 1, self.gw, device=x_pre.device)
        mass = f.sum(-1).clamp_min(1e-6)
        cx = (f * xcoord).sum(-1) / mass
        yrow = torch.linspace(0, 1, self.gh, device=x_pre.device)
        Vr = torch.stack([yrow ** k for k in range(4)], -1)
        coeff = torch.linalg.lstsq(Vr.unsqueeze(0).expand(B, -1, -1), cx.unsqueeze(-1)).solution
        Vq = torch.stack([ygrid ** k for k in range(4)], -1)
        return (Vq.unsqueeze(0) @ coeff).squeeze(-1)

    def tokens(self, x_pre):
        """-> (Btok (B,J,D), pi (B,J,N))."""
        Fm = self.stem(x_pre); _, D, Hf, Wf = Fm.shape
        Ff = Fm.flatten(2).transpose(1, 2)
        ygr = torch.linspace(0, 1, Hf, device=x_pre.device).view(Hf, 1).expand(Hf, Wf).reshape(-1)
        xgr = torch.linspace(0, 1, Wf, device=x_pre.device).view(1, Wf).expand(Hf, Wf).reshape(-1)
        xc = self._xc_cubic(x_pre, ygr)
        qn = F.normalize(self.q, dim=-1); fn = F.normalize(self.Wf(Ff), dim=-1)
        content = torch.einsum("jd,bnd->bjn", qn, fn)
        spatial = (-self.beta * (ygr[None, None, :] - self.mu[None, :, None]) ** 2
                   - self.eta * (xgr[None, None, :] - xc[:, None, :]) ** 2)
        pi = torch.softmax(content + spatial, dim=-1)
        Btok = torch.einsum("bjn,bnd->bjd", pi, Ff)
        return Btok, pi, Ff, (Hf, Wf)

    def basis(self, Btok):
        b = Btok.shape[0]
        z = self.basis_tok(Btok)
        z = F.gelu(self.basis_mix(z.reshape(b, -1)).reshape(b, self.J, -1))
        Q, _ = torch.linalg.qr(self.basis_out(z), mode="reduced")   # (B,J,K)
        return Q

    def _Aphi(self, a0, A, tt):                                    # tt (B,) -> (B,J,D)
        b = a0.shape[0]
        lt = self.leg.eval(tt)                                     # (B,Kt)
        s = sum(lt[:, k].view(b, 1, 1) * A[k] for k in range(self.Kt))
        ttv = tt.view(b, 1, 1)
        return (1 - ttv) * a0 + ttv * (1 - ttv) * s

    def forward(self, x_pre, r, t, t_emb=None, r_emb=None):
        b = x_pre.shape[0]
        Btok, pi, Ff, (Hf, Wf) = self.tokens(x_pre)
        a0 = self.f_corr(Btok)                                     # (B,J,D)
        A = [self.rmsna(self.g[k](Btok)) for k in range(self.Kt)]
        if self.cond_mode == "static":
            abar = a0
        elif self.cond_mode == "point":
            abar = self._Aphi(a0, A, (r + t) / 2)                  # single-point potential
        else:                                                      # secant (default)
            Delta = (t - r).view(b, 1, 1).clamp_min(1e-3)
            abar = (self._Aphi(a0, A, r) - self._Aphi(a0, A, t)) / Delta
        Q = self.basis(Btok)                                       # (B,J,K)
        Pi = Q @ Q.transpose(-1, -2)                               # (B,J,J)
        low = torch.einsum("bjk,bkd->bjd", Pi, abar)
        high = abar - low
        isum = torch.stack([t - r, (t + r) / 2], -1)               # (B,2)
        gamma = torch.sigmoid(self.hfgate(isum)).view(b, 1, 1)
        m_dyn = low + gamma * high                                 # (B,J,D)
        # 2D mass-preserving back-projection to patches (NO time-emb add)
        bw = pi.transpose(1, 2)                                    # (B,N,J)
        c_grid = torch.einsum("bnj,bjd->bnd", bw, m_dyn) * (bw.shape[1] / self.J)
        c_grid = c_grid.transpose(1, 2).reshape(b, self.dim, Hf, Wf)
        c_grid = F.interpolate(c_grid, size=(self.gh, self.gw), mode="bilinear", align_corners=False)
        c_patch = c_grid.flatten(2).transpose(1, 2)                # (B, gh*gw, D)
        aux = {"pi": pi, "Btok": Btok, "Q": Q, "Pi": Pi,
               "gamma": gamma.mean().detach(),
               "abar_rms": abar.detach().pow(2).mean().sqrt(),
               "a0_rms": a0.detach().pow(2).mean().sqrt()}
        return c_patch, aux

    def l_harm(self, Q):                                           # tr(Q^T Lpath Q) mean
        LQ = torch.einsum("jk,bkc->bjc", self.Lpath.to(Q.device), Q)
        return torch.einsum("bjc,bjc->b", Q, LQ).mean()


class ResidualConvHead(nn.Module):
    """Bias-free decode: input 0 -> output 0 (so c_dyn=0 => u_corr=0)."""
    def __init__(self, dim, patch_size, gh, gw, out_ch):
        super().__init__()
        self.gh, self.gw, self.dim = gh, gw, dim
        hid = max(32, dim // 2)
        self.proj = nn.Conv2d(dim, hid * patch_size * patch_size, 1, bias=False)
        self.ps = nn.PixelShuffle(patch_size)
        self.refine = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(hid, out_ch, 3, padding=1, bias=False))

    def forward(self, h):                                          # (B,T,D)
        x = h.transpose(1, 2).reshape(h.shape[0], self.dim, self.gh, self.gw)
        return self.refine(self.ps(self.proj(x)))


class DynamicResidualHead(nn.Module):
    """Multiplicative gating: h_corr = W_h(h_base) ⊙ tanh(W_g(c_dyn)) + W_c(c_dyn).
    All bias-free + bias-free decode => c_dyn=0 yields u_corr=0 EXACTLY."""
    def __init__(self, dim, patch_size, gh, gw, out_ch):
        super().__init__()
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.decode = ResidualConvHead(dim, patch_size, gh, gw, out_ch)

    def forward(self, h_base, c_dyn):
        g = torch.tanh(self.gate(c_dyn))
        h_corr = self.h_proj(h_base) * g + self.c_proj(c_dyn)
        return self.decode(h_corr)


class ResidualScoliCMF(nn.Module):
    def __init__(self, bridge, dyn_cond, corr_head):
        super().__init__()
        self.bridge = bridge
        self.dyn_cond = dyn_cond
        self.corr_head = corr_head
        self.dyn_off = False
        for p in self.bridge.parameters():
            p.requires_grad = False
        self.bridge.eval()

    def train(self, mode=True):                                   # keep frozen bridge in eval
        super().train(mode); self.bridge.eval(); return self

    def forward(self, z_t, r, t, x_pre, return_aux=False):
        with torch.no_grad():
            h_base, c_base, _ = self.bridge.forward_features(z_t, r, t, x_pre)
            u_base = self.bridge.head_forward(h_base, c_base)
        if self.dyn_off:
            u = u_base
            aux = {"u_base": u_base, "u_corr": torch.zeros_like(u_base)}
            return (u, aux) if return_aux else u
        c_dyn, aux = self.dyn_cond(x_pre, r, t)
        u_corr = self.corr_head(h_base.detach(), c_dyn)
        u = u_base + u_corr
        aux["u_base"] = u_base; aux["u_corr"] = u_corr
        return (u, aux) if return_aux else u
