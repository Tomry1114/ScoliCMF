"""SC-PGA: Spinal-Chain Potential Gating Adapter (S5).

Projector library Pi_spine (J x J). The rank-matched (G_graph) projectors are
ORTHOGONAL (Pi^T=Pi, Pi^2=Pi, rank=K_g); identity is the FULL-rank no-restriction ref so G_graph isolates *which
subspace*, not soft-vs-hard or rank (R33-二). v2-Frozen uses residual-W2 adjacency
(remove nominal axial spacing; Bures cov term; w_min>0 floor) on a DETACHED token graph.
Token extractor / condition field / projection-last injection / L_time follow next.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormNA(nn.Module):
    """No-affine RMSNorm (pure /std along feature dim). Applied BEFORE Pi (Norm->Pi)."""
    def forward(self, x):
        return F.normalize(x, dim=-1) * (x.shape[-1] ** 0.5)


def path_laplacian(J, w=None, device=None):
    A = torch.zeros(J, J, device=device)
    w = torch.ones(J - 1, device=device) if w is None else w
    idx = torch.arange(J - 1, device=device)
    A[idx, idx + 1] = w
    A[idx + 1, idx] = w
    d = A.sum(1).clamp_min(1e-9)
    Dm = torch.diag(d.pow(-0.5))
    return torch.eye(J, device=device) - Dm @ A @ Dm


def _topk_eigvecs(M, k, low=True):
    M = (M + M.transpose(-1, -2)) / 2
    w, V = torch.linalg.eigh(M)                       # ascending eigenvalues
    order = torch.arange(k) if low else torch.arange(M.shape[-1] - k, M.shape[-1])
    return V[:, order]                                # (J,k)


def build_static_projector(name, J, Kg, seed=0):
    """Patient-independent rank-Kg orthogonal projector (precompute as buffer)."""
    if name == "identity":
        return torch.eye(J)                            # no-restriction reference (full rank)
    if name == "dct":
        n = torch.arange(J).float()
        cols = []
        for k in range(Kg):
            v = torch.cos(math.pi * (n + 0.5) * k / J)
            cols.append(v / (v.norm() + 1e-9))
        U = torch.stack(cols, 1)
        return U @ U.T
    if name == "random":
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(J, Kg, generator=g))
        return Q @ Q.T
    if name == "gaussian":
        i = torch.arange(J).float()
        G = torch.exp(-((i[:, None] - i[None, :]) ** 2) / (2 * (J / 4.0) ** 2))
        U = _topk_eigvecs(G, Kg, low=False)            # top eigenspace of smoother
        return U @ U.T
    if name == "toeplitz":
        i = torch.arange(J).float()
        T = torch.clamp(1.0 - 0.3 * torch.abs(i[:, None] - i[None, :]), min=0.0)
        U = _topk_eigvecs(T, Kg, low=False)
        return U @ U.T
    if name == "v1":
        U = _topk_eigvecs(path_laplacian(J), Kg, low=True)   # low graph-freq of path
        return U @ U.T
    raise ValueError(name)


def build_v2_projector(res_means, sqrt_vars, Kg, tau=1.0, w_min=0.1, lam_sigma=0.5):
    """Per-sample patient-weighted path projector from DETACHED residual token stats.
    res_means: (B,J,2) residual centroid (nominal axial spacing removed, normalized).
    sqrt_vars: (B,J,2) sqrt of token spatial variance (for Bures cov term).
    Returns (B,J,J) rank-Kg orthogonal projectors (no grad through graph)."""
    res_means = res_means.detach()
    sqrt_vars = sqrt_vars.detach()
    B, J, _ = res_means.shape
    out = []
    for b in range(B):
        dm = ((res_means[b, 1:] - res_means[b, :-1]) ** 2).sum(-1)            # (J-1,)
        db = ((sqrt_vars[b, 1:] - sqrt_vars[b, :-1]) ** 2).sum(-1)            # Bures (diag)
        cost = dm + lam_sigma * db                                           # (J-1,)
        scale = (tau * cost.median()).clamp_min(1e-4)                        # issue-5: per-sample median scale (not fixed tau) -> weights actually spread
        w = w_min + (1 - w_min) * torch.exp(-cost / scale)                   # (J-1,) in (w_min,1]
        U = _topk_eigvecs(path_laplacian(J, w, device=res_means.device), Kg, low=True)
        out.append(U @ U.T)
    return torch.stack(out, 0)


# ===================== SC-PGA module (token extractor + condition + injection) =====================
from legendre import LegendreBasis


class ConvStem(nn.Module):
    """Small learnable feature stem (pixel-first stand-in for frozen AE features)."""
    def __init__(self, in_ch, dim, down=16):
        super().__init__()
        ch = max(32, dim // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(ch, dim, 3, stride=down // 4, padding=1),
        )

    def forward(self, x):
        return self.net(x)                       # (B, dim, Hf, Wf)


class SCPGA(nn.Module):
    """Returns per-patch conditioning (B, gh*gw, D); stashes last_l_time. Base PGA = proj 'identity'."""
    def __init__(self, img_size, dim, patch_size, J=12, Kg=4, Kt=2,
                 beta=40.0, eta=4.0, tau=1.0, w_min=0.1, lam_sigma=0.5, proj="v2", dyn_off=False,
                 cond_mode="secant_full"):
        super().__init__()
        self.dim, self.J, self.Kg, self.Kt = dim, J, Kg, Kt
        self.beta, self.eta, self.tau, self.w_min, self.proj = beta, eta, tau, w_min, proj
        self.lam_sigma = lam_sigma
        self.dyn_off = dyn_off   # #7 ablation: force m_dyn=0 (only static conditioning)
        self.cond_mode = cond_mode   # issue-6: static|point|secant_mean|secant_full (fair SCM ablation)
        ih, iw = img_size
        self.gh, self.gw = ih // patch_size, iw // patch_size

        self.stem = ConvStem(1, dim)
        self.q = nn.Parameter(torch.randn(J, dim) * 0.02)
        self.Wf = nn.Linear(dim, dim, bias=False)
        self.register_buffer("mu", torch.linspace(0, 1, J))                  # nominal axial centers
        self.A0 = nn.Linear(dim, dim)
        self.g = nn.ModuleList([nn.Linear(dim, dim) for _ in range(Kt)])     # dynamic mode adapters
        self.rmsna = RMSNormNA()
        self.M_static = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.M_dyn = nn.Sequential(nn.Linear(2 * dim, dim, bias=False), nn.GELU(), nn.Linear(dim, dim, bias=False))
        self.time = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.leg = LegendreBasis(Kt)
        if proj != "v2":
            self.register_buffer("Pi_static", build_static_projector(proj, J, Kg))
        self.last_l_time = 0.0
        self.perm = None  # set to a (J,) LongTensor for S_order topology-only intervention

    def _xc_cubic(self, x_pre, ygrid):
        """Fit cubic x_c(y) to per-row horizontal centroid of x_pre; eval at ygrid (N,). -> (B,N)."""
        B = x_pre.shape[0]
        f = F.adaptive_avg_pool2d(x_pre, (self.gh, self.gw))[:, 0]            # (B,gh,gw)
        xcoord = torch.linspace(0, 1, self.gw, device=x_pre.device)
        mass = f.sum(-1).clamp_min(1e-6)
        cx = (f * xcoord).sum(-1) / mass                                     # (B,gh) row centroid
        yrow = torch.linspace(0, 1, self.gh, device=x_pre.device)
        Vr = torch.stack([yrow ** k for k in range(4)], -1)                  # (gh,4)
        coeff = torch.linalg.lstsq(Vr.unsqueeze(0).expand(B, -1, -1), cx.unsqueeze(-1)).solution  # (B,4,1)
        Vq = torch.stack([ygrid ** k for k in range(4)], -1)                 # (N,4)
        return (Vq.unsqueeze(0) @ coeff).squeeze(-1)                         # (B,N)

    def _proj_apply(self, Pi, X):
        return torch.einsum("bjk,bkd->bjd", Pi, X) if Pi.dim() == 3 else torch.einsum("jk,bkd->bjd", Pi, X)

    def forward(self, x_pre, r, t, t_emb, r_emb):
        Bn = x_pre.shape[0]
        Fm = self.stem(x_pre)                                                # (B,D,Hf,Wf)
        _, D, Hf, Wf = Fm.shape
        Ff = Fm.flatten(2).transpose(1, 2)                                   # (B,N,D)
        ygr = torch.linspace(0, 1, Hf, device=x_pre.device).view(Hf, 1).expand(Hf, Wf).reshape(-1)
        xgr = torch.linspace(0, 1, Wf, device=x_pre.device).view(1, Wf).expand(Hf, Wf).reshape(-1)
        xc = self._xc_cubic(x_pre, ygr)                                      # (B,N)

        qn = F.normalize(self.q, dim=-1)                                     # (J,D)
        fn = F.normalize(self.Wf(Ff), dim=-1)                               # (B,N,D)
        content = torch.einsum("jd,bnd->bjn", qn, fn)                        # (B,J,N) cosine
        spatial = (-self.beta * (ygr[None, None, :] - self.mu[None, :, None]) ** 2
                   - self.eta * (xgr[None, None, :] - xc[:, None, :]) ** 2)
        pi = torch.softmax(content + spatial, dim=-1)                        # (B,J,N)

        Btok = torch.einsum("bjn,bnd->bjd", pi, Ff)                          # (B,J,D)
        grid = torch.stack([ygr, xgr], -1)                                  # (N,2)
        pos = torch.einsum("bjn,nc->bjc", pi, grid)                          # (B,J,2) mean (y,x)
        var = (torch.einsum("bjn,nc->bjc", pi, grid ** 2) - pos ** 2).clamp_min(0)
        res = torch.stack([pos[..., 0] - self.mu[None, :], pos[..., 1] - 0.5], -1)   # residual (y-mu, x-0.5)
        # token-diversity loss (combat collapse: tokens must cover distinct regions & decorrelated features)
        _eye = torch.eye(self.J, device=Btok.device)[None]
        _bn = F.normalize(Btok, dim=-1)
        _feat_off = (torch.einsum("bid,bjd->bij", _bn, _bn) - _eye).abs().sum((1, 2)) / (self.J * (self.J - 1))
        _pi_off = (torch.einsum("bin,bjn->bij", pi, pi) * (1 - _eye)).sum((1, 2)) / (self.J * (self.J - 1))
        l_tokdiv = (_feat_off + _pi_off).mean()

        if self.proj == "v2":
            Pi = build_v2_projector(res, var.sqrt(), self.Kg, self.tau, self.w_min, self.lam_sigma)  # (B,J,J) detached
        else:
            Pi = self.Pi_static
        if self.perm is not None:
            idx = self.perm.to(Pi.device)
            Pi = Pi[..., idx, :][..., :, idx]   # P^T Pi P : wrong chain topology, tokens unchanged

        A = [self.rmsna(self.g[k](Btok)) for k in range(self.Kt)]            # full-rank modes; projection happens ONCE at end (projection-last)
        # interval-aligned dynamic condition; cond_mode = point vs secant (issue-6 fair ablation, same params)
        zeros = torch.zeros_like(A[0])
        cbar = trend = zeros
        if self.cond_mode == "point":
            lm = self.leg.eval((r + t) / 2)                                  # (B,Kt) midpoint single-point
            cbar = sum(lm[:, k].view(Bn, 1, 1) * A[k] for k in range(self.Kt))
            dyn_in = torch.cat([cbar, zeros], -1)
        elif self.cond_mode in ("secant_mean", "secant_full"):
            dd = self.leg.potential_dd(r, t)                                 # (B,Kt) secant mean cbar_{r,t}
            cbar = sum(dd[:, k].view(Bn, 1, 1) * A[k] for k in range(self.Kt))
            if self.cond_mode == "secant_full":
                lt, lr = self.leg.eval(t), self.leg.eval(r)
                Delta = (t - r).clamp_min(1e-3)
                trend = sum(((1 - Delta) * (lt[:, k] - lr[:, k]) / Delta).view(Bn, 1, 1) * A[k] for k in range(self.Kt))
            dyn_in = torch.cat([cbar, trend], -1)
        else:                                                                # static: no dynamic branch
            dyn_in = None

        e = self.time(torch.cat([t_emb + r_emb, t_emb - r_emb], -1))         # (B,D); time enters static+patch ONLY
        m_static = self.M_static(Btok + self.A0(Btok) + e[:, None, :])       # (B,J,D)
        if self.dyn_off or dyn_in is None:
            raw_dyn = None
            m_dyn = torch.zeros_like(m_static)                               # full-span/static -> dynamic truly inert
        else:
            raw_dyn = self.M_dyn(dyn_in)                                     # (B,J,D) BEFORE projection
            m_dyn = self._proj_apply(Pi, raw_dyn)                            # projection-last; bias-free -> in@0 gives 0
        m = m_static + m_dyn                                                 # (B,J,D)
        aux = {"m_dyn_rms": m_dyn.detach().pow(2).mean().sqrt(),
               "m_static_rms": m_static.detach().pow(2).mean().sqrt(),
               "A_rms": A[0].detach().pow(2).mean().sqrt(),
               "cbar_rms": cbar.detach().pow(2).mean().sqrt(),
               "trend_rms": trend.detach().pow(2).mean().sqrt()}
        if raw_dyn is not None:                                             # R_removed: how much projection strips from dynamic feat
            _rd = raw_dyn.detach()
            aux["R_removed"] = (_rd - m_dyn.detach()).norm() / _rd.norm().clamp_min(1e-6)
        with torch.no_grad():                                              # P1 token-collapse diagnostics
            bn = F.normalize(Btok.detach(), dim=-1)
            cosm = torch.einsum("bid,bjd->bij", bn, bn) - torch.eye(self.J, device=bn.device)[None]
            aux["tok_cos"] = cosm.abs().sum((1, 2)).mean() / (self.J * (self.J - 1))
            if raw_dyn is not None:
                ev = torch.linalg.svdvals(raw_dyn.detach()).pow(2)          # (B, min(J,D))
                aux["E_top4"] = (ev[:, :4].sum(1) / ev.sum(1).clamp_min(1e-9)).mean()
        aux["l_tokdiv"] = l_tokdiv
        if getattr(self, "diag", False):                                   # diagnostic stash (off in training)
            self._diag = {"raw_dyn": raw_dyn, "Pi": Pi, "pi": pi, "m_dyn": m_dyn.detach(), "m_static": m_static.detach()}

        # 2D BACK-PROJECTION (priority-1 fix): scatter modulated spinal tokens to the patient s real
        # spine region via token attention pi (B,J,N), N=Hf*Wf -- NOT row-broadcast. Preserves lateral
        # (left/right) spine geometry instead of collapsing it into channels.
        # spatial-mass gate (P0-2): pi sums to 1 over N per token, so each pixel carries the token mass
        # it actually receives. DO NOT renormalize over J (that gives background a full-amplitude mix).
        # *N/J restores the mean modulation amplitude to ~1 while keeping spine>>background support.
        bw = pi.transpose(1, 2)                                              # (B,N,J)
        c_grid = torch.einsum("bnj,bjd->bnd", bw, m) * (bw.shape[1] / self.J)  # (B,N,D) mass-preserving
        c_grid = c_grid.transpose(1, 2).reshape(Bn, D, Hf, Wf)               # (B,D,Hf,Wf)
        c_grid = F.interpolate(c_grid, size=(self.gh, self.gw), mode="bilinear", align_corners=False)
        c_patch = c_grid.flatten(2).transpose(1, 2)                          # (B, gh*gw, D)
        c_patch = c_patch + e[:, None, :]

        # temporal Sobolev L_time = sum_kl R_kl <A_k,A_l>
        R = self.leg.R.to(x_pre.device)
        lt_val = sum(R[k, l] * (A[k] * A[l]).mean() for k in range(self.Kt) for l in range(self.Kt))
        aux["l_time"] = lt_val
        return c_patch, aux


def build_scpga(cfg, H, W, proj_override=None):
    """SINGLE source of truth for SCPGA construction -- train/eval/diag MUST all use this so
    cond_mode/tau/w_min/lam_sigma never silently diverge (cond_mode is NOT in state_dict)."""
    mc = cfg["model"]
    return SCPGA(
        img_size=(H, W), dim=mc["dim"], patch_size=mc["patch_size"],
        J=mc.get("J", 12), Kg=mc.get("Kg", 4), Kt=mc.get("Kt", 2),
        beta=mc.get("beta", 40.0), eta=mc.get("eta", 4.0),
        tau=mc.get("tau", 1.0), w_min=mc.get("w_min", 0.1), lam_sigma=mc.get("lam_sigma", 0.5),
        proj=proj_override or mc.get("proj", "v2"),
        dyn_off=mc.get("dyn_off", False),
        cond_mode=mc.get("cond_mode", "secant_full"),
    )
