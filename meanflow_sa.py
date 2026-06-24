"""Source-anchored compositional MeanFlow.
S2 L_span+L_end ; S3 +L_ST(detach JVP) ; S4 +L_comp+L_roll(EMA teacher + lambda ramp).

t=1 source(x_pre), t=0 target(x_post). T_{t->r}(z)=z-(t-r)u_theta.
L_comp (R32-fix): composed student path matches a DETACHED EMA-teacher single step
(not the live blurry single step) -> avoids early comp<->roll tug-of-war; lambda_comp
ramps from 0 while L_roll (anchor to prescribed z_r) leads early.
"""
import torch
from utils import adaptive_l2_loss, stopgrad
from path_sa import SourceAnchoredPath
from losses import sample_rt, sample_triplet


def _v4(x):
    return x.view(-1, 1, 1, 1)


class SourceAnchoredMeanFlow:
    def __init__(self, gamma=2.0, sigma_m=0.0, lambda_end=1.0, rho_end=0.25,
                 lambda_st=0.0, st_mode="detach", jvp_api="autograd",
                 lambda_comp=0.0, lambda_roll=0.0, comp_ramp_steps=2000, lambda_time=0.0):
        self.path = SourceAnchoredPath(gamma, sigma_m)
        self.sigma_m = sigma_m
        self.lambda_end, self.rho_end = lambda_end, rho_end
        self.lambda_st, self.st_mode, self.jvp_api = lambda_st, st_mode, jvp_api
        self.lambda_comp, self.lambda_roll = lambda_comp, lambda_roll
        self.comp_ramp_steps = comp_ramp_steps
        self.lambda_time = lambda_time

    def T(self, model, z, r, t, x_pre):
        return z - _v4(t - r) * model(z, r, t, x_pre)

    def _l_st(self, model, x_pre, x_post):
        B, device = x_pre.shape[0], x_pre.device
        r, t = sample_rt(B, device, local_only=True)
        t4 = _v4(t)
        eps = torch.randn_like(x_pre) if self.sigma_m > 0 else None
        z = self.path.z_t(x_pre, x_post, t4, eps)
        v_star = self.path.v_star(x_pre, x_post, t4, eps)

        def f(z_in, r_in, t_in):
            return model(z_in, r_in, t_in, x_pre)

        tangents = (v_star, torch.zeros_like(r), torch.ones_like(t))
        if self.jvp_api == "autograd":
            u, dudt = torch.autograd.functional.jvp(f, (z, r, t), tangents, create_graph=True)
        else:
            u, dudt = torch.func.jvp(f, (z, r, t), tangents)
        u_tgt = v_star - _v4(t - r) * dudt
        tgt = stopgrad(u_tgt) if self.st_mode == "detach" else u_tgt
        return adaptive_l2_loss(u - tgt)

    def _l_comp_roll(self, model, teacher, x_pre, x_post):
        B, device = x_pre.shape[0], x_pre.device
        r, s, t = sample_triplet(B, device)
        z_t = self.path.z_t(x_pre, x_post, _v4(t))
        z_r = self.path.z_t(x_pre, x_post, _v4(r))
        z_s = z_t - _v4(t - s) * model(z_t, s, t, x_pre)            # student step t->s
        z_r_comp = z_s - _v4(s - r) * model(z_s, r, s, x_pre)        # student step s->r
        with torch.no_grad():                                       # detached EMA-teacher direct t->r
            z_r_dir = z_t - _v4(t - r) * teacher(z_t, r, t, x_pre)
        l_comp = (z_r_comp - z_r_dir).abs().mean()
        l_roll = (z_r_comp - z_r).abs().mean()
        return l_comp, l_roll

    def loss(self, model, x_pre, x_post, teacher=None, step=0):
        B, device = x_pre.shape[0], x_pre.device
        r, t = sample_rt(B, device)
        r4, t4 = _v4(r), _v4(t)
        eps = torch.randn_like(x_pre) if self.sigma_m > 0 else None
        z_t = self.path.z_t(x_pre, x_post, t4, eps)
        z_r = self.path.z_t(x_pre, x_post, r4, eps)
        u, aux = model(z_t, r, t, x_pre, return_aux=True)
        target = (z_t - z_r) / (t4 - r4).clamp_min(1e-6)
        l_span = adaptive_l2_loss(u - stopgrad(target))

        logs = {"l_span": l_span.detach()}
        loss = l_span
        if "m_dyn_rms" in aux:                       # #7 SC-PGA shortcut diagnostic
            logs["m_dyn_rms"] = aux["m_dyn_rms"]; logs["m_static_rms"] = aux["m_static_rms"]
        if self.lambda_time > 0 and torch.is_tensor(aux.get("l_time")) and aux["l_time"].requires_grad:
            loss = loss + self.lambda_time * aux["l_time"]
            logs["l_time"] = aux["l_time"].detach()
        if torch.rand(()).item() < self.rho_end:
            z0 = self.T(model, x_pre, torch.zeros(B, device=device), torch.ones(B, device=device), x_pre)
            l_end = (z0 - x_post).abs().mean()
            loss = loss + self.lambda_end * l_end
            logs["l_end"] = l_end.detach()
        if self.lambda_st > 0:
            l_st = self._l_st(model, x_pre, x_post)
            loss = loss + self.lambda_st * l_st
            logs["l_st"] = l_st.detach()
        if teacher is not None and (self.lambda_comp > 0 or self.lambda_roll > 0):
            l_comp, l_roll = self._l_comp_roll(model, teacher, x_pre, x_post)
            lam_c = self.lambda_comp * min(1.0, step / max(1, self.comp_ramp_steps))   # ramp from 0
            loss = loss + lam_c * l_comp + self.lambda_roll * l_roll
            logs["l_comp"] = l_comp.detach()
            logs["l_roll"] = l_roll.detach()
        return loss, logs

    @torch.no_grad()
    def sample(self, model, x_pre, steps=4):
        model.eval()
        B, device = x_pre.shape[0], x_pre.device
        z = x_pre.clone()
        tv = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t = torch.full((B,), tv[i].item(), device=device)
            r = torch.full((B,), tv[i + 1].item(), device=device)
            z = self.T(model, z, r, t, x_pre)
        return z.clamp(0, 1)
