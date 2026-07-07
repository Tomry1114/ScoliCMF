import numpy as np
from typing import Optional, Tuple

import torch
from einops import rearrange
from tqdm import tqdm

from utils import Normalizer, stopgrad, adaptive_l2_loss


class MeanFlow:
    def __init__(
        self,
        channels: int = 1,
        image_size: int = 256,
        normalizer=['minmax', None, None],
        flow_ratio: float = 1.0,
        time_dist=['lognorm', -0.4, 1.0],
        jvp_api: str = 'autograd',
    ):
        self.channels = channels
        self.image_size = image_size
        if isinstance(image_size, (tuple, list)):
            self.height, self.width = image_size[0], image_size[1]
        else:
            self.height = self.width = image_size
        self.normer = Normalizer.from_list(normalizer)
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist

        assert jvp_api in ['funtorch', 'autograd']
        self.jvp_api = jvp_api
        self.noise_gen = None   # FIX3 optional generator for train noise
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        else:
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    def sample_t_r(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        else:
            raise ValueError(f"Unknown time_dist: {self.time_dist[0]}")

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        if num_selected > 0:
            indices = np.random.permutation(batch_size)[:num_selected]
            r_np[indices] = t_np[indices]

        return torch.tensor(t_np, device=device), torch.tensor(r_np, device=device)

    def loss(
        self,
        model: torch.nn.Module,
        y: torch.Tensor,
        cond_img: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, device = y.shape[0], y.device
        t, r = self.sample_t_r(B, device)

        t_ = rearrange(t, "b -> b 1 1 1").clone().detach()
        r_ = rearrange(r, "b -> b 1 1 1").clone().detach()

        e = torch.randn(y.shape, generator=self.noise_gen, device=y.device, dtype=y.dtype) if getattr(self,'noise_gen',None) is not None else torch.randn_like(y)
        y_norm = self.normer.norm(y)

        z = (1 - t_) * y_norm + t_ * e
        v = e - y_norm

        mk = model_kwargs or {}
        def f(z_in, t_in, r_in):
            return model(z_in, t_in, r_in, **mk) if cond_img is None else model(z_in, t_in, r_in, cond_img, **mk)

        inputs = (z, t, r)
        tangents = (v, torch.ones_like(t), torch.zeros_like(r))

        if self.jvp_api == 'autograd':
            u, dudt = self.jvp_fn(f, inputs, tangents, create_graph=self.create_graph)
        else:
            u, dudt = self.jvp_fn(f, inputs, tangents)

        u_tgt = v - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        
        loss = adaptive_l2_loss(error)
        mse_val = (stopgrad(error) ** 2).mean()
        
        return loss, mse_val

    @torch.no_grad()
    def sample_given_cond(
        self,
        model: torch.nn.Module,
        cond_img: torch.Tensor,
        sample_steps: int = 5,
        device: str = 'cuda',
        show_progress: bool = True,
    ) -> torch.Tensor:
        model.eval()
        cond_img = cond_img.to(device)
        B = cond_img.size(0)

        z = torch.randn(B, self.channels, self.height, self.width, device=device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        iterator = tqdm(range(sample_steps), desc="Sampling", dynamic_ncols=True) if show_progress else range(sample_steps)

        for i in iterator:
            t = torch.full((B,), t_vals[i], device=device)
            r = torch.full((B,), t_vals[i + 1], device=device)

            t_ = rearrange(t, "b -> b 1 1 1")
            r_ = rearrange(r, "b -> b 1 1 1")

            v = model(z, t, r, cond_img)
            z = z - (t_ - r_) * v

        return self.normer.unnorm(z)

    @torch.no_grad()
    def sample_unconditional(
        self,
        model: torch.nn.Module,
        batch_size: int,
        sample_steps: int = 5,
        device: str = 'cuda',
        show_progress: bool = True,
    ) -> torch.Tensor:
        model.eval()
        B = batch_size

        z = torch.randn(B, self.channels, self.height, self.width, device=device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        iterator = tqdm(range(sample_steps), desc="Sampling (uncond)", dynamic_ncols=True) if show_progress else range(sample_steps)

        for i in iterator:
            t = torch.full((B,), t_vals[i], device=device)
            r = torch.full((B,), t_vals[i + 1], device=device)

            t_ = rearrange(t, "b -> b 1 1 1")
            r_ = rearrange(r, "b -> b 1 1 1")

            v = model(z, t, r)
            z = z - (t_ - r_) * v

        return self.normer.unnorm(z)