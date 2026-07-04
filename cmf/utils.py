import torch
import yaml
import time

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def log_to_file(path, step, loss, mse, lr):
    current_time = time.asctime(time.localtime(time.time()))
    message = f"[{current_time}] Step: {step} | Loss: {loss:.6f} | MSE: {mse:.6f} | LR: {lr:.6f}\n"
    with open(path, "a") as f:
        f.write(message)

class Normalizer:
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std']
        self.mode = mode

        if mode == 'mean_std':
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        return cls(*config)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'minmax':
            return x * 2 - 1
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'minmax':
            return (x.clip(-1, 1) + 1) * 0.5
        return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def adaptive_l2_loss(error: torch.Tensor, gamma: float = 0.5, c: float = 1e-3) -> torch.Tensor:
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3))
    w = 1.0 / (delta_sq + c).pow(1.0 - gamma)
    return (stopgrad(w) * delta_sq).mean()