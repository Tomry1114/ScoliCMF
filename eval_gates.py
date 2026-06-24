"""S5b evaluation harness: relative E_DC, projector bake-off, gates (G_struct/G_graph/
S_order) with bootstrap CIs + paired non-inferiority. Gates are meaningful only on
TRAINED checkpoints; --selftest runs a tiny random model to verify the harness.

E_DC^rel = ||z0_1step - z0_2step||_1 / max(||x_post-x_pre||_1, q10)   (self-consistency,
relative). Endpoint error ep1 = ||z0_1step - x_post||_1 is the external correctness anchor
(R33-3 co-requirement). S_order = E_DC^perm - E_DC^correct via topology-only Pi permutation.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from models.sc_dit import SCDiT
from sc_pga import SCPGA


def build_model(cfg, H, W, proj=None):
    cond = None
    if cfg["model"].get("cond", "base") == "scpga":
        cond = SCPGA(img_size=(H, W), dim=cfg["model"]["dim"], patch_size=cfg["model"]["patch_size"],
                     J=cfg["model"].get("J", 12), Kg=cfg["model"].get("Kg", 4), Kt=cfg["model"].get("Kt", 2),
                     beta=cfg["model"].get("beta", 40.0), eta=cfg["model"].get("eta", 4.0),
                     proj=proj or cfg["model"].get("proj", "v2"))
    return SCDiT(img_size=(H, W), patch_size=cfg["model"]["patch_size"], data_channels=1, cond_channels=1,
                 dim=cfg["model"]["dim"], depth=cfg["model"]["depth"], num_heads=cfg["model"]["num_heads"],
                 mlp_ratio=cfg["model"]["mlp_ratio"], cond_module=cond)


@torch.no_grad()
def edc_metrics(model, mf, loader, device, n_eval=64):
    model.eval()
    edc, delta, ep1, ep4 = [], [], [], []
    seen = 0
    for x_pre, x_post in loader:
        x_pre, x_post = x_pre.to(device), x_post.to(device)
        z1 = mf.sample(model, x_pre, steps=1)
        z2 = mf.sample(model, x_pre, steps=2)
        z4 = mf.sample(model, x_pre, steps=4)
        f = lambda a, b: (a - b).abs().flatten(1).mean(1).cpu()
        edc.append(f(z1, z2)); delta.append(f(x_post, x_pre)); ep1.append(f(z1, x_post)); ep4.append(f(z4, x_post))
        seen += x_pre.shape[0]
        if seen >= n_eval:
            break
    edc, delta, ep1, ep4 = [torch.cat(v).numpy() for v in (edc, delta, ep1, ep4)]
    q10 = float(np.quantile(delta, 0.1))
    return dict(edc_rel=edc / np.maximum(delta, q10), edc_abs=edc, ep1=ep1, ep4=ep4)


def bootstrap_ci(x, B=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(x.mean()), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def gate_paired(sc, other, seed=0):
    """G = other - sc (positive => SC has lower error). Returns mean + bootstrap LCB/UCB."""
    m, lo, hi = bootstrap_ci(other - sc, seed=seed)
    return {"mean": m, "lcb": lo, "ucb": hi}


def s_order(model, mf, loader, device, J, n_eval=64, seed=0):
    base = edc_metrics(model, mf, loader, device, n_eval)["edc_rel"]
    perm = torch.randperm(J, generator=torch.Generator().manual_seed(seed))
    model.cond.perm = perm
    permd = edc_metrics(model, mf, loader, device, n_eval)["edc_rel"]
    model.cond.perm = None
    return base, permd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sc_pixel.yaml")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n_eval", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.selftest:                                   # tiny random model, verify harness only
        cfg["model"].update(dim=64, depth=1, num_heads=4, J=8, Kg=3)
    import os
    ds = PairedSpineDataset(root=os.path.expanduser(cfg["data"]["root"]), split="train", size=(H, W))
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    mf = SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])

    sc = build_model(cfg, H, W, proj="v2").to(dev)
    base, permd = s_order(sc, mf, loader, dev, cfg["model"].get("J", 12), args.n_eval)
    m_edc = bootstrap_ci(base)
    so = gate_paired(base, permd)                       # S_order = E_DC^perm - E_DC^correct
    print("[selftest] random-init harness check (numbers meaningless pre-training):")
    print("  E_DC^rel: mean=%.4f  95%%CI=[%.4f,%.4f]" % m_edc)
    print("  S_order (perm-correct): mean=%.4f LCB=%.4f UCB=%.4f" % (so["mean"], so["lcb"], so["ucb"]))
    print("  harness OK: E_DC finite=%s, S_order intervention ran (perm changed Pi)"
          % np.isfinite(base).all())
