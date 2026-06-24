"""S5b evaluation + gates (real). Loads a trained checkpoint, evaluates on a held-out
split, runs the projector bake-off and computes G_struct / G_graph / S_order with
patient(+permutation) bootstrap CIs and a paired non-inferiority check.

Examples:
  # single checkpoint, full metrics on val split:
  python eval_gates.py --ckpt sa_long_ckpts/step_8000.pt --split val --json out.json
  # bake-off across projector variants (needs one ckpt per variant):
  python eval_gates.py --bakeoff v2=ck_v2.pt,v1=ck_v1.pt,dct=ck_dct.pt,random=ck_rand.pt,identity=ck_id.pt --split test

E_DC^rel = ||z0_1step - z0_2step||_1 / max(||x_post-x_pre||_1, q10)   (self-consistency).
Endpoint anchor ep1/ep4 = ||z0_kstep - x_post||_1 (external correctness). S_order = mean
over several fixed perms of (E_DC^perm - E_DC^correct), topology-only (Pi permuted).
"""
import argparse
import json
import os
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
                     proj=proj or cfg["model"].get("proj", "v2"),
                     dyn_off=cfg["model"].get("dyn_off", False))
    return SCDiT(img_size=(H, W), patch_size=cfg["model"]["patch_size"], data_channels=1, cond_channels=1,
                 dim=cfg["model"]["dim"], depth=cfg["model"]["depth"], num_heads=cfg["model"]["num_heads"],
                 mlp_ratio=cfg["model"]["mlp_ratio"], cond_module=cond,
                 decode_head=cfg["model"].get("decode_head", "conv"),
                 xpre_mode=cfg["model"].get("xpre_mode", "full"))


def load_ckpt(path, cfg, H, W, proj, dev, use_ema=True):
    state = torch.load(path, map_location=dev)
    sd = state.get("ema") if (use_ema and isinstance(state, dict) and "ema" in state) else \
         state.get("model", state)                      # rich dict or bare state_dict
    m = build_model(cfg, H, W, proj).to(dev)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def metrics(model, mf, loader, dev, n_eval=10**9):
    edc, delta, ep1, ep4 = [], [], [], []
    seen = 0
    for x_pre, x_post in loader:
        x_pre, x_post = x_pre.to(dev), x_post.to(dev)
        z1, z2, z4 = (mf.sample(model, x_pre, steps=k) for k in (1, 2, 4))
        f = lambda a, b: (a - b).abs().flatten(1).mean(1).cpu().numpy()
        edc.append(f(z1, z2)); delta.append(f(x_post, x_pre)); ep1.append(f(z1, x_post)); ep4.append(f(z4, x_post))
        seen += x_pre.shape[0]
        if seen >= n_eval:
            break
    edc, delta, ep1, ep4 = (np.concatenate(v) for v in (edc, delta, ep1, ep4))
    q10 = float(np.quantile(delta, 0.1))
    return {"edc_rel": edc / np.maximum(delta, q10), "edc_abs": edc, "ep1": ep1, "ep4": ep4}


def boot(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return float(x.mean()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


@torch.no_grad()
def s_order(model, mf, loader, dev, J, n_perm=4, n_eval=10**9):
    base = metrics(model, mf, loader, dev, n_eval)["edc_rel"]
    diffs = []
    for p in range(n_perm):
        model.cond.perm = torch.randperm(J, generator=torch.Generator().manual_seed(100 + p))
        diffs.append(metrics(model, mf, loader, dev, n_eval)["edc_rel"] - base)   # paired per-patient
    model.cond.perm = None
    return base, np.stack(diffs)            # (n_perm, n_patients)


def shortcut_diag(model, mf, loader, dev, J, n_eval, seed=0):
    """P1: does the chain mechanism convert into ACCURACY (ep) or only self-consistency?
    Compare ep1 & E_DC under: full / dyn_off (m_dyn=0) / permuted Pi (wrong chain)."""
    def run():
        r = metrics(model, mf, loader, dev, n_eval)
        return float(r["ep1"].mean()), float(r["edc_rel"].mean())
    out = {"full": run()}
    if hasattr(model.cond, "dyn_off"):
        old = model.cond.dyn_off; model.cond.dyn_off = True
        out["dyn_off"] = run(); model.cond.dyn_off = old
    model.cond.perm = torch.randperm(J, generator=torch.Generator().manual_seed(seed))
    out["perm"] = run(); model.cond.perm = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--config", default="configs/sc_pixel_long.yaml")
    ap.add_argument("--split", default="val"); ap.add_argument("--proj", default=None)
    ap.add_argument("--bakeoff", default=None, help="name=ckpt,name=ckpt,... (proj inferred from name)")
    ap.add_argument("--n_eval", type=int, default=10**9); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ema", action="store_true"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    split_file = os.path.expanduser(f"~/ScoliCMF/splits/{a.split}.txt")
    ds = PairedSpineDataset(root=os.path.expanduser(cfg["data"]["root"]), size=(H, W), split_file=split_file)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    mf = SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"])
    J = cfg["model"].get("J", 12)
    print(f"[eval] split={a.split} n={len(ds)} dev={dev}")
    out = {}

    if a.bakeoff:
        variants = dict(kv.split("=") for kv in a.bakeoff.split(","))
        edc_by = {}
        for name, ck in variants.items():
            m = load_ckpt(ck, cfg, H, W, name, dev, use_ema=not a.no_ema)
            r = metrics(m, mf, loader, dev, a.n_eval)
            edc_by[name] = r["edc_rel"]
            mu, lo, hi = boot(r["edc_rel"], seed=a.seed)
            ep, _, _ = boot(r["ep1"], seed=a.seed)
            print(f"  {name:10s} E_DC^rel={mu:.4f} [{lo:.4f},{hi:.4f}]  ep1={ep:.4f}")
            out[name] = {"edc_rel": mu, "edc_ci": [lo, hi], "ep1": ep}
        # gates (paired diffs; positive => SC lower error). G_struct vs global_base if present.
        def gate(sc, other):
            d = edc_by[other] - edc_by[sc]; m, lo, hi = boot(d, seed=a.seed); return {"mean": m, "lcb": lo, "ucb": hi}
        if "identity" in edc_by:
            rk = [k for k in ("dct", "random", "v1") if k in edc_by]
            if "v2" in edc_by and rk:
                worst = min(rk, key=lambda k: edc_by[k].mean())            # v2 must beat best rank-matched
                out["G_graph"] = gate("v2", worst); print(f"  G_graph (v2 vs {worst}): {out['G_graph']}")
        print(json.dumps(out, indent=2))
    else:
        assert a.ckpt, "need --ckpt or --bakeoff"
        m = load_ckpt(a.ckpt, cfg, H, W, a.proj, dev, use_ema=not a.no_ema)
        r = metrics(m, mf, loader, dev, a.n_eval)
        for k in ("edc_rel", "ep1", "ep4"):
            mu, lo, hi = boot(r[k], seed=a.seed); out[k] = {"mean": mu, "ci": [lo, hi]}
            print(f"  {k:8s} = {mu:.4f}  95%CI=[{lo:.4f},{hi:.4f}]")
        if cfg["model"].get("cond") == "scpga":
            base, diffs = s_order(m, mf, loader, dev, J, n_perm=4, n_eval=a.n_eval)
            flat = diffs.flatten()
            mu, lo, hi = boot(flat, seed=a.seed)
            out["S_order"] = {"mean": mu, "lcb": lo, "ucb": hi, "n_perm": diffs.shape[0]}
            print(f"  S_order  = {mu:.4f}  95%CI=[{lo:.4f},{hi:.4f}]  (perm-correct, {diffs.shape[0]} perms)")
            d = shortcut_diag(m, mf, loader, dev, J, a.n_eval, seed=a.seed)
            out["shortcut_diag"] = d
            print("  -- shortcut diag (ep1 / E_DC) --")
            for k in ("full", "dyn_off", "perm"):
                if k in d:
                    print("     %-8s ep1=%.4f  E_DC=%.4f" % (k, d[k][0], d[k][1]))
            fu = d["full"]
            if "dyn_off" in d:
                print("     -> dyn_off ep1 delta = %+.4f  (~0 => m_dyn gives no accuracy)" % (d["dyn_off"][0] - fu[0]))
            print("     -> perm     ep1 delta = %+.4f ; E_DC delta = %+.4f" % (d["perm"][0] - fu[0], d["perm"][1] - fu[1]))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2); print("wrote", a.json)


if __name__ == "__main__":
    main()
