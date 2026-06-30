"""PMOS trainer: build on a trained APTD warpres backbone; learn K plan prototypes via
soft-min set loss. Eval = best-of-K (oracle) vs single-prototype."""
import os, sys, argparse, math, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config, cycle
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from losses import sample_rt
from metrics_img import ssim, lpips_fn
from pmos_model import PMOSNet

def _v4(x): return x.view(-1, 1, 1, 1)
HOME = os.path.expanduser("~/ScoliCMF")

def loader(cfg, H, W, split, bs, sh):
    ds = PairedSpineDataset(root=os.path.join(HOME, cfg["data"]["root"]), size=(H, W),
                            split_file=os.path.join(HOME, "splits", split),
                            augment=(sh and cfg["data"].get("augment", False)))
    return DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=2, drop_last=sh)

def psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10))

@torch.no_grad()
def evaluate(model, path, cfg, H, W, dev):
    model.eval(); bestS = []; bestP = []; bestL = []; s0 = []; l0 = []; usage = torch.zeros(model.K)
    for xp, xq in loader(cfg, H, W, "val.txt", 6, False):
        xp, xq = xp.to(dev), xq.to(dev); B = xp.shape[0]
        r0 = torch.zeros(B, device=dev); t1 = torch.ones(B, device=dev)
        xhat = model.forward_all(xp, r0, t1, xp)["xhat"].clamp(0, 1)   # (B,K,1,H,W)
        ssK = torch.stack([ssim(xhat[:, k], xq) for k in range(model.K)], 1)   # (B,K)
        lpK = torch.stack([lpips_fn(xhat[:, k], xq) for k in range(model.K)], 1)
        psK = torch.stack([psnr(xhat[:, k], xq) for k in range(model.K)], 1)
        bi = ssK.argmax(1)                                            # best-by-SSIM prototype
        bestS.append(ssK.max(1).values.cpu()); bestL.append(lpK.min(1).values.cpu())
        bestP.append(psK.gather(1, bi[:, None]).squeeze(1).cpu())
        s0.append(ssK[:, 0].cpu()); l0.append(lpK[:, 0].cpu())
        for k in bi.tolist(): usage[k] += 1
    import numpy as np
    return (float(torch.cat(bestS).mean()), float(torch.cat(bestP).mean()), float(torch.cat(bestL).mean()),
            float(torch.cat(s0).mean()), float(torch.cat(l0).mean()), usage.tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/s2_base.yaml")
    ap.add_argument("--init", default="runs/aptd_long_fs015/ckpts/step_2000.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--lambda_div", type=float, default=0.3)
    ap.add_argument("--div_margin", type=float, default=0.06)
    ap.add_argument("--flow_scale", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_bal", type=float, default=0.1)
    ap.add_argument("--lambda_smooth", type=float, default=0.05)
    ap.add_argument("--lambda_res", type=float, default=0.02)
    ap.add_argument("--save_step", type=int, default=500)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(os.path.join(HOME, a.cfg)); H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
    gamma, sm = cfg["meanflow"]["gamma"], cfg["meanflow"]["sigma_m"]
    mf = SourceAnchoredMeanFlow(gamma=gamma, sigma_m=sm); path = mf.path
    cfg["model"]["xpre_mode"] = "full"
    backbone = build_model(cfg, H, W).to(dev)
    model = PMOSNet(backbone, K=a.K, mode="warpres", flow_scale=a.flow_scale).to(dev)
    sd = torch.load(os.path.join(HOME, a.init), map_location=dev)["model"]
    miss = model.load_state_dict(sd, strict=False)
    print("init from %s | missing(proto expected)=%s" % (a.init, [m for m in miss.missing_keys]), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    ema = [p.detach().clone() for p in model.parameters()]
    def emaup(d=0.999):
        for e, p in zip(ema, model.parameters()): e.mul_(d).add_(p.detach(), alpha=1 - d)
    odir = os.path.join(HOME, "runs", a.out, "ckpts"); os.makedirs(odir, exist_ok=True)
    logf = open(os.path.join(HOME, "runs", a.out, "log.txt"), "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log("K=%d tau=%.3f trainable=%.2fM" % (a.K, a.tau, sum(p.numel() for p in model.parameters()) / 1e6))
    it = cycle(loader(cfg, H, W, "train.txt", a.bs, True)); model.train()
    for step in range(1, a.steps + 1):
        xp, xq = next(it); xp, xq = xp.to(dev), xq.to(dev); B = xp.shape[0]
        r, t = sample_rt(B, dev)
        eps = torch.randn_like(xp) if sm > 0 else None
        z_t = path.z_t(xp, xq, _v4(t), eps)
        out = model.forward_all(z_t, r, t, xp); xhat = out["xhat"]      # (B,K,1,H,W)
        d = (xhat - xq.unsqueeze(1)).abs().mean(dim=(2, 3, 4))          # (B,K) per-sample endpoint L1
        l_set = (-a.tau * (torch.logsumexp(-d / a.tau, dim=1) - math.log(a.K))).mean()
        asg = torch.softmax(-d / a.tau, dim=1); qbar = asg.mean(0)
        l_bal = (qbar * (qbar * a.K + 1e-9).log()).sum()
        l_sm = out["flow"].diff(dim=3).abs().mean() + out["flow"].diff(dim=4).abs().mean()
        l_rs = out["res"].abs().mean()
        xf = xhat.flatten(2)                                          # (B,K,HW)
        dij = (xf.unsqueeze(1) - xf.unsqueeze(2)).abs().mean(-1)       # (B,K,K) pairwise output L1
        offd = dij[:, ~torch.eye(a.K, dtype=torch.bool, device=dev)].reshape(B, -1)
        l_div = (a.div_margin - offd).clamp_min(0).mean()             # hinge: push prototypes >= margin apart
        loss = l_set + a.lambda_bal * l_bal + a.lambda_smooth * l_sm + a.lambda_res * l_rs + a.lambda_div * l_div
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); emaup()
        if step % 100 == 0:
            log("step %4d | loss %.4f set %.4f div %.4f bal %.4f | qbar %s" % (step, loss.item(), l_set.item(), l_div.item(), l_bal.item(), [round(x, 2) for x in qbar.tolist()]))
        if step % a.save_step == 0:
            bk = [p.detach().clone() for p in model.parameters()]
            for p, e in zip(model.parameters(), ema): p.data.copy_(e)
            bS, bP, bL, s0, l0, us = evaluate(model, path, cfg, H, W, dev)
            for p, b in zip(model.parameters(), bk): p.data.copy_(b)
            model.train()
            log("  [eval %d] best-of-K SSIM=%.4f PSNR=%.3f LPIPS=%.4f | proto0 SSIM=%.4f LPIPS=%.4f | usage=%s" % (step, bS, bP, bL, s0, l0, us))
            torch.save({"model": model.state_dict(), "ema": [e.cpu() for e in ema], "step": step}, os.path.join(odir, "step_%d.pt" % step))
    log("PMOS_DONE")

if __name__ == "__main__":
    main()
