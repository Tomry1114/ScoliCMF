"""APTD trainer (x0-param). Ablation modes: direct | residual | warp | warpres."""
import os, sys, argparse, math, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config, adaptive_l2_loss, cycle
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from losses import sample_rt
from metrics_img import ssim, lpips_fn
from aptd_model import APTDNet

def _v4(x): return x.view(-1, 1, 1, 1)
HOME = os.path.expanduser("~/ScoliCMF")

def loader(cfg, H, W, split, bs, sh):
    ds = PairedSpineDataset(root=os.path.join(HOME, cfg["data"]["root"]), size=(H, W),
                            split_file=os.path.join(HOME, "splits", split),
                            augment=(sh and cfg["data"].get("augment", False)))
    return DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=2, drop_last=sh)

def psnr(a, b):
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return (-10 * torch.log10(mse))

@torch.no_grad()
def evaluate(model, path, cfg, H, W, dev, nfe):
    model.eval(); SS = []; PS = []; LP = []
    for xp, xq in loader(cfg, H, W, "val.txt", 6, False):
        xp, xq = xp.to(dev), xq.to(dev); B = xp.shape[0]
        z = xp
        tv = torch.linspace(1.0, 0.0, nfe + 1, device=dev)
        xhat = None
        for i in range(nfe):
            t = torch.full((B,), tv[i].item(), device=dev); r = torch.full((B,), tv[i + 1].item(), device=dev)
            xhat = model(z, r, t, xp)["xhat"]
            z = xp + _v4(path.alpha(r)) * (xhat - xp)
        out = xhat.clamp(0, 1)
        SS.append(ssim(out, xq).cpu()); PS.append(psnr(out, xq).cpu()); LP.append(lpips_fn(out, xq).cpu())
    return float(torch.cat(SS).mean()), float(torch.cat(PS).mean()), float(torch.cat(LP).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/s2_base.yaml")
    ap.add_argument("--mode", required=True)          # direct|residual|warp|warpres
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_end", type=float, default=1.0)
    ap.add_argument("--lambda_smooth", type=float, default=0.05)
    ap.add_argument("--lambda_res", type=float, default=0.02)
    ap.add_argument("--save_step", type=int, default=500)
    ap.add_argument("--flow_scale", type=float, default=0.3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(os.path.join(HOME, a.cfg)); H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
    gamma, sm = cfg["meanflow"]["gamma"], cfg["meanflow"]["sigma_m"]
    mf = SourceAnchoredMeanFlow(gamma=gamma, sigma_m=sm); path = mf.path
    cfg["model"]["xpre_mode"] = "full"
    backbone = build_model(cfg, H, W).to(dev)
    model = APTDNet(backbone, a.mode, flow_scale=a.flow_scale).to(dev)
    nP = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("mode=%s trainable=%.2fM" % (a.mode, nP / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    ema = [p.detach().clone() for p in model.parameters()]
    def emaup(d=0.999):
        for e, p in zip(ema, model.parameters()): e.mul_(d).add_(p.detach(), alpha=1 - d)
    odir = os.path.join(HOME, "runs", a.out, "ckpts"); os.makedirs(odir, exist_ok=True)
    logf = open(os.path.join(HOME, "runs", a.out, "log.txt"), "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    it = cycle(loader(cfg, H, W, "train.txt", a.bs, True)); model.train()
    for step in range(1, a.steps + 1):
        xp, xq = next(it); xp, xq = xp.to(dev), xq.to(dev); B = xp.shape[0]
        r, t = sample_rt(B, dev)
        eps = torch.randn_like(xp) if sm > 0 else None
        z_t = path.z_t(xp, xq, _v4(t), eps)
        w = _v4((path.alpha(t) - path.alpha(r)) / (t - r).clamp_min(1e-3))
        out = model(z_t, r, t, xp); xhat = out["xhat"]
        l_span = adaptive_l2_loss(w * (xhat - xq))
        l_end = (xhat - xq).abs().mean()
        l_sm = torch.zeros((), device=dev); l_rs = torch.zeros((), device=dev)
        if out["flow"] is not None:
            fl = out["flow"]; l_sm = (fl[:, :, 1:] - fl[:, :, :-1]).abs().mean() + (fl[:, :, :, 1:] - fl[:, :, :, :-1]).abs().mean()
        if a.mode == "warpres" and out["res"] is not None:
            l_rs = out["res"].abs().mean()
        loss = l_span + a.lambda_end * l_end + a.lambda_smooth * l_sm + a.lambda_res * l_rs
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); emaup()
        if step % 100 == 0:
            log("step %4d | loss %.4f span %.4f end %.4f sm %.4f rs %.4f" % (step, loss.item(), l_span.item(), l_end.item(), float(l_sm), float(l_rs)))
        if step % a.save_step == 0:
            bk = [p.detach().clone() for p in model.parameters()]
            for p, e in zip(model.parameters(), ema): p.data.copy_(e)
            s1, p1, l1 = evaluate(model, path, cfg, H, W, dev, 1)
            s4, p4, l4 = evaluate(model, path, cfg, H, W, dev, 4)
            for p, b in zip(model.parameters(), bk): p.data.copy_(b)
            model.train()
            log("  [eval ema %d] 1NFE SSIM=%.4f PSNR=%.3f LPIPS=%.4f | 4NFE SSIM=%.4f PSNR=%.3f LPIPS=%.4f" % (step, s1, p1, l1, s4, p4, l4))
            torch.save({"model": model.state_dict(), "ema": [e.cpu() for e in ema], "step": step, "mode": a.mode}, os.path.join(odir, "step_%d.pt" % step))
    log("APTD_DONE mode=%s" % a.mode)

if __name__ == "__main__":
    main()
