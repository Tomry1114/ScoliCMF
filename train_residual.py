"""Step-4 pilot trainer: frozen Bridge + correction-grounded residual branch."""
import os, sys, argparse, copy, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config, adaptive_l2_loss, stopgrad, cycle
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import load_ckpt
from losses import sample_rt
from metrics_img import ssim, lpips_fn
from residual_model import DynamicCorrectionConditioner, DynamicResidualHead, ResidualScoliCMF

def _v4(x): return x.view(-1, 1, 1, 1)
ROOTHOME = os.path.expanduser("~/ScoliCMF")

def make_loader(cfg, H, W, split, bs, shuffle):
    ds = PairedSpineDataset(root=os.path.join(ROOTHOME, cfg["data"]["root"]), size=(H, W),
                            split_file=os.path.join(ROOTHOME, "splits", split))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2, drop_last=shuffle)

@torch.no_grad()
def endpoint(model, mf, cfg, H, W, dev, dyn_off):
    old = model.dyn_off; model.dyn_off = dyn_off; model.eval()
    ss, lp = [], []
    for xp, xq in make_loader(cfg, H, W, "val.txt", 6, False):
        xp, xq = xp.to(dev), xq.to(dev)
        z = mf.sample(model, xp, steps=4)
        ss.append(ssim(z, xq).cpu()); lp.append(lpips_fn(z, xq).cpu())
    model.dyn_off = old
    return float(torch.cat(ss).mean()), float(torch.cat(lp).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge_cfg", default="configs/s2_base.yaml")
    ap.add_argument("--bridge_ckpt", default="runs/s2_base/ckpts/step_5000.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond_mode", default="secant")    # secant | point | static
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--l_full", type=float, default=1.0)
    ap.add_argument("--l_corr", type=float, default=1.0)
    ap.add_argument("--l_sub", type=float, default=0.1)
    ap.add_argument("--l_harm", type=float, default=0.01)
    ap.add_argument("--save_step", type=int, default=500)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(os.path.join(ROOTHOME, a.bridge_cfg))
    H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
    dim, ps = cfg["model"]["dim"], cfg["model"]["patch_size"]
    gamma, sigma_m = cfg["meanflow"]["gamma"], cfg["meanflow"]["sigma_m"]
    mf = SourceAnchoredMeanFlow(gamma=gamma, sigma_m=sigma_m)

    bridge = load_ckpt(os.path.join(ROOTHOME, a.bridge_ckpt), cfg, H, W, None, dev, use_ema=True)
    dyn = DynamicCorrectionConditioner((H, W), dim, ps, J=cfg["model"].get("J",12),
                                       K=cfg["model"].get("Kg",4), Kt=cfg["model"].get("Kt",2),
                                       cond_mode=a.cond_mode).to(dev)
    head = DynamicResidualHead(dim, ps, H // ps, W // ps, 1).to(dev)
    model = ResidualScoliCMF(bridge, dyn, head).to(dev)
    train_params = [p for p in model.parameters() if p.requires_grad]
    nP = sum(p.numel() for p in train_params)
    print("trainable params = %.2fM  (cond_mode=%s)" % (nP/1e6, a.cond_mode), flush=True)

    # invariant check: c_dyn=0 => u_corr=0
    with torch.no_grad():
        hb = torch.randn(2, (H//ps)*(W//ps), dim, device=dev)
        z0 = head(hb, torch.zeros(2, (H//ps)*(W//ps), dim, device=dev))
        print("INVARIANT c_dyn=0 -> |u_corr|max = %.2e (expect ~0)" % float(z0.abs().max()), flush=True)

    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=1e-2)
    ema = [p.detach().clone() for p in train_params]
    def ema_update(d=0.999):
        for e, p in zip(ema, train_params): e.mul_(d).add_(p.detach(), alpha=1-d)
    odir = os.path.join(ROOTHOME, "runs", a.out, "ckpts"); os.makedirs(odir, exist_ok=True)
    logf = open(os.path.join(ROOTHOME, "runs", a.out, "log.txt"), "a")
    def log(s): print(s, flush=True); logf.write(s+"\n"); logf.flush()

    # baseline (frozen bridge) endpoint for reference
    bss, blp = endpoint(model, mf, cfg, H, W, dev, dyn_off=True)
    log("[baseline frozen-Bridge] val SSIM4=%.4f LPIPS4=%.4f" % (bss, blp))

    it = cycle(make_loader(cfg, H, W, "train.txt", a.bs, True))
    model.train()
    for step in range(1, a.steps+1):
        xp, xq = next(it); xp, xq = xp.to(dev), xq.to(dev); B = xp.shape[0]
        r, t = sample_rt(B, dev)
        eps = torch.randn_like(xp) if sigma_m > 0 else None
        z_t = mf.path.z_t(xp, xq, _v4(t), eps); z_r = mf.path.z_t(xp, xq, _v4(r), eps)
        target = (z_t - z_r) / (_v4(t) - _v4(r)).clamp_min(1e-6)
        u, aux = model(z_t, r, t, xp, return_aux=True)
        l_full = adaptive_l2_loss(u - stopgrad(target))
        l_corr = adaptive_l2_loss(aux["u_corr"] - stopgrad(target - aux["u_base"]))
        Ffq = model.dyn_cond.stem(xq).flatten(2).transpose(1, 2)
        B_post = torch.einsum("bjn,bnd->bjd", aux["pi"].detach(), Ffq)
        dB = (B_post - aux["Btok"]).detach()
        res = dB - torch.einsum("bjk,bkd->bjd", aux["Pi"], dB)
        l_sub = res.pow(2).sum() / (dB.pow(2).sum() + 1e-6)
        l_harm = model.dyn_cond.l_harm(aux["Q"])
        loss = a.l_full*l_full + a.l_corr*l_corr + a.l_sub*l_sub + a.l_harm*l_harm
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step(); ema_update()
        if step % 50 == 0:
            log("step %4d | loss %.4f | full %.4f corr %.4f sub %.4f harm %.4f | g %.3f cov %.3f" % (
                step, loss.item(), l_full.item(), l_corr.item(), l_sub.item(), l_harm.item(),
                float(aux["gamma"]), 1-l_sub.item()))
        if step % a.save_step == 0:
            bk = [p.detach().clone() for p in train_params]
            for p, e in zip(train_params, ema): p.data.copy_(e)
            fss, flp = endpoint(model, mf, cfg, H, W, dev, dyn_off=False)
            dss, dlp = endpoint(model, mf, cfg, H, W, dev, dyn_off=True)
            for p, b in zip(train_params, bk): p.data.copy_(b)
            model.train()
            log("  [eval ema step %d] FULL SSIM4=%.4f LPIPS4=%.4f | DYN-OFF SSIM4=%.4f LPIPS4=%.4f | dSSIM=%.4f" % (
                step, fss, flp, dss, dlp, fss-dss))
            torch.save({"dyn": model.dyn_cond.state_dict(), "head": model.corr_head.state_dict(),
                        "ema": [e.cpu() for e in ema], "step": step, "cond_mode": a.cond_mode},
                       os.path.join(odir, "step_%d.pt" % step))
    log("TRAIN_RESIDUAL_DONE")

if __name__ == "__main__":
    main()
