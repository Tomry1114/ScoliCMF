"""Train APTD on ADOC-cleaned targets; evaluate in the CANONICAL (cleaned) frame.
Reports the raw-trained APTD (aptd_long_fs015) on cleaned val as the baseline, so the
comparison isolates the effect of cleaning the SUPERVISION target."""
import os, sys, argparse, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

def xpre_inorder(cfg, H, W, split):
    ds = PairedSpineDataset(root=os.path.join(HOME, cfg["data"]["root"]), size=(H, W),
                            split_file=os.path.join(HOME, "splits", split))
    XP = [a for a, b in DataLoader(ds, batch_size=64, shuffle=False)]
    return torch.cat(XP)

def psnr(a, b): return -10 * torch.log10(((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10))

@torch.no_grad()
def evaluate(model, path, xpre, tgt, dev, nfe=1):
    model.eval(); S = []; P = []; L = []
    for i in range(0, xpre.shape[0], 6):
        xp = xpre[i:i + 6].to(dev); xq = tgt[i:i + 6].to(dev); B = xp.shape[0]
        z = xp; tv = torch.linspace(1, 0, nfe + 1, device=dev); xhat = None
        for k in range(nfe):
            t = torch.full((B,), tv[k].item(), device=dev); r = torch.full((B,), tv[k + 1].item(), device=dev)
            xhat = model(z, r, t, xp)["xhat"]; z = xp + _v4(path.alpha(r)) * (xhat - xp)
        o = xhat.clamp(0, 1)
        S.append(ssim(o, xq).cpu()); P.append(psnr(o, xq).cpu()); L.append(lpips_fn(o, xq).cpu())
    return float(torch.cat(S).mean()), float(torch.cat(P).mean()), float(torch.cat(L).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--flow_scale", type=float, default=0.15); ap.add_argument("--save_step", type=int, default=1000)
    a = ap.parse_args()
    dev = "cuda"; cfg = load_config(os.path.join(HOME, "configs/s2_base.yaml")); H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
    gamma, sm = cfg["meanflow"]["gamma"], cfg["meanflow"]["sigma_m"]
    mf = SourceAnchoredMeanFlow(gamma=gamma, sigma_m=sm); path = mf.path
    cfg["model"]["xpre_mode"] = "full"
    xptr = xpre_inorder(cfg, H, W, "train.txt"); xpva = xpre_inorder(cfg, H, W, "val.txt")
    cltr = torch.load(os.path.join(HOME, "runs/adoc/clean_train.pt"))["clean"]
    clva = torch.load(os.path.join(HOME, "runs/adoc/clean_val.pt"))["clean"]
    rawva = []  # raw x_post val for reference
    dsv = PairedSpineDataset(root=os.path.join(HOME, cfg["data"]["root"]), size=(H, W), split_file=os.path.join(HOME, "splits/val.txt"))
    for _, b in DataLoader(dsv, batch_size=64, shuffle=False): rawva.append(b)
    rawva = torch.cat(rawva)
    logf = open(os.path.join(HOME, "runs", a.out + ".log"), "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()

    # ---- baseline: raw-trained APTD evaluated on CLEANED val ----
    for st in [2000, 5000]:
        bb = build_model(cfg, H, W).to(dev); m0 = APTDNet(bb, "warpres", flow_scale=0.15).to(dev)
        m0.load_state_dict(torch.load(os.path.join(HOME, "runs/aptd_long_fs015/ckpts/step_%d.pt" % st), map_location=dev)["model"])
        sc, pc, lc = evaluate(m0, path, xpva, clva, dev); sr, pr, lr = evaluate(m0, path, xpva, rawva, dev)
        log("[BASELINE raw-APTD step%d] on CLEANED val SSIM=%.4f PSNR=%.3f LPIPS=%.4f | on RAW val SSIM=%.4f LPIPS=%.4f" % (st, sc, pc, lc, sr, lr))
    del bb, m0; torch.cuda.empty_cache()

    # ---- train APTD on CLEANED targets ----
    bb = build_model(cfg, H, W).to(dev); model = APTDNet(bb, "warpres", flow_scale=a.flow_scale).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    ema = [p.detach().clone() for p in model.parameters()]
    dl = cycle(DataLoader(TensorDataset(xptr, cltr), batch_size=a.bs, shuffle=True, drop_last=True))
    odir = os.path.join(HOME, "runs", a.out, "ckpts"); os.makedirs(odir, exist_ok=True); model.train()
    for step in range(1, a.steps + 1):
        xp, xq = next(dl); xp, xq = xp.to(dev), xq.to(dev); B = xp.shape[0]
        r, t = sample_rt(B, dev); eps = torch.randn_like(xp) if sm > 0 else None
        z_t = path.z_t(xp, xq, _v4(t), eps); w = _v4((path.alpha(t) - path.alpha(r)) / (t - r).clamp_min(1e-3))
        out = model(z_t, r, t, xp); xhat = out["xhat"]
        l_span = adaptive_l2_loss(w * (xhat - xq)); l_end = (xhat - xq).abs().mean()
        fl = out["flow"]; l_sm = (fl[:, :, 1:] - fl[:, :, :-1]).abs().mean() + (fl[:, :, :, 1:] - fl[:, :, :, :-1]).abs().mean()
        l_rs = out["res"].abs().mean()
        loss = l_span + l_end + 0.05 * l_sm + 0.02 * l_rs
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        for e, p in zip(ema, model.parameters()): e.mul_(0.999).add_(p.detach(), alpha=0.001)
        if step % 1000 == 0:
            log("step %4d | loss %.4f" % (step, loss.item()))
        if step % a.save_step == 0:
            bk = [p.detach().clone() for p in model.parameters()]
            for p, e in zip(model.parameters(), ema): p.data.copy_(e)
            sc, pc, lc = evaluate(model, path, xpva, clva, dev); sr, _, lr = evaluate(model, path, xpva, rawva, dev)
            for p, b in zip(model.parameters(), bk): p.data.copy_(b); model.train()
            log("  [ADOC-APTD step%d] on CLEANED val SSIM=%.4f PSNR=%.3f LPIPS=%.4f | on RAW val SSIM=%.4f LPIPS=%.4f" % (step, sc, pc, lc, sr, lr))
            torch.save({"model": model.state_dict(), "step": step}, os.path.join(odir, "step_%d.pt" % step))
    log("ADOC_TRAIN_DONE")

if __name__ == "__main__":
    main()
