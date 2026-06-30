"""Unified 2x2 / ablation trainer (x0-param). One trainer, vary {mode}x{target}.
Target = raw (paired loader) OR a cleaned-target file (TensorDataset on sorted-stem order).
EVERY checkpoint is evaluated on BOTH the canonical full-clean val frame AND the raw val frame,
so all runs are comparable on one fixed reference (the canonical frame)."""
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

def xq_inorder(split):
    ds = PairedSpineDataset(root=os.path.join(HOME, cfg["data"]["root"]), size=(H, W),
                            split_file=os.path.join(HOME, "splits", split))
    XP = []; XQ = []
    for x, y in DataLoader(ds, batch_size=64, shuffle=False): XP.append(x); XQ.append(y)
    return torch.cat(XP), torch.cat(XQ)

def psnr(a, b): return -10 * torch.log10(((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10))

@torch.no_grad()
def evaluate(model, path, xpre, tgt, dev, nfe=1):
    model.eval(); S = []; P = []; L = []
    for i in range(0, xpre.shape[0], 6):
        xp = xpre[i:i+6].to(dev); xq = tgt[i:i+6].to(dev); B = xp.shape[0]
        z = xp; tv = torch.linspace(1, 0, nfe + 1, device=dev); xhat = None
        for k in range(nfe):
            t = torch.full((B,), tv[k].item(), device=dev); r = torch.full((B,), tv[k+1].item(), device=dev)
            xhat = model(z, r, t, xp)["xhat"]; z = xp + _v4(path.alpha(r)) * (xhat - xp)
        o = xhat.clamp(0, 1)
        S.append(ssim(o, xq).cpu()); P.append(psnr(o, xq).cpu()); L.append(lpips_fn(o, xq).cpu())
    model.train(); return float(torch.cat(S).mean()), float(torch.cat(P).mean()), float(torch.cat(L).mean())

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True)                 # direct|warpres
ap.add_argument("--target", required=True)               # "raw" OR path to clean_*_train.pt
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--flow_scale", type=float, default=0.15)
ap.add_argument("--lambda_end", type=float, default=1.0)
ap.add_argument("--lambda_smooth", type=float, default=0.05); ap.add_argument("--lambda_res", type=float, default=0.02)
ap.add_argument("--save_step", type=int, default=1000)
a = ap.parse_args()
dev = "cuda"; cfg = load_config(os.path.join(HOME, "configs/s2_base.yaml")); H, W = cfg["data"]["size_h"], cfg["data"]["size_w"]
gamma, sm = cfg["meanflow"]["gamma"], cfg["meanflow"]["sigma_m"]
mf = SourceAnchoredMeanFlow(gamma=gamma, sigma_m=sm); path = mf.path
cfg["model"]["xpre_mode"] = "full"
bb = build_model(cfg, H, W).to(dev); model = APTDNet(bb, a.mode, flow_scale=a.flow_scale).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
ema = [p.detach().clone() for p in model.parameters()]

# ---- data ----
XPtr, XQtr_raw = xq_inorder("train.txt")
if a.target == "raw":
    tgt_tr = XQtr_raw
else:
    tgt_tr = torch.load(a.target)["clean"]
dl = cycle(DataLoader(TensorDataset(XPtr, tgt_tr), batch_size=a.bs, shuffle=True, drop_last=True))

# ---- eval references: canonical full-clean val (fixed) + raw val ----
XPva, XQva_raw = xq_inorder("val.txt")
clva = torch.load(os.path.join(HOME, "runs/adoc/clean_val.pt"))["clean"]   # canonical reference

odir = os.path.join(HOME, "runs", a.out, "ckpts"); os.makedirs(odir, exist_ok=True)
logf = open(os.path.join(HOME, "runs", a.out, "log.txt"), "a")
def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
nP = sum(p.numel() for p in model.parameters() if p.requires_grad)
log("=== mode=%s target=%s out=%s trainable=%.2fM ===" % (a.mode, a.target, a.out, nP / 1e6))
model.train()
for step in range(1, a.steps + 1):
    xp, xq = next(dl); xp, xq = xp.to(dev), xq.to(dev); B = xp.shape[0]
    r, t = sample_rt(B, dev); eps = torch.randn_like(xp) if sm > 0 else None
    z_t = path.z_t(xp, xq, _v4(t), eps)
    w = _v4((path.alpha(t) - path.alpha(r)) / (t - r).clamp_min(1e-3))
    out = model(z_t, r, t, xp); xhat = out["xhat"]
    l_span = adaptive_l2_loss(w * (xhat - xq)); l_end = (xhat - xq).abs().mean()
    l_sm = out["flow"].abs().mean() if out.get("flow") is not None else torch.zeros((), device=dev)
    l_rs = out["res"].abs().mean() if out.get("res") is not None else torch.zeros((), device=dev)
    loss = l_span + a.lambda_end * l_end + a.lambda_smooth * l_sm + a.lambda_res * l_rs
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    for e, p in zip(ema, model.parameters()): e.mul_(0.999).add_(p.detach(), alpha=0.001)
    if step % 100 == 0:
        log("step %4d | loss %.4f span %.4f end %.4f" % (step, loss.item(), l_span.item(), l_end.item()))
    if step % a.save_step == 0:
        bk = [p.detach().clone() for p in model.parameters()]
        for p, e in zip(model.parameters(), ema): p.data.copy_(e)
        sc, pc, lc = evaluate(model, path, XPva, clva, dev)
        sr, pr, lr = evaluate(model, path, XPva, XQva_raw, dev)
        for p, b in zip(model.parameters(), bk): p.data.copy_(b)
        log("  [eval %d] CANON SSIM=%.4f PSNR=%.3f LPIPS=%.4f | RAW SSIM=%.4f PSNR=%.3f LPIPS=%.4f" % (step, sc, pc, lc, sr, pr, lr))
        torch.save({"model": model.state_dict(), "ema": [e.cpu() for e in ema], "step": step, "mode": a.mode}, os.path.join(odir, "step_%d.pt" % step))
log("TRAIN2x2_DONE mode=%s target=%s" % (a.mode, a.target))
