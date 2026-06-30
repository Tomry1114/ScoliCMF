"""Self-supervised training of C_psi + consistency check vs per-pair cleaned + save C_psi-cleaned targets."""
import os, sys, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config
from dataset_sa import PairedSpineDataset
from adoc_net import AcquisitionCorrector, apply_A
dev = "cuda"; H, W = 480, 240; HOME = os.path.expanduser("~/ScoliCMF")
cfg = load_config(os.path.join(HOME, "configs/s2_base.yaml"))
def load_split(sp):
    ds = PairedSpineDataset(root=os.path.join(HOME, cfg["data"]["root"]), size=(H, W), split_file=os.path.join(HOME, "splits/%s" % sp))
    P = []; Q = []
    for a, b in DataLoader(ds, batch_size=64, shuffle=False): P.append(a); Q.append(b)
    return torch.cat(P), torch.cat(Q)
trp, trq = load_split("train.txt"); vap, vaq = load_split("val.txt")
pool = torch.cat([trp, trq], 0)                     # 864 single images for self-sup
print("pool=%d  val pairs=%d" % (pool.shape[0], vap.shape[0]), flush=True)
xcol = torch.linspace(0, 1, W, device=dev); Wacq = (1 - 0.8 * torch.exp(-((xcol - 0.5) ** 2) / (2 * 0.15 ** 2))).view(1, 1, 1, W)

C = AcquisitionCorrector().to(dev)
opt = torch.optim.AdamW(C.parameters(), lr=3e-4, weight_decay=1e-4)
g = torch.Generator().manual_seed(0)
def sample_batch(bs):
    idx = torch.randint(0, pool.shape[0], (bs,), generator=g)
    return pool[idx].to(dev)
C.train()
for step in range(1, 3001):
    x = sample_batch(16)
    nu = torch.randn(x.shape[0], 8, device=dev)         # random restricted acquisition (raw -> bounded in apply_A)
    xt = apply_A(x, nu)
    pred = C(xt, x)                                     # correction to align xt back to x
    xrec = apply_A(xt, pred)
    loss = (xrec - x).abs().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print("step %4d | recon L1 %.4f" % (step, loss.item()), flush=True)

C.eval()
with torch.no_grad():
    # self-sup held-out reconstruction sanity
    x = sample_batch(64); nu = torch.randn(64, 8, device=dev); xt = apply_A(x, nu)
    rec0 = (xt - x).abs().mean().item(); rec1 = (apply_A(xt, C(xt, x)) - x).abs().mean().item()
    print("[selfsup] perturbed L1=%.4f -> corrected L1=%.4f (lower=better)" % (rec0, rec1), flush=True)
    # apply C_psi to REAL pairs -> C_psi-cleaned; compare to per-pair gold
    def net_clean(P, Q):
        out = []
        for i in range(0, P.shape[0], 32):
            xp = P[i:i+32].to(dev); xq = Q[i:i+32].to(dev)
            out.append(apply_A(xq, C(xq, xp)).cpu())     # align x_post to x_pre frame
        return torch.cat(out)
    clv_net = net_clean(vap, vaq); clt_net = net_clean(trp, trq)
    gold = torch.load(os.path.join(HOME, "runs/adoc/clean_val.pt"))["clean"]
    agree = (clv_net - gold).abs().mean().item()
    # center-suppressed alignment to x_pre: net vs gold vs raw
    def al(c): return (Wacq * (vap.to(dev) - c.to(dev)).abs()).mean().item()
    print("[real pairs] |Cpsi_clean - perpair_gold|=%.4f" % agree, flush=True)
    print("   center-supp align-to-pre L1: raw x_post=%.4f | perpair=%.4f | Cpsi=%.4f" % (al(vaq), al(gold), al(clv_net)), flush=True)
    torch.save({"clean": clt_net}, os.path.join(HOME, "runs/adoc/clean_train_net.pt"))
    torch.save({"clean": clv_net}, os.path.join(HOME, "runs/adoc/clean_val_net.pt"))
    torch.save({"model": C.state_dict()}, os.path.join(HOME, "runs/adoc/cpsi.pt"))
print("ADOC_NET_DONE")
