"""Train original ScoliCMF baseline under the MATCHED regime (same data/split/res/optim/aug/EMA
as s5b). Only the METHOD (noise->image conditional MeanFlow + FGA, original arch) differs."""
import os, sys, copy, argparse, time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
sys.path.insert(0, os.path.dirname(__file__))
from dataset_sa import PairedSpineDataset
from utils import count_parameters
from baseline_orig import MFDiT_orig, MeanFlowOrig

ROOT = os.path.expanduser("~/ScoliCMF/data/Spine生成_Miccai数据集")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--out", default=os.path.expanduser("~/ScoliCMF/runs/orig_baseline"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "ckpts"), exist_ok=True)
    log = os.path.join(a.out, "log.txt")
    H, W = 480, 240; device = "cuda"
    torch.manual_seed(1114)
    ds = PairedSpineDataset(root=ROOT, size=(H, W), split_file=os.path.expanduser("~/ScoliCMF/splits/train.txt"), augment=True)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    model = MFDiT_orig(img_size=(H, W), patch_size=8, data_channels=1, cond_channels=1,
                       dim=384, depth=12, num_heads=6).to(device)
    mf = MeanFlowOrig(channels=1, flow_ratio=0.75)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    ema_decay = 0.999
    print(f"[orig-baseline] {count_parameters(model)/1e6:.2f}M params; train_pairs={len(ds)}; steps={a.steps}", flush=True)
    def cyc():
        while True:
            for b in loader: yield b
    it = cyc(); pbar = tqdm(range(1, a.steps + 1), dynamic_ncols=True)
    for step in pbar:
        x_pre, x_post = next(it)
        x_pre, x_post = x_pre.to(device), x_post.to(device)
        loss, mse = mf.loss(model, x_post, x_pre)   # y=x_post, cond=x_pre
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
        if step % 50 == 0:
            pbar.set_description(f"step {step} loss {loss.item():.4f} mse {mse.item():.4f}")
            with open(log, "a") as fh:
                fh.write(f"[{time.asctime()}] Step: {step} | Loss: {loss.item():.6f} | MSE: {mse.item():.6f}\n")
        if step % a.save_every == 0:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step},
                       os.path.join(a.out, "ckpts", f"step_{step}.pt"))
    print("TRAIN_DONE", flush=True)

if __name__ == "__main__":
    main()
