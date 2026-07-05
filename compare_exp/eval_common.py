"""Common evaluator for comparison experiments. Scores a method's predictions against val post-op.
Usage: python eval_common.py <method_dir>   # reads <method_dir>/preds/<stem>.png, prints SSIM/PSNR/LPIPS.
All methods use the IDENTICAL protocol/split/metrics as our method (fair comparison)."""
import os, sys, glob, numpy as np, torch
from PIL import Image
from torchvision import transforms as T
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from metrics_img import ssim, lpips_fn
HOME=os.path.expanduser("~/ScoliCMF"); ROOT=os.path.join(HOME,"data/Spine生成_Miccai数据集/train"); H,W=480,240
def psnr(a,b): return -10*torch.log10(((a-b)**2).mean(dim=(1,2,3)).clamp_min(1e-10))
def load(p): return T.Compose([T.Resize((H,W)),T.ToTensor()])(Image.open(p).convert("L"))
def main(mdir, split="val.txt"):
    stems=[l.strip() for l in open(os.path.join(HOME,"splits",split)) if l.strip()]
    post={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(ROOT,"postop_standardized","*"))}
    pred_dir=os.path.join(mdir,"preds"); S=[];P=[];L=[]; miss=0
    for s in stems:
        pp=os.path.join(pred_dir,s+".png")
        if not os.path.exists(pp) or s not in post: miss+=1; continue
        pr=load(pp).unsqueeze(0); gt=load(post[s]).unsqueeze(0)
        S.append(float(ssim(pr,gt))); P.append(float(psnr(pr,gt))); L.append(float(lpips_fn(pr,gt)))
    n=len(S)
    print("method=%s  n=%d (missing %d)  SSIM=%.4f  PSNR=%.3f  LPIPS=%.4f"%(
        os.path.basename(mdir.rstrip("/")), n, miss, np.mean(S) if n else 0, np.mean(P) if n else 0, np.mean(L) if n else 0), flush=True)
if __name__=="__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "val.txt")
