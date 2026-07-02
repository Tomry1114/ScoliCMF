"""IC-MF Gate C: matched short training. Vary training bridge {current sigma_old | IC sigma} and
whether a source-only main branch is used. Eval = 1-NFE SOURCE-ONLY raw val (the real inference).
Report L_src (source-only) and L_interior separately + SSIM/LPIPS."""
import os, sys, math, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config, adaptive_l2_loss, cycle
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from losses import sample_rt
from metrics_img import ssim, lpips_fn
from aptd_model import APTDNet
def _v4(x): return x.view(-1,1,1,1)
HOME=os.path.expanduser("~/ScoliCMF")
def loader(cfg,H,W,split,bs,sh):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits",split),augment=(sh and cfg["data"].get("augment",False)))
    return DataLoader(ds,batch_size=bs,shuffle=sh,num_workers=2,drop_last=sh)
def psnr(a,b): return -10*torch.log10(((a-b)**2).mean(dim=(1,2,3)).clamp_min(1e-10))
ap=argparse.ArgumentParser()
ap.add_argument("--path",required=True)      # current|ic
ap.add_argument("--src",type=int,default=0)  # 1 => add source-only main branch + SNR-downweight interior
ap.add_argument("--endpoint_only",type=int,default=0)
ap.add_argument("--out",required=True); ap.add_argument("--steps",type=int,default=3000)
ap.add_argument("--bs",type=int,default=8); ap.add_argument("--lr",type=float,default=2e-4); ap.add_argument("--save_step",type=int,default=1000)
a=ap.parse_args(); dev="cuda"
cfg=load_config(os.path.join(HOME,"configs/s2_base.yaml")); H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]
gamma=cfg["meanflow"]["gamma"]; sm=cfg["meanflow"]["sigma_m"]
mf=SourceAnchoredMeanFlow(gamma=gamma,sigma_m=sm); path=mf.path
cfg["model"]["xpre_mode"]="full"
# energy-matched kappa
ts=[i/2000 for i in range(1,2000)]
E_old=sum((sm*math.sin(math.pi*t)**2)**2 for t in ts)/len(ts)
_af=lambda t:(math.exp(gamma*(1-t))-1)/(math.exp(gamma)-1)
E_ai=sum(_af(t)*(1-_af(t)) for t in ts)/len(ts)
kappa=math.sqrt(E_old/E_ai)
def sigma_of(t):
    a_=path.alpha(t)
    if a.path=="ic": return kappa*(a_*(1-a_)).clamp_min(1e-12).sqrt()
    return sm*torch.sin(math.pi*t)**2
def snr_of(t):
    a_=path.alpha(t); s=sigma_of(t); return (a_**2)/(s**2).clamp_min(1e-12)
bb=build_model(cfg,H,W).to(dev); model=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-2)
ema=[p.detach().clone() for p in model.parameters()]
odir=os.path.join(HOME,"runs",a.out,"ckpts"); os.makedirs(odir,exist_ok=True)
logf=open(os.path.join(HOME,"runs",a.out,"log.txt"),"a")
def log(s): print(s,flush=True); logf.write(s+"\n"); logf.flush()
@torch.no_grad()
def evaluate():
    model.eval(); SS=[];PS=[];LP=[];Lsrc=[];Lint=[]
    for xp,xq in loader(cfg,H,W,"val.txt",6,False):
        xp,xq=xp.to(dev),xq.to(dev); B=xp.shape[0]
        # source-only 1-NFE
        ys=model(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)["xhat"].clamp(0,1)
        SS.append(ssim(ys,xq).cpu());PS.append(psnr(ys,xq).cpu());LP.append(lpips_fn(ys,xq).cpu())
        Lsrc.append((ys-xq).abs().mean(dim=(1,2,3)).cpu())
        # interior loss at a mid t (leaked input) for the gap
        t=torch.full((B,),0.6,device=dev); r=torch.zeros(B,device=dev)
        zt=xp+_v4(path.alpha(t))*(xq-xp)+_v4(sigma_of(t))*torch.randn_like(xp)
        yi=model(zt,r,t,xp)["xhat"].clamp(0,1); Lint.append((yi-xq).abs().mean(dim=(1,2,3)).cpu())
    model.train()
    return float(torch.cat(SS).mean()),float(torch.cat(PS).mean()),float(torch.cat(LP).mean()),float(torch.cat(Lsrc).mean()),float(torch.cat(Lint).mean())
it=cycle(loader(cfg,H,W,"train.txt",a.bs,True)); model.train()
log("=== path=%s src=%d endpoint_only=%d kappa=%.4f ==="%(a.path,a.src,a.endpoint_only,kappa))
for step in range(1,a.steps+1):
    xp,xq=next(it); xp,xq=xp.to(dev),xq.to(dev); B=xp.shape[0]
    loss=torch.zeros((),device=dev)
    if a.endpoint_only:
        ys=model(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)["xhat"]
        loss=adaptive_l2_loss(ys-xq)+(ys-xq).abs().mean()
    else:
        r,t=sample_rt(B,dev)
        zt=xp+_v4(path.alpha(t))*(xq-xp)+_v4(sigma_of(t))*torch.randn_like(xp)
        w=_v4((path.alpha(t)-path.alpha(r))/(t-r).clamp_min(1e-3))
        out=model(zt,r,t,xp); xh=out["xhat"]
        wint=1.0
        if a.src: wint=_v4(1.0/(1.0+snr_of(t)))
        l_span=adaptive_l2_loss(wint*w*(xh-xq)); l_end=(wint*(xh-xq)).abs().mean()
        l_sm=out["flow"].abs().mean() if out.get("flow") is not None else torch.zeros((),device=dev)
        l_rs=out["res"].abs().mean() if out.get("res") is not None else torch.zeros((),device=dev)
        loss=l_span+l_end+0.05*l_sm+0.02*l_rs
        if a.src:
            ys=model(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)["xhat"]
            loss=loss+adaptive_l2_loss(ys-xq)+(ys-xq).abs().mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    for e,p in zip(ema,model.parameters()): e.mul_(0.999).add_(p.detach(),alpha=0.001)
    if step%200==0: log("step %4d loss %.4f"%(step,loss.item()))
    if step%a.save_step==0:
        bk=[p.detach().clone() for p in model.parameters()]
        for p,e in zip(model.parameters(),ema): p.data.copy_(e)
        s,ps,lp,ls,li=evaluate()
        for p,b in zip(model.parameters(),bk): p.data.copy_(b)
        log("  [eval %d] SOURCE-ONLY SSIM=%.4f PSNR=%.3f LPIPS=%.4f | L_src=%.4f L_interior=%.4f gap=%.4f"%(step,s,ps,lp,ls,li,li-ls))
        torch.save({"model":model.state_dict(),"ema":[e.cpu() for e in ema],"step":step},os.path.join(odir,"step_%d.pt"%step))
log("ICMF_TRAIN_DONE path=%s src=%d"%(a.path,a.src))
