"""Definitive routing experiment: APTD + state-conditioned FiLM adapter, MATCHED vs SHUFFLED state.
Same architecture/capacity/config; only difference = whether the phenotype state fed during TRAINING
belongs to the image (matched) or is a random draw (shuffle). Both EVALUATED with the correct (matched)
state. matched >> shuffle on val SSIM/LPIPS => routing carries usable, trainable signal.
State = confidence-gated soft probs [g_loc*q_loc(3), g_dir*q_dir(2)] from the 7 VLM votes (>=3/7 usable,
continuous gate, no hard threshold). Adapter output zero-init => start == plain APTD."""
import os, sys, argparse, json, math, numpy as np, torch, torch.nn as nn
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
LOCN=["thoracic","thoracolumbar","lumbar"]; DIRN=["image_left","image_right"]
# ---- state table from votes ----
def build_states():
    st={}
    for l in open(os.path.join(HOME,"labels.json")):
        if not l.strip(): continue
        r=json.loads(l); vs=[v for v in r["votes"] if v and v[0]!="ERR"]
        cl=np.zeros(3); cd=np.zeros(2); nl=nd=0
        for ln,dn in vs:
            if ln in LOCN: cl[LOCN.index(ln)]+=1; nl+=1
            if dn in DIRN: cd[DIRN.index(dn)]+=1; nd+=1
        gl=nl/7.0; gd=nd/7.0
        ql=cl/nl if nl>0 else np.zeros(3); qd=cd/nd if nd>0 else np.zeros(2)
        st[r["stem"]]=np.concatenate([gl*ql, gd*qd]).astype(np.float32)  # 5-dim
    return st
STATE=build_states(); SDIM=5
ALL_STATES=np.stack(list(STATE.values()))
class CondWithState(nn.Module):
    def __init__(self, base, sdim, D):
        super().__init__(); self.base=base
        self.adapter=nn.Sequential(nn.Linear(sdim,D), nn.SiLU(), nn.Linear(D,D))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
        self.state=None
    def forward(self,x_pre,r,t,t_emb,r_emb):
        c,aux=self.base(x_pre,r,t,t_emb,r_emb)
        if self.state is not None: c=c+self.adapter(self.state)
        return c,aux
def loader(cfg,H,W,split,bs,sh):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),return_stem=True,
                          split_file=os.path.join(HOME,"splits",split),
                          augment=(sh and cfg["data"].get("augment",False)))
    return DataLoader(ds,batch_size=bs,shuffle=sh,num_workers=2,drop_last=sh)
def state_of(stems,dev):
    return torch.tensor(np.stack([STATE.get(s,np.zeros(SDIM,np.float32)) for s in stems]),device=dev)
def psnr(a,b): return -10*torch.log10(((a-b)**2).mean(dim=(1,2,3)).clamp_min(1e-10))
@torch.no_grad()
def evaluate(model,path,cfg,H,W,dev,nfe):  # ALWAYS matched state
    model.eval(); SS=[];PS=[];LP=[]
    for xp,xq,stm in loader(cfg,H,W,"val.txt",6,False):
        xp,xq=xp.to(dev),xq.to(dev); B=xp.shape[0]; z=xp
        tv=torch.linspace(1.0,0.0,nfe+1,device=dev); xhat=None
        for i in range(nfe):
            t=torch.full((B,),tv[i].item(),device=dev); r=torch.full((B,),tv[i+1].item(),device=dev)
            model.bb.cond.state=state_of(stm,dev)
            xhat=model(z,r,t,xp)["xhat"]; z=xp+_v4(path.alpha(r))*(xhat-xp)
        out=xhat.clamp(0,1); SS.append(ssim(out,xq).cpu());PS.append(psnr(out,xq).cpu());LP.append(lpips_fn(out,xq).cpu())
    return float(torch.cat(SS).mean()),float(torch.cat(PS).mean()),float(torch.cat(LP).mean())
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True,choices=["matched","shuffle"])
    ap.add_argument("--out",required=True); ap.add_argument("--steps",type=int,default=5000)
    ap.add_argument("--bs",type=int,default=8); ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--flow_scale",type=float,default=0.15); ap.add_argument("--save_step",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng=np.random.RandomState(a.seed)
    cfg=load_config(os.path.join(HOME,"configs/s2_base.yaml")); H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]
    gamma,sm=cfg["meanflow"]["gamma"],cfg["meanflow"]["sigma_m"]; mf=SourceAnchoredMeanFlow(gamma=gamma,sigma_m=sm); path=mf.path
    cfg["model"]["xpre_mode"]="full"
    backbone=build_model(cfg,H,W)
    D=backbone.pos_embed.shape[-1]; backbone.cond=CondWithState(backbone.cond,SDIM,D)
    backbone=backbone.to(dev); model=APTDNet(backbone,"warpres",flow_scale=a.flow_scale).to(dev)
    nP=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("state=%s trainable=%.2fM adapterD=%d"%(a.state,nP/1e6,D),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-2)
    ema=[p.detach().clone() for p in model.parameters()]
    def emaup(d=0.999):
        for e,p in zip(ema,model.parameters()): e.mul_(d).add_(p.detach(),alpha=1-d)
    odir=os.path.join(HOME,"runs",a.out,"ckpts"); os.makedirs(odir,exist_ok=True)
    logf=open(os.path.join(HOME,"runs",a.out,"log.txt"),"a")
    def log(s): print(s,flush=True); logf.write(s+"\n"); logf.flush()
    it=cycle(loader(cfg,H,W,"train.txt",a.bs,True)); model.train()
    for step in range(1,a.steps+1):
        xp,xq,stm=next(it); xp,xq=xp.to(dev),xq.to(dev); B=xp.shape[0]
        if a.state=="matched": stv=state_of(stm,dev)
        else: stv=torch.tensor(ALL_STATES[rng.randint(0,len(ALL_STATES),B)],device=dev)  # random draw
        model.bb.cond.state=stv
        r,t=sample_rt(B,dev); eps=torch.randn_like(xp) if sm>0 else None
        z_t=path.z_t(xp,xq,_v4(t),eps); w=_v4((path.alpha(t)-path.alpha(r))/(t-r).clamp_min(1e-3))
        out=model(z_t,r,t,xp); xhat=out["xhat"]
        l_span=adaptive_l2_loss(w*(xhat-xq)); l_end=(xhat-xq).abs().mean()
        l_sm=torch.zeros((),device=dev); l_rs=torch.zeros((),device=dev)
        if out["flow"] is not None:
            fl=out["flow"]; l_sm=(fl[:,:,1:]-fl[:,:,:-1]).abs().mean()+(fl[:,:,:,1:]-fl[:,:,:,:-1]).abs().mean()
        if out["res"] is not None: l_rs=out["res"].abs().mean()
        loss=l_span+1.0*l_end+0.05*l_sm+0.02*l_rs
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); emaup()
        if step%200==0: log("step %4d | loss %.4f span %.4f end %.4f"%(step,loss.item(),l_span.item(),l_end.item()))
        if step%a.save_step==0:
            bk=[p.detach().clone() for p in model.parameters()]
            for p,e in zip(model.parameters(),ema): p.data.copy_(e)
            s1,p1,l1=evaluate(model,path,cfg,H,W,dev,1); s4,p4,l4=evaluate(model,path,cfg,H,W,dev,4)
            for p,b in zip(model.parameters(),bk): p.data.copy_(b);
            model.train()
            log("  [eval ema %d] 1NFE SSIM=%.4f PSNR=%.3f LPIPS=%.4f | 4NFE SSIM=%.4f LPIPS=%.4f"%(step,s1,p1,l1,s4,l4))
            torch.save({"model":model.state_dict(),"ema":[e.cpu() for e in ema],"step":step,"state":a.state},os.path.join(odir,"step_%d.pt"%step))
    log("ROUTE_TRAIN_DONE state=%s"%a.state)
if __name__=="__main__": main()
