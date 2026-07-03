"""Continuous Shared-Basis Transport Router (Rui redesign P1-P4).
P1: K=4 SHARED basis adapters with continuous coeffs alpha(s) (not 44 class-experts).
P2: real 6-dim JOINT vote distribution + entropy-calibrated confidence c + state centering (s - s_bar).
P3: route ONLY the transport/flow branch (flow from h+Dh, residual from shared h).
P4: patient-adaptive Gaussian axial mask mu(s),sigma(s).
Frozen APTD (backbone+head). matched vs Hungarian-derange on joint state. state dropout 15%."""
import os, sys, argparse, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from utils import load_config, adaptive_l2_loss, cycle
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from losses import sample_rt
from metrics_img import ssim, lpips_fn
from aptd_model import APTDNet
from models.sc_dit import RMSNorm
try: from scipy.optimize import linear_sum_assignment; HAS=True
except Exception: HAS=False
def _v4(x): return x.view(-1,1,1,1)
HOME=os.path.expanduser("~/ScoliCMF"); LOCN=["thoracic","thoracolumbar","lumbar"]; DIRN=["image_left","image_right"]
SD=6  # joint state dim
def build_joint():
    q={};c={}
    for l in open(os.path.join(HOME,"labels.json")):
        if not l.strip(): continue
        r=json.loads(l); vs=[v for v in r["votes"] if v and v[0]!="ERR" and v[0] in LOCN and v[1] in DIRN]
        cnt=np.zeros(6)
        for ln,dn in vs: cnt[LOCN.index(ln)*2+DIRN.index(dn)]+=1
        nv=len(vs)
        if nv>0:
            p=cnt/nv; H=-(p[p>0]*np.log(p[p>0])).sum(); conf=(nv/7.)*(1-H/math.log(6))
        else: p=np.zeros(6); conf=0.0
        q[r["stem"]]=p.astype(np.float32); c[r["stem"]]=float(conf)
    return q,c
QJ,CF=build_joint()
TR=[l.strip() for l in open(os.path.join(HOME,"splits/train.txt")) if l.strip() and l.strip() in QJ]
QBAR=np.mean([QJ[s] for s in TR if CF[s]>0],0).astype(np.float32)  # train mean joint (for centering)
def derange():
    S=np.stack([QJ[s] for s in TR]); D=np.abs(S[:,None]-S[None]).sum(-1); np.fill_diagonal(D,-1e9)
    if HAS: _,perm=linear_sum_assignment(-D)
    else: perm=D.argmax(1)
    return {TR[i]:TR[perm[i]] for i in range(len(TR))}
DER=derange()
class LowRankAdapter(nn.Module):
    def __init__(s,dim,rank=8):
        super().__init__(); s.norm=RMSNorm(dim); s.down=nn.Linear(dim,rank,bias=False); s.up=nn.Linear(rank,dim,bias=False); nn.init.zeros_(s.up.weight)
    def forward(s,h): return s.up(F.silu(s.down(s.norm(h))))
class SharedBasisTransportRouter(nn.Module):
    def __init__(s,dim,gh,gw,K=4,rank=8):
        super().__init__(); s.gh,s.gw=gh,gw
        s.enc=nn.Sequential(nn.Linear(SD,64),nn.SiLU(),nn.Linear(64,64))
        s.coeff=nn.Linear(64,K); s.gate=nn.Linear(64,1); s.mc=nn.Linear(64,1); s.mw=nn.Linear(64,1)
        s.basis=nn.ModuleList([LowRankAdapter(dim,rank) for _ in range(K)])
        nn.init.zeros_(s.coeff.weight); nn.init.zeros_(s.coeff.bias); nn.init.zeros_(s.gate.weight); nn.init.constant_(s.gate.bias,-2.0)
        y=(torch.arange(gh*gw)//gw).float()/(gh-1); s.register_buffer("y",y)  # (T,)
    def forward(s,h,state,conf):   # state:(B,SD) centered, conf:(B,)
        e=s.enc(state); alpha=torch.tanh(s.coeff(e)); gate=torch.sigmoid(s.gate(e))[:,0]
        mu=torch.sigmoid(s.mc(e)); sig=0.08+0.35*torch.sigmoid(s.mw(e))          # (B,1)
        m=torch.exp(-0.5*((s.y[None,:]-mu)/sig)**2); m=m/(m.amax(1,keepdim=True)+1e-6)  # (B,T)
        d=torch.zeros_like(h)
        for k,a in enumerate(s.basis): d=d+alpha[:,k,None,None]*a(h)
        d=d*m[:,:,None]                                                          # patient-adaptive axial mask
        return (conf*gate)[:,None,None]*d                                        # entropy-conf x learned gate
def loader(cfg,H,W,split,bs,sh):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),return_stem=True,split_file=os.path.join(HOME,"splits",split),augment=False)
    return DataLoader(ds,batch_size=bs,shuffle=sh,num_workers=2,drop_last=sh)
def st_of(stems,dev,mode):   # RAW joint state + confidence (centering applied at use); derange swaps SOURCE case
    S=[];C=[]
    for s in stems:
        src=DER[s] if (mode=="derange" and s in DER) else s
        S.append(QJ.get(src,QBAR)); C.append(CF.get(src,0.0))
    return torch.tensor(np.stack(S),device=dev),torch.tensor(np.array(C,np.float32),device=dev)
def psnr(a,b): return -10*torch.log10(((a-b)**2).mean(dim=(1,2,3)).clamp_min(1e-10))
def run_head_transport(model,router,h,x_pre,state,conf):   # flow from h+Dh, residual from shared h
    B,_,H,W=x_pre.shape; base=F.affine_grid(model.theta.expand(B,2,3),(B,1,H,W),align_corners=False)
    dh=router(h,state,conf)
    flow=model.head(h+dh,x_pre,base)["flow"]; res=model.head(h,x_pre,base)["res"]
    warp=F.grid_sample(x_pre,base+flow.permute(0,2,3,1),align_corners=False,padding_mode="border")
    return warp+res, flow, res
@torch.no_grad()
def evaluate(model,router,path,cfg,H,W,dev,nfe):
    model.eval();router.eval();SS=[];PS=[];LP=[]
    for xp,xq,stm in loader(cfg,H,W,"val.txt",6,False):
        xp,xq=xp.to(dev),xq.to(dev);B=xp.shape[0];z=xp;tv=torch.linspace(1,0,nfe+1,device=dev);xhat=None
        st,cf=st_of(stm,dev,"matched"); st=st-torch.tensor(QBAR,device=dev)
        for i in range(nfe):
            t=torch.full((B,),tv[i].item(),device=dev);r=torch.full((B,),tv[i+1].item(),device=dev)
            h,_,_=model.bb.forward_features(z,r,t,xp); xhat,_,_=run_head_transport(model,router,h,xp,st,cf); xhat=xhat.clamp(0,1)
            z=xp+_v4(path.alpha(r))*(xhat-xp)
        SS.append(ssim(xhat,xq).cpu());PS.append(psnr(xhat,xq).cpu());LP.append(lpips_fn(xhat,xq).cpu())
    router.train();return float(torch.cat(SS).mean()),float(torch.cat(PS).mean()),float(torch.cat(LP).mean())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True,choices=["matched","derange"])
    ap.add_argument("--out",required=True); ap.add_argument("--steps",type=int,default=3000); ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--lr",type=float,default=5e-4); ap.add_argument("--K",type=int,default=4); ap.add_argument("--sdrop",type=float,default=0.15)
    ap.add_argument("--unfreeze",default="none",choices=["none","head"]); ap.add_argument("--save_step",type=int,default=1000); ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed); g=torch.Generator(device=dev).manual_seed(a.seed)
    cfg=load_config(os.path.join(HOME,"configs/s2_base.yaml")); H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]; cfg["model"]["xpre_mode"]="full"
    mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"],sigma_m=cfg["meanflow"]["sigma_m"]); path=mf.path
    bb=build_model(cfg,H,W).to(dev); model=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
    ck=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_5000.pt"),map_location=dev)
    for p,e in zip(model.parameters(),ck["ema"]): p.data.copy_(e.to(dev))
    router=SharedBasisTransportRouter(bb.pos_embed.shape[-1],bb.gh,bb.gw,a.K).to(dev)
    for p in model.parameters(): p.requires_grad=False
    tparams=list(router.parameters())
    if a.unfreeze=="head":
        for p in model.head.parameters(): p.requires_grad=True
        tparams=tparams+list(model.head.parameters())
    print("state=%s K=%d unfreeze=%s trainable=%.4fM scipy=%s"%(a.state,a.K,a.unfreeze,sum(p.numel() for p in tparams)/1e6,HAS),flush=True)
    opt=torch.optim.AdamW(tparams,lr=a.lr,weight_decay=1e-2); ema=[p.detach().clone() for p in tparams]
    def emaup(d=0.999):
        for e,p in zip(ema,tparams): e.mul_(d).add_(p.detach(),alpha=1-d)
    odir=os.path.join(HOME,"runs",a.out,"ckpts"); os.makedirs(odir,exist_ok=True); logf=open(os.path.join(HOME,"runs",a.out,"log.txt"),"a")
    def log(s): print(s,flush=True);logf.write(s+"\n");logf.flush()
    it=cycle(loader(cfg,H,W,"train.txt",a.bs,True)); model.train();router.train()
    for step in range(1,a.steps+1):
        xp,xq,stm=next(it);xp,xq=xp.to(dev),xq.to(dev);B=xp.shape[0]
        fmask=torch.rand(B,generator=g,device=dev)<0.5
        if fmask.any():
            idx=fmask.nonzero().squeeze(1); xp[idx]=torch.flip(xp[idx],[-1]); xq[idx]=torch.flip(xq[idx],[-1])
        qraw,cf=st_of(stm,dev,a.state)
        if fmask.any():  # mirror => swap L/R within each location pair on RAW joint (indices l*2+0/1)
            qr=qraw.clone()
            for lloc in range(3): qr[fmask,lloc*2],qr[fmask,lloc*2+1]=qraw[fmask,lloc*2+1],qraw[fmask,lloc*2]
            qraw=qr
        st=qraw-torch.tensor(QBAR,device=dev)
        keep=(torch.rand(B,generator=g,device=dev)>a.sdrop).float()  # state dropout
        st=st*keep[:,None]; cf=cf*keep
        r,t=sample_rt(B,dev); z_t=path.z_t(xp,xq,_v4(t),None)
        h,_,_=model.bb.forward_features(z_t,r,t,xp)
        xhat,flow,res=run_head_transport(model,router,h,xp,st,cf)
        w=_v4((path.alpha(t)-path.alpha(r))/(t-r).clamp_min(1e-3))
        l=adaptive_l2_loss(w*(xhat-xq))+1.0*(xhat-xq).abs().mean()+0.05*((flow[:,:,1:]-flow[:,:,:-1]).abs().mean()+(flow[:,:,:,1:]-flow[:,:,:,:-1]).abs().mean())
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(tparams,1.0); opt.step(); emaup()
        if step%200==0: log("step %4d loss %.4f"%(step,l.item()))
        if step%a.save_step==0:
            bk=[p.detach().clone() for p in tparams]
            for p,e in zip(tparams,ema): p.data.copy_(e)
            s1,p1,l1=evaluate(model,router,path,cfg,H,W,dev,1)
            for p,b in zip(tparams,bk): p.data.copy_(b)
            model.train();router.train(); log("  [eval ema %d] 1NFE SSIM=%.4f PSNR=%.3f LPIPS=%.4f"%(step,s1,p1,l1))
            torch.save({"router":router.state_dict(),"step":step,"state":a.state},os.path.join(odir,"step_%d.pt"%step))
    log("ROUTE3_TRAIN_DONE state=%s"%a.state)
if __name__=="__main__": main()
