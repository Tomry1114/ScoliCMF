"""True factorized state ROUTING (Rui's design). Load trained APTD, FREEZE backbone+head, train ONLY
low-rank state-routers in the last 4 DiT blocks: h' = h + sum_k q_loc_k A^loc_k(h) + sum_j q_dir_j A^dir_j(h),
A(h)=W_up sigma(W_down Norm(h)), up zero-init. State = confidence-gated soft probs q_loc(3)/q_dir(2).
matched vs fixed max-dist derangement, SAME init/seed, both eval'd with correct state. This tests whether
state-specific adapters explain residual case-dependent variance ON TOP of the shared pre->post map."""
import os, sys, argparse, json, types, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
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
def _v4(x): return x.view(-1,1,1,1)
HOME=os.path.expanduser("~/ScoliCMF"); LOCN=["thoracic","thoracolumbar","lumbar"]; DIRN=["image_left","image_right"]
def build_states():
    ql={};qd={}
    for l in open(os.path.join(HOME,"labels.json")):
        if not l.strip(): continue
        r=json.loads(l); vs=[v for v in r["votes"] if v and v[0]!="ERR"]
        cl=np.zeros(3);cd=np.zeros(2);nl=nd=0
        for ln,dn in vs:
            if ln in LOCN: cl[LOCN.index(ln)]+=1;nl+=1
            if dn in DIRN: cd[DIRN.index(dn)]+=1;nd+=1
        ql[r["stem"]]=((nl/7.)*(cl/nl if nl>0 else cl)).astype(np.float32)
        qd[r["stem"]]=((nd/7.)*(cd/nd if nd>0 else cd)).astype(np.float32)
    return ql,qd
QL,QD=build_states()
def derange_map():
    tr=[l.strip() for l in open(os.path.join(HOME,"splits/train.txt")) if l.strip() and l.strip() in QL]
    S=np.stack([np.concatenate([QL[s],QD[s]]) for s in tr]); mu=S.mean(0);_,_,Vt=np.linalg.svd(S-mu,full_matrices=False)
    pr=(S-mu)@Vt[0]; o=np.argsort(pr); N=len(o); perm=np.empty(N,int)
    for k in range(N): perm[o[k]]=o[(k+N//2)%N]
    return {tr[i]:(QL[tr[perm[i]]],QD[tr[perm[i]]]) for i in range(N)}
DER=derange_map()
class LowRankAdapter(nn.Module):
    def __init__(self,dim,rank=8):
        super().__init__(); self.norm=RMSNorm(dim); self.down=nn.Linear(dim,rank,bias=False); self.up=nn.Linear(rank,dim,bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self,h): return self.up(F.silu(self.down(self.norm(h))))
class FactorizedStateRouter(nn.Module):
    def __init__(self,dim,rank=8):
        super().__init__(); self.location=nn.ModuleList([LowRankAdapter(dim,rank) for _ in range(3)])
        self.direction=nn.ModuleList([LowRankAdapter(dim,rank) for _ in range(2)])
    def forward(self,h,ql,qd):
        d=torch.zeros_like(h)
        for k,a in enumerate(self.location): d=d+ql[:,k,None,None]*a(h)
        for k,a in enumerate(self.direction): d=d+qd[:,k,None,None]*a(h)
        return h+d
def ff_routed(self,z_t,r,t,x_pre):
    x=self.x_embedder(self._x_in(z_t,x_pre))+self.pos_embed
    c,aux=self.cond(x_pre,r,t,self.t_embedder(t),self.r_embedder(r)); nb=len(self.blocks)
    for i,blk in enumerate(self.blocks):
        x=blk(x,c)
        if i>=nb-4: x=self.state_routers[str(i)](x,self._ql,self._qd)
    return x,c,aux
def loader(cfg,H,W,split,bs,sh):
    ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),return_stem=True,
                          split_file=os.path.join(HOME,"splits",split),augment=(sh and cfg["data"].get("augment",False)))
    return DataLoader(ds,batch_size=bs,shuffle=sh,num_workers=2,drop_last=sh)
def qof(stems,dev,mode):
    ql=[];qd=[]
    for s in stems:
        if mode=="derange" and s in DER: a,b=DER[s]
        else: a,b=QL.get(s,np.zeros(3,np.float32)),QD.get(s,np.zeros(2,np.float32))
        ql.append(a);qd.append(b)
    return torch.tensor(np.stack(ql),device=dev),torch.tensor(np.stack(qd),device=dev)
def psnr(a,b): return -10*torch.log10(((a-b)**2).mean(dim=(1,2,3)).clamp_min(1e-10))
@torch.no_grad()
def evaluate(model,path,cfg,H,W,dev,nfe):
    model.eval();SS=[];PS=[];LP=[]
    for xp,xq,stm in loader(cfg,H,W,"val.txt",6,False):
        xp,xq=xp.to(dev),xq.to(dev);B=xp.shape[0];z=xp;tv=torch.linspace(1,0,nfe+1,device=dev);xhat=None
        ql,qd=qof(stm,dev,"matched")   # ALWAYS correct state at eval
        for i in range(nfe):
            t=torch.full((B,),tv[i].item(),device=dev);r=torch.full((B,),tv[i+1].item(),device=dev)
            model.bb._ql=ql;model.bb._qd=qd; xhat=model(z,r,t,xp)["xhat"]; z=xp+_v4(path.alpha(r))*(xhat-xp)
        o=xhat.clamp(0,1);SS.append(ssim(o,xq).cpu());PS.append(psnr(o,xq).cpu());LP.append(lpips_fn(o,xq).cpu())
    return float(torch.cat(SS).mean()),float(torch.cat(PS).mean()),float(torch.cat(LP).mean())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",required=True,choices=["matched","derange"])
    ap.add_argument("--out",required=True); ap.add_argument("--steps",type=int,default=3000); ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--lr",type=float,default=5e-4); ap.add_argument("--rank",type=int,default=8); ap.add_argument("--save_step",type=int,default=1000); ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg=load_config(os.path.join(HOME,"configs/s2_base.yaml")); H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]
    cfg["model"]["xpre_mode"]="full"; mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"],sigma_m=cfg["meanflow"]["sigma_m"]); path=mf.path
    bb=build_model(cfg,H,W).to(dev); model=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
    ck=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_5000.pt"),map_location=dev)  # load trained APTD
    for p,e in zip(model.parameters(),ck["ema"]): p.data.copy_(e.to(dev))
    D=bb.pos_embed.shape[-1]; nb=len(bb.blocks)
    bb.state_routers=nn.ModuleDict({str(i):FactorizedStateRouter(D,a.rank) for i in range(nb-4,nb)}).to(dev)
    bb.forward_features=types.MethodType(ff_routed,bb)
    for p in model.parameters(): p.requires_grad=False
    rparams=[p for r in bb.state_routers.values() for p in r.parameters()]
    for p in rparams: p.requires_grad=True
    print("state=%s ROUTER-only trainable=%.4fM (frozen APTD)"%(a.state,sum(p.numel() for p in rparams)/1e6),flush=True)
    opt=torch.optim.AdamW(rparams,lr=a.lr,weight_decay=1e-2); ema=[p.detach().clone() for p in rparams]
    def emaup(d=0.999):
        for e,p in zip(ema,rparams): e.mul_(d).add_(p.detach(),alpha=1-d)
    odir=os.path.join(HOME,"runs",a.out,"ckpts"); os.makedirs(odir,exist_ok=True); logf=open(os.path.join(HOME,"runs",a.out,"log.txt"),"a")
    def log(s): print(s,flush=True);logf.write(s+"\n");logf.flush()
    it=cycle(loader(cfg,H,W,"train.txt",a.bs,True)); model.train()
    for step in range(1,a.steps+1):
        xp,xq,stm=next(it);xp,xq=xp.to(dev),xq.to(dev);B=xp.shape[0]
        ql,qd=qof(stm,dev,a.state); model.bb._ql=ql; model.bb._qd=qd
        r,t=sample_rt(B,dev); z_t=path.z_t(xp,xq,_v4(t),None); w=_v4((path.alpha(t)-path.alpha(r))/(t-r).clamp_min(1e-3))
        out=model(z_t,r,t,xp); xhat=out["xhat"]
        l=adaptive_l2_loss(w*(xhat-xq))+1.0*(xhat-xq).abs().mean()
        if out["flow"] is not None:
            fl=out["flow"]; l=l+0.05*((fl[:,:,1:]-fl[:,:,:-1]).abs().mean()+(fl[:,:,:,1:]-fl[:,:,:,:-1]).abs().mean())
        if out["res"] is not None: l=l+0.02*out["res"].abs().mean()
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(rparams,1.0); opt.step(); emaup()
        if step%200==0: log("step %4d loss %.4f"%(step,l.item()))
        if step%a.save_step==0:
            bk=[p.detach().clone() for p in rparams]
            for p,e in zip(rparams,ema): p.data.copy_(e)
            s1,p1,l1=evaluate(model,path,cfg,H,W,dev,1)
            for p,b in zip(rparams,bk): p.data.copy_(b);
            model.train(); log("  [eval ema %d] 1NFE SSIM=%.4f PSNR=%.3f LPIPS=%.4f"%(step,s1,p1,l1))
            torch.save({"routers":bb.state_routers.state_dict(),"ema":[e.cpu() for e in ema],"step":step,"state":a.state},os.path.join(odir,"step_%d.pt"%step))
    log("ROUTE2_TRAIN_DONE state=%s"%a.state)
if __name__=="__main__": main()
