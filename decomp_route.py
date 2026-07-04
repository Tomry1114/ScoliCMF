"""Decompose WHERE the +0.0005 matched-vs-derange gap lives: per-case, per-phenotype-cell, per-region.
Also controllability: on the matched model, does CORRECT state move x_hat closer to x_post than WRONG
(derange) state, per case? Loads spatial seed-0 checkpoints (routers on frozen APTD)."""
import os, sys, json, types, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from metrics_img import ssim as ssim_fn
from aptd_model import APTDNet
from models.sc_dit import RMSNorm
HOME=os.path.expanduser("~/ScoliCMF"); dev="cuda" if torch.cuda.is_available() else "cpu"
LOCN=["thoracic","thoracolumbar","lumbar"]; DIRN=["image_left","image_right"]
def _v4(x): return x.view(-1,1,1,1)
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
try: from scipy.optimize import linear_sum_assignment; HAS=True
except Exception: HAS=False
def DERmap():
    tr=[l.strip() for l in open(os.path.join(HOME,"splits/train.txt")) if l.strip() and l.strip() in QL]
    S=np.stack([np.concatenate([QL[s],QD[s]]) for s in tr]); dist=np.abs(S[:,None]-S[None]).sum(-1); np.fill_diagonal(dist,-1e9)
    _,perm=linear_sum_assignment(-dist); return {tr[i]:(QL[tr[perm[i]]],QD[tr[perm[i]]]) for i in range(len(tr))}
DER=DERmap()
class LowRankAdapter(nn.Module):
    def __init__(s,dim,rank=8):
        super().__init__(); s.norm=RMSNorm(dim); s.down=nn.Linear(dim,rank,bias=False); s.up=nn.Linear(rank,dim,bias=False)
    def forward(s,h): return s.up(F.silu(s.down(s.norm(h))))
class FactorizedStateRouter(nn.Module):
    def __init__(s,dim,gh,gw,rank=8,spatial=True):
        super().__init__(); s.location=nn.ModuleList([LowRankAdapter(dim,rank) for _ in range(3)])
        s.direction=nn.ModuleList([LowRankAdapter(dim,rank) for _ in range(2)]); s.joint=nn.ModuleList([LowRankAdapter(dim,rank) for _ in range(6)]); s.spatial=spatial
        if spatial:
            T=gh*gw; u=(torch.arange(T)//gw).float()/(gh-1); cen=torch.tensor([0.25,0.5,0.75])
            s.register_buffer("mask",torch.softmax(-((u[None,:]-cen[:,None])**2)/0.03,dim=0))
    def forward(s,h,ql,qd):
        d=torch.zeros_like(h)
        for k,a in enumerate(s.location):
            g=ql[:,k,None,None]*a(h); d=d+(g*s.mask[k][None,:,None] if s.spatial else g)
        for k,a in enumerate(s.direction): d=d+qd[:,k,None,None]*a(h)
        for i in range(3):
            for j in range(2):
                w=(ql[:,i]*qd[:,j])[:,None,None]*s.joint[i*2+j](h); d=d+(w*s.mask[i][None,:,None] if s.spatial else w)
        return h+d
def ff(self,z_t,r,t,x_pre):
    x=self.x_embedder(self._x_in(z_t,x_pre))+self.pos_embed; c,aux=self.cond(x_pre,r,t,self.t_embedder(t),self.r_embedder(r)); nb=len(self.blocks)
    for i,blk in enumerate(self.blocks):
        x=blk(x,c)
        if i>=nb-4: x=self.state_routers[str(i)](x,self._ql,self._qd)
    return x,c,aux
cfg=load_config(os.path.join(HOME,"configs/s2_base.yaml")); H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]; cfg["model"]["xpre_mode"]="full"
mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"],sigma_m=cfg["meanflow"]["sigma_m"]); path=mf.path
def load(run):
    bb=build_model(cfg,H,W).to(dev); model=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
    ck=torch.load(os.path.join(HOME,"runs/aptd_long_fs015/ckpts/step_5000.pt"),map_location=dev)
    for p,e in zip(model.parameters(),ck["ema"]): p.data.copy_(e.to(dev))
    bb.state_routers=nn.ModuleDict({str(i):FactorizedStateRouter(bb.pos_embed.shape[-1],bb.gh,bb.gw,8,True) for i in range(len(bb.blocks)-4,len(bb.blocks))}).to(dev)
    bb.forward_features=types.MethodType(ff,bb)
    rck=torch.load(os.path.join(HOME,"runs/%s/ckpts/step_3000.pt"%run),map_location=dev)
    bb.state_routers.load_state_dict(rck["routers"]); model.eval(); return model
Mm=load("route2sp_matched_s0"); Md=load("route2sp_derange_s0")
ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),return_stem=True,split_file=os.path.join(HOME,"splits/val.txt"))
XP=[];XQ=[];ST=[]
for xp,xq,s in DataLoader(ds,batch_size=64,shuffle=False): XP.append(xp);XQ.append(xq);ST+=list(s)
XP=torch.cat(XP);XQ=torch.cat(XQ);Nv=XP.shape[0]
def qof(stems,mode):
    ql=[];qd=[]
    for s in stems:
        a,b=(DER[s] if (mode=="derange" and s in DER) else (QL.get(s,np.zeros(3,np.float32)),QD.get(s,np.zeros(2,np.float32)))); ql.append(a);qd.append(b)
    return torch.tensor(np.stack(ql),device=dev),torch.tensor(np.stack(qd),device=dev)
@torch.no_grad()
def percase(model,state_mode):
    S=[]
    for i in range(0,Nv,6):
        xp=XP[i:i+6].to(dev);q=XQ[i:i+6].to(dev);B=xp.shape[0]; ql,qd=qof(ST[i:i+6],state_mode)
        model.bb._ql=ql;model.bb._qd=qd; o=model(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)["xhat"].clamp(0,1)
        S.append(ssim_fn(o,q).cpu())
    return torch.cat(S).numpy()
sm=percase(Mm,"matched"); sd=percase(Md,"matched")  # both correct state; matched-model vs derange-model
gap=sm-sd
# phenotype cell per val case (argmax, uncertain if g low)
cell=[]
for s in ST:
    ql=QL.get(s,np.zeros(3)); qd=QD.get(s,np.zeros(2))
    if ql.sum()<3/7-1e-6 or qd.sum()<3/7-1e-6: cell.append("uncertain")
    else: cell.append("%s|%s"%(LOCN[ql.argmax()],DIRN[qd.argmax()]))
cell=np.array(cell)
print("=== matched-model vs derange-model, per-case SSIM gap (both eval correct state) ===",flush=True)
print("  overall gap mean=%.4f  median=%.4f  >0 frac=%.2f  (n=%d)"%(gap.mean(),np.median(gap),(gap>0).mean(),Nv),flush=True)
print("  by phenotype cell:",flush=True)
for c in sorted(set(cell)):
    m=cell==c; print("    %-22s n=%2d  gap mean=%+.4f  >0 frac=%.2f"%(c,m.sum(),gap[m].mean(),(gap[m]>0).mean()),flush=True)
# controllability: matched model, correct vs proper VAL-wrong state at inference
Sval=np.stack([np.concatenate([QL.get(s,np.zeros(3)),QD.get(s,np.zeros(2))]) for s in ST])
dv=np.abs(Sval[:,None]-Sval[None]).sum(-1); np.fill_diagonal(dv,-1e9); vperm=dv.argmax(1)  # each val -> most distant VAL state
VWQL={ST[i]:QL.get(ST[vperm[i]],np.zeros(3,np.float32)) for i in range(Nv)}; VWQD={ST[i]:QD.get(ST[vperm[i]],np.zeros(2,np.float32)) for i in range(Nv)}
@torch.no_grad()
def outs(model,mode):
    O=[]
    for i in range(0,Nv,6):
        xp=XP[i:i+6].to(dev);B=xp.shape[0]
        if mode=="valwrong":
            ql=torch.tensor(np.stack([VWQL[s] for s in ST[i:i+6]]),device=dev); qd=torch.tensor(np.stack([VWQD[s] for s in ST[i:i+6]]),device=dev)
        else: ql,qd=qof(ST[i:i+6],"matched")
        model.bb._ql=ql;model.bb._qd=qd
        O.append(model(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)["xhat"].clamp(0,1).cpu())
    return torch.cat(O)
oc=outs(Mm,"matched"); ow=outs(Mm,"valwrong")
sc=np.array([float(ssim_fn(oc[i:i+1].to(dev),XQ[i:i+1].to(dev))) for i in range(Nv)])
sw=np.array([float(ssim_fn(ow[i:i+1].to(dev),XQ[i:i+1].to(dev))) for i in range(Nv)])
print("=== controllability (matched model): correct vs wrong state at inference ===",flush=True)
print("  mean output change ||x_correct - x_wrong||_1 = %.5f"%(oc-ow).abs().mean().item(),flush=True)
print("  correct-state SSIM=%.4f  wrong-state SSIM=%.4f  correct>wrong frac=%.2f"%(sc.mean(),sw.mean(),(sc>sw).mean()),flush=True)
print("DECOMP_DONE",flush=True)
