"""State-sensitivity diagnostic (NO retrain): does the trained matched model USE the state path?
On the matched EMA checkpoint, eval val with 4 state inputs: correct | max-dist-shuffled | zero | pop-mean.
If all ~identical => the network ignored the conditioning path. Also report ||adapter(s)|| / ||c_base||
and mean ||xhat(correct)-xhat(shuffle)||_1."""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from meanflow_sa import SourceAnchoredMeanFlow
from eval_gates import build_model
from metrics_img import ssim as ssim_fn, lpips_fn
from aptd_model import APTDNet
HOME=os.path.expanduser("~/ScoliCMF"); dev="cuda" if torch.cuda.is_available() else "cpu"
LOCN=["thoracic","thoracolumbar","lumbar"]; DIRN=["image_left","image_right"]; SDIM=5
def _v4(x): return x.view(-1,1,1,1)
def build_states():
    st={}
    for l in open(os.path.join(HOME,"labels.json")):
        if not l.strip(): continue
        r=json.loads(l); vs=[v for v in r["votes"] if v and v[0]!="ERR"]
        cl=np.zeros(3);cd=np.zeros(2);nl=nd=0
        for ln,dn in vs:
            if ln in LOCN: cl[LOCN.index(ln)]+=1;nl+=1
            if dn in DIRN: cd[DIRN.index(dn)]+=1;nd+=1
        gl=nl/7.;gd=nd/7.; ql=cl/nl if nl>0 else np.zeros(3); qd=cd/nd if nd>0 else np.zeros(2)
        st[r["stem"]]=np.concatenate([gl*ql,gd*qd]).astype(np.float32)
    return st
STATE=build_states()
class CondWithState(nn.Module):
    def __init__(self,base,sdim,D):
        super().__init__();self.base=base
        self.adapter=nn.Sequential(nn.Linear(sdim,D),nn.SiLU(),nn.Linear(D,D))
        self.state=None
    def forward(self,x_pre,r,t,t_emb,r_emb):
        c,aux=self.base(x_pre,r,t,t_emb,r_emb)
        self._cbase=c.detach(); self._adapt=(self.adapter(self.state).detach() if self.state is not None else None)
        if self.state is not None: c=c+self.adapter(self.state)
        return c,aux
cfg=load_config(os.path.join(HOME,"configs/s2_base.yaml")); H,W=cfg["data"]["size_h"],cfg["data"]["size_w"]
cfg["model"]["xpre_mode"]="full"
mf=SourceAnchoredMeanFlow(gamma=cfg["meanflow"]["gamma"],sigma_m=cfg["meanflow"]["sigma_m"]); path=mf.path
bb=build_model(cfg,H,W); D=bb.pos_embed.shape[-1]; bb.cond=CondWithState(bb.cond,SDIM,D); bb=bb.to(dev)
model=APTDNet(bb,"warpres",flow_scale=0.15).to(dev)
ck=torch.load(os.path.join(HOME,"runs/route_matched/ckpts/step_5000.pt"),map_location=dev)
for p,e in zip(model.parameters(),ck["ema"]): p.data.copy_(e.to(dev))
model.eval()
# val data + stems
ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),return_stem=True,split_file=os.path.join(HOME,"splits/val.txt"))
XP=[];XQ=[];ST=[]
for xp,xq,s in DataLoader(ds,batch_size=64,shuffle=False): XP.append(xp);XQ.append(xq);ST+=list(s)
XP=torch.cat(XP);XQ=torch.cat(XQ);Nv=XP.shape[0]
S_correct=np.stack([STATE.get(s,np.zeros(SDIM,np.float32)) for s in ST])
popmean=S_correct.mean(0,keepdims=True).repeat(Nv,0).astype(np.float32)
# max-distance assignment (greedy argmax per case, fixed)
Dmat=np.linalg.norm(S_correct[:,None,:]-S_correct[None,:,:],axis=2); S_shuf=S_correct[Dmat.argmax(1)]
S_zero=np.zeros_like(S_correct)
@torch.no_grad()
def ev(states):
    m=torch.tensor(states,device=dev); SS=[];LP=[];outs=[]
    for i in range(0,Nv,6):
        xp=XP[i:i+6].to(dev); q=XQ[i:i+6].to(dev); B=xp.shape[0]
        model.bb.cond.state=m[i:i+6]
        o=model(xp,torch.zeros(B,device=dev),torch.ones(B,device=dev),xp)["xhat"].clamp(0,1)
        SS.append(ssim_fn(o,q).cpu());LP.append(lpips_fn(o,q).cpu());outs.append(o.cpu())
    return float(torch.cat(SS).mean()),float(torch.cat(LP).mean()),torch.cat(outs)
print("=== state-sensitivity of trained MATCHED model (val, 1-NFE) ===",flush=True)
res={}
for nm,stt in [("correct",S_correct),("max-dist shuffle",S_shuf),("zero",S_zero),("pop-mean",popmean)]:
    s,l,o=ev(stt); res[nm]=o; print("  %-16s SSIM=%.4f LPIPS=%.4f"%(nm,s,l),flush=True)
print("  ||xhat(correct)-xhat(shuffle)||_1 (mean over pixels) = %.6f"%(res["correct"]-res["max-dist shuffle"]).abs().mean().item(),flush=True)
print("  ||xhat(correct)-xhat(zero)||_1                        = %.6f"%(res["correct"]-res["zero"]).abs().mean().item(),flush=True)
# adapter vs base magnitude
model.bb.cond.state=torch.tensor(S_correct,device=dev)[:6]
xp=XP[:6].to(dev); _=model(xp,torch.zeros(6,device=dev),torch.ones(6,device=dev),xp)
ab=model.bb.cond._adapt.norm(dim=-1).mean().item(); cb=model.bb.cond._cbase.norm(dim=-1).mean().item()
print("  ||adapter(s)|| / ||c_base|| = %.5f / %.5f = %.4f"%(ab,cb,ab/max(cb,1e-9)),flush=True)
print("DIAG_DONE",flush=True)
