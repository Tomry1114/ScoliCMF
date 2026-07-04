"""ScoliCMF conditional MeanFlow + TEXT — trains original meanflow.py + MFDiT (FGA image cond)
on our preop->postop data, conditioned on (pre-op image + VLM phenotype text). Genuine velocity
MeanFlow (JVP), noise->postop. --text on|off|shuffle ablation."""
import os, sys, argparse, json, glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))     # for metrics_img
sys.path.insert(0, os.path.expanduser("~/ScoliCMF_cmf"))
from utils import cycle, load_config
from meanflow import MeanFlow
from models.dit import MFDiT
from metrics_img import ssim, lpips_fn
try: from scipy.optimize import linear_sum_assignment; HAS=True
except Exception: HAS=False
def _v4(x): return x.view(-1,1,1,1)
HOME=os.path.expanduser("~/ScoliCMF"); ROOT=os.path.join(HOME,"data/Spine生成_Miccai数据集/train")
LOCN=["thoracic","thoracolumbar","lumbar"]; DIRN=["image_left","image_right"]
# ---- phenotype text (6-dim joint) ----
def build_states():
    q={}
    for l in open(os.path.join(HOME,"labels.json")):
        if not l.strip(): continue
        r=json.loads(l); vs=[v for v in r["votes"] if v and v[0]!="ERR" and v[0] in LOCN and v[1] in DIRN]
        c=np.zeros(6)
        for ln,dn in vs: c[LOCN.index(ln)*2+DIRN.index(dn)]+=1
        q[r["stem"]]=(c/len(vs)).astype(np.float32) if len(vs)>0 else np.zeros(6,np.float32)
    return q
QJ=build_states(); TR=[l.strip() for l in open(os.path.join(HOME,"splits/train.txt")) if l.strip() and l.strip() in QJ]
QBAR=np.mean([QJ[s] for s in TR],0).astype(np.float32)
def derange():
    S=np.stack([QJ[s] for s in TR]); D=np.abs(S[:,None]-S[None]).sum(-1); np.fill_diagonal(D,-1e9)
    perm=linear_sum_assignment(-D)[1] if HAS else D.argmax(1); return {TR[i]:QJ[TR[perm[i]]] for i in range(len(TR))}
DER=derange()
def text_of(stems,dev,mode):
    if mode=="off": return None
    S=[(DER.get(s,QBAR) if mode=="shuffle" else QJ.get(s,QBAR))-QBAR for s in stems]
    return torch.tensor(np.stack(S),device=dev)
# ---- data ----
class Paired(Dataset):
    def __init__(self,split,H,W):
        keep=set(l.strip() for l in open(os.path.join(HOME,"splits",split)) if l.strip())
        pre={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(ROOT,"preop_standardized","*"))}
        post={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(ROOT,"postop_standardized","*"))}
        self.stems=sorted(set(pre)&set(post)&keep); self.pre=pre; self.post=post
        self.tf=T.Compose([T.Resize((H,W)),T.ToTensor()])
    def __len__(self): return len(self.stems)
    def __getitem__(self,i):
        s=self.stems[i]
        return self.tf(Image.open(self.pre[s]).convert("L")), self.tf(Image.open(self.post[s]).convert("L")), s
def loader(split,H,W,bs,sh):
    return DataLoader(Paired(split,H,W),batch_size=bs,shuffle=sh,num_workers=2,drop_last=sh)
def psnr(a,b): return -10*torch.log10(((a-b)**2).mean(dim=(1,2,3)).clamp_min(1e-10))
@torch.no_grad()
def sample(mf,model,cond,H,W,dev,steps):
    B=cond.shape[0]; z=torch.randn(B,1,H,W,device=dev); tv=torch.linspace(1,0,steps+1,device=dev)
    for i in range(steps):
        t=torch.full((B,),tv[i].item(),device=dev); r=torch.full((B,),tv[i+1].item(),device=dev)
        z=z-_v4(t-r)*model(z,t,r,cond)
    return mf.normer.unnorm(z).clamp(0,1)
@torch.no_grad()
def evaluate(mf,model,H,W,dev,steps,text_mode):
    model.eval();SS=[];PS=[];LP=[]
    for cond,tgt,stm in loader("val.txt",H,W,6,False):
        cond,tgt=cond.to(dev),tgt.to(dev); model.text_state=text_of(stm,dev,text_mode)
        o=sample(mf,model,cond,H,W,dev,steps)
        SS.append(ssim(o,tgt).cpu());PS.append(psnr(o,tgt).cpu());LP.append(lpips_fn(o,tgt).cpu())
    model.train(); return float(torch.cat(SS).mean()),float(torch.cat(PS).mean()),float(torch.cat(LP).mean())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--text",default="on",choices=["on","off","shuffle"])
    ap.add_argument("--steps",type=int,default=20000); ap.add_argument("--bs",type=int,default=6); ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--nfe",type=int,default=10); ap.add_argument("--save_step",type=int,default=4000); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--attn",default="vanilla",choices=["vanilla","ttt"]); ap.add_argument("--inner_lr",type=float,default=0.25)
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed)
    H,W=480,240
    model=MFDiT(img_size=(H,W),patch_size=8,data_channels=1,cond_channels=1,dim=384,depth=12,num_heads=6,text=(a.text!="off"),attn_type=a.attn,inner_lr=a.inner_lr).to(dev)
    mf=MeanFlow(channels=1,image_size=H,normalizer=['mean_std',[0.5],[0.5]],flow_ratio=0.75,time_dist=['lognorm',-0.4,1.0])
    print("CMF attn=%s inner_lr=%.2f text=%s trainable=%.2fM"%(a.attn,a.inner_lr,a.text,sum(p.numel() for p in model.parameters())/1e6),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.0); ema=[p.detach().clone() for p in model.parameters()]
    def emaup(d=0.999):
        for e,p in zip(ema,model.parameters()): e.mul_(d).add_(p.detach(),alpha=1-d)
    odir=os.path.join(HOME,"runs",a.out,"ckpts"); os.makedirs(odir,exist_ok=True); logf=open(os.path.join(HOME,"runs",a.out,"log.txt"),"a")
    def log(s): print(s,flush=True);logf.write(s+"\n");logf.flush()
    it=cycle(loader("train.txt",H,W,a.bs,True)); model.train()
    for step in range(1,a.steps+1):
        cond,tgt,stm=next(it); cond,tgt=cond.to(dev),tgt.to(dev)
        model.text_state=text_of(stm,dev,a.text)
        loss,mse=mf.loss(model,tgt,cond_img=cond)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); emaup()
        if step%200==0: log("step %5d loss %.4f mse %.5f"%(step,loss.item(),float(mse)))
        if step%a.save_step==0:
            bk=[p.detach().clone() for p in model.parameters()]
            for p,e in zip(model.parameters(),ema): p.data.copy_(e)
            s1,p1,l1=evaluate(mf,model,H,W,dev,a.nfe,a.text)
            for p,b in zip(model.parameters(),bk): p.data.copy_(b)
            log("  [eval ema %d] %dNFE SSIM=%.4f PSNR=%.3f LPIPS=%.4f"%(step,a.nfe,s1,p1,l1))
            torch.save({"model":model.state_dict(),"ema":[e.cpu() for e in ema],"step":step,"text":a.text},os.path.join(odir,"step_%d.pt"%step))
    log("CMF_DONE text=%s"%a.text)
if __name__=="__main__": main()
