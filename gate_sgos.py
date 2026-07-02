"""SGOS Gate A (oracle-control headroom, NO generative training, CPU/GPU-light).
SGOS changes the INFORMATION SET: I=(X_pre, A), A={K sparse spine-height horizontal-move targets}.
This is the FIRST idea that can lower the identifiability ceiling (A = low-bandwidth proxy of the
unobservable surgical plan). Cheap upper-bound test: derive oracle horizontal displacement d(y) from
the TRUE post-op midline, keep only K anchors, biharmonic(=natural cubic spline) extend to a control
field, WARP x_pre by it, measure how close to x_post vs K.

Arms (raw val, per-case):
  x_pre (no control)         floor
  warp K=1,3,5,8             sparse oracle control
  warp dense (all rows)      full oracle control (upper bound of the horizontal-shift channel)
  warp shuffled-K5           negative control (d from a random other case)
Reference: current APTD source-only WITHOUT control  step2000 SSIM 0.2554/LPIPS 0.4429 ; step5000 0.3018/0.6650.
PASS (SGOS headroom): K=3..5 oracle warp beats APTD-no-control clearly (dSSIM>=+0.02 or dLPIPS<=-0.03,
other not worse) AND >> shuffled. Also report dense-oracle to see the ceiling of the shift channel and
whether a generative model is even needed (warp alone vs post gap = room for new content R)."""
import os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from metrics_img import ssim as ssim_fn, lpips_fn
HOME=os.path.expanduser("~/ScoliCMF"); H,W=480,240; NR=48  # midline row resolution
dev="cuda" if torch.cuda.is_available() else "cpu"
def load(split):
    ds=PairedSpineDataset(root=os.path.join(HOME,load_config(os.path.join(HOME,"configs/s2_base.yaml"))["data"]["root"]),
                          size=(H,W),split_file=os.path.join(HOME,"splits/%s.txt"%split))
    XP=[];XQ=[]
    for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
    return torch.cat(XP),torch.cat(XQ)
xpva,xqva=load("val"); N=xpva.shape[0]
def midline_px(imgs):  # (N,1,H,W)->(N,NR) horizontal centroid in PIXELS, at NR row bands
    N=imgs.shape[0]; xs=np.arange(W)[None,:]; out=np.full((N,NR),np.nan)
    for n in range(N):
        im=imgs[n,0].numpy(); w=np.clip(im-im.mean(),0,None)
        num=(w*xs).sum(1); den=w.sum(1); c=np.where(den>1e-6,num/np.maximum(den,1e-8),np.nan)  # (H,)
        for r,b in enumerate(np.array_split(c,NR)):
            if np.any(~np.isnan(b)): out[n,r]=np.nanmean(b)
    col=np.nanmean(out,0); out=np.where(np.isnan(out),col[None,:],out); return out  # pixels
Cpre=midline_px(xpva); Cpost=midline_px(xqva); Dfull=Cpost-Cpre  # (N,NR) px displacement per row-band
rowc=(np.linspace(0,H-1,NR))  # band center rows

def spline_field(d_bands,anchor_idx):
    """natural cubic spline through K anchors -> displacement at every image row (H,)."""
    ax=rowc[anchor_idx]; ay=d_bands[anchor_idx]
    if len(anchor_idx)==1: return np.full(H,ay[0])
    # natural cubic spline
    xr=np.arange(H)
    return np.interp(xr,ax,ay) if len(anchor_idx)<4 else _cubic(ax,ay,xr)
def _cubic(x,y,xr):
    n=len(x); h=np.diff(x); A=np.zeros((n,n)); b=np.zeros(n); A[0,0]=A[-1,-1]=1
    for i in range(1,n-1):
        A[i,i-1]=h[i-1]; A[i,i]=2*(h[i-1]+h[i]); A[i,i+1]=h[i]
        b[i]=3*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
    c=np.linalg.solve(A,b); out=np.empty_like(xr,dtype=float)
    for k,xx in enumerate(xr):
        i=min(np.searchsorted(x,xx)-1,n-2); i=max(i,0); dx=xx-x[i]
        bb=(y[i+1]-y[i])/h[i]-h[i]*(2*c[i]+c[i+1])/3; dd=(c[i+1]-c[i])/(3*h[i])
        out[k]=y[i]+bb*dx+c[i]*dx**2+dd*dx**3
    return out
def warp(xp,drow):  # xp (1,1,H,W); drow (H,) px shift per row -> sample content shifted right by d
    bx=torch.linspace(-1,1,W,device=dev).view(1,W).repeat(H,1)
    sh=torch.tensor(drow,device=dev,dtype=torch.float32).view(H,1)*2/(W-1)
    gx=(bx-sh).clamp(-1,1); by=torch.linspace(-1,1,H,device=dev).view(H,1).repeat(1,W)
    grid=torch.stack([gx,by],-1).unsqueeze(0)
    return F.grid_sample(xp.to(dev),grid,align_corners=True,padding_mode="border")
def evalset(getd):
    S=[];L=[]
    for n in range(N):
        xp=xpva[n:n+1]; q=xqva[n:n+1].to(dev); dr=getd(n)
        o=warp(xp,dr).clamp(0,1) if dr is not None else xp.to(dev)
        S.append(float(ssim_fn(o,q))); L.append(float(lpips_fn(o,q)))
    return np.mean(S),np.mean(L)
def anchors(K): return np.linspace(0,NR-1,K).round().astype(int)
rng=np.random.RandomState(0); perm=rng.permutation(N)

print("== SGOS Gate A: oracle sparse-control warp headroom (raw val, N=%d) =="%N,flush=True)
print("  ref APTD no-control: step2000 SSIM .2554/LPIPS .4429  step5000 .3018/.6650",flush=True)
s,l=evalset(lambda n:None); print("  x_pre(no ctrl)   SSIM=%.4f LPIPS=%.4f"%(s,l),flush=True)
for K in [1,3,5,8]:
    ai=anchors(K)
    s,l=evalset(lambda n: spline_field(Dfull[n],ai))
    print("  warp K=%-2d         SSIM=%.4f LPIPS=%.4f"%(K,s,l),flush=True)
s,l=evalset(lambda n: spline_field(Dfull[n],np.arange(NR)))
print("  warp DENSE(oracle) SSIM=%.4f LPIPS=%.4f"%(s,l),flush=True)
ai=anchors(5)
s,l=evalset(lambda n: spline_field(Dfull[perm[n]],ai))
print("  warp shuffled K=5  SSIM=%.4f LPIPS=%.4f  (neg control)"%(s,l),flush=True)
print("SGOS_GATE_DONE",flush=True)
