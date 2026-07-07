import sys, torch
sys.path.insert(0, "cmf")
torch.backends.cudnn.enabled = False
from models.dit_drica import MFDiTDRICA
from meanflow import MeanFlow
dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W, B = 480, 240, 2

torch.manual_seed(0)
m = MFDiTDRICA(img_size=(H,W), patch_size=8, dim=384, depth=12, num_heads=6, drica_layer_ids=(2,6,10)).to(dev)
print("params=%.2fM  drica layers=%s" % (sum(p.numel() for p in m.parameters())/1e6, m.drica_layer_ids))

def diag(B):
    r=torch.rand(B,3,device=dev); r/=r.sum(1,keepdim=True)
    d=torch.rand(B,2,device=dev); d/=d.sum(1,keepdim=True)
    j=torch.rand(B,6,device=dev); j/=j.sum(1,keepdim=True)
    return {"region":r,"direction":d,"joint":j}
D = diag(B)

g=torch.Generator(device="cpu").manual_seed(20260707)
y=torch.randn(B,1,H,W,generator=g).to(dev); e=torch.randn(B,1,H,W,generator=g).to(dev); cond=torch.randn(B,1,H,W,generator=g).to(dev)
t=torch.rand(B,generator=g).to(dev).clamp(0.05,0.95); r=t*torch.rand(B,generator=g).to(dev)
t_=t.view(B,1,1,1); v=e-y; z=(1-t_)*y+t_*e

# (1) forward at init (should be ~0 due to zero-init final/output_proj)
m.eval()
with torch.no_grad(): o0=m(z,t,r,cond,D)
print("(1) forward OK out=%s  init |out|max=%.2e (zero-init => ~0)" % (tuple(o0.shape), o0.abs().max().item()))

# (2) perturb zero-inited params to a realistic regime, then JVP through meanflow-style f
torch.manual_seed(1)
for p in m.parameters():
    if float(p.abs().sum())==0.0: p.data.normal_(0,0.05)
def f(zz,ti,ri): return m(zz,ti,ri,cond,D)
u,dudt=torch.autograd.functional.jvp(f,(z,t,r),(v,torch.ones_like(t),torch.zeros_like(r)),create_graph=True)
print("(2) JVP finite=%s max|dudt|=%.3e NaN=%d Inf=%d  |u|max=%.3e" %
      (bool(torch.isfinite(dudt).all()), dudt[torch.isfinite(dudt)].abs().max().item(),
       int(torch.isnan(dudt).sum()), int(torch.isinf(dudt).sum()), u.abs().max().item()))

# (3) real meanflow loss with model_kwargs -> backward, all grad finite
mf=MeanFlow(channels=1,image_size=(H,W),jvp_api='autograd')
m.train()
loss,mse=mf.loss(m,y,cond_img=cond,model_kwargs={"diagnosis":D})
loss.backward()
gfin=all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
ngrad=sum(1 for p in m.parameters() if p.grad is not None)
print("(3) meanflow loss=%.4f  all-grad-finite=%s  (%d tensors got grad)" % (loss.item(), gfin, ngrad))

# (4) diagnosis actually matters: different diagnosis -> different output
with torch.no_grad():
    D2=diag(B); oa=m(z,t,r,cond,D); ob=m(z,t,r,cond,D2)
    print("(4) output rel-change under different diagnosis: %.3e (>0 => DRICA routing active)" % ((oa-ob).norm()/oa.norm()).item())
print("DONE")
