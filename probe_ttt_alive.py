"""Probe-2: is TTT's inner-loop actually doing anything, or is it an inert static mixer?
(1) forward output change  inner_lr=0.25 vs 0.0   (2) inner-update relative magnitude ||lr*g||/||w||
(3) after a real meanflow loss.backward(), grad norm on w1/w2/w3 vs qkv/proj."""
import sys, torch
sys.path.insert(0, "cmf")
torch.backends.cudnn.enabled = False
from models.dit import MFDiT
import ttt_block as TB

dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W, B = 480, 240, 2

def build(inner_lr):
    torch.manual_seed(0)
    m = MFDiT(img_size=(H, W), patch_size=8, data_channels=1, cond_channels=1,
              dim=384, depth=12, num_heads=6, text=False, attn_type="ttt", inner_lr=inner_lr).to(dev)
    torch.manual_seed(1)
    for p in m.parameters():
        if float(p.abs().sum()) == 0.0: p.data.normal_(0, 0.05)
    return m

g = torch.Generator(device="cpu").manual_seed(20260707)
y    = torch.randn(B,1,H,W,generator=g).to(dev); e = torch.randn(B,1,H,W,generator=g).to(dev)
cond = torch.randn(B,1,H,W,generator=g).to(dev)
t = torch.rand(B,generator=g).to(dev).clamp(0.05,0.95); r = t*torch.rand(B,generator=g).to(dev)
t_ = t.view(B,1,1,1); z = (1-t_)*y + t_*e

# (1) output change 0.25 vs 0.0  (weights identical; only inner_lr differs)
m25 = build(0.25); m00 = build(0.0)
with torch.no_grad():
    o25 = m25(z,t,r,cond); o00 = m00(z,t,r,cond)
    rel = (o25-o00).norm() / o00.norm()
print("(1) output rel-change  inner_lr 0.25 vs 0.0 : %.4e   (if ~0 -> TTT inert in forward)" % rel.item())

# (2) instrument one TTT block: capture ||lr*g|| / ||w|| for w1,w2,w3
blk = m25.blocks[0].attn
rec = {}
orig_sg = blk.inner_train_simplified_swiglu; orig_dw = blk.inner_train_3x3dwc
def wrap_sg(k,v,w1,w2,lr=1.0):
    nw1,nw2 = orig_sg(k,v,w1,w2,lr)
    rec["w1"] = ((nw1-w1).norm()/(w1.norm()+1e-12)).item()
    rec["w2"] = ((nw2-w2).norm()/(w2.norm()+1e-12)).item(); return nw1,nw2
def wrap_dw(k,v,w,lr=1.0,implementation='prod'):
    nw = orig_dw(k,v,w,lr,implementation)
    w_rep = w.repeat(nw.shape[0]//w.shape[0],1,1,1) if nw.shape[0]!=w.shape[0] else w
    rec["w3"] = ((nw-w_rep).norm()/(w_rep.norm()+1e-12)).item(); return nw
blk.inner_train_simplified_swiglu = wrap_sg; blk.inner_train_3x3dwc = wrap_dw
with torch.no_grad(): _ = m25(z,t,r,cond)
print("(2) inner-update rel-mag ||lr*g||/||w||  block0:  w1=%.4e  w2=%.4e  w3=%.4e  (if <<1 -> negligible)"
      % (rec["w1"], rec["w2"], rec["w3"]))
blk.inner_train_simplified_swiglu = orig_sg; blk.inner_train_3x3dwc = orig_dw

# (3) real meanflow loss backward -> grad norms on ttt params vs qkv/proj
sys.path.insert(0, "cmf")
from meanflow import MeanFlow
mf = MeanFlow(channels=1, image_size=(H,W), jvp_api='autograd')
m25.train()
loss, _ = mf.loss(m25, y, cond)
loss.backward()
def gnorm(sub):
    tot = 0.0; cnt = 0
    for n,p in m25.named_parameters():
        if sub in n and p.grad is not None:
            tot += p.grad.norm().item()**2; cnt += 1
    return (tot**0.5, cnt)
for name,key in [("qkv",".attn.qkv"),("proj",".attn.proj"),("w1",".attn.w1"),("w2",".attn.w2"),("w3",".attn.w3")]:
    gn,c = gnorm(key); print("(3) grad-norm %-4s (%2d tensors): %.4e" % (name, c, gn))
print("DONE")
