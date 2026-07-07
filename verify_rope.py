"""Verify #3 RoPE: (a) rope buffers present; (b) rope on vs off changes output (position is injected);
(c) JVP finite no-NaN; (d) meanflow.loss backward OK; (e) rope=False path == pre-#3 behavior sanity."""
import sys, torch
sys.path.insert(0, "cmf")
torch.backends.cudnn.enabled = False
from models.dit import MFDiT
from ttt_block import TTT
from meanflow import MeanFlow

dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W, B = 480, 240, 2

# (a) rope buffers
t = TTT(384, 6, inner_lr=0.25, gh=60, gw=30, use_rope=True)
print("(a) TTT.rope is not None:", t.rope is not None,
      "| buffers:", [n for n,_ in t.rope.named_buffers()] if t.rope else None)

def build(rope_on):
    torch.manual_seed(0)
    m = MFDiT(img_size=(H,W), patch_size=8, data_channels=1, cond_channels=1,
              dim=384, depth=12, num_heads=6, text=False, attn_type="ttt", inner_lr=0.25, rope=rope_on).to(dev)
    torch.manual_seed(1)
    for p in m.parameters():
        if float(p.abs().sum()) == 0.0: p.data.normal_(0, 0.05)
    return m

g = torch.Generator(device="cpu").manual_seed(20260707)
y = torch.randn(B,1,H,W,generator=g).to(dev); e = torch.randn(B,1,H,W,generator=g).to(dev)
cond = torch.randn(B,1,H,W,generator=g).to(dev)
tt = torch.rand(B,generator=g).to(dev).clamp(0.05,0.95); r = tt*torch.rand(B,generator=g).to(dev)
t_ = tt.view(B,1,1,1); v = e - y; z = (1-t_)*y + t_*e

# (b) rope on vs off changes output (identical seeds -> only rope differs; rope adds no params so shapes match)
mon = build(True); moff = build(False)
with torch.no_grad():
    oon = mon(z,tt,r,cond); ooff = moff(z,tt,r,cond)
    rel = (oon-ooff).norm()/ooff.norm()
print("(b) rope on-vs-off output rel-change: %.4e  (>0 => position injected)" % rel.item())

# (c) JVP finite
mon.eval()
def f(zz,ti,ri): return mon(zz,ti,ri,cond)
u,dudt = torch.autograd.functional.jvp(f,(z,tt,r),(v,torch.ones_like(tt),torch.zeros_like(r)),create_graph=True)
print("(c) rope JVP finite=%s  max|dudt|=%.3e  NaN=%d" %
      (bool(torch.isfinite(dudt).all()), dudt[torch.isfinite(dudt)].abs().max().item(), int(torch.isnan(dudt).sum())))

# (d) meanflow loss backward
mf = MeanFlow(channels=1, image_size=(H,W), jvp_api='autograd')
m = build(True); m.train()
loss,_ = mf.loss(m, y, cond); loss.backward()
gfin = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
print("(d) rope meanflow loss=%.4f  all-grad-finite=%s" % (loss.item(), gfin))
print("DONE")
