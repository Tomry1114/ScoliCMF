"""Verify #1+#2: (a) TTT now has qn/kn params; (b) vanilla & ttt share forward(x,gh,gw);
(c) JVP still finite (no regression); (d) meanflow.loss backward works for both."""
import sys, torch
sys.path.insert(0, "cmf")
torch.backends.cudnn.enabled = False
from models.dit import MFDiT, Attention
from ttt_block import TTT
from meanflow import MeanFlow

dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W, B = 480, 240, 2

# (a) TTT qk-norm params present
t = TTT(384, 6, inner_lr=0.25)
has = hasattr(t, "qn") and hasattr(t, "kn")
print("(a) TTT has qn/kn:", has, "| qk_norm flag =", getattr(t, "qk_norm", None))

# (b) unified interface: Attention now accepts (x, gh, gw)
a = Attention(384, 6)
x = torch.randn(B, 60*30, 384)
try:
    o = a(x, 60, 30); print("(b) Attention(x,gh,gw) OK  out", tuple(o.shape))
except TypeError as ex:
    print("(b) FAIL:", ex)

def build(attn):
    torch.manual_seed(0)
    m = MFDiT(img_size=(H,W), patch_size=8, data_channels=1, cond_channels=1,
              dim=384, depth=12, num_heads=6, text=False, attn_type=attn, inner_lr=0.25).to(dev)
    torch.manual_seed(1)
    for p in m.parameters():
        if float(p.abs().sum()) == 0.0: p.data.normal_(0, 0.05)
    return m

g = torch.Generator(device="cpu").manual_seed(20260707)
y = torch.randn(B,1,H,W,generator=g).to(dev); e = torch.randn(B,1,H,W,generator=g).to(dev)
cond = torch.randn(B,1,H,W,generator=g).to(dev)
tt = torch.rand(B,generator=g).to(dev).clamp(0.05,0.95); r = tt*torch.rand(B,generator=g).to(dev)
t_ = tt.view(B,1,1,1); v = e - y; z = (1-t_)*y + t_*e

# (c) JVP finite for both
for attn in ("vanilla","ttt"):
    m = build(attn); m.eval()
    def f(zz,ti,ri): return m(zz,ti,ri,cond)
    u,dudt = torch.autograd.functional.jvp(f,(z,tt,r),(v,torch.ones_like(tt),torch.zeros_like(r)),create_graph=True)
    print("(c) %-7s JVP finite=%s  max|dudt|=%.3e  NaN=%d" %
          (attn, bool(torch.isfinite(dudt).all()), dudt[torch.isfinite(dudt)].abs().max().item(),
           int(torch.isnan(dudt).sum())))

# (d) real meanflow loss backward for both
mf = MeanFlow(channels=1, image_size=(H,W), jvp_api='autograd')
for attn in ("vanilla","ttt"):
    m = build(attn); m.train()
    loss,_ = mf.loss(m, y, cond); loss.backward()
    gsum = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print("(d) %-7s loss=%.4f  grad-sum finite=%s" % (attn, loss.item(), str(gsum==gsum and gsum<float('inf'))))
print("DONE")
