"""Acceptance probe: does TTT's sqrt-norm grad-clip pollute the MeanFlow JVP (create_graph=True)?
Same (z,t,r,cond,v) fed through vanilla vs ttt attention; report dudt (the term that gets 2nd-order
differentiated in u_tgt = v - (t-r)*dudt) stats: max |dudt|, mean, NaN/Inf counts."""
import sys, torch
sys.path.insert(0, "cmf")
torch.backends.cudnn.enabled = False
from models.dit import MFDiT

dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 480, 240
B = 2

def build(attn):
    torch.manual_seed(0)                      # same init for shared submodules
    m = MFDiT(img_size=(H, W), patch_size=8, data_channels=1, cond_channels=1,
              dim=384, depth=12, num_heads=6, text=False, attn_type=attn, inner_lr=0.25).to(dev)
    # DiT zero-inits adaLN gates + final layer -> output==0 at init. Perturb ONLY the zero-inited
    # params to a small non-zero value so the net is in a realistic mid-training regime (fair: same for both).
    torch.manual_seed(1)
    n_perturbed = 0
    for p in m.parameters():
        if float(p.abs().sum()) == 0.0:
            p.data.normal_(0, 0.05); n_perturbed += 1
    m.eval()
    return m, n_perturbed

# one fixed batch
g = torch.Generator(device="cpu").manual_seed(20260707)
y   = torch.randn(B, 1, H, W, generator=g).to(dev)
e   = torch.randn(B, 1, H, W, generator=g).to(dev)
cond= torch.randn(B, 1, H, W, generator=g).to(dev)
t   = torch.rand(B, generator=g).to(dev).clamp(0.05, 0.95)
r   = (t * torch.rand(B, generator=g).to(dev))            # r < t
t_ = t.view(B,1,1,1); v = e - y
z  = (1 - t_) * y + t_ * e

def probe(attn):
    m, npz = build(attn)
    def f(z_in, t_in, r_in):
        return m(z_in, t_in, r_in, cond)
    inputs   = (z, t, r)
    tangents = (v, torch.ones_like(t), torch.zeros_like(r))
    u, dudt = torch.autograd.functional.jvp(f, inputs, tangents, create_graph=True)
    d = dudt.detach()
    nan = torch.isnan(d).sum().item(); inf = torch.isinf(d).sum().item()
    finite = d[torch.isfinite(d)]
    mx  = finite.abs().max().item() if finite.numel() else float("nan")
    mn  = finite.abs().mean().item() if finite.numel() else float("nan")
    p999= finite.abs().float().quantile(0.999).item() if finite.numel() else float("nan")
    umx = u.detach().abs().max().item()
    print(f"[{attn:7s}] perturbed_zero_params={npz}  dudt: max|.|={mx:.3e}  mean|.|={mn:.3e}  p99.9={p999:.3e}  NaN={nan}  Inf={inf}   (u max|.|={umx:.3e})")
    return d

print(f"device={dev}  grid={480//8}x{240//8}  B={B}")
dv = probe("vanilla")
dt = probe("ttt")
den = dv[torch.isfinite(dv)].abs().max().item()
if den > 0 and dt[torch.isfinite(dt)].numel():
    print("\nratio  ttt/vanilla  max|dudt| = %.1fx" %
          (dt[torch.isfinite(dt)].abs().max().item() / den))
print("DONE")
