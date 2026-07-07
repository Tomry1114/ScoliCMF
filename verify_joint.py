import sys, torch, numpy as np
sys.path.insert(0, "cmf")
torch.backends.cudnn.enabled = False
from models.dit import MFDiT
dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W, B = 480, 240, 2

def mk(text_emb, inject):
    m = MFDiT(img_size=(H,W), patch_size=8, dim=384, depth=2, num_heads=6,
              text=True, attn_type="vanilla", text_emb=text_emb, inject=inject).to(dev)
    return m

# (a) profile presence per mode
for te in ("factorized","joint","both"):
    m = mk(te, "spatial")
    has = lambda n: hasattr(m, n)
    print("(a) text_emb=%-10s spatial: region_vprof=%s dir_hprof=%s joint_vprof=%s joint_hprof=%s"
          % (te, has("region_vprofile"), has("direction_hprofile"), has("joint_vprofile"), has("joint_hprofile")))

# (b) spatial_bias actually depends on joint_prob (joint mode)
m = mk("joint","spatial");
torch.nn.init.normal_(m.joint_vprofile, std=0.1); torch.nn.init.normal_(m.joint_hprofile, std=0.1)
m.region_prob = torch.zeros(B,3,device=dev); m.direction_prob = torch.zeros(B,2,device=dev)
jp1 = torch.tensor([[1.,0,0,0,0,0],[0,0,0,0,0,1.]],device=dev)
jp2 = torch.tensor([[0,0,0,0,0,1.],[1.,0,0,0,0,0]],device=dev)
m.joint_prob = jp1; b1 = m.spatial_bias(B)
m.joint_prob = jp2; b2 = m.spatial_bias(B)
print("(b) joint spatial_bias changes with joint_prob: rel-diff=%.3e (must be >0)" %
      ((b1-b2).norm()/b1.norm()).item())

# (c) forward runs for joint+spatial and both+spatial
for te, inj in (("joint","spatial"),("both","both"),("factorized","spatial")):
    m = mk(te, inj)
    m.region_prob=torch.rand(B,3,device=dev); m.region_prob/=m.region_prob.sum(1,keepdim=True)
    m.direction_prob=torch.rand(B,2,device=dev); m.direction_prob/=m.direction_prob.sum(1,keepdim=True)
    m.joint_prob=torch.rand(B,6,device=dev); m.joint_prob/=m.joint_prob.sum(1,keepdim=True)
    x=torch.randn(B,1,H,W,device=dev); t=torch.rand(B,device=dev); r=t*0.5; cond=torch.randn(B,1,H,W,device=dev)
    o=m(x,t,r,cond)
    print("(c) forward %-10s + %-8s OK  out=%s" % (te, inj, tuple(o.shape)))

# (d) derangement distance space matches text_emb
sys.argv=["x"]
import importlib, train_cmf as TC
print("(d) _der_feat dims: factorized=%d joint=%d both=%d" % (
    len(TC._der_feat(TC.TR[0],"factorized")), len(TC._der_feat(TC.TR[0],"joint")), len(TC._der_feat(TC.TR[0],"both"))))
df = TC.build_derangement(TC.TR[:20],"factorized"); dj = TC.build_derangement(TC.TR[:20],"joint")
ndiff = sum(1 for k in df if df[k]!=dj.get(k))
print("(d) factorized-vs-joint derangement differ on %d/20 stems (joint uses Q_JOINT distance)" % ndiff)
print("DONE")
