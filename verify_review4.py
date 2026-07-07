import sys, os, inspect, torch
sys.path.insert(0, "cmf")
torch.backends.cudnn.enabled = False
from models.dit_drica import MFDiTDRICA
from meanflow import MeanFlow
print("[import] MFDiTDRICA <-", inspect.getfile(MFDiTDRICA))
print("[import] MeanFlow   <-", inspect.getfile(MeanFlow))
dev = "cuda" if torch.cuda.is_available() else "cpu"
H, W, B = 480, 240, 2

# (1) CPE zero-init when cpe=True
m = MFDiTDRICA(img_size=(H,W), dim=384, depth=12, num_heads=6, cpe=True, drica_layer_ids=(2,6,10)).to(dev)
cpe_blocks = [b for b in m.blocks if getattr(b,"cpe_on",False)]
allzero = all(float(b.cpe.weight.abs().sum())==0 and float(b.cpe.bias.abs().sum())==0 for b in cpe_blocks)
print("(1) cpe=True: %d cpe blocks, all zero-init=%s" % (len(cpe_blocks), allzero))

# (4) MeanFlow (H,W) + sample_given_cond model_kwargs
mf = MeanFlow(channels=1, image_size=(H,W))
print("(4a) MeanFlow height/width = %d/%d (expect 480/240)" % (mf.height, mf.width))
m2 = MFDiTDRICA(img_size=(H,W), dim=384, depth=12, num_heads=6, drica_layer_ids=(2,6,10)).to(dev)
D = {"region":torch.ones(B,3,device=dev)/3,"direction":torch.ones(B,2,device=dev)/2,"joint":torch.ones(B,6,device=dev)/6}
cond = torch.randn(B,1,H,W,device=dev)
try:
    out = mf.sample_given_cond(m2, cond, sample_steps=2, model_kwargs={"diagnosis":D}, device=dev, show_progress=False)
    print("(4b) sample_given_cond(model_kwargs) OK, out=%s" % (tuple(out.shape),))
except Exception as ex:
    print("(4b) FAIL:", type(ex).__name__, ex)
print("DONE")
