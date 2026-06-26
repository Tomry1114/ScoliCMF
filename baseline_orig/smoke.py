import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
sys.path.insert(0, os.path.dirname(os.path.expanduser("~/ScoliCMF/baseline_orig/x")))
from baseline_orig import MFDiT_orig, MeanFlowOrig
from utils import count_parameters
dev="cuda"
H,W=480,240
m=MFDiT_orig(img_size=(H,W),patch_size=8,dim=384,depth=12,num_heads=6).to(dev)
print(f"params={count_parameters(m)/1e6:.2f}M")
mf=MeanFlowOrig(channels=1,flow_ratio=0.75)
xpre=torch.rand(2,1,H,W,device=dev); xpost=torch.rand(2,1,H,W,device=dev)
loss,mse=mf.loss(m,xpost,xpre); print("loss ok:",float(loss),"mse",float(mse))
loss.backward(); print("backward ok")
with torch.no_grad():
    z=mf.sample_given_cond(m,xpre,sample_steps=4)
print("sample ok shape",tuple(z.shape),"range",[round(float(z.min()),3),round(float(z.max()),3)])
print("SMOKE_OK")
