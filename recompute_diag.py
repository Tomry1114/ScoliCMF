"""Recompute correctly-normalized acquisition/center diagnostics from saved cleaned val targets."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
HOME=os.path.expanduser("~/ScoliCMF"); H,W=480,240
cfg=load_config(os.path.join(HOME,"configs/s2_base.yaml"))
ds=PairedSpineDataset(root=os.path.join(HOME,cfg["data"]["root"]),size=(H,W),split_file=os.path.join(HOME,"splits/val.txt"))
XP=[];XQ=[]
for a,b in DataLoader(ds,batch_size=64,shuffle=False): XP.append(a);XQ.append(b)
XP=torch.cat(XP); XQ=torch.cat(XQ)
xcol=torch.linspace(0,1,W); cen=(xcol-0.5).abs()<0.15
cen_m=cen.view(1,1,1,W).float(); per_m=(~cen).view(1,1,1,W).float()
def mean_masked(d,m): return (d.abs()*m).sum()/(m.sum()*d.shape[0]*H)
cen_true=mean_masked(XP-XQ,cen_m).item()
print("== Exp3 center-protection table (correctly normalized, val) ==")
print("  %-12s %-12s %-14s %-10s"%("variant","periph_L1","central_kept","PRESERVED"))
for tag,label in [("clean","gauss(current)"),("cen_none","none"),("cen_strong","strong"),("geo","geo-only"),("photo","photo-only")]:
    fp=os.path.join(HOME,"runs/adoc",("clean_val.pt" if tag=="clean" else "clean_%s_val.pt"%tag))
    if not os.path.exists(fp): print("  %-14s MISSING"%label); continue
    cl=torch.load(fp)["clean"]
    pL=mean_masked(XP-cl,per_m).item(); ck=mean_masked(XP-cl,cen_m).item()
    print("  %-14s %-12.4f %-14.4f %-10.3f"%(label,pL,ck,ck/cen_true))
print("  central_change_true=%.4f"%cen_true)
print("RECOMPUTE_DONE")
