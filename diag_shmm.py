import os,sys,torch
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from utils import load_config
from dataset_sa import PairedSpineDataset
from eval_gates import load_ckpt
from sc_pga import path_laplacian, _topk_eigvecs
dev="cuda"; H,W=480,240
cfg=load_config(os.path.expanduser("~/ScoliCMF/configs/shmm_v2.yaml"))
m=load_ckpt(os.path.expanduser("~/ScoliCMF/runs/shmm_v2/ckpts/step_5000.pt"),cfg,H,W,None,dev,use_ema=True)
core=getattr(m,"module",m)
cond=core.cond; cond.diag=True
ds=PairedSpineDataset(root=os.path.expanduser("~/ScoliCMF/data/Spine生成_Miccai数据集"),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
xp,xq=next(iter(DataLoader(ds,batch_size=8))); xp=xp.to(dev)
B=xp.shape[0]; r=torch.full((B,),0.25,device=dev); t=torch.full((B,),0.5,device=dev)
z=xp.clone()
with torch.no_grad():
    u_normal=core(z,r,t,xp)
    d=cond._diag; raw=d["raw_dyn"]; Pi=d["Pi"]; pi=d["pi"]; m_dyn=d["m_dyn"]; m_static=d["m_static"]
    # R_removed (projection strips how much of dynamic feature)
    R_removed=float((raw-cond._proj_apply(Pi,raw)).norm()/raw.norm().clamp_min(1e-6))
    # R_action v1 vs v2: does swapping projector change the dynamic feature?
    J=cond.J; Uv1=_topk_eigvecs(path_laplacian(J,device=dev),cond.Kg,low=True); Pi_v1=(Uv1@Uv1.T)
    R_action=float((cond._proj_apply(Pi,raw)-cond._proj_apply(Pi_v1,raw)).norm()/raw.norm().clamp_min(1e-6))
    R_dyn=float(m_dyn.norm()/m_static.norm().clamp_min(1e-6))
    # interventions: output change
    cond.dyn_off=True; u_dynoff=core(z,r,t,xp); cond.dyn_off=False
    R_int_dyn=float((u_normal-u_dynoff).norm()/u_normal.norm().clamp_min(1e-6))
    g=torch.Generator(device="cpu").manual_seed(0); cond.perm=torch.randperm(J,generator=g)
    u_perm=core(z,r,t,xp); cond.perm=None
    R_int_perm=float((u_normal-u_perm).norm()/u_normal.norm().clamp_min(1e-6))
print("R_removed   = %.4f   (proj strips this frac of dynamic feat; ~0 => feat already low-freq, projector inert)"%R_removed)
print("R_action    = %.4f   (||(Pi_v2-Pi_v1) X||/||X||; ~0 => swapping patient<->path graph does nothing to the feat)"%R_action)
print("R_dyn       = %.4f   (||m_dyn||/||m_static||; small => static branch dominates, can bypass SHMM)"%R_dyn)
print("R_int_dynoff= %.4f   (output change when m_dyn->0; ~0 => dynamic branch barely controls output)"%R_int_dyn)
print("R_int_perm  = %.4f   (output change when chain order permuted; ~0 => spine topology irrelevant to output)"%R_int_perm)
print("SHMM_DIAG_DONE")
