import os,sys,torch,numpy as np
sys.path.insert(0,os.path.expanduser("~/ScoliCMF"))
from torch.utils.data import DataLoader
from dataset_sa import PairedSpineDataset
from sc_pga import SCPGA, build_v2_projector, build_static_projector, path_laplacian, _topk_eigvecs
dev="cuda"; H,W=480,240
ds=PairedSpineDataset(root=os.path.expanduser("~/ScoliCMF/data/Spine生成_Miccai数据集"),size=(H,W),split_file=os.path.expanduser("~/ScoliCMF/splits/val.txt"))
ld=DataLoader(ds,batch_size=8,num_workers=2)
m=SCPGA(img_size=(H,W),dim=256,patch_size=8,proj="v2").to(dev).eval()
xp,_=next(iter(ld)); xp=xp.to(dev)
# reproduce token stats path from forward
with torch.no_grad():
    Fm=m.stem(xp); Ff=Fm.flatten(2).transpose(1,2)
    _,Dd,Hf,Wf=Fm.shape
    ygr=torch.linspace(0,1,Hf,device=dev).view(Hf,1).expand(Hf,Wf).reshape(-1)
    xgr=torch.linspace(0,1,Wf,device=dev).view(1,Wf).expand(Hf,Wf).reshape(-1)
    xc=m._xc_cubic(xp,ygr)
    qn=torch.nn.functional.normalize(m.q,dim=-1); fn=torch.nn.functional.normalize(m.Wf(Ff),dim=-1)
    content=torch.einsum("jd,bnd->bjn",qn,fn)
    spatial=(-m.beta*(ygr[None,None,:]-m.mu[None,:,None])**2 - m.eta*(xgr[None,None,:]-xc[:,None,:])**2)
    pi=torch.softmax(content+spatial,dim=-1)
    grid=torch.stack([ygr,xgr],-1)
    pos=torch.einsum("bjn,nc->bjc",pi,grid); var=(torch.einsum("bjn,nc->bjc",pi,grid**2)-pos**2).clamp_min(0)
    res=torch.stack([pos[...,0]-m.mu[None,:],pos[...,1]-0.5],-1)
    # weights as in build_v2_projector
    B,J,_=res.shape
    ws=[]
    for b in range(B):
        dm=((res[b,1:]-res[b,:-1])**2).sum(-1); db=((var[b,1:].sqrt()-var[b,:-1].sqrt())**2).sum(-1)
        w=m.w_min+(1-m.w_min)*torch.exp(-(dm+0.5*db)/m.tau); ws.append(w)
    ws=torch.stack(ws)
    Pi_v2=build_v2_projector(res,var.sqrt(),m.Kg,m.tau,m.w_min)
    Uv1=_topk_eigvecs(path_laplacian(J,device=dev),m.Kg,low=True); Pi_v1=(Uv1@Uv1.T).unsqueeze(0).expand(B,-1,-1)
    rel=(torch.linalg.norm((Pi_v2-Pi_v1).reshape(B,-1),dim=1)/torch.linalg.norm(Pi_v1.reshape(B,-1),dim=1)).mean()
print("edge w: mean=%.4f std=%.4f min=%.4f max=%.4f"%(ws.mean(),ws.std(),ws.min(),ws.max()))
print("||Pi_v2 - Pi_v1||_F / ||Pi_v1||_F = %.4f  (near 0 => v2 ~ fixed path graph)"%float(rel))
print("V2DIAG_DONE")
