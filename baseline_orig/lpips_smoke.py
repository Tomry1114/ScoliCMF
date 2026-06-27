import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/ScoliCMF"))
from metrics_img import ssim, psnr, lpips_fn
x=torch.rand(3,1,480,240,device="cuda"); y=x.clone(); y2=torch.rand(3,1,480,240,device="cuda")
print("SSIM(x,x)=",[round(float(v),3) for v in ssim(x,y)])
print("PSNR(x,x)=",[round(float(v),1) for v in psnr(x,y)])
print("LPIPS(x,x)=",[round(float(v),4) for v in lpips_fn(x,y)],"(should ~0)")
print("LPIPS(x,rand)=",[round(float(v),4) for v in lpips_fn(x,y2)],"(should >0)")
print("LPIPS_SMOKE_OK")
