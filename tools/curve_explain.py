import os, numpy as np
from PIL import Image
ROOT=os.path.expanduser('~/ScoliCMF/data/Spine生成_Miccai数据集/train')
def P(sub,st): 
    for e in('.png','.jpg'):
        p=os.path.join(ROOT,sub,st+e)
        if os.path.exists(p): return p
def loadL(p): return Image.open(p).convert('L')
def curve_mask(p):
    a=np.asarray(loadL(p)); b=a>127
    if b.sum()>(~b).sum(): b=~b
    # thicken for visibility
    m=b.copy()
    for dy in(-1,0,1):
        for dx in(-1,0,1):
            m|=np.roll(np.roll(b,dy,0),dx,1)
    return m
def overlay(img_p,cur_p):
    im=loadL(img_p).convert('RGB'); a=np.asarray(im).copy()
    m=curve_mask(cur_p)
    a[m]=[255,0,0]
    return Image.fromarray(a)
stems=['001','050','200','400']
cols=[]
for st in stems:
    pre=loadL(P('preop_standardized',st)).convert('RGB')
    preo=overlay(P('preop_standardized',st),P('PreOPCurve',st))
    post=loadL(P('postop_standardized',st)).convert('RGB')
    posto=overlay(P('postop_standardized',st),P('PostOPCurve',st))
    row=Image.new('RGB',(240*4+30,480),(20,20,20))
    for i,im in enumerate([pre,preo,post,posto]):
        row.paste(im,(i*(240+10),0))
    cols.append(np.asarray(row))
big=Image.fromarray(np.concatenate(cols,axis=0))
out=os.path.expanduser('~/ScoliCMF/data/_inspect/curve_explain.png')
big.save(out); print("saved",out,big.size)
print("layout per row: [preop | preop+curve(red) | postop | postop+curve(red)]; rows=stems",stems)
