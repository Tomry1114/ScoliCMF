#!/usr/bin/env python3
# Stage 0 data-contract audit (numpy+PIL only). Decides case 1/2/3.
import os, glob, numpy as np
from PIL import Image

ROOT = os.path.expanduser('~/ScoliCMF/data/Spine生成_Miccai数据集/train')
D = {k: os.path.join(ROOT,k) for k in ['preop_standardized','postop_standardized','PreOPCurve','PostOPCurve']}

def stems(d):
    return {os.path.splitext(os.path.basename(f))[0]: f
            for f in glob.glob(os.path.join(d,'*'))}

S = {k: stems(v) for k,v in D.items()}
# ---- data contract: stem correspondence ----
keys = [set(S[k]) for k in S]
common = set.intersection(*keys)
allk   = set.union(*keys)
print("== STEM CONTRACT ==")
for k in S: print(f"  {k:22s} n={len(S[k])}")
print(f"  common stems = {len(common)} ; union = {len(allk)} ; mismatched = {len(allk-common)}")
exts = {}
for k in S:
    for f in S[k].values():
        e=os.path.splitext(f)[1].lower(); exts[e]=exts.get(e,0)+1
print("  extensions:", exts)

def load_bin(path):
    im = Image.open(path).convert('L'); a = np.asarray(im)
    b = a > 127
    # foreground = minority class (thin centerline)
    if b.sum() > (~b).sum(): b = ~b
    return b, a.shape  # (H,W)

def centerline(b):
    H,W = b.shape
    ys=[]; xs=[]
    for y in range(H):
        idx = np.where(b[y])[0]
        if idx.size:
            ys.append(y); xs.append(float(np.median(idx)))
    return np.array(ys,float), np.array(xs,float), (H,W)

def arclen(ys,xs):
    if len(ys)<2: return 0.0
    d=np.sqrt(np.diff(xs)**2+np.diff(ys)**2); return float(d.sum())

def border_flags(b):
    H,W=b.shape
    return dict(top=bool(b[0].any()), bot=bool(b[H-1].any()),
               left=bool(b[:,0].any()), right=bool(b[:,W-1].any()))

def polyfit_x_of_y(ys,xs,H,deg=4):
    t=(ys-ys.min())/max(1.0,(ys.max()-ys.min()))  # normalized along covered span
    c=np.polyfit(t,xs/H,deg)  # x normalized by H for scale-free
    return c,t

def curvature_max(c):
    # x(t) poly; kappa ~ |x''| / (1+x'^2)^1.5 sampled
    t=np.linspace(0,1,50); p=np.poly1d(c)
    d1=np.polyder(p,1)(t); d2=np.polyder(p,2)(t)
    k=np.abs(d2)/np.power(1+d1*d1,1.5); return float(k.max())

rows=[]
case2_flags=0
for st in sorted(common):
    bpre,_=load_bin(S['PreOPCurve'][st]); 
    bpost,_=load_bin(S['PostOPCurve'][st])
    yp,xp,(H,W)=centerline(bpre); yq,xq,_=centerline(bpost)
    if len(yp)<5 or len(yq)<5: 
        rows.append(dict(stem=st,bad=1)); continue
    pa=dict(ymin=yp.min()/H, ymax=yp.max()/H, arc=arclen(yp,xp)/H,
            x_top=xp[0]/W, x_bot=xp[-1]/W)
    qa=dict(ymin=yq.min()/H, ymax=yq.max()/H, arc=arclen(yq,xq)/H,
            x_top=xq[0]/W, x_bot=xq[-1]/W)
    # vertical coverage IoU (in row fraction)
    lo=max(pa['ymin'],qa['ymin']); hi=min(pa['ymax'],qa['ymax'])
    inter=max(0.0,hi-lo); uni=max(pa['ymax'],qa['ymax'])-min(pa['ymin'],qa['ymin'])
    iou=inter/uni if uni>0 else 0.0
    bf_p=border_flags(bpre); bf_q=border_flags(bpost)
    cut = bf_p['top'] or bf_p['bot'] or bf_q['top'] or bf_q['bot']
    drift_top=abs(pa['ymin']-qa['ymin']); drift_bot=abs(pa['ymax']-qa['ymax'])
    arc_ratio=qa['arc']/pa['arc'] if pa['arc']>0 else 0
    # poly interp sanity at s=0.5 (coeff interpolation)
    cp,_=polyfit_x_of_y(yp,xp,H); cq,_=polyfit_x_of_y(yq,xq,H)
    cmid=0.5*cp+0.5*cq
    kmax_mid=curvature_max(cmid); kmax_pre=curvature_max(cp); kmax_post=curvature_max(cq)
    rows.append(dict(stem=st,bad=0,H=H,W=W,covIoU=iou,cut=int(cut),
                     drift_top=drift_top,drift_bot=drift_bot,arc_ratio=arc_ratio,
                     kmax_mid=kmax_mid,kmax_env=max(kmax_pre,kmax_post),
                     p_top=bf_p['top'],p_bot=bf_p['bot'],q_top=bf_q['top'],q_bot=bf_q['bot']))

good=[r for r in rows if r.get('bad')==0]
def col(n): return np.array([r[n] for r in good],float)
def pc(a): return {p:round(float(np.percentile(a,p)),3) for p in (5,25,50,75,95)}

print(f"\n== PAIR AUDIT (good={len(good)}/{len(rows)}) ==")
print("vertical coverage IoU   :", pc(col('covIoU')))
print("endpoint drift TOP (|dymin|/H):", pc(col('drift_top')))
print("endpoint drift BOT (|dymax|/H):", pc(col('drift_bot')))
print("arc-length ratio post/pre     :", pc(col('arc_ratio')))
print("curvature max (interp s=0.5)  :", pc(col('kmax_mid')))
print("curvature max (pre/post env)  :", pc(col('kmax_env')))
cut_rate=col('cut').mean()
print(f"border-cut rate (curve touches top/bot): {cut_rate:.3f}")
for side in ['p_top','p_bot','q_top','q_bot']:
    print(f"  touches {side}: {np.array([r[side] for r in good]).mean():.3f}")

iou=col('covIoU'); 
frac_low_iou=(iou<0.85).mean()
print(f"\nfrac pairs covIoU<0.85: {frac_low_iou:.3f}")
print(f"frac pairs covIoU<0.70: {(iou<0.70).mean():.3f}")

# ---- case verdict heuristic ----
med_iou=np.median(iou); p5_iou=np.percentile(iou,5)
verdict = ("CASE 1 (stable endpoints; arc-length resample OK)" if (med_iou>0.9 and p5_iou>0.8 and cut_rate<0.1)
           else "CASE 2 (partial coverage; need validity masks / common-interval only)" if frac_low_iou>0.15
           else "CASE 1-ish but watch tails (median good, some low-coverage pairs)")
print("\n== CASE VERDICT ==\n ", verdict)
print(" NOTE: no per-vertebra labels exist -> correspondence by arc-length, not vertebra level (case-3 caveat applies to labels regardless).")

# raw image ghosting baseline (NCC of canonical-naive blend) on 30 pairs
print("\n== IMAGE-SPACE GHOSTING BASELINE (raw, no canonicalization) ==")
def ncc(a,b):
    a=a.astype(float)-a.mean(); b=b.astype(float)-b.mean()
    d=np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/d) if d>0 else 0.0
nccs=[]
for st in sorted(common)[:30]:
    ap=np.asarray(Image.open(S['preop_standardized'][st]).convert('L'),float)
    aq=np.asarray(Image.open(S['postop_standardized'][st]).convert('L'),float)
    if ap.shape==aq.shape: nccs.append(ncc(ap,aq))
print(f" raw preop/postop NCC (n={len(nccs)}): median={np.median(nccs):.3f} p25={np.percentile(nccs,25):.3f} p75={np.percentile(nccs,75):.3f}")
print(" -> low NCC confirms pixel-space pairs are unregistered; motivates canonicalization + latent bridge (true latent P3 check runs post-AE).")

# write CSV
import csv
with open(os.path.expanduser('~/ScoliCMF/doc/stage0_audit.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(good[0].keys())); w.writeheader()
    for r in good: w.writerow(r)
print("\nwrote doc/stage0_audit.csv")
