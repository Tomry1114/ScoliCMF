"""Build Palette ColorizationDataset structure for preop->postop finetune.
gray/{idx}.png = PRE-op (cond), color/{idx}.png = POST-op (gt); flists + idx2stem map.
256x256 (match CelebA checkpoint). ColorizationDataset + mask=None => pure conditional translation."""
import os, glob, json
from PIL import Image
HOME=os.path.expanduser("~/ScoliCMF"); ROOT=os.path.join(HOME,"data/Spine生成_Miccai数据集/train")
OUT=os.path.join(HOME,"compare_exp/01_palette/repo/datasets/scoli"); SZ=256
for sub in ["gray","color"]: os.makedirs(os.path.join(OUT,sub),exist_ok=True)
pre={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(ROOT,"preop_standardized","*"))}
post={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(ROOT,"postop_standardized","*"))}
def rs(p): return Image.open(p).convert("RGB").resize((SZ,SZ))   # 3ch, 256x256
idx2stem={}; flists={}
for split in ["train","val"]:
    stems=[l.strip() for l in open(os.path.join(HOME,"splits",split+".txt")) if l.strip() and l.strip() in pre and l.strip() in post]
    ids=[]
    base=0 if split=="train" else 100000   # disjoint index ranges
    for i,s in enumerate(stems):
        idx=base+i; fn=str(idx).zfill(5)+".png"
        rs(pre[s]).save(os.path.join(OUT,"gray",fn)); rs(post[s]).save(os.path.join(OUT,"color",fn))
        idx2stem[idx]=s; ids.append(idx)
    flists[split]=ids
    with open(os.path.join(OUT,split+".flist"),"w") as f: f.write("\n".join(str(i) for i in ids))
json.dump(idx2stem, open(os.path.join(OUT,"idx2stem.json"),"w"))
print("built palette data:", {k:len(v) for k,v in flists.items()}, "-> datasets/scoli/{gray,color}")
