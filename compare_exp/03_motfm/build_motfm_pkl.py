"""Build MOTFM .pkl from our paired preop/postop data.
image = POST-op (target, 1x480x240 float32 in [0,1]); mask = PRE-op (ControlNet condition); metadata.stem for mapping back.
No class (mask_conditioning.yaml: with_conditioning=false)."""
import os, glob, pickle, numpy as np
from PIL import Image
HOME=os.path.expanduser("~/ScoliCMF"); ROOT=os.path.join(HOME,"data/Spine生成_Miccai数据集/train"); H,W=480,240
def load(p): return (np.asarray(Image.open(p).convert("L").resize((W,H)),dtype=np.float32)/255.0)[None]  # (1,H,W) in [0,1]
pre={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(ROOT,"preop_standardized","*"))}
post={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(ROOT,"postop_standardized","*"))}
def split(name):
    keep=[l.strip() for l in open(os.path.join(HOME,"splits",name)) if l.strip()]
    out=[]
    for s in keep:
        if s in pre and s in post:
            out.append({"image":load(post[s]), "mask":load(pre[s]), "class":0, "metadata":{"stem":s}})
    return out
data={"train":split("train.txt"), "valid":split("val.txt"), "test":split("test.txt")}
outp=os.path.join(HOME,"compare_exp/03_motfm/data"); os.makedirs(outp,exist_ok=True)
with open(os.path.join(outp,"scoli.pkl"),"wb") as f: pickle.dump(data,f)
print("built scoli.pkl:", {k:len(v) for k,v in data.items()}, "image shape", data["train"][0]["image"].shape, "range",
      float(data["train"][0]["image"].min()), float(data["train"][0]["image"].max()))
