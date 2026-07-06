"""I2SB paired dataset: returns (clean=POSTOP, corrupt=PREOP, label=0) 3-tuples.
With --corrupt mixture, Runner.sample_batch takes the 3-tuple path -> uses preop directly as x1
(NO corruption operator). With --cond-x1, cond=preop. 256x256, [-1,1], 3ch (match ADM init)."""
import os, glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
HOME=os.path.expanduser("~/ScoliCMF"); ROOT=os.path.join(HOME,"data/Spine生成_Miccai数据集/train")
class PairedScoli(Dataset):
    def __init__(self, split, image_size=256):
        keep=set(l.strip() for l in open(os.path.join(HOME,"splits",split+".txt")) if l.strip())
        self.pre={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(ROOT,"preop_standardized","*"))}
        self.post={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(ROOT,"postop_standardized","*"))}
        self.stems=sorted(set(self.pre)&set(self.post)&keep)
        self.tf=T.Compose([T.Resize((image_size,image_size)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    def __len__(self): return len(self.stems)
    def __getitem__(self, i):
        s=self.stems[i]
        x0=self.tf(Image.open(self.post[s]).convert("RGB"))   # clean = postop (target)
        x1=self.tf(Image.open(self.pre[s]).convert("RGB"))    # corrupt = preop (prior/cond)
        return x0, x1, 0
    def stem(self, i): return self.stems[i]
