"""Imaging Agent: label preop spine X-rays with dominant-curve (location, direction) via micuapi VLM.
No geometry. Reliability = self-consistency (K samples of primary model, majority vote) + optional
cross-model agreement. Resumable (incremental JSON). Usage:
  python agent.py --n 24 --cross    # pilot: 24 cases, gpt-5.5 x K + cross-model agreement
  python agent.py                    # full run over all preop images (primary model only)
"""
import os, sys, glob, base64, json, re, argparse, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

KEY=os.environ.get("MICUAPI_KEY","")
BASE="https://www.micuapi.ai/v1"
PRIMARY="gpt-5.5"; CROSS=["gpt-5.4","gpt-5.4-mini"]
HOME=os.path.expanduser("~/ScoliCMF")
PREOP=os.path.join(HOME,"data/Spine生成_Miccai数据集/train/preop_standardized")
PROMPT="""You are analyzing a preoperative frontal whole-spine radiograph.
Identify only the dominant spinal curvature.
Return JSON only with exactly two fields:
1. dominant_location: thoracic | thoracolumbar | lumbar | uncertain
2. dominant_direction: image_left | image_right | uncertain
The direction refers to the convexity of the dominant curve in IMAGE COORDINATES, not patient anatomical left or right.
Do not infer a surgical plan. Do not describe postoperative outcomes. Do not add explanations."""
LOC={"thoracic","thoracolumbar","lumbar","uncertain"}
DIR={"image_left","image_right","uncertain"}
cli=OpenAI(api_key=KEY, base_url=BASE, timeout=60, max_retries=2)

def b64_of(path):
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode()
def parse(txt):
    m=re.search(r"\{.*\}", txt, re.S)
    if not m: return None
    try: d=json.loads(m.group(0))
    except Exception: return None
    loc=str(d.get("dominant_location","")).strip().lower()
    dr =str(d.get("dominant_direction","")).strip().lower()
    if loc not in LOC: loc="uncertain"
    if dr  not in DIR: dr ="uncertain"
    return loc,dr
def ask(model, ext, b64, temp):
    r=cli.chat.completions.create(model=model, temperature=temp, max_tokens=120,
        messages=[{"role":"user","content":[{"type":"text","text":PROMPT},
        {"type":"image_url","image_url":{"url":f"data:image/{ext};base64,{b64}"}}]}])
    return parse(r.choices[0].message.content or "")
def majority(votes):  # list of (loc,dir) -> (loc,dir, agree_loc, agree_dir)
    votes=[v for v in votes if v]
    if not votes: return "uncertain","uncertain",0.0,0.0
    lc=Counter(v[0] for v in votes); dc=Counter(v[1] for v in votes)
    loc,ln=lc.most_common(1)[0]; dr,dn=dc.most_common(1)[0]
    n=len(votes); return loc,dr,ln/n,dn/n

def label_one(stem, path, K, cross):
    ext=os.path.splitext(path)[1].lstrip(".").lower() or "png"; b=b64_of(path)
    votes=[]
    for k in range(K):
        try: votes.append(ask(PRIMARY, ext, b, 0.0 if K==1 else 0.5))
        except Exception as e: votes.append(None)
    loc,dr,al,ad=majority(votes)
    rec={"stem":stem,"dominant_location":loc,"dominant_direction":dr,
         "agree_loc":round(al,3),"agree_dir":round(ad,3),
         "votes":[v if v else ["ERR","ERR"] for v in votes]}
    if cross:
        cm={}
        for m in CROSS:
            try: cm[m]=ask(m, ext, b, 0.0)
            except Exception: cm[m]=None
        rec["cross"]={m:(v if v else ["ERR","ERR"]) for m,v in cm.items()}
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=0)
    ap.add_argument("--k",type=int,default=3); ap.add_argument("--cross",action="store_true")
    ap.add_argument("--workers",type=int,default=6); ap.add_argument("--out",default="labels.json")
    a=ap.parse_args()
    imgs=sorted(glob.glob(os.path.join(PREOP,"*")))
    stems=[(os.path.splitext(os.path.basename(p))[0],p) for p in imgs]
    if a.n: stems=stems[:a.n]
    outp=os.path.join(HOME,a.out)
    done={}
    if os.path.exists(outp):
        done={r["stem"]:r for r in (json.loads(l) for l in open(outp) if l.strip())}
    todo=[(s,p) for s,p in stems if s not in done]
    print(f"total={len(stems)} done={len(done)} todo={len(todo)} K={a.k} cross={a.cross}",flush=True)
    fh=open(outp,"a")
    recs=list(done.values())
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(label_one,s,p,a.k,a.cross):s for s,p in todo}
        for i,fu in enumerate(as_completed(futs)):
            r=fu.result(); recs.append(r); fh.write(json.dumps(r,ensure_ascii=False)+"\n"); fh.flush()
            if (i+1)%20==0: print(f"  {i+1}/{len(todo)}",flush=True)
    fh.close()
    # reliability summary
    print("=== RELIABILITY ===",flush=True)
    al=[r["agree_loc"] for r in recs]; ad=[r["agree_dir"] for r in recs]
    unc_l=sum(r["dominant_location"]=="uncertain" for r in recs); unc_d=sum(r["dominant_direction"]=="uncertain" for r in recs)
    print(f"  self-consistency: mean agree_loc={sum(al)/len(al):.3f} agree_dir={sum(ad)/len(ad):.3f}",flush=True)
    print(f"  full-agreement(3/3): loc={sum(x>=0.999 for x in al)}/{len(al)}  dir={sum(x>=0.999 for x in ad)}/{len(ad)}",flush=True)
    print(f"  labeled 'uncertain': loc={unc_l} dir={unc_d}",flush=True)
    from collections import Counter as C
    print("  loc dist:", dict(C(r["dominant_location"] for r in recs)),flush=True)
    print("  dir dist:", dict(C(r["dominant_direction"] for r in recs)),flush=True)
    if any("cross" in r for r in recs):
        cl=cd=n=0
        for r in recs:
            if "cross" not in r: continue
            n+=1
            allv=[[r["dominant_location"],r["dominant_direction"]]]+list(r["cross"].values())
            cl+=len(set(v[0] for v in allv))==1; cd+=len(set(v[1] for v in allv))==1
        print(f"  cross-model FULL agreement (primary+{len(CROSS)}): loc={cl}/{n} dir={cd}/{n}",flush=True)
    print("SUMMARY_DONE",flush=True)
if __name__=="__main__": main()
