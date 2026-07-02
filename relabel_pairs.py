"""Fix #6: joint-pair majority voting (loc & dir voted TOGETHER, not separately).
Re-derives labels from the K=7 votes already stored in labels.json -> labels_pair.json (NO API cost).
Also quantifies the separate-voting bug: how many cases got a (loc,dir) pair that appeared in ZERO
single complete answers, and how many change vs the old separate-voting label."""
import os, json
from collections import Counter
HOME=os.path.expanduser("~/ScoliCMF")
def majority_pair(votes):
    votes=[tuple(v) for v in votes if v and v[0]!="ERR"]
    if not votes: return "uncertain","uncertain",0.0
    pair,cnt=Counter(votes).most_common(1)[0]
    return pair[0],pair[1],cnt/len(votes)
def majority_sep(votes):  # old buggy separate voting (to measure impact)
    votes=[v for v in votes if v and v[0]!="ERR"]
    if not votes: return "uncertain","uncertain"
    lc=Counter(v[0] for v in votes); dc=Counter(v[1] for v in votes)
    return lc.most_common(1)[0][0], dc.most_common(1)[0][0]
recs=[json.loads(l) for l in open(os.path.join(HOME,"labels.json")) if l.strip()]
recs={r["stem"]:r for r in recs}.values()  # dedup
out=[]; recomb=0; changed=0
for r in recs:
    vts=r["votes"]
    ploc,pdir,pag=majority_pair(vts)
    sloc,sdir=majority_sep(vts)
    # did the separate-voting pair ever appear as a complete single answer?
    complete=set(tuple(v) for v in vts if v and v[0]!="ERR")
    if (sloc,sdir) not in complete and sloc!="uncertain": recomb+=1
    if (ploc,pdir)!=(sloc,sdir): changed+=1
    out.append({"stem":r["stem"],"dominant_location":ploc,"dominant_direction":pdir,"agree_pair":round(pag,3)})
with open(os.path.join(HOME,"labels_pair.json"),"w") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
n=len(out)
print(f"=== joint-pair relabel (n={n}) ===")
print(f"  BUG IMPACT: separate-voting produced a pair never in any single answer: {recomb}/{n} ({100*recomb/n:.0f}%)")
print(f"  joint-pair label differs from separate-voting label: {changed}/{n} ({100*changed/n:.0f}%)")
ag=[o["agree_pair"] for o in out]
for th in [4/7,5/7,6/7]:
    print(f"  pair agree >= {th*7:.0f}/7: {sum(a>=th-1e-6 for a in ag)}/{n} ({100*sum(a>=th-1e-6 for a in ag)/n:.0f}%)")
print("  pair dist (agree>=5/7):", dict(Counter(f'{o["dominant_location"]}|{o["dominant_direction"]}' for o in out if o["agree_pair"]>=5/7-1e-6)))
print("RELABEL_DONE")
