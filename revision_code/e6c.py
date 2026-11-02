import os
import sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
D = data()
rows=[]
for city in CITIES:
    out={}
    for lam in (1,2,5,10,25):
        cert=0
        for g in D[city]:
            S=g["S"]; n=g["n"]; ls=n/(2*(n-1))
            a=pick(S,"average"); C=S.mean(1)>=S[a].mean()-lam*S[a].var()
            R=S[C].max()-S[C].min()
            cert+= lam <= ls/R if R>0 else True
        out[lam]=cert
    rows.append(dict(city=city, **{f"certified_lam{l}":v for l,v in out.items()}))
    print(rows[-1])
with open(os.environ.get("FM_REV_OUT", "figdata_rev") + "/e6_certificate.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
