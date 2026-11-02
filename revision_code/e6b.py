import os
import sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
D = data()
grid = np.logspace(0, 3, 121)
rows=[]
for city in CITIES:
    firsts=[]
    for g in D[city]:
        S=g["S"]; f=np.inf
        for lam in grid:
            p=pick(S,"fair",lam)
            if np.any(np.all(S>=S[p],axis=1)&np.any(S>S[p],axis=1)): f=lam; break
        firsts.append(f)
    firsts=np.array(firsts); fin=firsts[np.isfinite(firsts)]
    rows.append(dict(city=city, first_dominated_min=round(fin.min(),1), first_dominated_p5=round(np.percentile(firsts,5),1),
          groups_ever_dominated_le1000=len(fin), groups_dominated_le_10=int((firsts<=10).sum()), groups_dominated_le_25=int((firsts<=25).sum())))
    print(rows[-1])
with open(os.environ.get("FM_REV_OUT", "figdata_rev") + "/e6_first_dominated.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
