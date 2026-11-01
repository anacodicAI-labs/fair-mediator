import os
import sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
D = data()
rows=[]
for city in CITIES:
    G=D[city]
    pipes=[("Average on whole-menu proxy","S","average",0),("Fair (lambda=0.5) on whole-menu proxy","S","fair",0.5),
           ("Fair (lambda=5) on whole-menu proxy","S","fair",5),("Least-Misery on whole-menu proxy","S","least_misery",0),
           ("Average on order-aware utility","D","average",0),("Fair (lambda=5) on order-aware utility","D","fair",5),
           ("Least-Misery on order-aware utility","D","least_misery",0)]
    for name,key,rule,lam in pipes:
        ps=[pick(g[key],rule,lam) for g in G]
        mins=[g["D"][p].min() for g,p in zip(G,ps)]; means=[g["D"][p].mean() for g,p in zip(G,ps)]
        lo,hi=boot_ci(mins)
        rows.append(dict(city=city,pipeline=name,exp_minsat=round(np.mean(mins),4),lo=round(lo,4),hi=round(hi,4),exp_mean=round(np.mean(means),4)))
        print(rows[-1])
    # paired gain order-aware Average vs paper pipeline Fair(0.5,S)
    a=[g["D"][pick(g["D"],"average")].min()-g["D"][pick(g["S"],"fair",0.5)].min() for g in G]
    print(city,"paired gain",np.mean(a),boot_ci(a), "groups worse:",sum(x< -1e-12 for x in a))
    # C4>=B2 by construction check not needed
with open(os.environ.get("FM_REV_OUT", "figdata_rev") + "/e2b_experienced.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
