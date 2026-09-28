import os
import sys, json, itertools, numpy as np, pandas as pd
L=pd.read_parquet("work/loss_val.parquet"); dv=pd.read_parquet("work/val_preds.parquet")
q_all=L.q.values; qpos={q:i for i,q in enumerate(q_all)}; ntrue=L.ntrue.values.astype(float)
# one-owner (threshold independent) precomputed once
d=dv[dv.prob>=dv.groupby("p").prob.transform("max")]; d=d[~d.duplicated("p")].copy()
d["mx"]=d.groupby("q").prob.transform("max"); d["first"]=d.prob>=d.mx
qi=d.q.map(qpos).values; prob=d.prob.values; mx=d.mx.values; first=d["first"].values; lab=d.label.values.astype(float)
def score(tf,te,rel,mask=None):
    keep=((first&(prob>=tf))|((~first)&(prob>=te)&(mx>=tf)))&(prob>=rel*mx)
    tp=np.bincount(qi[keep],weights=lab[keep],minlength=len(q_all)); npd=np.bincount(qi[keep],minlength=len(q_all))
    f=np.where((ntrue==0)&(npd==0),1.0,0.0); m=tp>0
    P=np.where(m,tp/np.maximum(npd,1),0); R=np.where(m,tp/np.maximum(ntrue,1),0)
    f[m]=1.25*P[m]*R[m]/(0.25*P[m]+R[m])
    if mask is not None: f=f[mask]
    return f.mean()
grid=list(itertools.product(np.round(np.arange(0.5,0.95,0.03),2),np.round(np.arange(0.2,0.8,0.03),2),[0,0.3,0.5,0.7]))
print("current rule (.68/.68/.7):",round(score(.68,.68,.7),4))
res=sorted(((score(*g),)+g for g in grid),reverse=True); print("top:",[tuple(np.round(r,4)) for r in res[:5]])
rng=np.random.RandomState(42); h=rng.rand(len(q_all))<0.5
for a,b in ((h,~h),(~h,h)):
    best=max((score(*g,mask=a),)+g for g in grid)
    print(f"half-split tuned {best[1:]}: held-out {score(*best[1:],mask=b):.4f} vs current {score(.68,.68,.7,mask=b):.4f}")
json.dump(dict(zip(["F05","thr_first","thr_extra","rel"],map(float,res[0]))),open("work/decision_b3.json","w"))
