import os
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import config as C
from rapidfuzz import fuzz
from run_pipeline import Source, make_split, load_gt
S1=Source("work","train","s1"); P=Source("work","train","pool")
ids=S1.col("entity_id").to_numpy(zero_copy_only=False); ctry=S1.col("country").to_numpy(zero_copy_only=False)
tr,va=make_split(ids,ctry,load_gt("dataset","work"))
gi=pd.read_parquet("work/gt_int.parquet"); gv=gi[np.isin(gi.q.values,va)].copy()
cands=pd.concat([pd.read_parquet(f,columns=["q","p"]) for f in glob.glob(f"work/cands_train_*_{C.BLOCK_VERSION}_*.parquet")])
key=lambda d: d.q.values.astype(np.int64)<<32|d.p.values.astype(np.int64)
gv["hit"]=np.isin(key(gv),key(cands))
miss=gv[~gv.hit]; hit=gv[gv.hit]
print("val true pairs",len(gv),"missed",len(miss), "S1 with a miss", miss.q.nunique(), "of which have >=1 retrieved sibling", miss.q.isin(set(hit.q)).mean().round(3))
m=miss.merge(hit[["q","p"]].rename(columns={"p":"sp"}),on="q")
up=np.unique(np.concatenate([m.p.values,m.sp.values,miss.p.values])); pf=P.frame(["name_core","addr_norm"],up); pos=pd.Series(np.arange(len(up)),index=up)
a=pos[m.p.values].values; b=pos[m.sp.values].values
nm=pf.name_core.values; ad=pf.addr_norm.values
m["ns"]=[fuzz.token_set_ratio(nm[x],nm[y]) for x,y in zip(a,b)]
m["as_"]=[fuzz.token_set_ratio(ad[x],ad[y]) if ad[x] and ad[y] else -1 for x,y in zip(a,b)]
best=m.groupby(["q","p"]).agg(ns=("ns","max"),as_=("as_","max")).reset_index()
print("missed pairs with a retrieved sibling:",len(best),"of",len(miss))
for thr in (95,90,85,80):
    print(f"  best sibling sim >= {thr}: name {(best.ns>=thr).mean():.3f} | addr {(best.as_>=thr).mean():.3f} | either {((best.ns>=thr)|(best.as_>=thr)).mean():.3f}")
mp=pos[miss.p.values].values
print("missed record empty address:", (ad[mp]=="").mean().round(3))
