import os
import sys, time, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from blocking import block_partition, select
from evaluate import rss_mb
if __name__ == "__main__":
    frac=float(sys.argv[1]); country=sys.argv[2]
    s1=pd.read_parquet("work/norm_train_s1.parquet", columns=["entity_id","country","name_core","addr_norm"])
    s1=s1[s1.country==country].sample(frac=frac,random_state=0).reset_index(drop=True)
    pool=pd.concat([pd.read_parquet(f"work/norm_train_s{i}.parquet", columns=["entity_id","country","name_core","addr_norm"]) for i in (2,3)],ignore_index=True)
    pool=pool[pool.country==country].reset_index(drop=True)
    gt=pd.read_parquet("work/gt_pairs.parquet"); gt=gt[gt.source1_entity_id.isin(set(s1.entity_id))]
    t=time.time(); df=block_partition(s1,pool); el=time.time()-t
    print(f"{country} n_s1={len(s1)} pool={len(pool)} block_time={el:.0f}s rss={rss_mb()}")
    key=set(zip(gt.source1_entity_id,gt.mid))
    df["y"]=np.fromiter(((a,b) in key for a,b in zip(s1.entity_id.values[df.q.values],pool.entity_id.values[df.p.values])),bool,len(df))
    df["s1"]=s1.entity_id.values[df.q.values]; df["cid"]=pool.entity_id.values[df.p.values]
    df.to_parquet(f"work/probe_{country}.parquet")
    T=len(gt)
    for col in ("r_name","r_addr","r_comb"):
        print(col, {k: round(df[(df[col]<=k)].y.sum()/T,4) for k in (5,10,15,20)})
    print("union all", round(df.y.sum()/T,4), "mean", len(df)/len(s1))
    rows=[]
    for a,b,c,mc in itertools.product((5,10,15,20),(5,10,15,20),(5,10,15,20),(10,12,15,20,25)):
        d=select(df,a,b,c,mc); n=d.groupby("q").size().reindex(range(len(s1)),fill_value=0)
        rows.append((a,b,c,mc,d.y.sum()/T,n.mean(),n.quantile(.95)))
    r=pd.DataFrame(rows,columns=["a","b","c","mc","recall","mean","p95"])
    r.to_csv(f"work/probe_grid_{country}.csv",index=False)
    for thr in (0.93,0.95,0.97):
        x=r[r.recall>=thr].sort_values("mean").head(3); print(f"smallest with recall>={thr}:\n", x.to_string(index=False))
