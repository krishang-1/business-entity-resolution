import os
import sys, json, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import config as C
from run_pipeline import Source, make_split
from decide import decide
from evaluate import f05
S1=Source("work","train","s1")
ids=S1.col("entity_id").to_numpy(zero_copy_only=False); ctry=S1.col("country").to_numpy(zero_copy_only=False)
gtt=pq.read_table("work/gt_pairs.parquet")
gt_s1=gtt.column("source1_entity_id").to_numpy()
tr,va=make_split(ids,ctry,pd.DataFrame({"source1_entity_id":np.unique(gt_s1)}))
va_ids=ids[va]; vset=set(va_ids)
m=pc.is_in(gtt.column("source1_entity_id"),value_set=pa.array(va_ids))
g=gtt.filter(m).to_pandas(); del gtt
P=Source("work","train","pool")
dv=pd.read_parquet("work/val_preds.parquet"); d=json.load(open("work/decision.json"))
acc=decide(dv,d["thr"],d["rel"])
pid=P.col("entity_id")
dv["cid"]=pid.take(pa.array(dv.p.values)).to_numpy(zero_copy_only=False); acc=acc.assign(cid=dv.loc[acc.index,"cid"].values)
del pid
q2id=dict(zip(va,va_ids))
cand=dv.groupby("q")["cid"].agg(set); pred=acc.groupby("q")["cid"].agg(set); true=g.groupby("source1_entity_id")["mid"].agg(set)
rows=[]
for q,sid in zip(va,va_ids):
    t=true.get(sid,set()); c=cand.get(q,set()); p=pred.get(q,set())
    fa=f05(p,t); fo=f05(c&t,t) if t else 1.0; fnofp=f05(p&t,t) if t else fa
    if not t: rows.append((q,0,1-fa,0,0,0,fa,fo,len(c)))
    else: rows.append((q,len(t),0,1-fo,fo-fnofp,fnofp-fa,fa,fo,len(c)))
L=pd.DataFrame(rows,columns=["q","ntrue","singleton_FP","missed_by_blocking","model_FN","model_FP","f","f_oracle","ncand"])
L["country"]=ctry[L.q.values]; L["bucket"]=L.ntrue.clip(upper=3).astype(str).replace({"3":"3+"})
n=len(L); B=["singleton_FP","missed_by_blocking","model_FN","model_FP"]
print(f"val S1={n} F05={L.f.mean():.4f} oracle_ceiling={L.f_oracle.mean():.4f} pair_recall={sum(len(cand.get(q,set())&true.get(s,set())) for q,s in zip(va,va_ids))/sum(len(v) for v in true.values):.4f} cands mean={L.ncand.mean():.2f} median={L.ncand.median()} p95={L.ncand.quantile(.95)}")
print("LOSS (points of macro F0.5):", {b: round(L[b].sum()/n,4) for b in B}, "total", round(1-L.f.mean(),4))
for k in ("country","bucket"):
    print(f"by {k}:"); print((L.groupby(k)[B].sum()/n).round(4).assign(n=L.groupby(k).size()).to_string())
L.to_parquet("work/loss_val.parquet")
# examples of the 2 largest buckets
top=sorted(B,key=lambda b:-L[b].sum())[:2]; print("largest buckets:",top)
S1f=S1.frame(["name_core","addr_norm"],None); Pn=P.col("name_core"); Pa_=P.col("addr_norm")
def ptxt(idx): return Pn.take(pa.array(idx)).to_numpy(zero_copy_only=False), Pa_.take(pa.array(idx)).to_numpy(zero_copy_only=False)
dv["label"]=dv.label.astype(int); dv["acc"]=0; dv.loc[acc.index,"acc"]=1
for b in top:
    if b=="missed_by_blocking":
        qs=L[L[b]>0].sample(25,random_state=0).q.values
        for q in qs:
            sid=q2id[q]; miss=list(true.get(sid,set())-cand.get(q,set()))[:1]
            print(f"[MISS] {S1f.name_core[q]} | {S1f.addr_norm[q][:55]}  ->  missed {miss}")
        continue
    sel={"singleton_FP":(dv.acc==1)&dv.q.isin(L[L.singleton_FP>0].q),"model_FP":(dv.acc==1)&(dv.label==0)&dv.q.isin(L[L.ntrue>0].q),
         "model_FN":(dv.acc==0)&(dv.label==1)}[b]
    x=dv[sel].sample(25,random_state=0); a,bb=ptxt(x.p.values)
    print(f"==== {b}")
    for i,(q,pr) in enumerate(zip(x.q.values,x.prob.values)):
        print(f"{pr:.2f} | {S1f.name_core[q]} | {S1f.addr_norm[q][:50]}\n     | {a[i]} | {bb[i][:50]}")
