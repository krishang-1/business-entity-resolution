import os
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import config as C
from run_pipeline import Source, make_split, load_gt, add_labels, model_feats
from features import context_features
from decide import tune
from blocking import select
if __name__=="__main__":
    gt=load_gt("dataset","work"); S1=Source("work","train","s1"); P=Source("work","train","pool")
    ids=S1.col("entity_id").to_numpy(zero_copy_only=False); ctry=S1.col("country").to_numpy(zero_copy_only=False)
    tr,va=make_split(ids,ctry,gt)
    F=pd.concat([pd.read_parquet("work/feats_train_US_b3_185309_f2.parquet").assign(cp="US"),
                 pd.read_parquet("work/feats_train_India_b3_123646_f2.parquet").assign(cp="India")],ignore_index=True)
    add_labels(F,S1,P,gt); feats=[c for c in model_feats(F) if c!="cp"]
    tm=gt[gt.source1_entity_id.isin(set(ids[va]))].groupby("source1_entity_id").size()
    bst=lgb.Booster(model_file="work/lgb.txt"); d=json.load(open("work/decision.json"))
    va_set=set(va); isva=F.q.isin(va_set).values
    def evalset(Fv, q_all, bst_):
        dv=Fv[["q","p","label"]].copy(); dv["prob"]=bst_.predict(Fv[feats].values.astype(np.float32),num_threads=7)
        tc=pd.Series(tm.reindex(ids[q_all]).fillna(0).values,index=q_all)
        return tune(dv,tc,q_all)[0]
    # candidate shrinking on val: re-select smaller sets from the per-list ranks, recompute context feats, same model
    Fv=F[isva].reset_index(drop=True)
    for a,b,c,mc in []:
        s=select(Fv,a,b,c,mc).reset_index(drop=True)
        cf=context_features(s); s[cf.columns]=cf.values
        f=evalset(s,va,bst); print(f"cands a={a} b={b} c={c} mc={mc}: mean={len(s)/len(va):.2f} pair_recall={s.label.sum()/tm.sum():.4f} F05={f[0]:.4f} {f[1]}",flush=True)
    # leave-one-country-out
    for src_c,dst_c in (("US","India"),("India","US")):
        trm=(~isva)&(F.cp==src_c).values; vm=isva&(F.cp==dst_c).values
        b=lgb.train(C.LGB_PARAMS|{"num_threads":7},lgb.Dataset(F.loc[trm,feats].values.astype(np.float32),F.loc[trm,"label"].values),479)
        qv=np.array(sorted(set(va)&set(np.flatnonzero(ctry==dst_c))))
        f=evalset(F[vm],qv,b); fin=evalset(F[vm],qv,bst)
        print(f"LOCO {src_c}->{dst_c}: F05={f[0]:.4f} {f[1]} | in-domain model on {dst_c}: {fin[0]:.4f}",flush=True)
