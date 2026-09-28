import os
import sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rapidfuzz import fuzz
from normalize import skeleton
country=sys.argv[1]; frac=float(sys.argv[2])
cols=["entity_id","country","name_core","addr_norm","addr_nums"]
pool=pd.concat([pd.read_parquet(f"work/norm_train_s{i}.parquet",columns=cols) for i in (2,3)],ignore_index=True)
pool=pool[pool.country==country].reset_index(drop=True)
gt=pd.read_parquet("work/gt_pairs.parquet"); ow=pd.Series(gt.source1_entity_id.values,index=gt.mid.values)
pool["owner"]=ow.reindex(pool.entity_id.values).fillna("").values
if frac<1:
    rng=np.random.RandomState(0); own=pool.owner.unique(); keep=set(rng.choice(own,int(len(own)*frac),replace=False))
    pool=pool[pool.owner.isin(keep)|((pool.owner=="")&(np.random.RandomState(1).rand(len(pool))<frac))].reset_index(drop=True)
name=pool.name_core.values; addr=pool.addr_norm.values; owner=pool.owner.values
skel=np.array([" ".join(skeleton(t) for t in s.split()) for s in name],dtype=object)
nr=pd.DataFrame([(i,x) for i,s in enumerate(pool.addr_nums.values) for x in s.split()[:2]],columns=["i","num"])
def edges_from(keys, idx, maxk=50):
    g=pd.Series(keys).groupby(keys).indices; E=[]
    for v in g.values():
        if 2<=len(v)<=maxk:
            v=idx[v]; E+= [(v[a],v[b]) for a in range(len(v)) for b in range(a+1,len(v))]
    return np.array(E) if E else np.zeros((0,2),int)
def report(tag,E):
    if len(E)==0: print(tag,"no edges"); return
    o1,o2=owner[E[:,0]],owner[E[:,1]]; both=(o1!="")&(o2!="")
    print(f"{tag}: edges={len(E)} precision(both owned)={((o1==o2)&both).sum()/max(both.sum(),1):.4f} unowned-involved={(~both).mean():.3f}")
ia=np.arange(len(pool))
ne=(name!="")&(addr!="")
report("L1 name+addr", edges_from((name+"|"+addr)[ne], ia[ne]))
report("L2 name+num", edges_from(name[nr.i.values]+"|"+nr.num.values, nr.i.values))
report("L5 skel+num", edges_from(skel[nr.i.values]+"|"+nr.num.values, nr.i.values))
E=edges_from(addr[addr!=""], ia[addr!=""]); 
if len(E): 
    s=np.array([fuzz.token_set_ratio(skel[a],skel[b]) for a,b in E]); report("L4 addr+skel>=80",E[s>=80]); report("L4 addr+skel>=95",E[s>=95]); report("addr only",E)
# stricter: name+addr+first number equal
report("L1 & freq<=5", edges_from((name+"|"+addr)[ne], ia[ne], maxk=5))
