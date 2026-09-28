"""End-to-end pipeline: raw TSV -> normalization -> per-country blocking -> pair features -> LightGBM -> decision -> TSVs.
  python src/run_pipeline.py --data-dir <dataset> --out-dir <output> --work-dir <work> --mode {val,full}
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from evaluate import Stage, score, blocking_report, sanity
from io_utils import read_tsv, write_id_lists

FEAT_COLS = ["name_core", "name_norm", "name_legal", "addr_norm", "addr_main", "addr_nums", "postcode"]


def load_raw(data_dir, work, split, force=False):
    for i in (1, 2, 3):
        dst = f"{work}/raw_{split}_s{i}.parquet"
        if force or not os.path.exists(dst):
            read_tsv(f"{data_dir}/{split}/{split}_source{i}.tsv").to_parquet(dst, index=False)


class Source:
    """Lazy column access to normalized parquet (S1, or pool = S2 then S3) via Arrow; rows fetched by int index."""

    def __init__(self, work, split, which):
        self.paths = [f"{work}/norm_{split}_s1.parquet"] if which == "s1" else \
            [f"{work}/norm_{split}_s2.parquet", f"{work}/norm_{split}_s3.parquet"]

    def col(self, c):
        import pyarrow as pa
        return pa.concat_arrays([pq.read_table(p, columns=[c]).column(c).combine_chunks() for p in self.paths])

    def frame(self, cols, idx=None):
        import pyarrow as pa
        out = {}
        for c in cols:
            a = self.col(c)
            if idx is not None:
                a = a.take(pa.array(idx))
            out[c] = a.to_numpy(zero_copy_only=False)
            del a
        return pd.DataFrame(out)


def load_gt(data_dir, work):
    p = f"{work}/gt_pairs.parquet"
    if not os.path.exists(p):
        gt = read_tsv(f"{data_dir}/train/train_ground_truth.tsv")
        gt = gt.assign(mid=gt["matched_entity_ids"].str.split(",")).explode("mid")
        gt = gt[gt["mid"] != ""][["source1_entity_id", "mid"]]
        gt.to_parquet(p, index=False)
    return pd.read_parquet(p)


def make_split(s1_ids, s1_country, gt):
    """Positional (train_idx, val_idx): per country x has-match stratified subsample of train S1 (seed 42)."""
    has = pd.Series(s1_ids).isin(set(gt["source1_entity_id"])).values
    strata = pd.Series(s1_country).astype(str).values + "_" + has.astype(str)
    rng = np.random.RandomState(C.SEED)
    tr, va = [], []
    for _, idx in pd.Series(np.arange(len(s1_ids))).groupby(strata):
        idx = rng.permutation(idx.values)
        ntr, nva = int(round(len(idx) * C.TRAIN_FRAC)), int(round(len(idx) * C.VAL_FRAC))
        tr.append(idx[:ntr]); va.append(idx[ntr:ntr + nva])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def safe(lab):
    return "".join(ch if ch.isalnum() else "_" for ch in str(lab))


def wide_country(work, split, lab, q_glob, p_glob, S1, POOL, pruner=None, force=False):
    """Wide-net blocking for one country (pruned inline when pruner is given). Global int32 q/p."""
    from blocking import block_partition
    path = f"{work}/{'cands' if pruner else 'wide'}_{split}_{safe(lab)}_{C.BLOCK_VERSION}_{len(q_glob)}.parquet"
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)
    with Stage(f"blocking {split} {lab} n_s1={len(q_glob)} n_pool={len(p_glob)} pruned={pruner is not None}"):
        s1t = S1.frame(["name_core", "addr_norm"], q_glob)
        pt = POOL.frame(["name_core", "addr_norm"], p_glob)
        df = block_partition(s1t, pt, pruner=pruner)
        del s1t, pt
        df["q"] = q_glob[df["q"].values].astype(np.int32)
        df["p"] = p_glob[df["p"].values].astype(np.int32)
        df.to_parquet(path, index=False)
    gc.collect()
    return df


def process_country(work, split, lab, q_glob, p_glob, S1, POOL, force=False, predictor=None, pruner=None,
                    chunk_s1=150_000):
    """Block + features for one country, features computed in S1 chunks. Returns DataFrame with global int32 q/p and
    feature columns, or (when predictor=(booster, feats) is given) only q, p, prob."""
    from features import string_features, context_features
    tag = f"{split}_{safe(lab)}_{C.BLOCK_VERSION}_{len(q_glob)}"
    cpath = f"{work}/cands_{tag}.parquet"
    kind = ("preds_p2" if len(predictor) > 2 else "preds") if predictor else "feats"
    fpath = f"{work}/{kind}_{tag}_{C.FEAT_VERSION}.parquet"
    if os.path.exists(fpath) and not force:
        return pd.read_parquet(fpath)
    if os.path.exists(cpath) and not force:
        cand = pd.read_parquet(cpath)
    else:
        cand = wide_country(work, split, lab, q_glob, p_glob, S1, POOL, pruner=pruner, force=force)
    with Stage(f"name/addr frequency {split} {lab}"):
        pnc = POOL.frame(["name_core", "addr_norm"], p_glob)
        name_freq = pnc["name_core"].value_counts()
        addr_freq = pnc["addr_norm"].value_counts()
        del pnc
    uq_all = np.unique(cand["q"].values)
    outs = []
    for c0 in range(0, len(uq_all), chunk_s1):
        with Stage(f"features {split} {lab} chunk {c0}"):
            part = cand[cand["q"].isin(uq_all[c0:c0 + chunk_s1])].reset_index(drop=True)
            uq, qi = np.unique(part["q"].values, return_inverse=True)
            up, pi = np.unique(part["p"].values, return_inverse=True)
            s1f = S1.frame(FEAT_COLS, uq)
            pf = POOL.frame(FEAT_COLS + ["entity_id"], up)
            part["src_s3"] = pd.Series(pf["entity_id"].values[pi]).str.startswith("S3-").values.astype(np.int8)
            sf = string_features(s1f, pf, qi.astype(np.int64), pi.astype(np.int64))
            cf = context_features(part)
            cf["p_name_freq"] = pf["name_core"].map(name_freq).fillna(0).values[pi].astype(np.float32)
            cf["q_name_freq"] = s1f["name_core"].map(name_freq).fillna(0).values[qi].astype(np.float32)
            pa_ = pf["addr_norm"].values[pi]
            cf["p_addr_freq"] = np.where(pa_ == "", -1, pf["addr_norm"].map(addr_freq).fillna(0).values[pi]).astype(np.float32)
            F = pd.concat([part, sf, cf.reset_index(drop=True)], axis=1)
            del s1f, sf, cf, part
            if predictor:
                bst, feats = predictor[0], predictor[1]
                p1 = bst.predict(F[feats].values.astype(np.float32), num_threads=6).astype(np.float32)
                if len(predictor) > 2:
                    import stack
                    bst2, feats2 = predictor[2], predictor[3]
                    T = F.copy()
                    T["p1"] = p1
                    T["p"] = pi.astype(np.int32)
                    T = stack.group_features(T)
                    T = pd.concat([T, stack.sibling_features(T, pf["name_core"].values, pf["addr_norm"].values)], axis=1)
                    prob = bst2.predict(T[feats2].values.astype(np.float32), num_threads=6)
                    del T
                else:
                    prob = p1
                F = pd.DataFrame({"q": F["q"].values, "p": F["p"].values, "p1": p1, "prob": prob})
            del pf
            outs.append(F)
            gc.collect()
    F = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(columns=["q", "p", "prob"])
    F.to_parquet(fpath, index=False)
    return F


def run_split(work, split, s1_idx, force=False, predictor=None, pruner=None):
    S1, POOL = Source(work, split, "s1"), Source(work, split, "pool")
    s1c = S1.col("country").to_numpy(zero_copy_only=False)
    pc = POOL.col("country").to_numpy(zero_copy_only=False)
    from blocking import partitions
    parts = []
    for lab, s1m, poolm in partitions(s1c[s1_idx], pc):
        F = process_country(work, split, lab, s1_idx[s1m].astype(np.int32), np.flatnonzero(poolm).astype(np.int32),
                            S1, POOL, force, predictor, pruner)
        F["country_part"] = lab
        parts.append(F)
        gc.collect()
    return pd.concat(parts, ignore_index=True), S1, POOL


def model_feats(F):
    drop = {"q", "p", "country_part", "label", "prob", "p1"}
    return [c for c in F.columns if c not in drop]


def ids_for(src, idx):
    import pyarrow as pa
    return src.col("entity_id").take(pa.array(idx)).to_numpy(zero_copy_only=False)


def gt_int(work, S1, POOL, gt):
    """Ground-truth pairs as global int indices (q into S1, p into pool), via Arrow hash lookups."""
    import pyarrow as pa
    import pyarrow.compute as pc
    path = f"{work}/gt_int.parquet"
    if not os.path.exists(path):
        q = pc.index_in(pa.array(gt["source1_entity_id"].values), value_set=S1.col("entity_id")).to_numpy(zero_copy_only=False)
        p = pc.index_in(pa.array(gt["mid"].values), value_set=POOL.col("entity_id")).to_numpy(zero_copy_only=False)
        pd.DataFrame({"q": q.astype(np.int64), "p": p.astype(np.int64)}).to_parquet(path, index=False)
    return pd.read_parquet(path)


def int_labels(df, gi):
    key = df["q"].values.astype(np.int64) << 32 | df["p"].values.astype(np.int64)
    gkey = gi["q"].values << 32 | gi["p"].values
    return np.isin(key, gkey).astype(np.int8)


def train_stage1(Wd, lab, is_va, n_true_va, work):
    """Learned first-stage blocker on cheap features; threshold keeps >= STAGE1_RECALL of wide-net val positives."""
    import lightgbm as lgb
    from blocking import STAGE1_FEATS, prune
    prm = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=100, seed=C.SEED, verbose=-1,
               num_threads=6)
    bst = lgb.train(prm, lgb.Dataset(Wd.loc[~is_va, STAGE1_FEATS].values.astype(np.float32), lab[~is_va]), 100)
    bst.save_model(f"{work}/stage1.txt")
    V = Wd.loc[is_va].copy()
    V["label"] = lab[is_va]
    wide_pos = V["label"].sum()
    print(f"stage1: val wide-net pair_recall={wide_pos / n_true_va:.4f} mean_cands={len(V) / V['q'].nunique():.2f}", flush=True)
    best = None
    for thr in [0.5, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]:
        d = prune(V.copy(), (bst, thr, C.STAGE1_MAX_CAND))
        kept = d["label"].sum() / wide_pos
        n = d.groupby("q").size()
        print(f"  thr={thr}: kept {kept:.4f} of wide positives, pair_recall={d['label'].sum() / n_true_va:.4f} "
              f"mean={n.mean():.2f} p95={n.quantile(.95):.0f}", flush=True)
        if kept >= C.STAGE1_RECALL:
            best = thr
            break
    best = best if best is not None else 0.001
    json.dump({"thr": best}, open(f"{work}/stage1.json", "w"))
    return (bst, best, C.STAGE1_MAX_CAND)


def add_labels(F, S1, POOL, gt):
    s1_ids = ids_for(S1, F["q"].values)
    p_ids = ids_for(POOL, F["p"].values)
    g = gt[gt["source1_entity_id"].isin(set(s1_ids))]
    key = set(zip(g["source1_entity_id"], g["mid"]))
    F["label"] = np.fromiter(((a, b) in key for a, b in zip(s1_ids, p_ids)), np.int8, len(F))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--mode", choices=["val", "full"], required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    W = a.work_dir
    os.makedirs(W, exist_ok=True); os.makedirs(a.out_dir, exist_ok=True)
    sanity()
    from normalize import normalize_split
    from model import train as lgb_train
    from decide import decide, tune
    with Stage("load+normalize"):
        for sp in ["train"] + (["test"] if a.mode == "full" else []):
            load_raw(a.data_dir, W, sp)
            normalize_split(W, sp)
    gt = load_gt(a.data_dir, W)
    S1 = Source(W, "train", "s1")
    s1_ids = S1.col("entity_id").to_numpy(zero_copy_only=False)
    s1_country = S1.col("country").to_numpy(zero_copy_only=False)
    tr_idx, va_idx = make_split(s1_ids, s1_country, gt)
    print(f"subsample: train S1={len(tr_idx)} val S1={len(va_idx)} of {len(s1_ids)} (per-country stratified, seed {C.SEED})", flush=True)
    va_ids, va_country = s1_ids[va_idx], s1_country[va_idx]
    true_map = gt[gt["source1_entity_id"].isin(set(va_ids))].groupby("source1_entity_id")["mid"].agg(set).to_dict()
    del s1_ids, s1_country
    gc.collect()

    sub = np.sort(np.concatenate([tr_idx, va_idx]))
    POOL = Source(W, "train", "pool")
    gi = gt_int(W, S1, POOL, gt)
    va_arr = np.asarray(va_idx)
    n_true_va = int(np.isin(gi["q"].values, va_arr).sum())
    from blocking import partitions, prune
    s1c = S1.col("country").to_numpy(zero_copy_only=False)
    pcn = POOL.col("country").to_numpy(zero_copy_only=False)
    parts = [(lab, sub[s1m].astype(np.int32), np.flatnonzero(poolm).astype(np.int32))
             for lab, s1m, poolm in partitions(s1c[sub], pcn)]
    del s1c, pcn
    if os.path.exists(f"{W}/stage1.json") and os.path.exists(f"{W}/stage1.txt") and not a.force:
        import lightgbm as lgb
        pruner = (lgb.Booster(model_file=f"{W}/stage1.txt"), json.load(open(f"{W}/stage1.json"))["thr"], C.STAGE1_MAX_CAND)
    else:
        Wd = pd.concat([wide_country(W, "train", lab, q, p, S1, POOL, force=a.force) for lab, q, p in parts],
                       ignore_index=True)
        with Stage("stage1 blocker train + threshold"):
            wl = int_labels(Wd, gi)
            pruner = train_stage1(Wd, wl, np.isin(Wd["q"].values, va_arr), n_true_va, W)
        del Wd, wl
        gc.collect()
    print(f"stage1 pruner thr={pruner[1]} max_cand={pruner[2]}", flush=True)
    for lab, q, p in parts:
        cpath = f"{W}/cands_train_{safe(lab)}_{C.BLOCK_VERSION}_{len(q)}.parquet"
        if not os.path.exists(cpath) or a.force:
            with Stage(f"prune train {lab}"):
                wd = wide_country(W, "train", lab, q, p, S1, POOL)
                prune(wd, pruner).to_parquet(cpath, index=False)
                del wd
                gc.collect()
    F, S1, POOL = run_split(W, "train", sub, a.force)
    F["label"] = int_labels(F, gi)
    feats = model_feats(F)
    is_va = F["q"].isin(set(va_idx)).values

    # blocking metrics on validation S1
    Fv = F.loc[is_va, ["q", "p", "label"]]
    blocking_report(pd.DataFrame({"s1": ids_for(S1, Fv["q"].values), "cid": ids_for(POOL, Fv["p"].values)}),
                    gt, va_ids, true_map)

    with Stage("lgb train (train split)"):
        bst = lgb_train(F.loc[~is_va, feats].values.astype(np.float32), F.loc[~is_va, "label"].values,
                        F.loc[~is_va, "q"].values, feats, W)
    dv = F.loc[is_va, ["q", "p", "label"]].copy()
    dv["prob"] = bst.predict(F.loc[is_va, feats].values.astype(np.float32), num_threads=6)
    tc = pd.Series({q: len(true_map.get(s, ())) for q, s in zip(va_idx, va_ids)})
    with Stage("tune decision"):
        (f, (thr, rel, P, R)), grid = tune(dv, tc, va_idx)
        f_no = tune(dv, tc, va_idx, one_owner=False)[0]
    print(f"VAL BEST F05={f:.4f} thr={thr} rel={rel} P={P:.4f} R={R:.4f} | without one-owner F05={f_no[0]:.4f} {f_no[1]}", flush=True)
    json.dump({"thr": thr, "rel": rel, "val_F05": f, "best_iter": bst.best_iteration}, open(f"{W}/decision.json", "w"))
    grid.to_csv(f"{W}/decision_grid.csv", index=False)
    acc = decide(dv, thr, rel)
    pm = pd.DataFrame({"s": ids_for(S1, acc["q"].values), "c": ids_for(POOL, acc["p"].values)}).groupby("s")["c"].agg(set).to_dict()
    score(pm, true_map, list(va_ids), country=va_country)
    dv.to_parquet(f"{W}/val_preds.parquet", index=False)

    # ---- pass 2 (stacking): OOF pass-1 probs on train rows + per-S1 context + sibling support ----
    import stack
    p2_feats = stack.PASS2_BASE + stack.SIB_COLS
    p2_path = f"{W}/pass2_train_{C.BLOCK_VERSION}_{C.FEAT_VERSION}.parquet"
    if os.path.exists(p2_path) and not a.force:
        D = pd.read_parquet(p2_path)
    else:
        oof = stack.oof_pass1(F.loc[~is_va, feats].values.astype(np.float32), F.loc[~is_va, "label"].values,
                              F.loc[~is_va, "q"].values, max(bst.best_iteration, 50))
        D = F[["q", "p", "label"]].copy()
        D["p1"] = np.float32(0)
        D.loc[~is_va, "p1"] = oof
        D.loc[is_va, "p1"] = dv["prob"].values.astype(np.float32)
        up, inv = np.unique(D["p"].values, return_inverse=True)
        pf = POOL.frame(["name_core", "addr_norm"], up)
        gp = D["p"].values
        D["p"] = inv.astype(np.int32)
        D = stack.pass2_frame(D, pf["name_core"].values, pf["addr_norm"].values)
        D["p"] = up[D["p"].values].astype(np.int32)
        del pf, oof
        D.to_parquet(p2_path, index=False)
    D = D.merge(F[["q", "p"] + [c for c in feats if c not in D.columns]], on=["q", "p"], how="left")
    p2_feats = feats + [c for c in stack.PASS2_BASE + stack.SIB_COLS if c not in feats]
    d_va = D["q"].isin(set(va_idx)).values
    with Stage("pass2 train"):
        bst2 = lgb_train(D.loc[~d_va, p2_feats].values.astype(np.float32), D.loc[~d_va, "label"].values,
                         D.loc[~d_va, "q"].values, p2_feats, W, name="lgb_pass2")
    dv2 = D.loc[d_va, ["q", "p", "label"]].copy()
    dv2["prob"] = bst2.predict(D.loc[d_va, p2_feats].values.astype(np.float32), num_threads=6)
    (f2, (thr2, rel2, P2, R2)), _ = tune(dv2, tc, va_idx)
    use_p2 = f2 >= f + 0.002
    print(f"PASS2 VAL F05={f2:.4f} thr={thr2} rel={rel2} P={P2:.4f} R={R2:.4f} | pass1 {f:.4f} -> use_pass2={use_p2}", flush=True)
    if use_p2:
        acc2 = decide(dv2, thr2, rel2)
        pm2 = pd.DataFrame({"s": ids_for(S1, acc2["q"].values), "c": ids_for(POOL, acc2["p"].values)}).groupby("s")["c"].agg(set).to_dict()
        score(pm2, true_map, list(va_ids), country=va_country)
        thr, rel = thr2, rel2
        json.dump({"thr": thr, "rel": rel, "val_F05": f2, "pass2": True}, open(f"{W}/decision.json", "w"))
        dv2.to_parquet(f"{W}/val_preds_pass2.parquet", index=False)
    del D
    gc.collect()
    if a.mode == "val":
        return

    # full: predict test with the model validated above (trained on the train split of the subsample)
    del F, dv
    gc.collect()
    TS1 = Source(W, "test", "s1")
    n_test = len(TS1.col("entity_id"))
    pred = (bst, feats, bst2, p2_feats) if use_p2 else (bst, feats)
    Ft, TS1, TPOOL = run_split(W, "test", np.arange(n_test), a.force, predictor=pred, pruner=pruner)
    with Stage("decide test"):
        acc = decide(Ft[["q", "p", "prob"]], thr, rel)
    write_outputs(TS1, TPOOL, Ft[["q", "p"]], acc, a.out_dir, n_test)


def write_outputs(TS1, TPOOL, cand, acc, out_dir, n_test):
    ids1 = TS1.col("entity_id").to_numpy(zero_copy_only=False)
    country = TS1.col("country").to_numpy(zero_copy_only=False)
    cp = ids_for(TPOOL, cand["p"].values); ap_ = ids_for(TPOOL, acc["p"].values)
    cl = pd.Series(cp).groupby(cand["q"].values).agg(list)
    ml = pd.Series(ap_).groupby(acc["q"].values).agg(list)
    cl_all = [cl.get(i, []) for i in range(n_test)]
    ml_all = [ml.get(i, []) for i in range(n_test)]
    write_id_lists(f"{out_dir}/candidate_pairs.tsv", ids1, cl_all, "candidate_entity_ids")
    write_id_lists(f"{out_dir}/matching_results.tsv", ids1, ml_all, "matched_entity_ids")
    n = np.array([len(x) for x in cl_all]); m = np.array([len(x) for x in ml_all])
    print(f"test: S1={n_test} cands mean={n.mean():.2f} median={np.median(n)} p95={np.percentile(n, 95)} max={n.max()} "
          f"| matched mean={m.mean():.2f} empty_frac={(m == 0).mean():.4f}", flush=True)
    for c in pd.unique(country):
        mk = country == c
        print(f"  {c}: n={mk.sum()} mean_cands={n[mk].mean():.2f} mean_matches={m[mk].mean():.2f} empty={(m[mk] == 0).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
