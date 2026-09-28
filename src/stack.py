"""Pass-2 stacking: per-S1 probability context + sibling support features on top of pass-1 probabilities.
Train rows use 5-fold out-of-fold pass-1 probabilities (grouped by S1); val/test use the pass-1 model trained on the
train split. Sibling support = similarity of a candidate to the S1's most confident other candidates (records of one
business tend to be near-duplicates of each other)."""
import gc
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

import config as C
from evaluate import Stage

N_SIB = 3
SIB_MIN_P = 0.3
PASS2_BASE = ["p1", "p1_rank", "p1_gap", "s1_pmax", "s1_p2", "s1_n05", "s1_n09", "s1_psum", "s1_ncand"]
SIB_COLS = ["sib_name", "sib_addr", "sib_name_p", "sib_addr_p", "sib_both", "sib_n"]


def oof_pass1(X, y, groups, n_rounds, n_folds=5):
    """Out-of-fold pass-1 probabilities for training rows, folds grouped by S1."""
    ug = np.unique(groups)
    rng = np.random.RandomState(C.SEED)
    fold_of = dict(zip(ug, rng.randint(0, n_folds, len(ug))))
    f = np.fromiter((fold_of[g] for g in groups), np.int8, len(groups))
    oof = np.zeros(len(y), np.float32)
    for k in range(n_folds):
        with Stage(f"oof pass1 fold {k}"):
            m = f == k
            b = lgb.train(C.LGB_PARAMS, lgb.Dataset(X[~m], y[~m]), n_rounds)
            oof[m] = b.predict(X[m], num_threads=C.LGB_PARAMS["num_threads"])
            del b
            gc.collect()
    return oof


def group_features(df):
    """df: q, p, p1. Adds per-S1 context of pass-1 probabilities."""
    g = df.groupby("q")["p1"]
    df["p1_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    df["s1_pmax"] = g.transform("max").astype(np.float32)
    df["p1_gap"] = (df["s1_pmax"] - df["p1"]).astype(np.float32)
    srt = df[["q", "p1"]].sort_values(["q", "p1"], ascending=[True, False])
    second = srt[srt.groupby("q").cumcount() == 1].set_index("q")["p1"]
    df["s1_p2"] = df["q"].map(second).fillna(0).values.astype(np.float32)
    df["s1_n05"] = (df["p1"] >= 0.5).groupby(df["q"]).transform("sum").astype(np.float32)
    df["s1_n09"] = (df["p1"] >= 0.9).groupby(df["q"]).transform("sum").astype(np.float32)
    df["s1_psum"] = g.transform("sum").astype(np.float32)
    df["s1_ncand"] = g.transform("size").astype(np.float32)
    return df


def _sib_chunk(args):
    na, nb, aa, ab = args
    out = np.empty((len(na), 2), np.float32)
    for i in range(len(na)):
        out[i, 0] = fuzz.token_set_ratio(na[i], nb[i])
        out[i, 1] = fuzz.token_set_ratio(aa[i], ab[i]) if aa[i] and ab[i] else -1
    return out


def sibling_features(df, pool_name, pool_addr, n_jobs=6, chunk=200_000):
    """df: q, p, p1, p1_rank (row order preserved). pool_name/pool_addr: arrays indexed by p (global)."""
    sib = df.loc[(df["p1_rank"] <= N_SIB) & (df["p1"] >= SIB_MIN_P), ["q", "p", "p1"]].rename(
        columns={"p": "sp", "p1": "sp1"})
    pairs = df[["q", "p"]].reset_index().merge(sib, on="q")
    pairs = pairs[pairs["p"] != pairs["sp"]].reset_index(drop=True)
    res = np.zeros((len(pairs), 2), np.float32)
    if len(pairs):
        pv, sv = pairs["p"].values, pairs["sp"].values
        jobs = [(pool_name[pv[i:i + chunk]].tolist(), pool_name[sv[i:i + chunk]].tolist(),
                 pool_addr[pv[i:i + chunk]].tolist(), pool_addr[sv[i:i + chunk]].tolist())
                for i in range(0, len(pairs), chunk)]
        with Pool(n_jobs) as pl:
            res = np.vstack(pl.map(_sib_chunk, jobs))
        del jobs
    pairs["sn"], pairs["sa"] = res[:, 0], res[:, 1]
    pairs["both"] = np.minimum(pairs["sn"], pairs["sa"])
    out = pd.DataFrame(index=df.index)
    out["sib_n"] = pairs.groupby("index").size().reindex(df.index).fillna(0).astype(np.float32)
    for col, src in (("sib_name", "sn"), ("sib_addr", "sa"), ("sib_both", "both")):
        out[col] = pairs.groupby("index")[src].max().reindex(df.index).fillna(-2).astype(np.float32)
    for col, src in (("sib_name_p", "sn"), ("sib_addr_p", "sa")):
        best = pairs.sort_values(["index", src], ascending=[True, False]).drop_duplicates("index")
        out[col] = best.set_index("index")["sp1"].reindex(df.index).fillna(0).astype(np.float32)
    return out


def pass2_frame(df, pool_name, pool_addr, q_chunk=150_000):
    """df: q, p, p1 (+ anything). Returns df with pass-2 features, processed in S1 chunks (features are per-S1)."""
    uq = np.unique(df["q"].values)
    outs = []
    for c0 in range(0, len(uq), q_chunk):
        with Stage(f"pass2 features chunk {c0}"):
            part = df[df["q"].isin(uq[c0:c0 + q_chunk])].reset_index(drop=True)
            part = group_features(part)
            sf = sibling_features(part, pool_name, pool_addr)
            outs.append(pd.concat([part, sf], axis=1))
            gc.collect()
    return pd.concat(outs, ignore_index=True)


def pass2_predict_frame(df, pool_name, pool_addr, bst, feats, q_chunk=150_000):
    """Like pass2_frame but keeps only q, p, p1, prob per chunk (bounded memory for test)."""
    uq = np.unique(df["q"].values)
    outs = []
    for c0 in range(0, len(uq), q_chunk):
        with Stage(f"pass2 predict chunk {c0}"):
            part = df[df["q"].isin(uq[c0:c0 + q_chunk])].reset_index(drop=True)
            part = group_features(part)
            part = pd.concat([part, sibling_features(part, pool_name, pool_addr)], axis=1)
            outs.append(pd.DataFrame({"q": part["q"].values, "p": part["p"].values, "p1": part["p1"].values,
                                      "prob": bst.predict(part[feats].values.astype(np.float32),
                                                          num_threads=C.LGB_PARAMS["num_threads"])}))
            del part
            gc.collect()
    return pd.concat(outs, ignore_index=True)
