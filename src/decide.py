import numpy as np
import pandas as pd

from evaluate import f05


def decide(df, thr, rel, one_owner=True):
    """df: q, p, prob. Returns filtered DataFrame of accepted (q, p)."""
    d = df[df["prob"] >= thr]
    if one_owner and len(d):
        d = d[d["prob"] >= d.groupby("p")["prob"].transform("max")]
        d = d[~d.duplicated("p")]
    if rel > 0 and len(d):
        d = d[d["prob"] >= rel * d.groupby("q")["prob"].transform("max")]
    return d


def fast_macro_f05(acc, true_cnt, q_all):
    """acc: accepted DataFrame with q and label (1 = true pair). true_cnt: Series q -> #true matches (0 for singletons)."""
    g = acc.groupby("q")["label"].agg(["sum", "size"])
    tc = true_cnt.reindex(q_all).fillna(0).values
    tp = g["sum"].reindex(q_all).fillna(0).values
    npred = g["size"].reindex(q_all).fillna(0).values
    f = np.zeros(len(q_all))
    f[(tc == 0) & (npred == 0)] = 1.0
    m = (tp > 0)
    P = np.where(m, tp / np.maximum(npred, 1), 0); R = np.where(m, tp / np.maximum(tc, 1), 0)
    f[m] = 1.25 * P[m] * R[m] / (0.25 * P[m] + R[m])
    return f.mean(), tp.sum() / max(npred.sum(), 1), tp.sum() / max(tc.sum(), 1)


def tune(df, true_cnt, q_all, one_owner=True):
    best = (-1, None)
    rows = []
    for thr in np.arange(0.20, 0.94, 0.04):
        for rel in [0.0, 0.5, 0.7]:
            acc = decide(df, thr, rel, one_owner)
            f, P, R = fast_macro_f05(acc, true_cnt, q_all)
            rows.append((round(thr, 2), rel, f, P, R))
            if f > best[0]:
                best = (f, (round(float(thr), 2), rel, P, R))
    return best, pd.DataFrame(rows, columns=["thr", "rel", "F05", "P", "R"])
