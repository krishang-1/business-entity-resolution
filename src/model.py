import json
import lightgbm as lgb
import numpy as np

import config as C

BASE_FEATS = ["name_cos", "addr_cos", "comb", "from_name", "from_addr", "from_comb", "src_s3"]


def train(X, y, groups, feats, work, name="lgb"):
    """Train LightGBM with early stopping on a held-out 10% of S1 groups."""
    rng = np.random.RandomState(C.SEED)
    ug = np.unique(groups)
    hold = set(rng.choice(ug, size=max(1, len(ug) // 10), replace=False))
    m = np.fromiter((g in hold for g in groups), bool, len(groups))
    dtr = lgb.Dataset(X[~m], y[~m], feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(X[m], y[m], feature_name=feats, reference=dtr)
    bst = lgb.train(C.LGB_PARAMS, dtr, C.NUM_ROUNDS, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(C.EARLY_STOP, verbose=False), lgb.log_evaluation(200)])
    bst.save_model(f"{work}/{name}.txt")
    json.dump(feats, open(f"{work}/{name}_features.json" if name != "lgb" else f"{work}/features.json", "w"))
    imp = sorted(zip(feats, bst.feature_importance("gain")), key=lambda x: -x[1])
    print("best_iter", bst.best_iteration, "top gain:", [(f, int(g)) for f, g in imp[:15]], flush=True)
    return bst


def load(work):
    return lgb.Booster(model_file=f"{work}/lgb.txt"), json.load(open(f"{work}/features.json"))
