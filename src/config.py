SEED = 42

# train/val subsample (fraction of train S1 per country); val is a disjoint slice
TRAIN_FRAC = 0.10
VAL_FRAC = 0.04
MIN_PARTITION = 50  # country labels with fewer records fall back to the whole pool

# blocking
S1_CHUNK = 2000
POOL_CHUNK = 1_000_000
NAME_MAX_DF = 2000   # absolute doc freq; more frequent tokens are not used for retrieval
ADDR_MAX_DF = 2000
K_NAME = 20   # retrieval depth per list (wide net)
K_ADDR = 30
K_COMB = 40
A_NAME = 5    # per-list rank caps used for the final candidate set
A_ADDR = 10
A_COMB = 15
MAX_CAND = 20       # b3 rank-cap selection (kept for reference)
STAGE1_RECALL = 0.995   # fraction of wide-net true pairs the learned pruner must keep (val)
STAGE1_MAX_CAND = 40
TOPK_JOBS = 4
BLOCK_VERSION = "b4"  # bump when blocking/normalization changes (cache key)
FEAT_VERSION = "f3"
STACK_VERSION = "s2"

# model
LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=SEED,
                  verbose=-1, num_threads=6)
NUM_ROUNDS = 1500
EARLY_STOP = 50

# decision (tuned on validation; overwritten by tuning output in work/decision.json when present)
THR = 0.5
REL = 0.0
