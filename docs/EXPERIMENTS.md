# Experiments log

## 16:10 IST — P0 EDA facts (measured)
- Machine: 8 cores, 7.8 GB RAM (only ~0.7 GB free at start; Chrome/Discord resident). Python 3.10 venv.
- Rows: train S1 2,206,821 (US 1,323,633 / India 883,188); S2 5,034,616; S3 5,285,603.
- Test S1 1,732,544 (India 809,986 / US 663,106 / France 259,452); S2 4,887,273 (France 703,378); S3 5,082,316 (France 731,615).
- Empty names: 0%. Empty address: S1 0%, S2/S3 ~3%. Median name len ~25, address ~40 chars.
- GT: singleton fraction 0.0558. Matches/S1: 0:123k 1:119k 2:375k 3:531k 4:484k 5:322k 6:165k 7:64k 8+:23k. Total pairs 7,638,365; S2 share 0.484.
- No S2/S3 id is matched to >1 S1 (max 1) -> one-owner assignment is valid.
- 81% of matched S1 have >1 match from the same source -> NO per-source top-1 rule.
- Unmatched fraction of pool: S2 0.266, S3 0.254.
- Noise seen: typos + leet (Principa1, Tayl0r, Fami1y), U+FFFD replacement chars, website names (johnsviking.com),
  junk names with correct address (Zephyumacalo), legal suffix add/drop/reorder, "Services"/"Center" suffixes, [brackets],
  address component reordering, ST/ROAD/DR abbreviations, state code vs full name (TX/Texas, RJ/Rajasthan) and Devanagari state names,
  "NULL" tokens, "##" prefixes, leading zeros (02620), number ranges (3100-3104), missing house numbers.
- France (test only): SARL/SAS/EURL/SCI/SA, RUE/R/R., BD, AV, BIS, 5-digit postcodes sometimes present, regions vs departments.

## 16:35 IST — blocking probe v1 (1% India train S1 = 8,832 vs 4,133,346 India pool; word tokens, df cap 2000)
config: K_NAME=10 K_ADDR=10, no combined list, no cap
timing: hash 83s, topk name 3.3s, topk addr 6.7s
blocking: pair_recall=0.890 ceiling_F05=0.955 mean_cands=18.3 p95=20
cap by comb rank: 6->0.725, 8->0.748, 10->0.756, 12->0.800, 15->0.873, 20->0.890
decision: too low. Misses = native Indian scripts (Tamil/Telugu/Bengali/Gujarati), TN->tennessee bug, common names above df cap.

## 16:57 IST — blocking probe v2 (same probe) + skeleton name token, state name->code, concat token always, combined list K_COMB=15
timing: hash 302s (contended with background normalization), topk name 4.8s, addr 7.3s, comb 13.0s; peak RSS 3.04 GB
blocking: pair_recall=0.916 ceiling_F05=0.966 mean_cands=21.8 median=22 p95=25 max=35
cap by comb rank: 6->0.772, 10->0.806, 15->0.834, 20->0.909
decision: keep. Next: per-list rank cap (B4), parallel hashing, recall per list, miss buckets.

## 17:25 IST — blocking config selection (K=20 per list; per-list rank caps simulated offline)
probes: India 1% (8,832 S1 vs 4,133,346 pool), US 0.7% (9,265 S1 vs 6,186,873 pool)
list recall @20: India name 0.528 / addr 0.834 / comb 0.866, union 0.928 (38.7 cands); US name 0.647 / addr 0.843 / comb 0.950, union 0.967 (38.5)
India misses (2,217 of 30,610): 16% empty pool address, 10% identical but common name (median 64 same-name pool records) with truncated address,
  rest = native-script transliterations (Tamil/Telugu/Bengali/Malayalam/Gujarati) and/or heavily truncated addresses.
chosen: a(name)=5 b(addr)=10 c(comb)=15 MAX_CAND=20 -> India recall 0.913 mean 19.1 p95 20; US recall 0.953 mean 18.1 p95 20
timing: single-process topk ~2.6 ms/S1 (India), ~3.4 ms/S1 (US 6.2M pool). Projected single-process test blocking ~70 min -> added 4-process topk.
  3% India probe (26,496 S1): topk 126 s total parallel (fixed spawn/pickle overhead ~110 s dominates at this size); peak RSS 3.9 GB (hashing).
decision: keep; rev_* (reverse-competition) features dropped for first submission (subsample bias), one-owner by probability only.

## 18:00 IST — run1: first LightGBM (train S1=220,683 / val S1=88,272 subsample = 10% / 4% per country x has-match, seed 42; blocked vs FULL per-country train pool)
config: a=5 b=10 c=15 MAX_CAND=20, df cap 2000, LGB lr0.05 leaves63 (hit 2000-round cap, still improving), no rev_* features
blocking (val): pair_recall=0.9354 ceiling_F05=0.9769 mean_cands=18.45 median=20 p95=20 max=20 total_pairs=1,628,671
timing: train blocking US 185,309 S1 354s, India 123,646 S1 409s (topk ~1.9 ms/S1 India with 4 procs); features ~140s/150k S1; LGB 684s; peak RSS 4.2 GB (LGB train)
val: F0.5=0.9491 P=0.9863 R=0.8932 singleton_acc=0.9223 | US=0.9626 India=0.9290 | m0=0.922 m1=0.880 m2=0.939 m3+=0.959
decision: thr=0.60 rel=0.7 one-owner on (no-one-owner F0.5=0.9490: little S1 competition inside subsample). Keep -> full run for first submission.
top gain: comb, a_tset, num_conflict, n_ratio, num_jacc, r_comb, n_partial, a_len_b, am_tset, n_nospace

## 18:37 IST — run1 test phase crashed (my error) -> run2
cause: features.py edited while run1 was running; spawned workers imported the new 31-col _pair_chunk vs 28 cols in the main process.
lesson: never edit src/ during a run. India test candidates (cands_test_India_b3_809986) were cached and are reused.
timing measured before the crash: full-subsample LGB retrain 1050s; test India blocking 870s (810k S1 vs 4.72M pool; topk 421s total); features ~118s per 150k S1.
run2 changes: feature v2 (+num_near, n_skel, n_contain, p_name_freq, q_name_freq, p_addr_freq), lr 0.08, 1500 rounds;
  full mode now reuses the validated train-split model (no retrain) to save ~17 min.

## run2 val — features v2 (same blocking, same subsample)
blocking (val): pair_recall=0.9354 ceiling_F05=0.9769 mean_cands=18.45 median=20 p95=20 max=20
val: F0.5=0.9512 P=0.9876 R=0.8983 singleton_acc=0.9363 | US=0.9638 India=0.9322 | m0=0.936 m1=0.875 m2=0.939 m3+=0.961
decision: thr=0.68 rel=0.7 one-owner on (off: 0.9511). LGB best_iter 479 (lr 0.08). vs run1: +0.0021 F0.5 -> keep.
top gain: comb, a_tset, r_comb, n_partial, num_jacc, n_ratio, num_conflict, am_tset, a_len_b, n_nospace, n_ntok_b, n_skel, nn_tset, p_addr_freq
test features v2: ~224 s per 150k S1 (India)

## 20:13 IST — run2 test output -> PROMOTED to output/ (validator PASS)
test blocking: India 809,986 S1 vs 4,717,565 pool (cached from run1, 870s); US 663,106 vs 3,817,031 522s; France 259,452 vs 1,434,993 153s; peak RSS 3.83 GB
test candidates/S1: mean 18.33 median 19 p95 20 max 20 | US 17.77, France 18.14, India 18.85
test predictions: mean matches/S1 3.21, empty 6.33% | US 3.34 / 5.82%, France 3.09 / 6.36%, India 3.14 / 6.74%  (France in line with US/India)
val (this model): F0.5=0.9512 P=0.9876 R=0.8983

## 20:29 IST — P4 measurements (val, run2 model; cached val features)
candidate shrinking (re-select from cached list ranks, recompute context feats, same model, re-tune decision):
  a5 b10 c15 mc20: mean 18.45 recall 0.9354 F0.5 0.9512 (current)
  a5 b10 c10 mc15: mean 14.29 recall 0.9259 F0.5 0.9478 (-0.0034)
  a5 b5  c10 mc12: mean 11.44 recall 0.9151 F0.5 0.9451
  a3 b5  c10 mc12: mean 11.16 recall 0.9172 F0.5 0.9458
  a5 b5  c5  mc10: mean  8.84 recall 0.8850 F0.5 0.9396
  a3 b5  c5  mc8 : mean  7.32 recall 0.8675 F0.5 0.9363
  decision: keep mc20 (every smaller set loses > 0.002).
leave-one-country-out (479 rounds, lr 0.08, decision re-tuned on target):
  US -> India: F0.5 0.8072 (P 0.932 R 0.714) vs in-domain 0.9322
  India -> US: F0.5 0.9413 (P 0.979 R 0.891) vs in-domain 0.9638
  note: India's native-script transliterations are unseen when training on US only; France (Latin script) resembles US. Final model trains on both.
20:30 IST — code frozen; clean end-to-end reproduction run started in fresh work_repro/ + output_repro/.

## 20:45 IST — Step 1 loss decomposition (val 88,272 S1, run2 model, thr .68 rel .7 one-owner)
F0.5=0.9512 | oracle ceiling (cands ∩ truth)=0.9769 | pair recall 0.9354 | cands mean 18.45 median 20 p95 20
loss points (total 0.0488): missed_by_blocking 0.0231 | model_FN 0.0148 | model_FP 0.0073 | singleton_FP 0.0036
by country (points of overall F0.5): India blk 0.0137 FN 0.0073 FP 0.0042 sFP 0.0019 | US blk 0.0094 FN 0.0075 FP 0.0031 sFP 0.0016
by true-count bucket: 0 -> sFP 0.0036 | 1 -> blk 0.0034 FN 0.0025 FP 0.0009 | 2 -> blk 0.0049 FN 0.0035 FP 0.0019 | 3+ -> blk 0.0148 FN 0.0088 FP 0.0045
model_FN pattern: probs 0.4-0.68 just under threshold; off-by-one house numbers (4313/4314, 911/910), junk names at correct address, empty address.
target 0.985 is above the blocking ceiling (0.977); Track A (re-blocking train+test + features, ~1.5-2 h here) cannot ship before 21:45 -> not attempted.

## 21:01 IST — Track B3: split thresholds THR_FIRST/THR_EXTRA + REL + one-owner (grid on cached val probs)
best: thr_first 0.68, thr_extra 0.53, rel 0.7 -> F0.5 0.9513 (current 0.9512). half-split held-out: +0.0002, +0.0000.
decision: REVERT (gain < 0.002). Current rule kept; output/ unchanged.

## 22:30 IST — P5 clean reproduction
command: python code/business_entity_resolution/src/run_pipeline.py --data-dir dataset --out-dir output_repro --work-dir work_repro --mode full (fresh empty work dir)
result: exit 0, runtime 6,793 s (113 min), peak RSS 4.17 GB; val F0.5 0.9512 (identical); outputs byte-identical to output/ (cmp), validator PASS.

## 23:30 IST — run3: Track A wide net + learned stage-1 blocker (BLOCK_VERSION b4, FEAT_VERSION f3)
config: K_NAME=20 K_ADDR=30 K_COMB=40, full union (no rank caps); stage-1 LightGBM (100 trees, 13 cheap blocking feats) trained on train-split S1,
  threshold = largest keeping >= 99.5% of wide-net val positives (+ always keep each S1's top-1 by comb and by name_cos), cap 40.
train wide-net blocking: US 185,309 S1 330s, India 123,646 S1 242s; peak RSS 3.72 GB
val wide net: pair_recall 0.9571, 58.8 cands/S1
stage-1 sweep (thr: kept-of-wide / pair_recall / mean cands / p95):
  0.5 0.863/0.826/3.87/7 | 0.3 0.917/0.878/4.33/7 | 0.2 0.944/0.903/4.68/8 | 0.1 0.961/0.920/5.08/8 | 0.07 0.967/0.925/5.30/8
  0.05 0.971/0.929/5.55/9 | 0.03 0.977/0.935/6.00/10 | 0.02 0.980/0.938/6.47/11 | 0.015 0.983/0.941/6.90/12 | 0.01 0.986/0.944/7.65/13
  0.007 0.988/0.946/8.44/15 | 0.005 0.990/0.947/9.41/17 | 0.003 0.993/0.950/11.79/22 | 0.002 0.994/0.952/14.58/28 | 0.001 0.997/0.954/21.37/40 (chosen)
blocking (val, final): pair_recall 0.9543 ceiling_F05 0.9836 mean 21.37 median 20 p95 40 max 40 (was 0.9354 / 0.9769 / 18.45)
val: F0.5=0.9558 P=0.9866 R=0.9130 (was 0.9512 / 0.9876 / 0.8983) thr 0.68 rel 0.0; one-owner off 0.9556. best_iter 443.
decision: KEEP (+0.0046). Test run in progress -> output/_tmp.
note: at thr 0.03 the stage-1 set matches the old recall (0.935) with 6.0 cands/S1 vs 18.45.

## 00:13 IST (28 Sep) — run3 test output -> PROMOTED (validator PASS incl. --check-ids); previous files kept in output/_backup_v2/
test blocking (stage-1 pruned inline): India 628s, US 438s, France 175s; peak RSS 3.85 GB
test candidates/S1: mean 23.38 median 22 p95 40 max 40 | US 19.61, India 22.39, France 36.08 (pruner less confident on unseen country -> more kept, often the 40 cap)
test predictions: mean 3.28 matches/S1, empty 6.08% | US 3.38/5.74%, France 3.19/5.90%, India 3.23/6.41%
val: F0.5=0.9558 P=0.9866 R=0.9130 singleton_acc=0.9339 | US 0.9677 India 0.9378 | m0 0.934 m1 0.883 m2 0.944 m3+ 0.966

## 00:20 IST (28 Sep) — package rebuilt for v3
team_submission.zip (271 MB): output/ (v3 files), code/business_entity_resolution/{src,README.md,requirements.txt}, Documentation_template.md (v3 numbers).
kaggle/entity_resolution_kaggle.ipynb regenerated from v3 src (compiles).
not done for v3: second clean end-to-end repro run (v2 repro was byte-identical, 113 min); LOCO re-run (v2 numbers reported as such).

## run4 — pass-2 stacking, p1-only features (5-fold OOF pass-1 on train rows, grouped by S1)
pass-2 feats: p1, p1_rank, p1_gap, s1_pmax, s1_p2, s1_n05, s1_n09, s1_psum, s1_ncand + sibling support (token_set name/addr
  vs the S1's top-3 other candidates with p1>=0.3: max sims, p1 of most similar sibling, count). OOF folds ~190-245 s each.
val: F0.5=0.9572 P=0.9873 R=0.9147 (pass-1 0.9558) -> +0.0014 < 0.002 -> NOT shipped.

## run5 — pass-2 with all 55 pass-1 features + 15 stacking features
pass-2 LGB best_iter 96; top gain: p1, p1_gap, s1_pmax, sib_both, sib_addr, p1_rank, a_tset, s1_psum, s1_p2, sib_addr_p
val: F0.5=0.9589 P=0.9877 R=0.9175 singleton_acc=0.9353 | US 0.9699 India 0.9424 | m0 0.935 m1 0.886 m2 0.948 m3+ 0.969
decision: thr 0.44 rel 0.7 one-owner. +0.0031 vs pass-1 -> KEEP; test recomputed with chained pass-1 -> pass-2 per chunk.

## Phase 1 — voiced/unvoiced skeleton merge (b/p d/t g/k j/c v/p z/s), 1% India probe (8,832 S1 vs 4,133,346 pool), K 20/30/40 union
off: wide recall 0.9376, 61.85 cands/S1, name-list recall 0.5275
on : wide recall 0.9377, 61.56 cands/S1, name-list recall 0.5294
gate (+0.003): FAIL -> default ER_SKEL_VOICE=0 (code kept behind the switch).

## Phase 2 — pool clustering (unsupervised, S2∪S3 per country) + cluster expansion
per-rule edge precision (5% owner-sampled slice): L1 name+addr 1.000/1.000 (IN/US), L4 addr+skel>=80 0.9999/1.000, L2 name+num 0.9857/0.9968, L5 skel+num 0.9723/0.9962, addr-only 0.9743/0.9838
(first component run had an index bug in key_groups -> 0.686; fixed.)
5% slice components: L1+L2+L4 India 0.9787 / US 0.9951 precision (recall 0.33/0.37); India L1+L4 0.9999 (recall 0.11)
FULL train pool (10.3M): L1+L2+L4 precision 0.9146 recall 0.342 (FAIL gate 0.98); L1+L4 precision 0.9997 recall 0.160
2c expansion (val 88,272 S1, cluster-mates of top-3 by stage-1 p, cap +8): current 0.9543 / ceiling 0.9836 / 21.37
  L1+L2+L4: 0.9604 / 0.9853 / 21.65 (added-pair precision 0.077)   L1+L4: 0.9554 / 0.9839 / 21.40
gate (+0.01 recall): FAIL both -> not integrated. Missed pairs are not exact-key duplicates of retrieved siblings.

## Missed-pair analysis (val, v5 blocking): 13,959 of 305,179 true pairs missed; 95% of affected S1 have a retrieved true sibling;
missed record vs best retrieved sibling token_set: >=95 name 0.443 / addr 0.464 / either 0.710; >=90 either 0.793; >=80 either 0.903; 28.6% of missed records have empty address.

## Second-hop retrieval probe (queries = top-3 stage-1 candidates, k=10 per list name/addr/comb, pool-only IDF), 10k val S1 per country
India: current 0.9315 / ceiling 0.9735 / 21.74 cands -> +3 new: 0.9456 / 0.9774 / 24.69 | +8: 0.9494 / 0.9786 / 29.34 | +20: 0.9531 / 0.9802 / 37.90
US:    current 0.9667 / ceiling 0.9894 / 21.09 cands -> +3 new: 0.9731 / 0.9908 / 24.02 | +8: 0.9747 / 0.9913 / 28.51 | +20: 0.9760 / 0.9917 / 36.20
new-pair precision at +3: India 0.016, US 0.007. Weighted (val is ~60% US / 40% India): recall ~+0.009, ceiling ~+0.0024 at +3 cands.
gate (+0.01 recall at <= +8 cands): borderline (India pass, US below). Not integrated yet.

## 11:10 IST (28 Sep) — v5 (pass-2 stacking) test output -> PROMOTED (validator PASS incl. --check-ids); v3 files kept in output/_backup_v3/
val: F0.5=0.9589 P=0.9877 R=0.9175 (thr 0.44, rel 0.7, one-owner) | candidates unchanged from v3 (mean 23.38, median 22, p95 40, max 40)
test predictions: mean 3.31 matches/S1, empty 5.96% | US 3.41/5.71%, France 3.20/5.90%, India 3.26/6.18%
