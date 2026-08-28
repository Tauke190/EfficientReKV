#!/bin/bash
# ReKV + RLT token reduction (llava_ov_* only). Each config writes to its own results
# dir. Use eval.sh for the untouched baseline.
#
# Two independent stages -- set either, both, or neither. Measured split of encode cost
# (RVS-Ego @0.5fps, llava_ov_0.5b, GPU preprocessing): preprocessing 2.5%, vision tower
# 13.6%, LM prefill 84.0%.
#
#   PRUNE_*  (stage 2) drops redundant tokens before the LM. Attacks the 84%, and shrinks
#            KV RAM and retrieval cost with it. Measured +29% frames/s alone.
#   VISION_* (stage 1) skips SigLIP work on unchanged patches. Attacks the 13.6% only;
#            KV size is unchanged and it stores reused (approximate) features, so it costs
#            accuracy for little throughput. Measured +1.6% alone, +58% with stage 2.
#
# Thresholds are on different scales and must be calibrated separately, against accuracy.
set -euo pipefail
cd "$(dirname "$0")/.."   # run_eval.py resolves video_qa/, data/, results/ from the repo root

NUM_CHUNKS=1              # parallel procs, 1 GPU each
MODEL=llava_ov_0.5b
DATASETS="rvs_ego rvs_movie"
SAMPLE_FPS=0.5
N_LOCAL=15000             # sliding window, tokens
RETRIEVE_SIZE=64          # blocks retrieved per question
DEBUG=false

VISION_METHOD=none        # registry key in model/vision_reduction.py; 'none' = off
VISION_MASK_SPACE=embed   # embed (SigLIP patch embeddings) | pixel
VISION_METRIC=cosine      # cosine = chunk-size invariant; l2 is not
VISION_REFRESH_EVERY=0    # force-encode every Nth frame in full, 0 = off
# Stage-1 threshold. Redundancy is a direct function of frame spacing and of the footage,
# so it does not transfer between SAMPLE_FPS values or datasets. Measured (patches encoded
# / mean relative error in the encoder output):
#   MLVU movie clip @1fps    2e-4 -> 65%/0.34   1e-3 -> 53%/0.52   2e-3 -> 47%/0.60
#   RVS-Ego @0.5fps          1e-3 -> 96%        5e-3 -> 92%        0.05 -> 75%
# Egocentric footage at 0.5fps is nearly incompressible -- constant camera motion at
# 2-second spacing leaves almost nothing to reuse. VISION_REFRESH_EVERY=4 buys back a
# meaningful chunk of fidelity for a few points of keep rate.
VISION_THRESHOLD=0.001

# --- stage 2: memory-side (model/token_pruning.py) -- the one that moves throughput ----
PRUNE_METHOD=rlt          # registry key in model/token_pruning.py; 'none' = off
PRUNE_METRIC=cosine       # cosine = chunk-size invariant; l2 is not
PRUNE_REFRESH_EVERY=0     # force-keep every Nth frame, 0 = off
# On projected+pooled LLM-space tokens -- ~100x looser than VISION_THRESHOLD. Scales with
# SAMPLE_FPS: fewer frames/sec = less redundancy = higher threshold. Measured keep rate on
# MLVU (4 videos):
#   @0.5fps  0.03->94%  0.1->83%  0.2->71%  0.3->56%  0.4->40%
#   @4.0fps  0.03->71%  0.1->47%  0.2->30%  0.3->19%  0.4->12%
PRUNE_THRESHOLDS="0.25"

# Per-video keep rates land in results.csv (see BaseVQA.reduction_stats). Echo the
# aggregate here too, so the log answers "what did this threshold actually drop?" without
# opening the CSV -- and flag videos that fit entirely inside n_local, where retrieval
# never runs and the accuracy number is not testing what it looks like it is testing.
summarize_reduction () {
  python - "$1" <<'PYEOF'
import sys, os, pandas as pd
d = sys.argv[1]; f = os.path.join(d, 'results.csv')
if not os.path.exists(f):
    print(f'  (no results.csv in {d})'); raise SystemExit
df = pd.read_csv(f)
if 'token_keep_rate' not in df.columns and 'patch_keep_rate' not in df.columns:
    print('  reduction: none (baseline run)'); raise SystemExit
v = df.drop_duplicates('video_id')
print(f'  videos: {len(v)}   questions: {len(df)}')
for col, label in (('token_keep_rate', 'tokens kept  '), ('patch_keep_rate', 'patches enc. ')):
    if col in v.columns:
        print(f'  {label}: mean {100*v[col].mean():5.1f}%   '
              f'min {100*v[col].min():5.1f}%   max {100*v[col].max():5.1f}%')
if {'tokens_kept', 'tokens_seen'} <= set(v.columns):
    print(f'  tokens overall: {int(v.tokens_kept.sum())}/{int(v.tokens_seen.sum())}'
          f'  ({100*v.tokens_kept.sum()/max(v.tokens_seen.sum(),1):.1f}%)')
if {'n_tokens_fed', 'n_local'} <= set(v.columns):
    below = int((v.n_tokens_fed <= v.n_local).sum())
    if below:
        print(f'  WARNING: {below}/{len(v)} videos fit inside n_local -- no offload, no '
              f'retrieval on those')
PYEOF
}

for DATASET in ${DATASETS}; do
  for THR in ${PRUNE_THRESHOLDS}; do
    echo "=== ${MODEL} | ${DATASET} | s1=${VISION_METHOD} | s2=${PRUNE_METHOD}@${THR} ==="
    python -m video_qa.run_eval \
        --num_chunks ${NUM_CHUNKS} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --sample_fps ${SAMPLE_FPS} \
        --n_local ${N_LOCAL} \
        --retrieve_size ${RETRIEVE_SIZE} \
        --debug ${DEBUG} \
        --vision_method ${VISION_METHOD} \
        --vision_threshold ${VISION_THRESHOLD} \
        --vision_mask_space ${VISION_MASK_SPACE} \
        --vision_metric ${VISION_METRIC} \
        --vision_refresh_every ${VISION_REFRESH_EVERY} \
        --prune_method ${PRUNE_METHOD} \
        --prune_threshold ${THR} \
        --prune_metric ${PRUNE_METRIC} \
        --prune_refresh_every ${PRUNE_REFRESH_EVERY}

    TAG=""
    [ "${VISION_METHOD}" != "none" ] && TAG="${TAG}-v${VISION_METHOD}${VISION_THRESHOLD}${VISION_METRIC}"
    [ "${PRUNE_METHOD}" != "none" ] && TAG="${TAG}-${PRUNE_METHOD}${THR}${PRUNE_METRIC}"
    echo "--- reduction summary | ${DATASET} | thr=${THR} ---"
    summarize_reduction "results/${MODEL}/${DATASET}/${RETRIEVE_SIZE}-${SAMPLE_FPS}${TAG}"
  done
done
