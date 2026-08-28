#!/bin/bash
# OVO-Bench evaluation. Run scripts/prepare_ovobench.sh first.
#
# Only the realtime (RVP) and backward (BT) modes are wired up. FAR (REC/SSR/CRR) is
# Yes/No-or-count rather than multiple choice and needs its own solver.

num_chunks=1  # must equal the number of visible GPUs; one worker per GPU
# llava_ov_0.5b exists for smoke tests; its OVO-Bench numbers are not worth reporting.
model=llava_ov_7b

# Space-separated: ovobench_realtime ovobench_backward
datasets="ovobench_backward"

# 1 FPS rather than the 0.5 used for RVS: OVO-Bench's fine-grained realtime tasks (OCR,
# ATR) key on detail that a 2-second stride drops. The cost is real -- the longest query
# sits at 1695 s, so ~1700 frames x 196 tokens for that one video.
sample_fps=1

# --- stage 2: token pruning (model/token_pruning.py) ---------------------------------
# Empty PRUNE_THRESHOLDS = the untouched baseline, byte-for-byte the command line that
# produced results/${model}/ovobench_realtime/64-1.0. Set a space-separated list to sweep:
#
#     PRUNE_THRESHOLDS="0.1 0.2 0.3" scripts/eval_ovobench.sh
#
# Each threshold writes to its own directory (run_eval.reduction_tag appends e.g.
# '-rlt0.2cosine'), so a sweep never overwrites the baseline or the previous arm.
#
# DO NOT transplant the 0.25/0.5 values from scripts/measure_speed.sh. Those were measured
# on RVS-Ego at 0.5 fps, and this run is at 1 fps -- frames one second apart are far more
# alike than frames two seconds apart, so the same threshold prunes considerably harder
# here. Measured keep rate on MLVU shows how steeply it moves with frame spacing:
#     @0.5fps  0.03->94%  0.1->83%  0.2->71%  0.3->56%  0.4->40%
#     @4.0fps  0.03->71%  0.1->47%  0.2->30%  0.3->19%  0.4->12%
# At 1 fps expect to land between those rows. Start at 0.1-0.2 and read the achieved keep
# rate off the summary below rather than assuming the threshold got you what you wanted.
#
# Stage 1 (--vision_method) is deliberately not offered here: it leaves the KV-Cache size
# unchanged, so on a memory-bound benchmark it buys nothing this script would show.
PRUNE_METHOD=${PRUNE_METHOD:-rlt}
PRUNE_METRIC=${PRUNE_METRIC:-cosine}
PRUNE_REFRESH_EVERY=${PRUNE_REFRESH_EVERY:-0}
PRUNE_THRESHOLDS=${PRUNE_THRESHOLDS:-""}

# What the threshold actually did. A threshold is an input, not a result -- redundancy
# varies enormously between videos, so the same value gives different keep rates on
# different footage, and an accuracy delta is unreadable without the rate it came with.
summarize_reduction () {
  python - "$1" <<'PYEOF'
import sys, os, pandas as pd
f = os.path.join(sys.argv[1], 'results.csv')
if not os.path.exists(f):
    print(f'  (no results.csv in {sys.argv[1]})'); raise SystemExit
df = pd.read_csv(f)
if 'token_keep_rate' not in df.columns:
    print('  reduction: none (baseline run)'); raise SystemExit
v = df.drop_duplicates('video_id')
print(f'  videos {len(v)}  questions {len(df)}')
print(f'  tokens kept: mean {100*v.token_keep_rate.mean():5.1f}%  '
      f'min {100*v.token_keep_rate.min():5.1f}%  max {100*v.token_keep_rate.max():5.1f}%')
# Queries whose whole visible window fits inside n_local never trigger retrieval, so
# their accuracy is not testing the thing this benchmark is for.
if {'n_tokens_fed', 'n_local'} <= set(df.columns):
    inside = (df.n_tokens_fed <= df.n_local).sum()
    print(f'  queries never reaching retrieval: {inside}/{len(df)} '
          f'({100*inside/len(df):.0f}%)')
PYEOF
}

# run_eval builds the results path from Python's own formatting of these values
# (reduction_tag uses '%g' for the threshold, and sample_fps arrives as a float), so derive
# the directory the same way instead of pasting strings together -- '0.5' would otherwise
# become '0.5.0', and '0.20' would not match Python's '0.2'.
FPS_TAG=$(python -c "print(float('${sample_fps}'))")
thr_tag () { python -c "print('%g' % float('$1'))"; }

run_arm () {   # $1 = dataset, remaining args = reduction flags
    local dataset=$1; shift
    python -m video_qa.run_eval \
        --num_chunks $num_chunks \
        --model ${model} \
        --dataset ${dataset} \
        --sample_fps ${sample_fps} \
        --n_local 15000 \
        --retrieve_size 64 \
        "$@"
}

for dataset in ${datasets}; do
    if [ -z "${PRUNE_THRESHOLDS}" ]; then
        echo "=== ${model} | ${dataset} | baseline ==="
        run_arm ${dataset}
        summarize_reduction "results/${model}/${dataset}/64-${FPS_TAG}"
    else
        for thr in ${PRUNE_THRESHOLDS}; do
            echo "=== ${model} | ${dataset} | ${PRUNE_METHOD} ${thr} ${PRUNE_METRIC} ==="
            run_arm ${dataset} \
                --prune_method ${PRUNE_METHOD} \
                --prune_threshold ${thr} \
                --prune_metric ${PRUNE_METRIC} \
                --prune_refresh_every ${PRUNE_REFRESH_EVERY}
            summarize_reduction \
                "results/${model}/${dataset}/64-${FPS_TAG}-${PRUNE_METHOD}$(thr_tag ${thr})${PRUNE_METRIC}"
        done
    fi
done
