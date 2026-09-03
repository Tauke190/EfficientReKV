#!/bin/bash
# StreamingBench evaluation. Run scripts/prepare_streamingbench.sh first.
#
# Four subsets are wired up. Proactive Output is not: it is scored on *when* the model
# speaks against a ground-truth timestamp, not on a letter, and needs its own solver.
#
#   streamingbench_real     2500 q / 500 videos / 10 tasks
#
#  Omni-Source Understanding task is not supported because Rekv vision only and that requires audio also

num_chunks=1  # must equal the number of visible GPUs; one worker per GPU
# llava_ov_0.5b exists for smoke tests; its StreamingBench numbers are not worth reporting.
model=llava_ov_7b

# Space-separated. Overridable:  DATASETS="streamingbench_real" scripts/eval_streamingbench.sh
datasets=${DATASETS:-"streamingbench_real"}

sample_fps=1

# --- stage 2: token pruning (model/token_pruning.py) ---------------------------------

PRUNE_METHOD=${PRUNE_METHOD:-rlt}
PRUNE_METRIC=${PRUNE_METRIC:-cosine}
PRUNE_REFRESH_EVERY=${PRUNE_REFRESH_EVERY:-0}
PRUNE_THRESHOLDS=${PRUNE_THRESHOLDS:-""}

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
