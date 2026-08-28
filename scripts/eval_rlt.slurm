#!/bin/bash
#SBATCH --job-name=rekv_eval_rlt
#SBATCH --output=/home/av354855/EfficientVideoXLPro/slurmlogs/rekv_eval_rlt-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# ReKV + encoder-side RLT token reduction (llava_ov_* only). Sweeps thresholds.
# See scripts/eval_rlt.sh for what the knobs mean and how to calibrate them.
module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate Rekv
cd /home/av354855/EfficientVideoXLPro/ReKV

LOGDIR=/home/av354855/EfficientVideoXLPro/slurmlogs

# Bail out on a full home quota rather than burning a GPU allocation on a run whose
# results can't be written. This cannot catch the quota being full at *submit* time:
# slurmd opens the --output file before this script runs, so a full quota kills the batch
# step in ~1s with ExitCode 0:53 and no .out file at all. A job that vanishes leaving no
# log is a quota problem, not a bug in this script -- check quota before debugging here.
if ! touch "${LOGDIR}/.writetest" 2>/dev/null; then
    echo "FATAL: ${LOGDIR} not writable -- home quota is full. Free space and resubmit." >&2
    exit 1
fi
rm -f "${LOGDIR}/.writetest"

# Keep the log small. Left unset, matplotlib picks an inline-image backend that dumps
# base64 PNGs straight into stdout and tqdm writes a line per update -- together those
# grew the older apt_videoxlpro logs to 25MB each, feeding the quota problem above.
export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

NUM_CHUNKS=4              # must match --gres=gpu:N above
MODEL=llava_ov_0.5b
DATASETS="rvs_ego rvs_movie"
SAMPLE_FPS=0.5
N_LOCAL=15000
RETRIEVE_SIZE=64
DEBUG=false

VISION_METHOD=none         # stage 1, encoder-side; 'none' = off (see eval_rlt.sh)
VISION_MASK_SPACE=embed
VISION_METRIC=cosine       # cosine = chunk-size invariant; l2 is not
VISION_REFRESH_EVERY=0     # force-encode every Nth frame in full, 0 = off
VISION_THRESHOLD=0.001     # does NOT transfer across SAMPLE_FPS or datasets

PRUNE_METHOD=rlt           # stage 2, memory-side -- the one that moves throughput
PRUNE_METRIC=cosine        # cosine = chunk-size invariant; l2 is not
PRUNE_REFRESH_EVERY=0
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

conda deactivate
