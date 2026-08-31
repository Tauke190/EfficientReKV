#!/bin/bash
#SBATCH --job-name=rekv_pareto_0.5b
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_pareto_0.5b-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# The high end of the memory-side pruning sweep for llava_ov_0.5b on OVO-Bench
# realtime: rlt at cosine thresholds 0.6, 0.7, 0.8 and 0.9. One model, one
# benchmark, one axis -- everything else is held at the values the rest of the
# grid uses (64 retrieve, 1 fps, 256 decode window, cosine, no refresh), so these
# four points extend the existing 0.25/0.5 runs upward on the same curve.
#
# What the numbers are. --prune_threshold is a *distance* cutoff, not a rate: a
# visual token is kept iff its cosine distance to the feature its position was
# last kept with exceeds the threshold (model/token_pruning.py). Raising it drops
# more, so accuracy falls and the KV-Cache shrinks -- the achieved keep rate is
# reported per video in the worker logs, and is what the accuracy here has to be
# plotted against. There is no flag that targets a keep rate directly.
#
# Reading the results. The baseline (no --prune_* flags) and the blind floor for
# this model/benchmark pair come from scripts/eval_remaining_grid.slurm and sit in
# the same results/llava_ov_0.5b/ovobench_realtime/ directory as 64-1.0 and
# 64-1.0-blind; these runs land beside them as 64-1.0-rlt<thr>cosine.

module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate Rekv

export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi
free -g

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

MODEL=llava_ov_0.5b
DATASET=ovobench_realtime
SUBDIR=ovobench_realtime

SAMPLE_FPS=1
N_LOCAL=15000
RETRIEVE_SIZE=64
DECODE_WINDOW=256

PRUNE_METHOD=rlt
PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

# The swept axis. Cosine-distance thresholds, ascending: more is dropped as it rises.
PRUNE_THRESHOLDS="0.6 0.7 0.8 0.9"

# Must match #SBATCH --gres=gpu:N above.
GPUS=2

# Host RAM all workers together may occupy. ReKV pins the whole KV-Cache for the length of
# a video and the pool does not come back between videos, so a worker's peak is set by the
# longest video it touches. Bounded by the job's own allocation, not the node's:
# --cpus-per-gpu=8 x --mem-per-cpu=8G x 2 GPUs = 128 GB, so this leaves ~18 GB of headroom.
MEM_BUDGET_GB=110

# Per-worker memory model for this model, and the longest last-query timestamp in
# realtime.json -- the two together set the peak. Pruning only ever shrinks the cache,
# so sizing off the unpruned figures is the conservative choice.
MEM_FIXED_GB=24
MEM_GB_PER_FRAME=0.0035
MAX_STREAM_SEC=1695

# Set to 0 to force a re-run of anything already on disk.
SKIP_EXISTING=1

if [ ! -f data/ovo_bench/realtime.json ]; then
    echo "FATAL: data/ovo_bench/realtime.json missing -- build the annotations first:" >&2
    echo "  python video_qa/convert_ovobench.py" >&2
    exit 1
fi

read PER_WORKER_GB NUM_CHUNKS <<< "$(awk -v fps="${SAMPLE_FPS}" -v sec="${MAX_STREAM_SEC}" \
    -v fixed="${MEM_FIXED_GB}" -v slope="${MEM_GB_PER_FRAME}" \
    -v budget="${MEM_BUDGET_GB}" -v gpus="${GPUS}" 'BEGIN{
      per = fixed + fps * sec * slope;
      n = int(budget / per); if (n > gpus) n = gpus;
      printf "%.0f %d", per, n }')"
if [ "${NUM_CHUNKS}" -lt 1 ]; then
    echo "FATAL: one worker would peak near ${PER_WORKER_GB} GB, over the" >&2
    echo "       ${MEM_BUDGET_GB} GB budget for the whole job." >&2
    exit 1
fi
if [ "${NUM_CHUNKS}" -lt "${GPUS}" ]; then
    echo "NOTE: ${MODEL} peaks near ${PER_WORKER_GB} GB per worker, so it runs on"
    echo "      ${NUM_CHUNKS} of ${GPUS} GPUs to stay inside ${MEM_BUDGET_GB} GB."
fi

for THR in ${PRUNE_THRESHOLDS}; do
    TAG="-${PRUNE_METHOD}${THR}${PRUNE_METRIC}"
    SAVE_DIR="results/${MODEL}/${SUBDIR}/${RETRIEVE_SIZE}-${SAMPLE_FPS}.0${TAG}"
    if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${SAVE_DIR}/results.csv" ]; then
        echo "=== SKIP (already on disk): ${SAVE_DIR}"
        continue
    fi

    echo "=========================================================================="
    echo "=== ${MODEL} | ${DATASET} | ${SAMPLE_FPS} fps | ${PRUNE_METHOD} ${THR} ${PRUNE_METRIC} ==="
    echo "=== -> ${SAVE_DIR} ==="
    echo "=== ${NUM_CHUNKS} workers x ~${PER_WORKER_GB} GB host memory at the peak ==="
    echo "=== started $(date) ==="
    python -m video_qa.run_eval \
        --num_chunks ${NUM_CHUNKS} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --sample_fps ${SAMPLE_FPS} \
        --n_local ${N_LOCAL} \
        --retrieve_size ${RETRIEVE_SIZE} \
        --decode_window ${DECODE_WINDOW} \
        --prune_method ${PRUNE_METHOD} \
        --prune_threshold ${THR} \
        --prune_metric ${PRUNE_METRIC} \
        --prune_refresh_every ${PRUNE_REFRESH_EVERY}
    echo "=== finished $(date) ==="
    free -g
done

echo "Job finished at $(date)"

conda deactivate
