#!/bin/bash
#SBATCH --job-name=rekv_pareto_0.5b
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_pareto_0.5b-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# The high end of the memory-side pruning sweep for llava_ov_0.5b: rlt at cosine
# thresholds 0.6, 0.7, 0.8 and 0.9, on ovobench_realtime, ovobench_backward and
# odvbench. One model, one axis -- everything else is held at the values the rest
# of the grid uses (64 retrieve, 1 fps, 256 decode window, cosine, no refresh), so
# these points extend the existing 0.25/0.5 runs upward on the same curve.
#
# What the numbers are. --prune_threshold is a *distance* cutoff, not a rate: a
# visual token is kept iff its cosine distance to the feature its position was
# last kept with exceeds the threshold (model/token_pruning.py). Raising it drops
# more, so accuracy falls and the KV-Cache shrinks -- the achieved keep rate lands
# in results.csv per question (token_keep_rate), and is what the accuracy here has
# to be plotted against. There is no flag that targets a keep rate directly.
#
# Why 1 fps on odvbench. Its clips are 5-90 s, so 1 fps gives an early question
# very few frames and 2+ fps would be the better protocol in isolation -- but the
# 0.25/0.5 odvbench runs these extend were measured at 1 fps, and a Pareto curve
# that mixes two frame rates is not a curve. Change it only by re-running the
# baseline and the 0.25/0.5 arms to match.
#
# Reading the results. The baseline (no --prune_* flags) and the blind floor for
# each pair come from scripts/eval_remaining_grid.slurm and sit in the same
# results/llava_ov_0.5b/<dataset>/ directory as 64-1.0 and 64-1.0-blind; these
# runs land beside them as 64-1.0-rlt<thr>cosine.

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

# Space-separated, and it must stay quoted: a bare `A=x y` assignment runs `y` as a
# command with A in its environment and leaves A empty in the script.
DATASETS="ovobench_backward odvbench"

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

# Per-worker memory model for llava_ov_0.5b. Combined with each dataset's longest stream
# below, this sets the peak. Pruning only ever shrinks the cache, so sizing off the
# unpruned figures is the conservative choice.
MEM_FIXED_GB=24
MEM_GB_PER_FRAME=0.0035

# Set to 0 to force a re-run of anything already on disk.
SKIP_EXISTING=1

for DATASET in ${DATASETS}; do
    # Longest last-query timestamp in each annotation, which is what sets the per-worker
    # memory peak, plus the annotation the run needs and where its results go.
    case "${DATASET}" in
        ovobench_realtime)
            SUBDIR=ovobench_realtime ; MAX_STREAM_SEC=1695
            ANNO=data/ovo_bench/realtime.json
            BUILD="python video_qa/convert_ovobench.py" ;;
        ovobench_backward)
            SUBDIR=ovobench_backward ; MAX_STREAM_SEC=1198
            ANNO=data/ovo_bench/backward.json
            BUILD="python video_qa/convert_ovobench.py" ;;
        odvbench)
            SUBDIR=odvbench ; MAX_STREAM_SEC=90
            ANNO=data/odvbench/full_mc.json
            BUILD="python scripts/setup_odvbench.py" ;;
        *) echo "FATAL: unknown dataset ${DATASET}" >&2 ; exit 1 ;;
    esac

    if [ ! -f "${ANNO}" ]; then
        echo "FATAL: ${ANNO} missing -- build the annotations first:" >&2
        echo "  ${BUILD}" >&2
        exit 1
    fi

    read PER_WORKER_GB NUM_CHUNKS <<< "$(awk -v fps="${SAMPLE_FPS}" -v sec="${MAX_STREAM_SEC}" \
        -v fixed="${MEM_FIXED_GB}" -v slope="${MEM_GB_PER_FRAME}" \
        -v budget="${MEM_BUDGET_GB}" -v gpus="${GPUS}" 'BEGIN{
          per = fixed + fps * sec * slope;
          n = int(budget / per); if (n > gpus) n = gpus;
          printf "%.0f %d", per, n }')"
    if [ "${NUM_CHUNKS}" -lt 1 ]; then
        echo "=== SKIP ${DATASET}: one worker would peak near ${PER_WORKER_GB} GB,"
        echo "=== over the ${MEM_BUDGET_GB} GB budget for the whole job."
        continue
    fi
    if [ "${NUM_CHUNKS}" -lt "${GPUS}" ]; then
        echo "NOTE: ${MODEL} on ${DATASET} peaks near ${PER_WORKER_GB} GB per worker, so it"
        echo "      runs on ${NUM_CHUNKS} of ${GPUS} GPUs to stay inside ${MEM_BUDGET_GB} GB."
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
done

echo "Job finished at $(date)"

conda deactivate
