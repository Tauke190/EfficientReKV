#!/bin/bash
#SBATCH --job-name=rekv_remaining
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_remaining-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# The 11 runs still missing from the grid, all at 1 fps:
#   llava_ov_0.5b  ovobench_realtime   blind, baseline, rlt 0.25, rlt 0.5
#   llava_ov_0.5b  ovobench_backward   blind, baseline, rlt 0.25, rlt 0.5
#   llava_ov_0.5b  odvbench            blind, baseline
#   llava_ov_7b    ovobench_backward   rlt 0.5
# Blind goes first inside each group: it is the cheapest run and it is the floor the
# sighted numbers in that group have to be read against.

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

SAMPLE_FPS=1
N_LOCAL=15000
RETRIEVE_SIZE=64
DECODE_WINDOW=256
PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

# Must match #SBATCH --gres=gpu:N above.
GPUS=2

# Host RAM all workers together may occupy. ReKV pins the whole KV-Cache for the length of
# a video and the pool does not come back between videos, so a worker's peak is set by the
# longest video it touches. Bounded by the job's own allocation, not the node's:
# --cpus-per-gpu=8 x --mem-per-cpu=8G x 2 GPUs = 128 GB, so this leaves ~18 GB of headroom.
MEM_BUDGET_GB=110

# Set to 0 to force a re-run of anything already on disk.
SKIP_EXISTING=1

# model|dataset|arm   with arm one of: blind, none, rlt:<threshold>
RUNS="
llava_ov_0.5b|ovobench_realtime|blind
llava_ov_0.5b|ovobench_realtime|none
llava_ov_0.5b|ovobench_realtime|rlt:0.25
llava_ov_0.5b|ovobench_realtime|rlt:0.5
llava_ov_0.5b|ovobench_backward|blind
llava_ov_0.5b|ovobench_backward|none
llava_ov_0.5b|ovobench_backward|rlt:0.25
llava_ov_0.5b|ovobench_backward|rlt:0.5
llava_ov_0.5b|odvbench|blind
llava_ov_0.5b|odvbench|none
llava_ov_7b|ovobench_backward|rlt:0.5
"

for f in data/ovo_bench/realtime.json data/ovo_bench/backward.json data/odvbench/full_mc.json; do
    if [ ! -f "${f}" ]; then
        echo "FATAL: ${f} missing -- build the annotations first:" >&2
        echo "  python video_qa/convert_ovobench.py" >&2
        echo "  python scripts/setup_odvbench.py" >&2
        exit 1
    fi
done

for RUN in ${RUNS}; do
    MODEL="$(echo "${RUN}" | cut -d'|' -f1)"
    DATASET="$(echo "${RUN}" | cut -d'|' -f2)"
    ARM="$(echo "${RUN}" | cut -d'|' -f3)"
    METHOD="${ARM%%:*}"
    THR="${ARM##*:}"

    case "${MODEL}" in
        llava_ov_0.5b) MEM_FIXED_GB=24 ; MEM_GB_PER_FRAME=0.0035 ;;
        llava_ov_7b)   MEM_FIXED_GB=32 ; MEM_GB_PER_FRAME=0.0163 ;;
        *) echo "FATAL: no memory figures for ${MODEL}; measure them before running." >&2
           exit 1 ;;
    esac

    # Longest last-query timestamp in each annotation; sets the per-worker memory peak.
    case "${DATASET}" in
        ovobench_realtime) MAX_STREAM_SEC=1695 ; SUBDIR=ovobench_realtime ;;
        ovobench_backward) MAX_STREAM_SEC=1198 ; SUBDIR=ovobench_backward ;;
        odvbench)          MAX_STREAM_SEC=90   ; SUBDIR=odvbench ;;
        *) echo "FATAL: unknown dataset ${DATASET}" >&2 ; exit 1 ;;
    esac

    REDUCE_ARGS=""
    BLIND_ARGS=""
    TAG=""
    if [ "${METHOD}" = "blind" ]; then
        BLIND_ARGS="--blind"
        TAG="-blind"
    elif [ "${METHOD}" != "none" ]; then
        # A bare --prune_threshold already reads as "enable rlt", so the baseline arm must
        # pass no reduction flags at all rather than a threshold of 0.
        REDUCE_ARGS="--prune_method ${METHOD} --prune_threshold ${THR} \
            --prune_metric ${PRUNE_METRIC} --prune_refresh_every ${PRUNE_REFRESH_EVERY}"
        TAG="-${METHOD}${THR}${PRUNE_METRIC}"
    fi

    SAVE_DIR="results/${MODEL}/${SUBDIR}/${RETRIEVE_SIZE}-${SAMPLE_FPS}.0${TAG}"
    if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${SAVE_DIR}/results.csv" ]; then
        echo "=== SKIP (already on disk): ${SAVE_DIR}"
        continue
    fi

    if [ "${METHOD}" = "blind" ]; then
        # No frame is decoded and no KV-Cache is built, so the memory model does not apply.
        NUM_CHUNKS=${GPUS}
        PER_WORKER_GB=0
    else
        read PER_WORKER_GB NUM_CHUNKS <<< "$(awk -v fps="${SAMPLE_FPS}" -v sec="${MAX_STREAM_SEC}" \
            -v fixed="${MEM_FIXED_GB}" -v slope="${MEM_GB_PER_FRAME}" \
            -v budget="${MEM_BUDGET_GB}" -v gpus="${GPUS}" 'BEGIN{
              per = fixed + fps * sec * slope;
              n = int(budget / per); if (n > gpus) n = gpus;
              printf "%.0f %d", per, n }')"
        if [ "${NUM_CHUNKS}" -lt 1 ]; then
            echo "=== SKIP ${MODEL} | ${DATASET}: one worker would peak near ${PER_WORKER_GB} GB,"
            echo "=== over the ${MEM_BUDGET_GB} GB budget for the whole node."
            continue
        fi
        if [ "${NUM_CHUNKS}" -lt "${GPUS}" ]; then
            echo "NOTE: ${MODEL} peaks near ${PER_WORKER_GB} GB per worker, so it runs on"
            echo "      ${NUM_CHUNKS} of ${GPUS} GPUs to stay inside ${MEM_BUDGET_GB} GB."
        fi
    fi

    echo "=========================================================================="
    echo "=== ${MODEL} | ${DATASET} | ${SAMPLE_FPS} fps | arm=${ARM} ==="
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
        ${BLIND_ARGS} ${REDUCE_ARGS}
    echo "=== finished $(date) ==="
    free -g
done

echo "Job finished at $(date)"

conda deactivate
