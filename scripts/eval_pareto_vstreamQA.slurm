#!/bin/bash
#SBATCH --job-name=rekv_pareto_rvs
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_pareto_rvs-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# The high end of the memory-side pruning sweep on VStream-QA: rlt at cosine thresholds
# 0.6, 0.7, 0.8 and 0.9, on rvs_ego and rvs_movie, for both models. One axis -- everything
# else is held at the values the rest of the grid uses (64 retrieve, 1 fps, cosine, no
# refresh), so these points sit on the same curve as the 0.25/0.5 arms. 2 models x 2
# datasets x 4 thresholds = 16 runs.
#
# Scoring is deliberately NOT run here. rvs_* answers are free-form, so a number only
# exists once a judge has read them, and the judge has to be the same one across every arm
# or the table compares judges rather than thresholds. Each run therefore stops at
# results.csv (--skip_scoring) and the sweep is scored afterwards in one pass:
#
#     OPENAI_API_KEY=... scripts/score_llmjudge_gpt.sh \
#         results/llava_ov_{0.5b,7b}/rvs_{ego,movie}/64-1.0-rlt*cosine/results.csv
#
# (scripts/score_open_ended.sh is the local-judge equivalent, no API key; its scores are
# self-consistent but not comparable to published gpt-3.5-turbo numbers.)
#
# Reading the results. These land in results/<model>/rvs_{ego,movie}/ as
# 64-1.0-rlt<thr>cosine.
#
# NOTE: nothing rvs is on disk yet -- results/ holds only odvbench and ovobench_*. This
# gives the 0.6-0.9 end of a curve that still has no unpruned baseline (no --prune_* flags,
# 64-1.0), no blind floor (64-1.0-blind) and no 0.25/0.5 arms; all of those have to be run
# before anything can be plotted. And per the README, thresholds do not transfer across
# datasets -- 0.6-0.9 was calibrated on ovobench_realtime, so on rvs this may already be
# deep in the tail. One 0.25/0.5 arm anchors that.

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

MODELS="llava_ov_0.5b llava_ov_7b"

DATASETS="rvs_ego rvs_movie"

SAMPLE_FPS=1
N_LOCAL=15000
RETRIEVE_SIZE=64

PRUNE_METHOD=rlt
PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

# The swept axis. Cosine-distance thresholds, ascending: more is dropped as it rises.
PRUNE_THRESHOLDS="0.6 0.7 0.8 0.9"

# No --decode_window here on purpose. run_eval forwards it only to the streaming datasets
# (stream_args, called from the ovobench/odvbench/fpsbench branches); the rvs branches do
# not pass it on, so the flag would be accepted and silently dropped. The solver's own
# default is already 256 (video_qa/base.py), which is what the rest of the grid ran at.

# Must match #SBATCH --gres=gpu:N above.
GPUS=4

# Host RAM all workers together may occupy. ReKV pins the whole KV-Cache for the length of
# a video and the pool does not come back between videos, so a worker's peak is set by the
# longest video it touches. Bounded by the job's own allocation, not the node's:
# --cpus-per-gpu=8 x --mem-per-cpu=8G x 4 GPUs = 256 GB, so this leaves ~26 GB of headroom.
MEM_BUDGET_GB=230

# Set to 0 to force a re-run of anything already on disk.
SKIP_EXISTING=1

for MODEL in ${MODELS}; do
    # Per-worker memory model: MEM_FIXED_GB + frames x MEM_GB_PER_FRAME, combined with each
    # dataset's longest stream below. Pruning only ever shrinks the cache, so sizing off the
    # unpruned figures is the conservative choice.
    #
    # These MUST track MODEL. The 0.5b pair undercounts a 7b worker by ~2x, and because
    # NUM_CHUNKS is budget/per_worker, undercounting the worker OVER-schedules the GPUs.
    case "${MODEL}" in
        llava_ov_0.5b) MEM_FIXED_GB=24 ; MEM_GB_PER_FRAME=0.0035 ;;
        llava_ov_7b)   MEM_FIXED_GB=32 ; MEM_GB_PER_FRAME=0.0163 ;;
        *) echo "FATAL: no memory figures for ${MODEL}; measure them before running." >&2
           exit 1 ;;
    esac

    for DATASET in ${DATASETS}; do
        case "${DATASET}" in
            rvs_ego)   SUBDIR=rvs_ego   ; ANNO=data/rvs/ego/ego4d_oe.json ;;
            rvs_movie) SUBDIR=rvs_movie ; ANNO=data/rvs/movie/movienet_oe.json ;;
            *) echo "FATAL: unknown dataset ${DATASET}" >&2 ; exit 1 ;;
        esac

        # rvs_* annotations ship with the repo -- unlike ovobench/odvbench there is no
        # build step to point at.
        if [ ! -f "${ANNO}" ]; then
            echo "FATAL: ${ANNO} missing; rvs_* annotations ship with the repo." >&2
            exit 1
        fi

        # Longest last-question timestamp in the annotation, which is what sets the
        # per-worker peak: rekv_stream_vqa encodes only up to end_time x sample_fps of each
        # question and the cursor never rewinds, so nothing past the last question of the
        # longest video is ever encoded.
        #
        # Derived rather than hard-coded like the ovobench figures. rvs streams are an
        # order of magnitude longer than the ones this script was originally sized for, and
        # a stale constant here is exactly what over-schedules the GPUs.
        MAX_STREAM_SEC="$(python - "${ANNO}" <<'PY'
import json, math, sys
anno = json.load(open(sys.argv[1]))
print(math.ceil(max(max(float(c['end_time']) for c in v['conversations']) for v in anno)))
PY
)"
        if ! [ "${MAX_STREAM_SEC}" -gt 0 ] 2>/dev/null; then
            echo "FATAL: could not read the longest stream out of ${ANNO}." >&2
            exit 1
        fi

        read PER_WORKER_GB NUM_CHUNKS <<< "$(awk -v fps="${SAMPLE_FPS}" -v sec="${MAX_STREAM_SEC}" \
            -v fixed="${MEM_FIXED_GB}" -v slope="${MEM_GB_PER_FRAME}" \
            -v budget="${MEM_BUDGET_GB}" -v gpus="${GPUS}" 'BEGIN{
              per = fixed + fps * sec * slope;
              n = int(budget / per); if (n > gpus) n = gpus;
              printf "%.0f %d", per, n }')"
        if [ "${NUM_CHUNKS}" -lt 1 ]; then
            echo "=== SKIP ${MODEL} | ${DATASET}: one worker would peak near ${PER_WORKER_GB} GB,"
            echo "=== over the ${MEM_BUDGET_GB} GB budget for the whole job."
            continue
        fi
        if [ "${NUM_CHUNKS}" -lt "${GPUS}" ]; then
            echo "NOTE: ${MODEL} on ${DATASET} (longest stream ${MAX_STREAM_SEC} s) peaks near"
            echo "      ${PER_WORKER_GB} GB per worker, so it runs on ${NUM_CHUNKS} of ${GPUS} GPUs"
            echo "      to stay inside ${MEM_BUDGET_GB} GB."
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
                --prune_method ${PRUNE_METHOD} \
                --prune_threshold ${THR} \
                --prune_metric ${PRUNE_METRIC} \
                --prune_refresh_every ${PRUNE_REFRESH_EVERY} \
                --skip_scoring
            echo "=== finished $(date) ==="
            free -g
        done
    done
done

echo "Job finished at $(date)"

conda deactivate
