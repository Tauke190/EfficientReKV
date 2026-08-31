#!/bin/bash
#SBATCH --job-name=rekv_ovobench
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_ovobench-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# OVO-Bench baseline grid: both wired-up modes x both models x the stage-2 arms, the same
# shape as scripts/eval_odvbench_baseline.slurm.
#
# What is being run. OVO-Bench asks its questions *mid-stream*: a query carries a
# `realtime` timestamp and may only be answered from frames in [0, t]. ReKV honours that
# natively -- one ingestion pass per video, questioned in between, with the question and
# its answer kept out of the KV-Cache (video_qa/rekv_ovobench_vqa.py) -- so a video is
# encoded once no matter how many queries it carries. Two modes are wired up here:
#     ovobench_realtime  237 videos, 837 queries, 22 h of footage   (OCR ACR ATR STU FPD OJR)
#     ovobench_backward  275 videos, 631 queries, 18 h              (EPM ASI HLD)
# The third, FAR (REC/SSR/CRR), is not wired up: those tasks are Yes/No-or-count rather
# than multiple choice and need their own prompts and scorer, so run_eval has no
# ovobench_forward.
#
# Scoring is video_qa/eval/eval_ovobench.py, which is not eval_multiple_choice.py for two
# reasons: OVO-Bench's headline number is the mean of the per-task accuracies inside a
# mode (so FPD's 101 queries count as much as OCR's 149; pooling would reweight the
# benchmark), and it aborts rather than scores if any row saw a frame past its own
# timestamp. It also writes ovo_results.json in OVO-Bench's own format, so a number here
# can be reproduced with their scorer instead of trusted from ours.
#
# Why SAMPLE_FPS=1 (see the comment on the knob below for the alternatives): it is the
# rate at which this benchmark actually exercises ReKV. n_local=15000 tokens is ~76 frames
# of llava_ov context, so a query only reaches past the sliding window -- i.e. only
# involves retrieval at all -- once its visible prefix is longer than that:
#     0.5 fps: median  83/89 frames visible, 53%/52% of queries past n_local
#       1 fps: median 165/178 frames visible, 76%/70% past n_local
#       2 fps: median 329/355 frames visible, 89%/82% past n_local
# (realtime/backward). 0.5 fps also leaves 5 realtime queries with two frames or fewer to
# answer from. Note that the runs already in results/ were 0.5 fps, and 1 fps writes to a
# different directory (64-1.0), so nothing here overwrites them -- but the two are not
# comparable either.
#
# Cost. 78k + 65k = 143k frames to encode per (model, arm) pass over both modes at 1 fps.
# llava_ov_0.5b sustains ~6-7 frames/s per GPU end to end, so that is ~1.5 h on 4 GPUs;
# llava_ov_7b is roughly 3x slower, so ~4.5 h. The grid below is 2 models x 3 arms, i.e.
# about 18 h in total. Cut MODELS or ARMS to run it in pieces.

module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate Rekv

LOGDIR=/home/av354855/EfficientReKV/slurmlogs

export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi
free -g

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

MODELS="llava_ov_0.5b llava_ov_7b"
DATASETS="ovobench_realtime ovobench_backward"

# 1 fps: see the note in the header. 0.5 halves the cost but leaves nearly half the
# queries inside the local window, where the run measures the backbone rather than ReKV's
# memory; 2 doubles it for a smaller further gain.
SAMPLE_FPS=1

N_LOCAL=15000
RETRIEVE_SIZE=64

# Slots decoded at a time. OVO-Bench videos run to 1695 s and up to 8 MB/frame at source
# resolution, so a whole-video decode (DECODE_WINDOW=0, the old behaviour) can hold 14 GB
# of pixels per worker on the longest video; a window bounds that at ~2 GB. The frames it
# hands the model are identical either way.
DECODE_WINDOW=256

# Stage-2 memory-side arms, as in scripts/eval_odvbench_baseline.slurm: "<method>:<threshold>",
# 'none:0' being the untouched baseline. Unlike odvbench, retrieval genuinely fires here
# (see the n_local figures above), so these arms are measured on both of the things
# pruning touches -- what the model can perceive, and what retrieval has left to search.
ARMS="none:0 rlt:0.25 rlt:0.5"
PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

# GPUs this job asked for; must match #SBATCH --gres=gpu:N above.
GPUS=2
# Host RAM all workers together may occupy, out of the 256G the job is allocated. ReKV
# holds the whole KV-Cache in pinned host memory for the length of a video, and the pool
# does not come back between videos, so a worker's peak is set by the longest video it
# touches: MEM_FIXED_GB + MEM_GB_PER_FRAME x frames, both measured for llava_ov_0.5b and
# scaled by per-token KV for llava_ov_7b. At 1 fps over the longest OVO-Bench video that
# is ~30 GB per 0.5b worker and ~60 GB per 7b worker, so the 7b arms give up a GPU.
MEM_BUDGET_GB=200
MAX_STREAM_SEC=1695   # longest last-query timestamp in either mode

for f in data/ovo_bench/realtime.json data/ovo_bench/backward.json; do
    if [ ! -f "${f}" ]; then
        echo "FATAL: ${f} missing -- build both first:" >&2
        echo "  python video_qa/convert_ovobench.py" >&2
        exit 1
    fi
done

for MODEL in ${MODELS}; do
  case "${MODEL}" in
      llava_ov_0.5b) MEM_FIXED_GB=24 ; MEM_GB_PER_FRAME=0.0035 ;;
      llava_ov_7b)   MEM_FIXED_GB=32 ; MEM_GB_PER_FRAME=0.0163 ;;
      *) echo "FATAL: no memory figures for ${MODEL}; measure them before running." >&2
         exit 1 ;;
  esac
  read PER_WORKER_GB NUM_CHUNKS <<< "$(awk -v fps="${SAMPLE_FPS}" -v sec="${MAX_STREAM_SEC}" \
      -v fixed="${MEM_FIXED_GB}" -v slope="${MEM_GB_PER_FRAME}" \
      -v budget="${MEM_BUDGET_GB}" -v gpus="${GPUS}" 'BEGIN{
        per = fixed + fps * sec * slope;
        n = int(budget / per); if (n > gpus) n = gpus;
        printf "%.0f %d", per, n }')"

  if [ "${NUM_CHUNKS}" -lt 1 ]; then
    echo "=== SKIP ${MODEL}: one worker would peak near ${PER_WORKER_GB} GB on the longest"
    echo "=== video, over the ${MEM_BUDGET_GB} GB budget for the whole node."
    continue
  fi
  if [ "${NUM_CHUNKS}" -lt "${GPUS}" ]; then
    echo "NOTE: ${MODEL} peaks near ${PER_WORKER_GB} GB per worker, so it runs on"
    echo "      ${NUM_CHUNKS} of ${GPUS} GPUs to stay inside ${MEM_BUDGET_GB} GB."
  fi

  for DATASET in ${DATASETS}; do
    for ARM in ${ARMS}; do
      METHOD="${ARM%%:*}"
      THR="${ARM##*:}"

      REDUCE_ARGS=""
      if [ "${METHOD}" != "none" ]; then
          REDUCE_ARGS="--prune_method ${METHOD} --prune_threshold ${THR} \
              --prune_metric ${PRUNE_METRIC} --prune_refresh_every ${PRUNE_REFRESH_EVERY}"
      fi

      echo "=========================================================================="
      echo "=== ${MODEL} | ${DATASET} | ${SAMPLE_FPS} fps | s1=none | s2=${METHOD}@${THR} ==="
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
          ${REDUCE_ARGS}
      echo "=== finished $(date) ==="
      free -g
    done
  done
done

echo "Job finished at $(date)"

conda deactivate
