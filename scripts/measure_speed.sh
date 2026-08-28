#!/bin/bash
# Speed benchmark: one 1-hour RVS-Ego video at 0.5 FPS (1800 frames), 100 questions
# injected mid-stream, 64-token questions and 128-token answers. Reports Video Enc. (FPS)
# and Latency (s/question), separately.
#
# Runs baseline and two stage-2 thresholds for each model given:
#   scripts/measure_speed.sh                    # llava_ov_0.5b llava_ov_7b
#   scripts/measure_speed.sh llava_ov_7b        # just the 7B
#
# The first run extracts the video's frames with ffmpeg into FRAME_CACHE (~12 min, ~400 MB
# for 1800 frames); every run after that reuses them. Without the cache, decord decodes
# this stride at ~1 frame/s -- ~30 min per run, inside untimed calls, which looks like a
# hang rather than slow progress.
#
# Budget: at ENCODE_CHUNK_SIZE=1 encoding is ~2x slower than the batched default, so allow
# ~5 min encoding per 0.5b run and considerably more for the 7B, plus QA. `--skip_qa true`
# in EXTRA drops QA entirely -- answering never mutates the video KV-Cache, so Video Enc.
# is identical either way and this is the fast path when only throughput is wanted.
# NUM_QUESTIONS=25 also cuts QA time to a quarter without moving the latency mean much.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- measurement protocol -----------------------------------------------------------
# ENCODE_CHUNK_SIZE=1 is the strict frame-by-frame number: the paper describes frames as
# arriving one at a time, and batching 64 of them would add ~128 s of lag at 0.5 FPS input.
# It costs ~1.6x against the batched default because batch-1 kernels leave the GPU idle
# between launches -- that idle time is real for a true streaming deployment.
#
# GPU_PREPROCESS=true is a deliberate mixed choice, and worth stating whenever these
# numbers are reported. Batching changes what "streaming" MEANS, so it is set to the
# paper's semantics above. Preprocessing is only an IMPLEMENTATION of identical
# resize/normalize work -- the HF processor does it single-threaded on the CPU at ~37-57 ms
# per 1080p frame, this does the same thing on the GPU at ~2 ms. Measured on one A100 80GB,
# 768 frames of RVS-Ego, llava_ov_0.5b:
#
#     preprocess   chunk   baseline   prune 0.5   speedup
#     CPU              1      14.42       22.69     1.57x   <- directly comparable to a published 17 FPS
#     CPU             64      22.92           -         -
#     GPU             64      40.63       88.55     2.18x
#
# So absolute FPS here will sit ABOVE a published CPU-preprocessing figure. Say so when
# reporting it, and never quote a speedup from one protocol against a baseline from
# another -- the ratio itself moves, because preprocessing and per-launch overhead are
# fixed costs that token reduction cannot touch.
MODELS=${@:-"llava_ov_0.5b llava_ov_7b"}
NUM_FRAMES=${NUM_FRAMES:-1800}
NUM_QUESTIONS=${NUM_QUESTIONS:-100}
ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-1}    # 1 = frame-by-frame, as the paper describes
GPU_PREPROCESS=${GPU_PREPROCESS:-true}       # same work, GPU implementation; see note above
PRUNE_METHOD=${PRUNE_METHOD:-rlt}
OUT_DIR=${OUT_DIR:-results/speed}
FRAME_CACHE=${FRAME_CACHE:-/home/av354855/EfficientVideoXLPro/ReKV/data/frame_cache}
EXTRA=${EXTRA:-}                             # e.g. EXTRA="--skip_qa true"

mkdir -p "${OUT_DIR}"

# Stage 2 only. It is the reduction that moves this metric: ingesting a frame means
# prefilling its tokens through the LM to produce KV, and that call is the bulk of the
# encode cost, so feeding it fewer tokens is the only lever on it. Stage 1 (--vision_method)
# attacks the vision tower, a much smaller share, and is left out here rather than being
# reported as if it contributed; add a row below if you want it measured.
#   label            extra flags
CONFIGS=(
  "baseline|"
  "rlt0.25|--prune_method ${PRUNE_METHOD} --prune_threshold 0.25"
  "rlt0.5|--prune_method ${PRUNE_METHOD} --prune_threshold 0.5"
)

for model in ${MODELS}; do
  # Baseline first: every other number is only meaningful next to it on the same GPU.
  for entry in "${CONFIGS[@]}"; do
    label="${entry%%|*}"
    flags="${entry#*|}"
    echo "=== ${model} | ${label} | chunk=${ENCODE_CHUNK_SIZE} gpu_preprocess=${GPU_PREPROCESS} ==="
    python video_qa/measure_encoding_fps.py \
        --model "${model}" \
        --num_frames ${NUM_FRAMES} \
        --num_questions ${NUM_QUESTIONS} \
        --frame_cache_dir "${FRAME_CACHE}" \
        --encode_chunk_size ${ENCODE_CHUNK_SIZE} \
        --gpu_preprocess ${GPU_PREPROCESS} \
        ${EXTRA} \
        ${flags} \
        --save_path "${OUT_DIR}/${model}-${label}.csv"
  done
done
