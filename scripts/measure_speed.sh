#!/bin/bash
# Speed benchmark: one 1-hour RVS-Ego video at 0.5 FPS (1800 frames), 100 questions
# injected mid-stream, 64-token questions and 128-token answers. Reports Video Enc. (FPS)
# and Latency (s/question), separately.
#
# Runs baseline and two stage-2 thresholds for each model given:
#   scripts/measure_speed.sh                    # llava_ov_0.5b llava_ov_7b
#   scripts/measure_speed.sh llava_ov_7b        # just the 7B
#
# Frames are decoded in-process with decord, the same way the eval does -- there is no
# pre-extraction step to run first. The decode is one batched call before timing starts, so
# it does not move the reported numbers, but NUM_FRAMES frames stay in RAM for the run
# (~11 GB at 1800 1080p frames) -- lower NUM_FRAMES if the box cannot hold that.
#
# Budget: at ENCODE_CHUNK_SIZE=1 encoding is ~2x slower than the batched default, so allow
# ~5 min encoding per 0.5b run and considerably more for the 7B, plus QA. `--skip_qa true`
# in EXTRA drops QA entirely -- answering never mutates the video KV-Cache, so Video Enc.
# is identical either way and this is the fast path when only throughput is wanted.
# NUM_QUESTIONS=25 also cuts QA time to a quarter without moving the latency mean much.
set -euo pipefail
cd "$(dirname "$0")/.."

# cd alone does not put the repo root on sys.path: the benchmark is launched as
# `python video_qa/measure_encoding_fps.py`, so sys.path[0] is video_qa/. Without this the
# `video_qa`/`model` imports resolve through the Rekv env's editable rekv-1.0 install, which
# maps both packages to a different clone -- the benchmark would time that tree's model code
# instead of this one's. See the same note in scripts/eval.sh.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

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
# Must be large enough that every arm reaches steady state, which is the only regime the
# throughput number is measured over. The n_local=15000-token window fills after
# 15000 / (196 * keep_rate) frames -- 77 at baseline, but 379 at a keep rate of 0.20,
# because pruning means each frame contributes fewer tokens. A 300-frame run therefore
# never leaves the warm-up on the aggressive arms and reports the cold start as if it were
# the sustained rate. 800 leaves ~400 steady frames at the lowest keep rate seen here.
#
# It is still a subset: at the default 0.5 fps this is 1600 s of the source video, not the
# whole 3600 s that 1800 frames covered. That matters because ego4d video_idx 0 is a
# 2.21 GB 1080p file that trips decord's threaded decoder and falls back to
# single-threaded decoding, so decode is the dominant cost of this script.
NUM_FRAMES=${NUM_FRAMES:-800}
NUM_QUESTIONS=${NUM_QUESTIONS:-100}
ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-1}    # 1 = frame-by-frame, as the paper describes
GPU_PREPROCESS=${GPU_PREPROCESS:-true}       # same work, GPU implementation; see note above
PRUNE_METHOD=${PRUNE_METHOD:-rlt}
OUT_DIR=${OUT_DIR:-results/speed}
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
        --encode_chunk_size ${ENCODE_CHUNK_SIZE} \
        --gpu_preprocess ${GPU_PREPROCESS} \
        ${EXTRA} \
        ${flags} \
        --save_path "${OUT_DIR}/${model}-${label}.csv"
  done
done
