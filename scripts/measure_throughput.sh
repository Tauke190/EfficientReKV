#!/bin/bash
# Streaming ingestion throughput: frames/s the model can absorb, per reduction arm.
#
# Measures encode only (--skip_qa) at encode_chunk_size=1, i.e. one frame per forward
# pass, which is what "streaming" means here. Reports steady-state FPS -- the rate once
# the local window is full -- which is the number to quote; the warm-up chunks before
# that run faster than the sustained rate and would flatter the result.
#
# Uses FPS-Bench-Stream videos rather than the RVS-Ego default: they are 720p/30fps/600s
# and normalised, so a 200 s span decodes in seconds, whereas ego4d video_idx 0 is a
# 2.21 GB 1080p file that trips decord's threaded decoder and falls back to
# single-threaded decoding (~17 min of dead decode per run).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

MODELS=${MODELS:-"llava_ov_0.5b llava_ov_7b"}
VIDEO_IDXS=${VIDEO_IDXS:-"0 1 2"}
ANNO=${ANNO:-data/fpsbench_stream/test_mc.json}
SAMPLE_FPS=${SAMPLE_FPS:-2}
NUM_FRAMES=${NUM_FRAMES:-400}
ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-1}
GPU_PREPROCESS=${GPU_PREPROCESS:-true}
OUT_DIR=${OUT_DIR:-results/throughput}

# THRESHOLDS is the sweep axis; baseline is always measured first because every other
# number here is only meaningful as a ratio against it on the same GPU.
THRESHOLDS=${THRESHOLDS:-"0.25 0.5 0.6 0.7 0.8 0.9"}
CONFIGS=("baseline|")
for thr in ${THRESHOLDS}; do
  CONFIGS+=("rlt${thr}|--prune_method rlt --prune_threshold ${thr}")
done

mkdir -p "${OUT_DIR}"

for model in ${MODELS}; do
  for idx in ${VIDEO_IDXS}; do
    for entry in "${CONFIGS[@]}"; do
      label="${entry%%|*}"
      flags="${entry#*|}"
      out="${OUT_DIR}/${model}-v${idx}-${label}.csv"
      if [ -s "${out}" ]; then
        echo "=== skip (exists): ${out}"
        continue
      fi
      echo "=== ${model} | video ${idx} | ${label} | ${NUM_FRAMES} frames @ ${SAMPLE_FPS} fps ==="
      python video_qa/measure_encoding_fps.py \
          --model "${model}" \
          --anno_path "${ANNO}" \
          --video_idx "${idx}" \
          --sample_fps "${SAMPLE_FPS}" \
          --num_frames "${NUM_FRAMES}" \
          --encode_chunk_size "${ENCODE_CHUNK_SIZE}" \
          --gpu_preprocess "${GPU_PREPROCESS}" \
          --skip_qa true \
          ${flags} \
          --save_path "${out}"
    done
  done
done

echo
echo "=== summary ==="
python scripts/summarize_throughput.py --dir "${OUT_DIR}"
