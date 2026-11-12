#!/bin/bash
# Token retention of the three ablation pruners (rlt_ref / rlt_prev / rlt_frame) across the
# pruning-threshold grid -- the keep-rate curve the accuracy ablation
# (scripts/eval/ablation.slurm) sits against.
#
# This needs no LLM and no QA run: stage-2 pruning decides keep/drop from the vision
# features alone, so retention is measured straight off the encoder output. A smoke test on
# one video is seconds; the whole StreamBench benchmark is minutes, not the GPU-day the
# accuracy ablation costs. See scripts/efficiency/token_retention_ablation.py.
#
# Needs a GPU. On a cluster:
#   srun -p gpu --gres=gpu:1 bash scripts/efficiency/token_retention_ablation.sh
#
# Smoke test ONE video first, then the full benchmark:
#   VIDEO=/path/to/test.mp4 bash scripts/efficiency/token_retention_ablation.sh   # smoke
#   bash scripts/efficiency/token_retention_ablation.sh                           # full
#
# Overrides (env vars):
#   VIDEO       single video file -> smoke test on it alone (default: sweep ANNO)
#   ANNO        annotation to sweep              (default: data/streambench/full_oe.json)
#   NUM_VIDEOS  stratified subsample, <=0 = all  (default: -1; ignored with VIDEO)
#   MAX_FRAMES  cap frames per clip, 0 = no cap  (default: 0; set e.g. 120 for a fast smoke)
#   MODEL       backbone                         (default: llava_ov_7b)
#   METHODS     pruners to compare               (default: "rlt_ref rlt_prev rlt_frame")
#   THRESHOLDS  grid                             (default: "0.25 0.5 0.6 0.7 0.8 0.9")
#   FPS         sampling rate                    (default: 1; StreamBench is scored at 1)
#   METRIC      cosine | l2                      (default: cosine)
#   OUT_DIR     where the CSVs go  (default: results/analysis/token_retention_ablation)

set -euo pipefail

cd "$(dirname "$0")/../.."
# The repo root has to be on sys.path, not just be the cwd -- see scripts/eval/eval.slurm:
# without it the workers import the other ReKV clone's code.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg

VIDEO=${VIDEO:-}
ANNO=${ANNO:-data/streambench/full_oe.json}
NUM_VIDEOS=${NUM_VIDEOS:--1}
MAX_FRAMES=${MAX_FRAMES:-0}
MODEL=${MODEL:-llava_ov_7b}
METHODS=${METHODS:-"rlt_ref rlt_prev rlt_frame"}
THRESHOLDS=${THRESHOLDS:-"0.25 0.5 0.6 0.7 0.8 0.9"}
FPS=${FPS:-1}
METRIC=${METRIC:-cosine}
OUT_DIR=${OUT_DIR:-results/analysis/token_retention_ablation}

# A single VIDEO picks the smoke-test path (one clip, ANNO/NUM_VIDEOS ignored); otherwise
# sweep the annotation.
if [ -n "${VIDEO}" ]; then
  source_args=(--video "${VIDEO}")
else
  source_args=(--anno_path "${ANNO}" --num_videos "${NUM_VIDEOS}")
fi

python scripts/efficiency/token_retention_ablation.py \
    "${source_args[@]}" \
    --model "${MODEL}" \
    --methods ${METHODS} \
    --thresholds ${THRESHOLDS} \
    --sample_fps "${FPS}" \
    --prune_metric "${METRIC}" \
    --max_frames "${MAX_FRAMES}" \
    --out_dir "${OUT_DIR}"
