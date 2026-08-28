#!/bin/bash
# Score open-ended (free-form answer) predictions with a LOCAL judge -- no API key.
# For rvs_ego / rvs_movie / qaego4d / activitynet_qa, whose answers cannot be scored by
# string match.
#
# Takes the results.csv files to score:
#   scripts/score_open_ended.sh results/llava_ov_0.5b/rvs_ego/64-0.5/results.csv
#   scripts/score_open_ended.sh results/llava_ov_0.5b/rvs_{ego,movie}/64-0.5/results.csv
# bash scripts/score_open_ended.sh /home/av354855/EfficientVideoXLPro/ReKV/results/llava_ov_0.5b/rvs_{ego,movie}/64-0.5-rlt0.25cosine/results.csv
#
# ---- picking the judge ---------------------------------------------------------------
# The default is Prometheus 2 (prometheus-eval/prometheus-7b-v2.0): an open model trained
# specifically to grade responses against a rubric, rather than a general chat model asked
# to behave like a grader. Swap it per run with JUDGE=<preset>:
#
#   JUDGE=prometheus       scripts/score_open_ended.sh <csv>   # default, 7B
#   JUDGE=prometheus8x7b   scripts/score_open_ended.sh <csv>   # 8x7B MoE, ~94 GB in bf16
#   JUDGE=qwen             scripts/score_open_ended.sh <csv>   # the previous default, 32B
#   JUDGE=qwen7b           scripts/score_open_ended.sh <csv>
#   JUDGE=some-org/some-instruct-model scripts/score_open_ended.sh <csv>   # any HF id
#
# Each preset carries its own output paths (below), so judges never overwrite each other
# and re-running a judge you already used costs nothing. Bringing an old judge back is
# just the env var -- its verdicts are still sitting in their own directory.
#
# Judges are NOT interchangeable as numbers. Prometheus grades 1-5 against a rubric and
# its yes/no is a threshold on that (YES_THRESHOLD, default 4); the qwen style asks for a
# yes/no plus a 0-5 score directly. Compare arms scored by the same judge; a table mixing
# the two is comparing judges, not models. The judge id is stamped into every verdict, and
# the scorer aborts rather than topping up a cache another judge wrote.
#
# ---- outputs -------------------------------------------------------------------------
# Verdicts land next to each CSV, with a per-item cache alongside (re-running skips what is
# already cached):
#   prometheus     -> results_local_prometheus.json      tmp_local_prometheus/
#   prometheus8x7b -> results_local_prometheus8x7b.json  tmp_local_prometheus8x7b/
#   qwen           -> results_local.json                 tmp_local/        (legacy paths)
# The qwen preset deliberately keeps the unsuffixed names: existing caches stay valid and
# blind/compare_blind.py reads results_local.json. API-judge outputs (results.json / tmp/)
# are untouched by all of them.
#
# For rvs_ego / rvs_movie this also prints a per-answer_type breakdown under the pooled
# Accuracy. Read it: 50.6% of RVS-Ego and 35.7% of RVS-Movie are categories the benchmark
# itself marks (Y/N), which have a ~54% majority-class floor no matter what the model sees,
# so the pooled figure averages a coin-flip with a real task. The labels are rejoined from
# the official release (video_qa/answer_types.py) because ReKV's data conversion drops
# them. Re-running a fully cached CSV costs nothing and prints the breakdown, so old
# results need no re-judging.
#
# The eval itself writes results.csv and stops there under --skip_scoring, so this is a
# separate step: no video inference is repeated, and the judge can be swapped freely.
#
# Overridable from the environment:
#   JUDGE          preset name or any HF id (see above)
#   JUDGE_MODEL    checkpoint, overriding the preset's
#   JUDGE_STYLE    auto | prometheus | qwen -- the prompt/parser pair, overriding the preset
#   JUDGE_SUFFIX   output-path suffix, overriding the preset's
#   YES_THRESHOLD  prometheus only: lowest rubric score counted correct (default 4)
#   BATCH_SIZE     prompts per forward pass (default 16)
set -euo pipefail

JUDGE=${JUDGE:-prometheus}
BATCH_SIZE=${BATCH_SIZE:-16}
YES_THRESHOLD=${YES_THRESHOLD:-4}

if [ $# -eq 0 ]; then
  echo "usage: [JUDGE=prometheus|prometheus8x7b|qwen|qwen7b|<hf-id>] $0 <results.csv> [results.csv ...]" >&2
  exit 1
fi

# Resolve before the cd, so relative paths work from wherever this was invoked.
CSVS=()
for arg in "$@"; do
  CSVS+=("$(realpath "${arg}")")
done
cd "$(dirname "$0")/.."

# The preset table lives in video_qa/eval/judges.py (PRESETS), not here: run_eval.py reads
# the same table, and a second copy in this file would eventually send the two entry points
# to different output paths for the same judge. An unknown JUDGE comes back as a bare
# checkpoint with style 'auto'.
IFS=$'\t' read -r preset_style preset_model preset_suffix < <(python video_qa/eval/judges.py "${JUDGE}")

JUDGE_MODEL=${JUDGE_MODEL:-${preset_model}}
JUDGE_STYLE=${JUDGE_STYLE:-${preset_style}}
JUDGE_SUFFIX=${JUDGE_SUFFIX-${preset_suffix}}   # no ':' -- an empty JUDGE_SUFFIX is a choice

# Compute nodes often have no outbound network. Fetch the judge once, on a node that
# does, then everything below runs offline:
#   hf download ${JUDGE_MODEL}
# (use `hf`, not the deprecated `huggingface-cli` wrapper -- it crashes on this box's
#  latin-1 locale while printing its own deprecation warning)

for csv in "${CSVS[@]}"; do
  if [ ! -f "${csv}" ]; then
    echo "skip ${csv}: not found" >&2
    continue
  fi
  dir=$(dirname "${csv}")
  echo "=== scoring ${csv} with ${JUDGE_MODEL} (${JUDGE_STYLE}) ==="
  python video_qa/eval/eval_open_ended_local.py \
      --pred_path "${csv}" \
      --output_dir "${dir}/tmp_local${JUDGE_SUFFIX}" \
      --output_json "${dir}/results_local${JUDGE_SUFFIX}.json" \
      --judge_style "${JUDGE_STYLE}" \
      --judge_model "${JUDGE_MODEL}" \
      --yes_threshold ${YES_THRESHOLD} \
      --batch_size ${BATCH_SIZE}
done
