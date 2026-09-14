#!/bin/bash
# Re-score finished StreamBench runs with a judge OTHER than upstream's.
#
# StreamBench's answers are sentences, so accuracy comes from an LLM judge, and the
# official one is meta-llama/Meta-Llama-3-8B-Instruct (StreamChat's
# eval_ego_streaming_with_llama3.py; reproduced prompt-for-prompt by judges.py's
# `streambench` preset, which run_eval uses by default). That is what makes a number
# comparable to the published table -- and also the reason to have this script: an 8B
# judge is the weakest link in the pipeline, and the only way to know whether a gap
# between two pruning arms is real or is the judge's noise is to grade the same
# predictions with a second, independent judge and check the ranking survives.
#
# This takes runs that ALREADY have results.csv and only re-grades them -- no GPU-hours
# of inference are repeated, and the official Llama-3 verdicts are left untouched. Every
# judge writes to its own cache, its own results_local*.json and its own
# streambench_scores_*.json, so nothing here can overwrite a published number.
#
# Usage -- pass results.csv paths, or the run directories holding them:
#   scripts/score_streambench_judge.sh results/llava_ov_7b/streambench/64-1.0-rlt0.5cosine
#   scripts/score_streambench_judge.sh results/llava_ov_7b/streambench/*/results.csv
#   JUDGE=prometheus scripts/score_streambench_judge.sh results/*/streambench/*/
#
# Knobs (environment):
#   JUDGE=qwen          preset from judges.PRESETS, or any HF id / local path.
#                       qwen | qwen7b | prometheus | prometheus8x7b | streambench
#   JUDGE_MODEL=...     swap the checkpoint, keep the preset's prompt/parser. Use this to
#                       run upstream's exact prompt through a different model:
#                         JUDGE=streambench JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
#   BATCH_SIZE=16       prompts per forward pass; lower it if the judge OOMs.
#   YES_THRESHOLD=4     prometheus only -- lowest rubric score counted as correct.
#   LIMIT=              score only the first N items, to smoke-test a judge cheaply.
#
# Needs a GPU (the judge is a local HF model, not an API). Unlike
# scripts/score_llmjudge_gpt.sh this is fine on a compute node -- no outbound network is
# used once the checkpoint is in the HF cache.
#
# WHAT YOU MAY REPORT. Judge-to-judge gaps are larger than most effects being measured
# here, so a second judge's accuracy is NOT a StreamBench score and must never be put in
# the same column as a Llama-3-scored one. Its job is agreement: if rlt0.5 beats rlt0.25
# under both judges, that ordering is a property of the models; if it flips, the gap was
# inside the judge's margin. Score the whole sweep with whichever judge you quote.
set -euo pipefail

# Same activation dance as scripts/eval/eval.sh: a bare `bash scripts/...` otherwise runs
# the cluster's system python and dies on the first import.
if ! python -c "import torch, transformers" >/dev/null 2>&1; then
    command -v module >/dev/null 2>&1 && { module load anaconda3; module load cuda; }
    # `set -u` off across the activation only: conda's shell hook and the module files
    # read PS1 and other variables that are unset in a non-interactive shell, and under
    # -u that aborts the script with a line number pointing inside the eval'd hook.
    set +u
    eval "$(conda shell.bash hook)"
    conda activate Rekv
    set -u
fi
python -c "import torch, transformers" >/dev/null 2>&1 || {
    echo "FATAL: the Rekv env is not active and could not be activated here." >&2
    echo "       Run 'conda activate Rekv' first, then re-run this script." >&2
    exit 1; }

if [ $# -eq 0 ]; then
    echo "usage: $0 <run_dir|results.csv> [...]" >&2
    echo "       JUDGE=<preset|hf-id> $0 results/llava_ov_7b/streambench/*/" >&2
    exit 1
fi

# Resolve the arguments before the cd, so relative paths work from wherever this was run.
CSVS=()
for arg in "$@"; do
    path=$(realpath "${arg}")
    [ -d "${path}" ] && path="${path}/results.csv"
    CSVS+=("${path}")
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/.."
# run_eval's workers need the repo root importable, and so does eval_open_ended_local
# (it imports video_qa.eval.judges).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

JUDGE=${JUDGE:-qwen}
BATCH_SIZE=${BATCH_SIZE:-16}
YES_THRESHOLD=${YES_THRESHOLD:-4}

# judges.py is the single source of truth for (style, checkpoint, cache suffix) -- do not
# keep a second copy of that table here, or a judge ends up writing to two places
# depending on which entry point launched it.
read -r STYLE MODEL SUFFIX < <(python video_qa/eval/judges.py "${JUDGE}")
MODEL=${JUDGE_MODEL:-${MODEL}}

# The cache path is the preset's, so verdicts are shared with run_eval and a re-run costs
# nothing. The summary filename is per *checkpoint*, because JUDGE_MODEL can change the
# model while keeping the preset -- two different judges, one suffix.
SLUG=$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | sed 's/-\{1,\}/-/g; s/^-//; s/-$//')
OUT_NAME="streambench_scores_${SLUG}.json"
# The one exception: upstream's own judge keeps the canonical filename that run_eval and
# the README already refer to, so passing JUDGE=streambench here reproduces exactly what
# the normal pipeline writes rather than a second copy under another name.
if [ "${STYLE}" = "streambench" ] && [ "${MODEL}" = "meta-llama/Meta-Llama-3-8B-Instruct" ]; then
    OUT_NAME="streambench_scores.json"
    echo "NOTE: JUDGE=${JUDGE} is StreamBench's OFFICIAL judge, not a different one." >&2
    echo "      This will write the canonical streambench_scores.json, same as run_eval." >&2
fi

echo "judge preset : ${JUDGE}"
echo "judge style  : ${STYLE}        (prompt + parser, video_qa/eval/judges.py)"
echo "judge model  : ${MODEL}"
echo "verdict cache: <run>/tmp_local${SUFFIX}.jsonl"
echo "summary      : <run>/${OUT_NAME}"
echo

for csv in "${CSVS[@]}"; do
    if [ ! -f "${csv}" ]; then
        echo "skip ${csv}: not found" >&2
        continue
    fi
    dir=$(dirname "${csv}")
    # A results.csv without the streaming-audit columns was written by a solver that does
    # not gate on the breakpoint time. eval_streambench.py aborts on it -- catch it here
    # instead, before spending judge time on rows that cannot be reported.
    if ! head -1 "${csv}" | grep -q 'n_frames_seen'; then
        echo "skip ${dir}: results.csv has no leak-check columns, so it is not a " \
             "StreamBench run (re-run with --dataset streambench)." >&2
        continue
    fi

    echo "=== judging ${dir} with ${MODEL} ==="
    python video_qa/eval/eval_open_ended_local.py \
        --pred_path "${csv}" \
        --output_dir "${dir}/tmp_local${SUFFIX}" \
        --output_json "${dir}/results_local${SUFFIX}.json" \
        --judge_style "${STYLE}" \
        --judge_model "${MODEL}" \
        --yes_threshold "${YES_THRESHOLD}" \
        --batch_size "${BATCH_SIZE}" \
        ${LIMIT:+--limit ${LIMIT}}

    # Pooled accuracy above is not the StreamBench headline: the six classes are what the
    # benchmark is for, and KG is answerable with the video off. This rejoins the verdicts
    # to the class/source columns, re-checks the no-future-frames invariant, and writes the
    # per-judge summary.
    echo "=== breaking ${dir} down by class ==="
    python video_qa/eval/eval_streambench.py \
        --save_dir "${dir}" \
        --cache_dir "${dir}/tmp_local${SUFFIX}" \
        --out_name "${OUT_NAME}"
    echo
done

echo "done. Compare judges with:"
echo "  python - <<'PY'"
echo "  import json, glob"
echo "  for f in sorted(glob.glob('results/*/streambench/*/streambench_scores*.json')):"
echo "      d = json.load(open(f))"
echo "      print(f\"{f:70s} {d.get('judge','?'):45s} macro_no_kg={d['macro_average_no_kg']}\")"
echo "  PY"
