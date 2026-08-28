#!/bin/bash
# Score open-ended (free-form answer) predictions with the OpenAI API judge -- the judge
# the ReKV paper used. For rvs_ego / rvs_movie / qaego4d / activitynet_qa, whose answers
# cannot be scored by string match.
#
# The API-key twin of score_open_ended.sh (which runs a local judge on your own GPU).
#
# Takes the results.csv files to score:
#   scripts/score_llmjudge_gpt.sh results/llava_ov_0.5b/rvs_ego/64-0.5/results.csv
#   scripts/score_llmjudge_gpt.sh results/llava_ov_0.5b/rvs_{ego,movie}/64-0.5/results.csv
#
# Verdicts go next to each CSV as results.json, with per-item cache in tmp/ (re-running
# skips what is already cached, so a Ctrl-C costs nothing). Local-judge outputs
# (results_local*.json / tmp_local*/, one pair per judge -- see scripts/score_open_ended.sh)
# are on separate paths and are left untouched.
#
# RUN THIS ON A LOGIN NODE, not through eval.slurm: compute nodes here have no outbound
# network. It is pure API calls -- no GPU needed.
#
# The key is read from scripts/.openai_key (gitignored, mode 600), so it never lands in
# a tracked file or in your shell history. An already-exported OPENAI_API_KEY wins.
#
# Overridable from the environment:
#   OPENAI_JUDGE_MODEL=gpt-4o scripts/score_llmjudge_gpt.sh <csv>
#   OPENAI_BASE_URL=...       # non-official / proxied endpoint
set -euo pipefail

# gpt-3.5-turbo-0613 is what the paper scored with; that snapshot 404s now. gpt-3.5-turbo
# still serves, resolving to -0125, which is what we pin here -- the closest available
# judge to the paper's. Pinned rather than the floating alias so a future realias cannot
# silently change scores. Same family, different weights: results are self-consistent but
# NOT directly comparable to the published numbers; say so if these go in a table.
export OPENAI_JUDGE_MODEL=${OPENAI_JUDGE_MODEL:-gpt-3.5-turbo-0125}

# Every judged number in this project uses gpt-3.5-turbo-0125. Do not mix in another
# model to save money -- a table whose rows were scored by different judges compares
# judges, not systems.
if [ "${OPENAI_JUDGE_MODEL}" != "gpt-3.5-turbo-0125" ]; then
  echo "WARNING: OPENAI_JUDGE_MODEL=${OPENAI_JUDGE_MODEL}, not the project default " \
       "gpt-3.5-turbo-0125. Results will not be comparable to the rest of the sweep." >&2
fi

# Each judge gets its own cache and output file, so one judge's verdicts can never be
# reused and relabelled as another's (eval_open_ended.py refuses that outright now).
SLUG=$(echo "${OPENAI_JUDGE_MODEL}" | tr './' '__')
OUT_TMP=${OUT_TMP:-tmp_${SLUG}}
OUT_JSON=${OUT_JSON:-results_${SLUG}.json}

if [ $# -eq 0 ]; then
  echo "usage: $0 <results.csv> [results.csv ...]" >&2
  exit 1
fi

# Resolve before the cd, so relative paths work from wherever this was invoked.
CSVS=()
for arg in "$@"; do
  CSVS+=("$(realpath "${arg}")")
done
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}/.."

if [ -z "${OPENAI_API_KEY:-}" ]; then
  KEY_FILE=${OPENAI_KEY_FILE:-${SCRIPT_DIR}/.openai_key}
  if [ ! -f "${KEY_FILE}" ]; then
    echo "no API key: set OPENAI_API_KEY, or put OPENAI_API_KEY=sk-... in ${KEY_FILE}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${KEY_FILE}"
  export OPENAI_API_KEY
fi

# Preflight one cheap call before the loop. eval_open_ended.py swallows per-item
# exceptions and retries until every item is written, so a bad key or an unserved model
# does not fail it -- it spins forever printing errors. Catch that here instead.
echo "=== preflight: ${OPENAI_JUDGE_MODEL} ==="
python - <<'PY'
import os, sys, openai
try:
    r = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None,
                      api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
        model=os.environ["OPENAI_JUDGE_MODEL"],
        messages=[{"role": "user", "content": "Reply with exactly: {'pred': 'yes', 'score': 5}"}],
        max_tokens=32, temperature=0)
except Exception as e:
    sys.exit(f"preflight failed: {e}")
out = (r.choices[0].message.content or "").strip()
print(f"judge replied: {out!r}")
# The judge must emit a bare Python dict literal -- eval_open_ended.py runs the reply
# through ast.literal_eval, and markdown fences make that raise, which is exactly the
# failure that turns into an infinite retry loop.
if out.startswith("```"):
    sys.exit("judge wraps replies in markdown fences; ast.literal_eval will fail on "
             "every item and the scorer will loop forever. Pick another judge model.")
PY

for csv in "${CSVS[@]}"; do
  if [ ! -f "${csv}" ]; then
    echo "skip ${csv}: not found" >&2
    continue
  fi
  dir=$(dirname "${csv}")
  echo "=== scoring ${csv} with ${OPENAI_JUDGE_MODEL} ==="
  python video_qa/eval/eval_open_ended.py \
      --pred_path "${csv}" \
      --output_dir "${dir}/${OUT_TMP}" \
      --output_json "${dir}/${OUT_JSON}"
done
