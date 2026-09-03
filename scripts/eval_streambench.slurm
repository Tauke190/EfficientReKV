#!/bin/bash
#SBATCH --job-name=rekv_streambench
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_streambench-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00

# StreamBench: the streaming sweep on both models.
#
#   dataset  streambench   1838 questions / 275 videos / 6 classes / 3 sources
#   models   llava_ov_0.5b, llava_ov_7b
#   arms     blind, baseline (no pruning at all), rlt 0.25 0.5 0.6 0.7 0.8 0.9
#
# 16 arms, run sequentially. Each writes its own results directory (run_eval.reduction_tag
# appends e.g. '-rlt0.6cosine', --blind appends '-blind'), so nothing overwrites anything
# and an interrupted job can be resumed by rerunning with the finished arms removed from
# ARMS.
#
# WHAT MAKES THIS ONE DIFFERENT: EVERY ARM RUNS A JUDGE. StreamBench answers are free-form
# sentences, so there is no letter to match -- each arm is QA followed by
# eval_open_ended_local.py grading all 1838 answers with Meta-Llama-3-8B-Instruct, then
# eval_streambench.py rejoining those verdicts to the class/source columns. That second
# model has to fit alongside nothing else (the QA model is unloaded by then), but it does
# mean an arm is not finished when the QA is.
#
# The judge is pinned to upstream's on purpose. StreamChat (arXiv 2501.13468) publishes
# StreamBench numbers under eval_ego_streaming_with_llama3.py, and
# judges.StreamBenchLlamaJudge reproduces that prompt byte-for-byte; judge-to-judge gaps
# are larger than most of the effects this sweep measures, so changing JUDGE_PRESET makes
# the results incomparable with published figures AND with the other arms here.
#
# Size: 17.3 h of video at 1 fps, ~62k frames -- a quarter of ovbench -- so the full 16-arm
# sweep does fit in 48 h. Split it anyway if the queue is tight:
#     sbatch --export=ALL,MODELS="llava_ov_7b",ARMS="none:0 rlt:0.5" scripts/eval_streambench.slurm
#
# Why 1 fps. It matches the ovobench / streamingbench / ovbench runs, and a Pareto curve
# that mixes frame rates is not a curve. Breakpoints here sit at 120-447 s (median 190), so
# even the earliest question gets ~120 frames; this is not a dataset where 1 fps starves
# the model the way odvbench's 5-90 s clips would.

module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate Rekv

LOGDIR=/home/av354855/EfficientReKV/slurmlogs

# A full home quota kills the batch step in ~1s with ExitCode 0:53 and no .out file, before
# this script runs at all. This guard only catches the quota filling later; a job that
# vanishes leaving no log is a quota problem, not a code problem.
if ! touch "${LOGDIR}/.writetest" 2>/dev/null; then
    echo "FATAL: ${LOGDIR} not writable -- home quota is full. Free space and resubmit." >&2
    exit 1
fi
rm -f "${LOGDIR}/.writetest"

export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

NUM_CHUNKS=${NUM_CHUNKS:-4}

DATASET=streambench
MODELS=${MODELS:-"llava_ov_0.5b llava_ov_7b"}
ARMS=${ARMS:-"blind:0 none:0 rlt:0.25 rlt:0.5 rlt:0.6 rlt:0.7 rlt:0.8 rlt:0.9"}

SAMPLE_FPS=${SAMPLE_FPS:-1}
N_LOCAL=${N_LOCAL:-15000}
RETRIEVE_SIZE=${RETRIEVE_SIZE:-64}

PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

JUDGE_PRESET=${JUDGE_PRESET:-streambench}
JUDGE_BATCH=${JUDGE_BATCH:-16}

ANNO=${ANNO:-data/streambench/full_oe.json}
ANNO_ARGS=""
[ "${ANNO}" != "data/streambench/full_oe.json" ] && ANNO_ARGS="--anno_path ${ANNO}"

if [ ! -f "${ANNO}" ]; then
    echo "FATAL: ${ANNO} missing -- run scripts/prepare_streambench.sh first." >&2
    exit 1
fi

# Preflight the judge BEFORE any QA runs. Meta-Llama-3-8B-Instruct is a gated repo: without
# an accepted licence and a token the download 401s. Discovering that after 17 h of
# ingestion would waste the whole job, and the QA results would sit there ungraded -- so
# fail in the first minute instead. Only the tokenizer is fetched here (a few hundred KB);
# the weights come down on first use.
JUDGE_MODEL=$(python video_qa/eval/judges.py "${JUDGE_PRESET}" | cut -f2)
echo "judge: ${JUDGE_PRESET} -> ${JUDGE_MODEL}"
if ! python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('${JUDGE_MODEL}')
print('judge checkpoint reachable')
" ; then
    echo "FATAL: cannot load ${JUDGE_MODEL}. It is a gated HuggingFace repo -- accept the" >&2
    echo "licence on its model page and run 'huggingface-cli login', or pass another" >&2
    echo "judge with --export=ALL,JUDGE_PRESET=qwen7b (NOT comparable to published" >&2
    echo "StreamBench numbers)." >&2
    exit 1
fi

# run_eval builds the results path from Python's own formatting of these values
# (reduction_tag uses '%g' for the threshold, and sample_fps arrives as a float), so derive
# the directory the same way instead of pasting strings together -- '1' would otherwise not
# match Python's '1.0', and '0.60' would not match its '0.6'.
FPS_TAG=$(python -c "print(float('${SAMPLE_FPS}'))")
thr_tag () { python -c "print('%g' % float('$1'))"; }

# What the threshold actually did. A threshold is an input, not a result -- redundancy
# varies enormously between videos, so the same value gives different keep rates on
# different footage, and an accuracy delta is unreadable without the rate it came with.
summarize_reduction () {
  python - "$1" <<'PYEOF'
import sys, os, pandas as pd
f = os.path.join(sys.argv[1], 'results.csv')
if not os.path.exists(f):
    print(f'  (no results.csv in {sys.argv[1]})'); raise SystemExit
df = pd.read_csv(f)
if 'blind' in df.columns:
    print(f'  blind control: {len(df)} questions, no frames ingested'); raise SystemExit
if 'token_keep_rate' not in df.columns:
    print('  reduction: none (baseline run)'); raise SystemExit
v = df.drop_duplicates('video_id')
print(f'  videos {len(v)}  questions {len(df)}')
print(f'  tokens kept: mean {100*v.token_keep_rate.mean():5.1f}%  '
      f'min {100*v.token_keep_rate.min():5.1f}%  max {100*v.token_keep_rate.max():5.1f}%')
if {'n_tokens_fed', 'n_local'} <= set(df.columns):
    inside = (df.n_tokens_fed <= df.n_local).sum()
    print(f'  queries never reaching retrieval: {inside}/{len(df)} '
          f'({100*inside/len(df):.0f}%)')
PYEOF
}

for MODEL in ${MODELS}; do
  for ARM in ${ARMS}; do
    METHOD="${ARM%%:*}"
    THR="${ARM##*:}"

    # --blind and the reduction flags are mutually exclusive: with no frames ingested there
    # is nothing to prune, and run_eval.py rejects the combination rather than inventing
    # distinct 'blind at 0.25' and 'blind at 0.5' directories holding identical runs.
    EXTRA_ARGS=""
    SAVE_DIR="results/${MODEL}/${DATASET}/${RETRIEVE_SIZE}-${FPS_TAG}"
    if [ "${METHOD}" = "blind" ]; then
        EXTRA_ARGS="--blind"
        SAVE_DIR="${SAVE_DIR}-blind"
    elif [ "${METHOD}" != "none" ]; then
        EXTRA_ARGS="--prune_method ${METHOD} --prune_threshold ${THR} \
            --prune_metric ${PRUNE_METRIC} --prune_refresh_every ${PRUNE_REFRESH_EVERY}"
        SAVE_DIR="${SAVE_DIR}-${METHOD}$(thr_tag ${THR})${PRUNE_METRIC}"
    fi

    echo "=========================================================================="
    echo "=== ${MODEL} | ${DATASET} | ${SAMPLE_FPS} fps | s1=none | s2=${METHOD}@${THR} ==="
    echo "=== judge ${JUDGE_PRESET} | started $(date) ==="
    python -m video_qa.run_eval \
        --num_chunks ${NUM_CHUNKS} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --sample_fps ${SAMPLE_FPS} \
        --n_local ${N_LOCAL} \
        --retrieve_size ${RETRIEVE_SIZE} \
        --judge_preset ${JUDGE_PRESET} \
        --judge_batch_size ${JUDGE_BATCH} \
        ${ANNO_ARGS} \
        ${EXTRA_ARGS}
    summarize_reduction "${SAVE_DIR}"
    echo "=== finished $(date) ==="
  done
done

echo "Job finished at $(date)"

conda deactivate
