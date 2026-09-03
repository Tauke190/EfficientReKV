#!/bin/bash
#SBATCH --job-name=sb_real_rekv
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_sb_real-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# StreamingBench / Real-Time Visual Understanding: the full stage-2 sweep on both models.
#
#   dataset  streamingbench_real   2500 questions / 500 videos / 10 tasks
#   models   llava_ov_0.5b, llava_ov_7b
#   arms     blind, baseline (no pruning at all), rlt 0.25 0.5 0.6 0.7 0.8 0.9
#
# 16 arms, run sequentially. Each writes its own results directory (run_eval.reduction_tag
# appends e.g. '-rlt0.6cosine', --blind appends '-blind'), so nothing overwrites anything
# and an interrupted job can be resumed by rerunning with the finished arms removed from
# ARMS -- see MODELS/ARMS below.
#
# Arm order is deliberate: blind first, then baseline, then rising thresholds. Blind costs
# minutes and gives the floor immediately; a baseline that lands before the sweep means a
# job killed halfway still yields a readable comparison. 0.5b before 7b for the same
# reason.
#
# Only `real` is run here. The other three subsets are still wired up in run_eval.py
# (streamingbench_{omni,context,sqa}); point DATASET at one of them for a separate job.
# Note that streamingbench_sqa cannot take --blind -- its sighted run carries the earlier
# questions and their gold answers, and blind_vqa does not, so the two would not be a
# control pair. run_eval.py refuses it rather than writing a misleading directory.

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

# Must match #SBATCH --gres=gpu:N -- run_eval.py starts one worker per chunk, each pinned
# to its own GPU. Chunking splits the *video* list, so each GPU owns a disjoint set of
# videos and no video is ingested twice.
NUM_CHUNKS=2

DATASET=${DATASET:-streamingbench_real}

# Overridable so the sweep can be split across jobs or resumed:
#     sbatch --export=ALL,MODELS="llava_ov_7b",ARMS="rlt:0.7 rlt:0.8 rlt:0.9" \
#            scripts/eval_streamingbench.slurm
MODELS=${MODELS:-"llava_ov_0.5b llava_ov_7b"}
# method:threshold. 'blind' ingests nothing; 'none' is ReKV untouched, the reference every
# pruned arm is measured against.
ARMS=${ARMS:-"blind:0 none:0 rlt:0.25 rlt:0.5 rlt:0.6 rlt:0.7 rlt:0.8 rlt:0.9"}

# 1 FPS, matching the OVO-Bench runs. StreamingBench's queries key on fine detail
# (Text-Rich Understanding, Counting) that a 2-second stride drops. The cost is real: each
# video is streamed only as far as its last query, and even so `real` is 70 h of video --
# ~252k frames at 1 fps x 196 tokens, or ~126k frames per GPU at NUM_CHUNKS=2.
SAMPLE_FPS=1
N_LOCAL=15000
RETRIEVE_SIZE=64

PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

if [ ! -f data/StreamingBench/real.json ]; then
    echo "FATAL: data/StreamingBench/real.json missing -- run scripts/prepare_streamingbench.sh first." >&2
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
# In particular, do not carry the keep rates measured in results/throughput/ over to here:
# those were taken at 2 fps, where consecutive frames are half a second apart and far more
# alike, so the same threshold prunes considerably harder there than it will at 1 fps.
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
# Queries whose whole visible window fits inside n_local never trigger retrieval, so
# their accuracy is not testing the thing this benchmark is for.
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
    echo "=== started $(date) ==="
    python -m video_qa.run_eval \
        --num_chunks ${NUM_CHUNKS} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --sample_fps ${SAMPLE_FPS} \
        --n_local ${N_LOCAL} \
        --retrieve_size ${RETRIEVE_SIZE} \
        ${EXTRA_ARGS}
    summarize_reduction "${SAVE_DIR}"
    echo "=== finished $(date) ==="
  done
done

echo "Job finished at $(date)"

conda deactivate
