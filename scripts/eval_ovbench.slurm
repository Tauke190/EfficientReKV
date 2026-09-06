#!/bin/bash
#SBATCH --job-name=ovbench_rekv
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_ovbench-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:4
#SBATCH --mem=360G
#SBATCH --cpus-per-gpu=8

# The old 8G/cpu x 16 cpus = 128 GiB was not enough: job 362087 died OUT_OF_MEMORY
# with MaxRSS pinned at exactly 128.0 GiB, and an earlier silent oom_kill took out
# one worker of the 7b baseline arm leaving no traceback at all -- a SIGKILL loses
# the stdout buffer, so it just looked like the run stopped. Only StreamingBench and
# OVBench provoke it: their videos reach ~190k tokens of KV against ~4k on ODV-Bench
# and OVO-Bench, and pin_memory=True makes that cache unswappable host RAM, one copy
# per worker.
#
# 190G is the ceiling, not a preference: slurm assigns the QOS from the GPU count,
# and group3-2 (2 GPUs) caps a job at cpu=24,mem=190G. Asking for more does not
# queue, it is refused. So this is the most memory two workers can be given --
# 95 GiB each, against the 64 GiB that died. Note the cap does scale with GPUs:
# group3-1 is 90G, group3-2 190G, group3-3 280G. Three GPUs running NUM_CHUNKS=2
# would leave one idle but give each worker 140 GiB; see NUM_CHUNKS below.

# OVBench: the streaming sweep on both models.
#
#   dataset  ovbench       7090 questions / 1463 videos / 16 sub-tasks / 6 types
#   models   llava_ov_0.5b, llava_ov_7b
#   arms     blind, baseline (no pruning at all), rlt 0.25 0.5 0.6 0.7 0.8 0.9
#
# SIZE WARNING. This is much the largest streaming set in the repo: 78 hours of video at
# 1 fps, ~281k frames, against odvbench's short clips. One arm of the 7b is a long run even
# across 2 GPUs, and 16 arms will not fit in 48 h. Split it -- one model or a few arms per
# job:
#     sbatch --export=ALL,MODELS="llava_ov_7b",ARMS="none:0 rlt:0.5" scripts/eval_ovbench.slurm
#
# Why 1 fps. It matches the OVO-Bench and StreamingBench runs, and a Pareto curve that
# mixes frame rates is not a curve. It is also the rate the sources can agree on: the three
# container sources are full-length videos where a question's timestamp can sit 1000 s in
# (AVA_RAW), while the seven transcoded sources run tens of seconds, so a higher rate costs
# disproportionately on the former without helping the latter.

module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate Rekv

LOGDIR=/home/av354855/EfficientReKV/slurmlogs


export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# Must match #SBATCH --gres=gpu:N -- run_eval.py starts one worker per chunk, each pinned

NUM_CHUNKS=${NUM_CHUNKS:-4}

DATASET=ovbench
MODELS=${MODELS:-"llava_ov_7b"}
# method:threshold. 'blind' ingests nothing; 'none' is ReKV untouched, the reference every
# pruned arm is measured against.
ARMS=${ARMS:-"none:0 rlt:0.25 rlt:0.5 rlt:0.6 rlt:0.7 rlt:0.8 rlt:0.9"}

SAMPLE_FPS=${SAMPLE_FPS:-1}
N_LOCAL=${N_LOCAL:-15000}
RETRIEVE_SIZE=${RETRIEVE_SIZE:-64}

PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

ANNO=${ANNO:-data/OVBench/full_mc.json}
ANNO_ARGS=""
[ "${ANNO}" != "data/OVBench/full_mc.json" ] && ANNO_ARGS="--anno_path ${ANNO}"

if [ ! -f "${ANNO}" ]; then
    echo "FATAL: ${ANNO} missing -- run scripts/prepare_ovbench.sh first." >&2
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
# OVBench makes this sharper than the other sets: its ten sources range from static
# trail-cam footage to driving video, so one pooled keep rate hides a wide spread.
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
    echo "=== started $(date) ==="
    python -m video_qa.run_eval \
        --num_chunks ${NUM_CHUNKS} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --sample_fps ${SAMPLE_FPS} \
        --n_local ${N_LOCAL} \
        --retrieve_size ${RETRIEVE_SIZE} \
        ${ANNO_ARGS} \
        ${EXTRA_ARGS}
    summarize_reduction "${SAVE_DIR}"
    echo "=== finished $(date) ==="
  done
done

echo "Job finished at $(date)"

conda deactivate
