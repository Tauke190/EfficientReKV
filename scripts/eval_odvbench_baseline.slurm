#!/bin/bash
#SBATCH --job-name=rekv_odvbench
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_odvbench-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate Rekv

LOGDIR=/home/av354855/EfficientReKV/slurmlogs

if ! touch "${LOGDIR}/.writetest" 2>/dev/null; then
    echo "FATAL: ${LOGDIR} not writable -- home quota is full. Free space and resubmit." >&2
    exit 1
fi
rm -f "${LOGDIR}/.writetest"

export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

NUM_CHUNKS=4
MODELS="llava_ov_0.5b llava_ov_7b"
DATASET=odvbench
SAMPLE_FPS=1
N_LOCAL=15000
RETRIEVE_SIZE=64

ARMS="none:0 rlt:0.25 rlt:0.5"

PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

if [ ! -f data/odvbench/full_mc.json ]; then
    echo "FATAL: data/odvbench/full_mc.json missing -- run scripts/setup_odvbench.py first." >&2
    exit 1
fi

for MODEL in ${MODELS}; do
  for ARM in ${ARMS}; do
    METHOD="${ARM%%:*}"
    THR="${ARM##*:}"

    REDUCE_ARGS=""
    if [ "${METHOD}" != "none" ]; then
        REDUCE_ARGS="--prune_method ${METHOD} --prune_threshold ${THR} \
            --prune_metric ${PRUNE_METRIC} --prune_refresh_every ${PRUNE_REFRESH_EVERY}"
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
        ${REDUCE_ARGS}
    echo "=== finished $(date) ==="
  done
done

echo "Job finished at $(date)"

conda deactivate
