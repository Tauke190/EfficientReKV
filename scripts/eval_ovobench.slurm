#!/bin/bash
#SBATCH --job-name=rekv_ovo
#SBATCH --output=/home/av354855/EfficientVideoXLPro/slurmlogs/rekv_ovo-%j.out
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

LOGDIR=/home/av354855/EfficientVideoXLPro/slurmlogs

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

cd /home/av354855/EfficientVideoXLPro/ReKV

# Must match #SBATCH --gres=gpu:N -- run_eval.py starts one worker per chunk, each pinned
# to its own GPU. Chunking splits the *video* list, so each GPU owns a disjoint set of
# videos and no video is ingested twice. (llava_ov_72b needs 4 GPUs per chunk: gres/4.)
NUM_CHUNKS=4
# llava_ov_0.5b exists for smoke tests; its OVO-Bench numbers are not worth reporting.
MODEL=llava_ov_7b

# ovobench_realtime (RVP: OCR/ACR/ATR/STU/FPD/OJR) and ovobench_backward (BT: EPM/ASI/HLD).
# FAR is not wired up -- see video_qa/run_eval.py:eval_ovobench.
#
# Overridable so this file and scripts/eval_ovobench.sh no longer have to be edited in
# lockstep -- they disagreed (sh on backward, slurm on realtime), which is the kind of
# split that gets the wrong split evaluated:
#     sbatch --export=ALL,DATASETS=ovobench_realtime scripts/eval_ovobench.slurm
DATASETS=${DATASETS:-"ovobench_realtime ovobench_backward"}

# Stage-2 token pruning. Empty = baseline. Space-separated list to sweep; each threshold
# lands in its own results directory, so arms never overwrite each other:
#     sbatch --export=ALL,PRUNE_THRESHOLDS="0.1 0.2 0.3" scripts/eval_ovobench.slurm
# Do not reuse measure_speed.sh's 0.25/0.5 -- those were calibrated at 0.5 fps and this
# runs at 1 fps, where the same threshold prunes much harder. See scripts/eval_ovobench.sh
# for the measured keep-rate-vs-frame-spacing table.
PRUNE_METHOD=${PRUNE_METHOD:-rlt}
PRUNE_METRIC=${PRUNE_METRIC:-cosine}
PRUNE_REFRESH_EVERY=${PRUNE_REFRESH_EVERY:-0}
PRUNE_THRESHOLDS=${PRUNE_THRESHOLDS:-"0.25"}

for DATASET in ${DATASETS}; do
    if [ -z "${PRUNE_THRESHOLDS}" ]; then
        echo "=== ${MODEL} | ${DATASET} | baseline ==="
        python -m video_qa.run_eval \
            --num_chunks $NUM_CHUNKS --model ${MODEL} --dataset ${DATASET} \
            --sample_fps 1 --n_local 15000 --retrieve_size 64
    else
        for THR in ${PRUNE_THRESHOLDS}; do
            echo "=== ${MODEL} | ${DATASET} | ${PRUNE_METHOD} ${THR} ==="
            python -m video_qa.run_eval \
                --num_chunks $NUM_CHUNKS --model ${MODEL} --dataset ${DATASET} \
                --sample_fps 1 --n_local 15000 --retrieve_size 64 \
                --prune_method ${PRUNE_METHOD} \
                --prune_threshold ${THR} \
                --prune_metric ${PRUNE_METRIC} \
                --prune_refresh_every ${PRUNE_REFRESH_EVERY}
        done
    fi
done

conda deactivate
