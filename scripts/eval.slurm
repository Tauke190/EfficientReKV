#!/bin/bash
#SBATCH --job-name=rekv_eval
#SBATCH --output=/home/av354855/EfficientVideoXLPro/slurmlogs/rekv_eval-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# Load necessary modules
module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"

# Activate the ReKV conda environment
conda activate Rekv

LOGDIR=/home/av354855/EfficientVideoXLPro/slurmlogs

# See eval_rlt.slurm for the full note: a full home quota kills the batch step in ~1s
# with ExitCode 0:53 and no .out file, before this script ever runs. This guard only
# catches the quota filling later; a job that vanishes leaving no log is a quota problem.
if ! touch "${LOGDIR}/.writetest" 2>/dev/null; then
    echo "FATAL: ${LOGDIR} not writable -- home quota is full. Free space and resubmit." >&2
    exit 1
fi
rm -f "${LOGDIR}/.writetest"

# Stop matplotlib dumping base64 PNGs into stdout and throttle tqdm; unset, these grew
# past logs to 25MB each.
export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

echo "Checking PyTorch CUDA version and availability:"
python -c "import torch; print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available:', torch.cuda.is_available())"

# Change directory
cd /home/av354855/EfficientVideoXLPro/ReKV

# NUM_CHUNKS below must match #SBATCH --gres=gpu:N above -- run_eval.py
# spawns one process per chunk, each pinned to its own GPU via CUDA_VISIBLE_DEVICES.
# Exception: llava_ov_72b needs 4 GPUs per chunk, so NUM_CHUNKS must be gres/4 for that model.
NUM_CHUNKS=4

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
MODEL=llava_ov_7b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
# Space-separated: each dataset is evaluated in turn.
DATASETS="rvs_ego rvs_movie"

# rvs_* answers are free-form text, so accuracy needs an LLM judge to decide whether a
# prediction means the same as the reference -- there is no string-match accuracy for
# them. We skip that step and just write predictions to results.csv; score it later with
# video_qa/eval/eval_open_ended.py (needs OPENAI_API_KEY). See eval.sh for the command.
SKIP_SCORING=--skip_scoring

for DATASET in ${DATASETS}; do
    echo "=== Evaluating ${MODEL} on ${DATASET} ==="
    python -m video_qa.run_eval \
        --num_chunks $NUM_CHUNKS \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --sample_fps 0.5 \
        --n_local 15000 \
        --retrieve_size 64 \
        ${SKIP_SCORING}
done

conda deactivate
