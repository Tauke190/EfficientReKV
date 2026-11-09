#!/bin/bash
#SBATCH --job-name=token_retention
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/token_retention-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:1
#SBATCH --qos=group2-1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# Token-retention visualization (scripts/qualitative/token_retention.py): which patches
# rlt_ref keeps vs. drops, on a near-static clip and a high-motion clip side by side.
#
# Clips were chosen by eye from `token_retention.py --preview 8` (CPU only):
#     static   sample_224 @ 421 s  -- math lecture: fixed slide, annotations appear
#     dynamic  sample_365 @ 583 s  -- egocentric kitchen: camera and hands moving
# Override with STATIC=/DYNAMIC= (VIDEO_ID:START_SEC), THRESHOLDS="0.5 0.7", MODEL=.

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate Rekv

export MPLBACKEND=Agg
cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

echo "Job started at $(date) on $(hostname)"
nvidia-smi

python scripts/qualitative/token_retention.py \
    --model "${MODEL:-llava_ov_7b}" \
    --thresholds ${THRESHOLDS:-0.5 0.7} \
    --static "${STATIC:-Real-Time_Visual_Understanding__sample_224:421}" \
    --dynamic "${DYNAMIC:-Real-Time_Visual_Understanding__sample_365:583}" \
    --gif \
    --out "${OUT:-figures/token_retention}"

echo "Job finished at $(date)"
