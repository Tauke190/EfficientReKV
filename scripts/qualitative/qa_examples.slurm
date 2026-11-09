#!/bin/bash
#SBATCH --job-name=qa_examples
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/qa_examples-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:1
#SBATCH --qos=group2-1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# Token retention on one question TTS answered correctly and one it got wrong, both from
# StreamingBench (Real) at threshold 0.5 (scripts/qualitative/qa_examples.py). Override
# with CORRECT=/INCORRECT= (question ids; list them with qa_examples.py --list), THRESHOLD=.

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate Rekv

export MPLBACKEND=Agg
cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

echo "Job started at $(date) on $(hostname)"
nvidia-smi

python scripts/qualitative/qa_examples.py \
    --model "${MODEL:-llava_ov_7b}" \
    --threshold "${THRESHOLD:-0.5}" \
    --correct "${CORRECT:-Real-Time Visual Understanding_sample_263_3}" \
    --incorrect "${INCORRECT:-Real-Time Visual Understanding_sample_263_4}" \
    --out "${OUT:-figures/qa_examples}"

echo "Job finished at $(date)"
