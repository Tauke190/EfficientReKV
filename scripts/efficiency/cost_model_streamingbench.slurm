#!/bin/bash
#SBATCH --job-name=rekv_cm_sb
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_cm_sb-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:1
#SBATCH --qos=group2-1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00

# Throughput measured on StreamingBench videos, so it can be quoted in the same row as the
# StreamingBench accuracy and token-reduction numbers.
#
# Why not the FPS-Bench-Stream sweep: that one measures a different operating point. At
# threshold 0.5 llava_ov keeps 0.337 on FPS-Bench but 0.252 on StreamingBench -- 34% more
# tokens surviving into the LM prefill, which is the term throughput scales on. Pairing a
# FPS-Bench speed-up with a StreamingBench accuracy drop silently joins two keep rates.
#
# Video selection (scripts/efficiency/streamingbench_selection.json, regenerate with the
# snippet recorded there): 20 of the 253 videos that are >=500 frames, present in all six
# backbone x threshold eval runs, and on disk. Chosen to match the FULL 2495-query keep
# rate -- every one of the six combinations lands within 0.45% of its population mean, so
# this is the same operating point as the accuracy table, not merely the same dataset.
# real.json is ordered by task category, so a contiguous index range would be one task
# type; these indices are a matched draw, hence N_STREAMS=1 per call.
#
# NUM_FRAMES=500 is a deliberate trade. Steady state needs n_local/(n_frame_tokens*keep)
# frames:
#     flash_vstream @0.5 214   @0.7 390     longva @0.5 264   @0.7 486
#     llava_ov      @0.5 303   @0.7 847
# Everything but llava_ov@0.7 reaches it inside 500. Covering llava_ov@0.7 would need
# >=847-frame videos, of which only 15 exist -- and their keep rate is 0.3185 against the
# population's 0.2524, a 26% bias, because long StreamingBench videos are less
# compressible. An unbiased sample and a steady llava_ov@0.7 cannot both be had; matching
# the accuracy operating point wins, and that row stays flagged PRE-STEADY exactly as it
# is in the FPS-Bench table.
#
# GPU preprocessing on, matching results/cost_model/llava_ov_7b-fps1-gpuprep-table.csv.
# FLOPS_FRAMES=0: the counter collides with torch.inference_mode() on the Vicuna backbones
# and GFLOPs are not what this job measures.

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate Rekv

export MPLBACKEND=Agg
export TQDM_MININTERVAL=120

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

ANNO=${ANNO:-data/StreamingBench/real.json}
THRESHOLDS=${THRESHOLDS:-"0.5 0.7"}
NUM_FRAMES=${NUM_FRAMES:-500}
GPU_PREPROCESS=${GPU_PREPROCESS:-true}
OUT_DIR=${OUT_DIR:-results/cost_model_streamingbench}
VIDEO_IDS=${VIDEO_IDS:-$(cat scripts/efficiency/streamingbench_video_idx.txt)}
if [ "${GPU_PREPROCESS}" = "true" ]; then TABLE_FLAG="--gpu_preprocess"; else TABLE_FLAG=""; fi

mkdir -p "${OUT_DIR}"
echo "videos: ${VIDEO_IDS}"

# model:n_local:retrieve_size -- these are not interchangeable across backbones.
for spec in "llava_ov_7b:15000:64" "longva_7b:11020:64" "flash_vstream_7b:3648:16"; do
    IFS=: read -r model n_local retrieve <<< "${spec}"
    echo
    echo "##########################################################################"
    echo "### ${model} | n_local=${n_local} retrieve=${retrieve} | thr ${THRESHOLDS}"
    echo "##########################################################################"
    for idx in ${VIDEO_IDS}; do
        MODEL="${model}" N_LOCAL="${n_local}" RETRIEVE_SIZE="${retrieve}" \
        THRESHOLDS="${THRESHOLDS}" VIDEO_IDX="${idx}" N_STREAMS=1 \
        ANNO="${ANNO}" OUT_DIR="${OUT_DIR}" FPS=1 NUM_FRAMES="${NUM_FRAMES}" \
        TIMING=span GPU_PREPROCESS="${GPU_PREPROCESS}" COLLECT=false FLOPS_FRAMES=0 \
            bash scripts/efficiency/cost_model.sh \
            || echo "!!! ${model} video ${idx} FAILED -- continuing"
    done
done

echo
echo "=== throughput, StreamingBench videos ==="
python scripts/efficiency/throughput_table.py --dir "${OUT_DIR}" ${TABLE_FLAG} \
    --out "${OUT_DIR}/streamingbench-fps1-gpuprep-throughput.csv"
echo "Job finished at $(date)"
