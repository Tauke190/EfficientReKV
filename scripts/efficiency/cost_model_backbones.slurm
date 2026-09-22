#!/bin/bash
#SBATCH --job-name=rekv_costmodel_bb
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_costmodel_bb-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:1
#SBATCH --qos=group2-1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00

# Streaming throughput for the backbone comparison, at the thresholds the accuracy table
# reports: baseline, 0.5 and 0.7, every arm measured in this one job.
#
# GPU. `-C gmem80` pins this to the A100 80GB nodes (c4-4, c4-5, c5-2); the A6000s carry
# gmem48 and not gmem80. That constraint is the whole point of the separate OUT_DIR: the
# span CSVs record no device, so an A6000 row and an A100 row are indistinguishable once
# written, and a speedup computed across the two is meaningless. The earlier A6000 sweep
# stays in results/cost_model_backbones/ untouched -- same split as results/speed vs
# results/speed_a6000. Do not point this job at that directory.
#
# Every arm is re-measured here, including the baselines, rather than reusing the A6000
# ones, because a speedup is only valid against a baseline from the same device.
#
# Protocol, held constant across arms:
#     span timing, CPU preprocessing (no -gpuprep tag), chunk size 1, 1 fps x 600 frames,
#     3 streams (v0/v1/v2).
#
# Per-backbone budgets, from scripts/eval/backbones.slurm -- these are not interchangeable:
#     llava_ov_7b       196 tok/frame  Qwen2   32k   n_local 15000 / retrieve 64
#     longva_7b         144 tok/frame  Qwen2  224k   n_local 11020 / retrieve 64
#     video_llava_7b    257 tok/frame  Vicuna   4k   n_local  3084 / retrieve  8
#     flash_vstream_7b   64 tok/frame  Vicuna   4k   n_local  3648 / retrieve 16
# Flash-VStream must use retrieve_size 16: at 64 its loader asserts, because
# n_init + topk*n_frame_tokens overruns the 4k context.
#
# Steady state. Throughput is quoted from chunks encoded after the local window filled,
# which takes n_local / (n_frame_tokens x keep_rate) frames. At threshold 0.7 that is
#     longva_7b        ~484 frames   reached within 600
#     flash_vstream_7b ~390 frames   reached within 600
#     llava_ov_7b      ~683 frames   NOT reached in a 600 s stream at 1 fps
# so llava_ov's 0.7 row comes back flagged reached_steady_state=False and is a pre-steady
# figure -- optimistic, since eviction/offload never engaged. Getting a steady number for
# it needs more frames than a 600 s stream holds at 1 fps (FPS=2 NUM_FRAMES=1200), which
# is a different operating point and must not be tabled beside the 1 fps rows.
#
# FLOPS_FRAMES=0, matching every span already in this directory (they all record
# flops_frames=0). Two reasons not to turn the counter on here: it would make these arms a
# different protocol from the baseline/0.5 rows they are quoted beside, and on
# Flash-VStream it raises outright -- FlopCounterMode tracks autograd, which collides with
# the torch.inference_mode() the encode path runs under ("Inference tensors cannot be
# saved for backward"). GFLOPs are not what this job is for; throughput is.
#
# COLLECT=false: the collector derives its KV/GFLOPs columns from config.json and only
# handles the HF-nested Qwen2 configs, so it raises on LongVA and Flash-VStream. The span
# CSVs carry the measured throughput and are written before it runs.

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate Rekv

export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

THRESHOLDS=${THRESHOLDS:-"0.5 0.7"}
N_STREAMS=${N_STREAMS:-3}
OUT_DIR=${OUT_DIR:-results/cost_model_backbones_a100}

# GPU preprocessing. The two paths write different filenames (-gpuprep) and are never
# pooled, so both can live in one directory. true reproduces the protocol behind
# results/cost_model/llava_ov_7b-fps1-gpuprep-table.csv; false is the CPU path.
# The trade, from cost_model.sh: GPU resampling is not bit-identical to PIL, so keep
# rates shift by a hair and these are not the byte-identical numbers the accuracy runs
# used. The gain is small at baseline and large under hard pruning, because once the
# LM work shrinks there is less GPU time left to hide the HF processor behind.
GPU_PREPROCESS=${GPU_PREPROCESS:-false}
if [ "${GPU_PREPROCESS}" = "true" ]; then TABLE_FLAG="--gpu_preprocess"; else TABLE_FLAG=""; fi

# model:n_local:retrieve_size
# video_llava_7b (3084/8) is deliberately not here: it OOMs partway through a 600-frame
# stream (SIGKILL at ~frame 308 in job 370357) and is not a backbone we report.
for spec in "llava_ov_7b:15000:64" "longva_7b:11020:64" \
            "flash_vstream_7b:3648:16"; do
    IFS=: read -r model n_local retrieve <<< "${spec}"
    echo
    echo "##########################################################################"
    echo "### ${model} | n_local=${n_local} retrieve_size=${retrieve} | thr ${THRESHOLDS}"
    echo "##########################################################################"
    # A failed backbone must not take the others down with it.
    MODEL="${model}" N_LOCAL="${n_local}" RETRIEVE_SIZE="${retrieve}" \
    THRESHOLDS="${THRESHOLDS}" N_STREAMS="${N_STREAMS}" OUT_DIR="${OUT_DIR}" \
    FPS=1 NUM_FRAMES=600 TIMING=span GPU_PREPROCESS="${GPU_PREPROCESS}" COLLECT=false \
    FLOPS_FRAMES="${FLOPS_FRAMES:-0}" \
        bash scripts/efficiency/cost_model.sh || echo "!!! ${model} FAILED -- continuing"
done

echo
echo "=== steady-state throughput, all arms on disk ==="
python scripts/efficiency/throughput_table.py --dir "${OUT_DIR}" ${TABLE_FLAG}
echo "Job finished at $(date)"
