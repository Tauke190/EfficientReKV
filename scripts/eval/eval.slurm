#!/bin/bash
#SBATCH --job-name=rekv_eval
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_eval-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=8G

# Run one evaluation through slurm:  sbatch scripts/eval/eval.slurm
# Edit the knobs below, and the #SBATCH header above to match (num_chunks = --gres=gpu:N).
# scripts/eval/eval.sh is the same thing without the scheduler, for a GPU you already hold.
#
# Ask for more than the header gives with flags, which override #SBATCH lines:
#   sbatch -C gmem80 --mem=360G --time=48:00:00 scripts/eval/eval.slurm
# ovbench and streamingbench need that: their videos reach ~190k tokens of KV, and ReKV
# pins the whole cache in host RAM, one copy per worker.

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate Rekv

export MPLBACKEND=Agg        # else matplotlib dumps base64 PNGs into the log
export TQDM_MININTERVAL=60

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/EfficientReKV
# The repo root has to be on sys.path, not just be the cwd: run_eval launches its workers
# as `python video_qa/<solver>.py`, so without this their `video_qa`/`model` imports resolve
# through the env's editable rekv-1.0 install, which points at a different ReKV clone.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# One process per GPU -- must match --gres=gpu:N above.
# llava_ov_72b needs 4x 80GB GPUs, so set it to num_gpus//4 for that model.
num_chunks=4

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
#                  flash_vstream_7b
model=llava_ov_7b

# Supported dataset. Counts are the annotations currently on disk.
#     ovobench_realtime        237 vid /   837 q   python video_qa/convert_ovobench.py
#     ovobench_backward        275 vid /   631 q   python video_qa/convert_ovobench.py
#     odvbench                1190 vid /  6348 q   python scripts/dataset_prep/setup_odvbench.py
#     ovbench                 1463 vid /  7090 q   bash scripts/dataset_prep/prepare_ovbench.sh
#     streambench              275 vid /  1838 q   bash scripts/dataset_prep/prepare_streambench.sh   [judge]
#     streamingbench_real      500 vid /  2500 q   bash scripts/dataset_prep/prepare_streamingbench.sh
#     fpsbench_stream          990 vid /   990 q   python video_qa/convert_fpsbench_stream.py
#     rvs_ego                   10 vid /  1465 q   ships with the repo   [judge]
#     rvs_movie                 22 vid /  1905 q   ships with the repo   [judge]
#     activitynet_qa           800 vid /  8000 q   ships with the repo   [judge]
#     mlvu                    1242 vid /  2172 q   ships with the repo
#     mlvu_test                349 vid /   502 q   ships with the repo
#     egoschema               5031 vid /  5031 q   ships with the repo
#     cgbench                 1219 vid / 12129 q   ships with the repo
#     qaego4d                    not built         ships with the repo
#
#   [judge] = answers are free-form sentences, so there is no letter to match. Add
#   --skip_scoring below and score afterwards with scripts/score_llmjudge_gpt.sh.
#   MLVU has an extremely long video (~9hr); drop it from the annotation if RAM is tight.
dataset=fpsbench_stream

# Space-separated -- each rate is a separate run into its own results directory.
# Pruning thresholds do NOT transfer across frame rates (frames are more alike at a higher
# rate), so sweep one axis at a time rather than crossing them and reading it as a curve.
sample_fps="2 4"

n_local=15000        # local window, in TOKENS (llava_ov spends 196 per frame)
retrieve_size=64     # blocks retrieved per query; one block is one frame

# Token pruning -- the thing this fork adds. Space-separated, each value its own run and
# its own results directory, so arms never overwrite each other. 'none' is the untouched
# ReKV baseline; put it first so the reference lands before the sweep.
prune_method=rlt
prune_thresholds="none"

# rvs_* and activitynet_qa answers are free-form, so their scorer is an LLM judge. Add
# --skip_scoring below and judge the whole sweep afterwards with one judge:
#   scripts/score_llmjudge_gpt.sh results/*/rvs_*/64-1.0*/results.csv
#
# fpsbench_stream only: passing --sample_fps_list 2,4 instead of looping decodes each video
# once at the top rate and reads the lower rates as strided views of it -- same results,
# less decode. Everything else needs the loop.

for fps in ${sample_fps}; do
  for thr in ${prune_thresholds}; do

    # A bare --prune_threshold already reads as "enable rlt" in run_eval, so the baseline
    # has to pass no pruning flags at all rather than a threshold of 0.
    prune_args=""
    [ "${thr}" != "none" ] && prune_args="--prune_method ${prune_method} \
        --prune_threshold ${thr}"

    echo "=== ${model} | ${dataset} | ${fps} fps | prune ${thr} ==="
    python -m video_qa.run_eval \
        --num_chunks ${num_chunks} \
        --model ${model} \
        --dataset ${dataset} \
        --sample_fps ${fps} \
        --n_local ${n_local} \
        --retrieve_size ${retrieve_size} \
        ${prune_args}
  done
done
