#!/bin/bash
#SBATCH --job-name=rekv_eval
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_eval-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# ReKV evaluation under sbatch. scripts/eval.sh is the run-it-directly twin -- same knobs,
# same loop; keep the two in step when changing either.

# Load necessary modules
module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"

# Activate the ReKV conda environment
conda activate Rekv

LOGDIR=/home/av354855/EfficientReKV/slurmlogs


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
cd /home/av354855/EfficientReKV

# The repo root has to be on sys.path, not just be the cwd: run_eval.py launches its
# workers as `python video_qa/<solver>.py`, which puts video_qa/ on sys.path[0] and leaves
# the root off entirely. The `video_qa`/`model` imports inside a worker would then resolve
# through the Rekv env's editable rekv-1.0 install, which maps both packages to the
# EfficientVideoXLPro/ReKV clone -- so the workers would run that tree's code while
# run_eval itself ran this one. See the same note in scripts/eval.sh.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# NUM_CHUNKS below must match #SBATCH --gres=gpu:N above -- run_eval.py
# spawns one process per chunk, each pinned to its own GPU via CUDA_VISIBLE_DEVICES.
# Exception: llava_ov_72b needs 4 GPUs per chunk, so NUM_CHUNKS must be gres/4 for that model.
NUM_CHUNKS=4

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
MODEL=llava_ov_7b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
# ovobench_realtime / ovobench_backward / fpsbench_stream / odvbench are valid --dataset
# values too, but each needs its annotation built first (video_qa/convert_ovobench.py,
# video_qa/convert_fpsbench_stream.py, scripts/setup_odvbench.py) and a sample_fps of its
# own. --trigger is a knob below; --anno_path is not, so add it to the run_eval call if a
# subset run needs it.
# Space-separated: each dataset is evaluated in turn.
DATASETS="rvs_ego rvs_movie"

# odvbench is the outlier here: it is streaming (each question may only see frames up to
# its own end_time, like ovobench), and its clips run 5-90 s -- median 33 s -- so 0.5 fps
# gives an early question a single frame. Use 2 or more. Note also that no odvbench clip
# comes close to n_local, so the retrieval path never fires: the run measures the
# reduction stages' effect on perception, not on retrieval.
SAMPLE_FPS=0.5

# fpsbench_stream only -- when each question fires. 'query' asks at query_time_sec, while
# the needle is still the newest thing in the cache: that measures realtime perception, the
# model's ability to answer about what it just saw. 'end' asks only after all 600 s have
# been ingested, so the needle has to be retrieved back out of a memory dominated by
# unrelated footage -- a retrieval measurement, and the benchmark's own default protocol.
# The two arms write to different results dirs ('query' adds a -query suffix), so neither
# overwrites the other. Ignored by every other dataset. For a frame-rate sweep over
# fpsbench_stream (1..32 fps, with the chunk count sized to host memory at each rate), use
# scripts/eval_fpsbenchstream.slurm rather than this file.
TRIGGER=query
N_LOCAL=15000
RETRIEVE_SIZE=64

# rvs_* answers are free-form text, so there is no string-match accuracy for them -- they
# need an LLM judge. Skip scoring here and just write predictions; score later with
# scripts/score_open_ended.sh (local judge) or score_llmjudge_gpt.sh (needs an API key).
SKIP_SCORING=--skip_scoring

# --- stage 1: encoder-side (model/vision_reduction.py) -- skips SigLIP work on unchanged
# patches. Attacks only the vision-tower share of encode cost; KV size is unchanged and it
# stores reused (approximate) features, so it costs accuracy for comparatively little
# throughput (measured +1.6% alone, +58% combined with stage 2). 'none' = off.
VISION_METHOD=none
VISION_MASK_SPACE=embed   # embed (SigLIP patch embeddings) | pixel
VISION_METRIC=cosine      # cosine = chunk-size invariant; l2 is not
VISION_REFRESH_EVERY=0    # force-encode every Nth frame in full, 0 = off
# Redundancy is a direct function of frame spacing and footage, so this does not transfer
# between SAMPLE_FPS values or datasets. Calibrate it rather than copying a value.
VISION_THRESHOLD=0.001

# --- stage 2: memory-side (model/token_pruning.py) -- the one that moves throughput.
# Drops redundant visual tokens before the LM; shrinks KV RAM and retrieval cost with it
# (measured +29% frames/s alone). 'none' = off, the untouched baseline. Space-separated
# list to sweep -- each threshold writes to its own results dir, so arms never overwrite
# each other or the baseline. Scales with SAMPLE_FPS: fewer frames/sec = less redundancy
# = higher threshold, so do not transplant a value across frame rates either.
PRUNE_METHOD=rlt
PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0
PRUNE_THRESHOLDS="0.2"

# Per-video keep rates land in results.csv (see BaseVQA.reduction_stats) -- read them
# there to see what a threshold actually dropped.

for DATASET in ${DATASETS}; do
  for THR in ${PRUNE_THRESHOLDS}; do
    # Forward each stage's flags only when that stage is on. Not cosmetic: run_eval.py's
    # pruning_on() treats a --prune_threshold on its own as "enable rlt" (so older scripts
    # that only set a threshold keep working), so passing the threshold unconditionally
    # would run stage 2 even with PRUNE_METHOD=none -- and the TAG below would not know it.
    REDUCE_ARGS=""
    [ "${VISION_METHOD}" != "none" ] && REDUCE_ARGS="${REDUCE_ARGS} \
        --vision_method ${VISION_METHOD} --vision_threshold ${VISION_THRESHOLD} \
        --vision_mask_space ${VISION_MASK_SPACE} --vision_metric ${VISION_METRIC} \
        --vision_refresh_every ${VISION_REFRESH_EVERY}"
    [ "${PRUNE_METHOD}" != "none" ] && REDUCE_ARGS="${REDUCE_ARGS} \
        --prune_method ${PRUNE_METHOD} --prune_threshold ${THR} \
        --prune_metric ${PRUNE_METRIC} --prune_refresh_every ${PRUNE_REFRESH_EVERY}"

    # --trigger is fpsbench_stream's alone; forwarding it elsewhere would put an
    # inapplicable flag on every other dataset's command line.
    TRIGGER_ARGS=""
    [ "${DATASET}" = "fpsbench_stream" ] && TRIGGER_ARGS="--trigger ${TRIGGER}"

    echo "=== ${MODEL} | ${DATASET} | s1=${VISION_METHOD} | s2=${PRUNE_METHOD}@${THR} ==="
    python -m video_qa.run_eval \
        --num_chunks ${NUM_CHUNKS} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --sample_fps ${SAMPLE_FPS} \
        --n_local ${N_LOCAL} \
        --retrieve_size ${RETRIEVE_SIZE} \
        ${SKIP_SCORING} \
        ${TRIGGER_ARGS} \
        ${REDUCE_ARGS}
  done
done

conda deactivate
