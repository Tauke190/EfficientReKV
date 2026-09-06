#!/bin/bash
# ReKV evaluation, run directly (no scheduler). scripts/eval.slurm is the sbatch twin --
# same knobs, same loop; keep the two in step when changing either.

# Import this checkout, not whatever `pip install -e` last registered. The Rekv env has an
# editable install of rekv-1.0 whose finder hard-maps the `video_qa` and `model` packages to
# a different clone (EfficientVideoXLPro/ReKV). run_eval launches its workers as
# `python video_qa/<solver>.py`, so sys.path[0] is video_qa/ -- the repo root is not on the
# path at all, PathFinder misses, and the editable finder answers with the other clone. Every
# `from video_qa.base import ...` in a worker then loads code from outside this tree. Putting
# the repo root on PYTHONPATH lets PathFinder win, since the editable finder is appended to
# sys.meta_path rather than prepended.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.
num_chunks=1

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
model=longva_7b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
# ovobench_realtime / ovobench_backward / fpsbench_stream / odvbench / ovbench are valid
# --dataset values too, but each needs its annotation built first
# (video_qa/convert_ovobench.py, video_qa/convert_fpsbench_stream.py,
# scripts/setup_odvbench.py, scripts/prepare_ovbench.sh) and a sample_fps of its own.
# --trigger is a knob below; --anno_path is not, so add it to the run_eval call if a
# subset run needs it.
# ovbench is much the largest of them -- 7090 questions over 1463 videos, ~78 h of video
# at 1 fps -- so prefer scripts/eval_ovbench.slurm, which sweeps the arms and splits by
# model, over listing it here beside a short dataset.
# Space-separated: each dataset is evaluated in turn.
datasets="rvs_ego rvs_movie"

# odvbench is the outlier here: it is streaming (each question may only see frames up to
# its own end_time, like ovobench), and its clips run 5-90 s -- median 33 s -- so 0.5 fps
# gives an early question a single frame. Use 2 or more. Note also that no odvbench clip
# comes close to n_local, so the retrieval path never fires: the run measures the
# reduction stages' effect on perception, not on retrieval.
sample_fps=0.5

# fpsbench_stream only -- when each question fires. 'query' asks at query_time_sec, while
# the needle is still the newest thing in the cache: that measures realtime perception, the
# model's ability to answer about what it just saw. 'end' asks only after all 600 s have
# been ingested, so the needle has to be retrieved back out of a memory dominated by
# unrelated footage -- a retrieval measurement, and the benchmark's own default protocol.
# The two arms write to different results dirs ('query' adds a -query suffix), so neither
# overwrites the other. Ignored by every other dataset.
trigger=query
n_local=15000
retrieve_size=64

# rvs_* answers are free-form text, so there is no string-match accuracy for them -- they
# need an LLM judge. Skip scoring here and just write predictions; score later with
# scripts/score_open_ended.sh (local judge) or score_llmjudge_gpt.sh (needs an API key).
skip_scoring=--skip_scoring

# --- stage 1: encoder-side (model/vision_reduction.py) -- skips SigLIP work on unchanged
# patches. Attacks only the vision-tower share of encode cost; KV size is unchanged and it
# stores reused (approximate) features, so it costs accuracy for comparatively little
# throughput (measured +1.6% alone, +58% combined with stage 2). 'none' = off.
vision_method=none
vision_mask_space=embed   # embed (SigLIP patch embeddings) | pixel
vision_metric=cosine      # cosine = chunk-size invariant; l2 is not
vision_refresh_every=0    # force-encode every Nth frame in full, 0 = off
# Redundancy is a direct function of frame spacing and footage, so this does not transfer
# between sample_fps values or datasets. Calibrate it rather than copying a value.
vision_threshold=0.001

# --- stage 2: memory-side (model/token_pruning.py) -- the one that moves throughput.
# Drops redundant visual tokens before the LM; shrinks KV RAM and retrieval cost with it
# (measured +29% frames/s alone). 'none' = off, the untouched baseline. Space-separated
# list to sweep -- each threshold writes to its own results dir, so arms never overwrite
# each other or the baseline. Scales with sample_fps: fewer frames/sec = less redundancy
# = higher threshold, so do not transplant a value across frame rates either.
prune_method=none
prune_metric=cosine
prune_refresh_every=0
prune_thresholds="0.2"

# Per-video keep rates land in results.csv (see BaseVQA.reduction_stats) -- read them
# there to see what a threshold actually dropped.

for dataset in ${datasets}; do
  for thr in ${prune_thresholds}; do
    # Forward each stage's flags only when that stage is on. Not cosmetic: run_eval.py's
    # pruning_on() treats a --prune_threshold on its own as "enable rlt" (so older scripts
    # that only set a threshold keep working), so passing the threshold unconditionally
    # would run stage 2 even with prune_method=none -- and the tag below would not know it.
    reduce_args=""
    [ "${vision_method}" != "none" ] && reduce_args="${reduce_args} \
        --vision_method ${vision_method} --vision_threshold ${vision_threshold} \
        --vision_mask_space ${vision_mask_space} --vision_metric ${vision_metric} \
        --vision_refresh_every ${vision_refresh_every}"
    [ "${prune_method}" != "none" ] && reduce_args="${reduce_args} \
        --prune_method ${prune_method} --prune_threshold ${thr} \
        --prune_metric ${prune_metric} --prune_refresh_every ${prune_refresh_every}"

    # --trigger is fpsbench_stream's alone; forwarding it elsewhere would put an
    # inapplicable flag on every other dataset's command line.
    trigger_args=""
    [ "${dataset}" = "fpsbench_stream" ] && trigger_args="--trigger ${trigger}"

    echo "=== ${model} | ${dataset} | s1=${vision_method} | s2=${prune_method}@${thr} ==="
    python -m video_qa.run_eval \
        --num_chunks ${num_chunks} \
        --model ${model} \
        --dataset ${dataset} \
        --sample_fps ${sample_fps} \
        --n_local ${n_local} \
        --retrieve_size ${retrieve_size} \
        ${skip_scoring} \
        ${trigger_args} \
        ${reduce_args}
  done
done
