#!/bin/bash
# ReKV evaluation, run directly (no scheduler). scripts/eval.slurm is the sbatch twin --
# same knobs, same loop; keep the two in step when changing either.

# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.
num_chunks=1

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
model=llava_ov_0.5b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
# ovobench_realtime / ovobench_backward / fpsbench_stream are valid --dataset values too,
# but need their own annotation build and sample_fps -- use their own scripts instead.
# Space-separated: each dataset is evaluated in turn.
datasets="rvs_ego rvs_movie"
sample_fps=0.5
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
prune_method=rlt
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

    echo "=== ${model} | ${dataset} | s1=${vision_method} | s2=${prune_method}@${thr} ==="
    python -m video_qa.run_eval \
        --num_chunks ${num_chunks} \
        --model ${model} \
        --dataset ${dataset} \
        --sample_fps ${sample_fps} \
        --n_local ${n_local} \
        --retrieve_size ${retrieve_size} \
        ${skip_scoring} \
        ${reduce_args}
  done
done
