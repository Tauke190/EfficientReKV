#!/bin/bash
# Run one evaluation. Edit the knobs below, then:  scripts/eval/eval.sh
# For a scheduled job use scripts/eval/eval.slurm -- same knobs, plus an #SBATCH header.

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# The repo root has to be on sys.path, not just be the cwd: run_eval launches its workers
# as `python video_qa/<solver>.py`, so without this their `video_qa`/`model` imports resolve
# through the env's editable rekv-1.0 install, which points at a different ReKV clone.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.
num_chunks=1

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
#                  flash_vstream_7b
model=llava_ov_0.5b

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
#
#   [judge] = answers are free-form sentences, so there is no letter to match. Add
#   --skip_scoring below and score afterwards with scripts/score_llmjudge_gpt.sh.
#   MLVU has an extremely long video (~9hr); drop it from the annotation if RAM is tight.
dataset=odvbench

# Space-separated -- each rate is a separate run into its own results directory.
# Pruning thresholds do NOT transfer across frame rates (frames are more alike at a higher
# rate), so sweep one axis at a time rather than crossing them and reading it as a curve.
sample_fps="1"

n_local=15000        # local window, in TOKENS (llava_ov spends 196 per frame)
retrieve_size=64     # blocks retrieved per query; one block is one frame

# Token pruning -- the thing this fork adds. Space-separated, each value its own run and
# its own results directory, so arms never overwrite each other. 'none' is the untouched
# ReKV baseline; put it first so the reference lands before the sweep.
prune_method=rlt
prune_thresholds="none 0.5"

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
