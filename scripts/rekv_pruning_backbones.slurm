#!/bin/bash
#SBATCH --job-name=rekv_prune_backbones
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_prune_backbones-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# Cross-backbone stage-2 pruning comparison on OVO-Bench: llava_ov_7b vs longva_7b.
#
# The question. Stage-2 pruning (model/token_pruning.py) was developed and swept on
# llava-onevision. Does it behave the same way on a different backbone, or was the
# measured win a property of that one vision stack? Same benchmark, same fps, same
# retrieval budget, same arms -- only the backbone changes.
#
# ---------------------------------------------------------------------------------
# Matching the two backbones. This is the whole design of the script, so read it before
# changing a knob.
#
# n_local is denominated in TOKENS (model/abstract_rekv.py:356), and the two backbones do
# not spend tokens at the same rate:
#     llava_ov_7b   196 tokens/frame   (model/llava_onevision_rekv.py:209)
#     longva_7b     144 tokens/frame   (model/longva_rekv.py:137)
# So a shared n_local=15000 would give llava_ov a 76.5-frame local window and longva a
# 104-frame one -- longva would see 36% further back before retrieval ever engages, and
# every throughput and accuracy difference would be confounded by that. Frames are the
# unit the comparison has to hold fixed, because frames are what both models perceive and
# what frames/s is measured in.
#
# Hence n_local is DERIVED per backbone from a shared frame budget, not set per backbone
# by hand:
#     frame budget = REF_N_LOCAL / REF_FRAME_TOKENS = 15000 / 196 = 76.5 frames
#     n_local(m)   = round(76.5 x tokens_per_frame(m))
#         llava_ov_7b  ->  15000  (76.5 frames, the reference, unchanged)
#         longva_7b    ->  11020  (76.5 frames)
# Computed in-script rather than hardcoded so that changing REF_N_LOCAL re-derives every
# backbone consistently instead of silently un-matching them.
#
# retrieve_size deliberately stays 64 for BOTH and is not rescaled. It counts BLOCKS, and
# a block is n_frame_tokens wide (Abstract_ReKV.block_size), so 64 blocks is 64 frames on
# either backbone at baseline -- already frame-equal. Scaling it the way n_local is scaled
# would break the match, not preserve it.
#
# ---------------------------------------------------------------------------------
# The one thing this script CANNOT match, and what to do about it.
#
# The rlt threshold is a cosine distance in the backbone's own feature space, and those
# spaces are not shared -- different tower (SigLIP vs CLIP ViT-L), different projector,
# different pooling (model/longva_rekv.py:128-131 says so explicitly). rlt:0.7 on
# llava_ov and rlt:0.7 on longva are NOT the same amount of pruning, and reading them as
# a matched pair is the one mistake this comparison invites.
#
# So do not compare arm-to-arm. Compare the two backbones' accuracy-vs-keep-rate CURVES,
# interpolated to a matched keep rate. `token_keep_rate` is written per video into
# results.csv (BaseVQA.reduction_stats) and is the x-axis that is actually common to both.
# That is why ARMS is a spread of thresholds rather than a single operating point: the
# curve is the measurement, an individual arm is not.
#
# Calibration hint for longva, the backbone with no ovobench history: rlt:0.2 on
# rvs_movie held a 90.4% keep rate (mean 90.7%, min 88.3%, max 95.0% over 10 videos), so
# longva's useful range starts well below llava_ov's. If the arms below all come back
# above ~90% keep, push the longva thresholds up rather than concluding pruning does
# nothing here.
#
# ---------------------------------------------------------------------------------
# What already exists, and why this script refuses to overwrite it.
#
# n_local is NOT part of the results path (run_eval.py:409 -- it is
# {retrieve_size}-{sample_fps}{reduction_tag}), so a run at n_local=11020 lands in exactly
# the directory a run at 15000 would. results/llava_ov_7b/ovobench_* already holds a
# completed sweep at n_local=15000, 1 fps:
#     realtime   rlt 0.6 0.7 0.8 0.9  (+ blind)     839 rows each
#     backward   rlt 0.5 0.6 0.7 0.8 0.9 (+ blind)  632 rows each
# Those were run at this script's reference settings, so they are reusable as-is -- and
# they would be destroyed by a careless re-run. OVERWRITE=0 (the default) skips any arm
# whose results.csv is already there and says so. Set OVERWRITE=1 only when you mean it.
#
# Note the gap that leaves: there is no unpruned llava_ov_7b baseline (`64-1.0`) for
# either mode -- only `-blind` and the pruned arms. Without it the pruned arms have
# nothing to be a delta from, so 'none:0' below is doing real work for llava_ov even
# though most of its pruned arms are already on disk.
#
# ---------------------------------------------------------------------------------
# Cost. 78k (realtime) + 65k (backward) = 143k frames per model-arm pass over both modes
# at 1 fps. llava_ov_7b runs ~4.5 h for that on 4 GPUs, so ~9 h on the 2 here; longva_7b
# is lighter per frame (144 vs 196 tokens through the same Qwen2-7B) but carries a CLIP
# ViT-L tower instead of SigLIP, so budget it as comparable. With OVERWRITE=0 the work
# below is roughly: llava_ov 2 baseline passes + realtime rlt:0.5, longva 8 passes --
# call it 40+ h in total. Trim MODELS or ARMS to run it in pieces; every arm is
# independent and writes to its own directory.

module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate Rekv

LOGDIR=/home/av354855/EfficientReKV/slurmlogs

export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi
free -g

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

MODELS="llava_ov_7b longva_7b"
DATASETS="ovobench_realtime ovobench_backward"

# 1 fps, matching scripts/eval_ovobench_baseline.slurm -- and not negotiable here, because
# the existing llava_ov results this script reuses are at 1.0. At 0.5 fps roughly half the
# queries never reach past the local window, so the run would measure the backbone rather
# than ReKV's memory.
SAMPLE_FPS=1

# The frame budget every backbone is matched to, expressed as one backbone's token budget.
# llava_ov_7b at n_local=15000 is the reference because that is what the existing results
# on disk were run at.
REF_N_LOCAL=15000
REF_FRAME_TOKENS=196

# Blocks, not tokens -- 64 blocks is 64 frames on both backbones. See the header.
RETRIEVE_SIZE=64

# Slots decoded at a time. OVO-Bench videos run to 1695 s at up to 8 MB/frame, so a
# whole-video decode can hold 14 GB of pixels per worker; a window bounds that at ~2 GB.
# The frames handed to the model are identical either way.
DECODE_WINDOW=256

PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

# Stage-2 arms as "<method>:<threshold>", 'none:0' being the untouched baseline. The same
# list is used for both backbones ON PURPOSE -- not because the thresholds mean the same
# thing (they do not, see the header), but because a shared spread is what lets the two
# accuracy-vs-keep-rate curves be interpolated to a common keep rate afterwards. Widen it
# per backbone if one of them bunches up at one end of the keep-rate range.
ARMS="none:0 rlt:0.5 rlt:0.7 rlt:0.9"

# 0 = skip an arm whose results.csv already exists (default; protects the completed
# llava_ov sweep). 1 = re-run and overwrite it.
OVERWRITE=0

# GPUs this job asked for; must match #SBATCH --gres=gpu:N above.
GPUS=2
# Host RAM all workers together may occupy. ReKV holds the whole KV-Cache in pinned host
# memory for the length of a video and the pool does not come back between videos, so a
# worker's peak is MEM_FIXED_GB + MEM_GB_PER_FRAME x frames on the longest video it sees.
MEM_BUDGET_GB=200
MAX_STREAM_SEC=1695   # longest last-query timestamp in either mode

for f in data/ovo_bench/realtime.json data/ovo_bench/backward.json; do
    if [ ! -f "${f}" ]; then
        echo "FATAL: ${f} missing -- build both first:" >&2
        echo "  python video_qa/convert_ovobench.py" >&2
        exit 1
    fi
done

for MODEL in ${MODELS}; do
  # N_FRAME_TOKENS must track the value the loader actually uses; if you change a backbone's
  # tokens/frame, change it here too or the frame matching silently stops holding.
  #
  # The memory figures: llava_ov_7b's are measured (carried over from
  # scripts/eval_ovobench_baseline.slurm). longva_7b's are DERIVED, not measured --
  # both backbones run the identical Qwen2-7B LM (28 layers, 28 heads, 4 KV heads,
  # hidden 3584, verified from both config.json), so per-token KV bytes are identical and
  # per-frame host memory scales by exactly 144/196: 0.0163 x 144/196 = 0.0120. The fixed
  # term is weights plus workspace and is the same model size, so it carries over. Treat
  # them as an estimate until a longva run is actually watched with `free -g`.
  case "${MODEL}" in
      llava_ov_7b) N_FRAME_TOKENS=196 ; MEM_FIXED_GB=32 ; MEM_GB_PER_FRAME=0.0163 ;;
      longva_7b)   N_FRAME_TOKENS=144 ; MEM_FIXED_GB=32 ; MEM_GB_PER_FRAME=0.0120 ;;
      *) echo "FATAL: ${MODEL} has no tokens/frame or memory figures here; add them" >&2
         echo "       before running, or the frame matching below is meaningless." >&2
         exit 1 ;;
  esac

  # The frame-matched local window. See the header for why this is derived and not typed.
  read N_LOCAL FRAME_BUDGET <<< "$(awk -v r="${REF_N_LOCAL}" -v rt="${REF_FRAME_TOKENS}" \
      -v ft="${N_FRAME_TOKENS}" 'BEGIN{ printf "%.0f %.1f", r*ft/rt, r/rt }')"

  read PER_WORKER_GB NUM_CHUNKS <<< "$(awk -v fps="${SAMPLE_FPS}" -v sec="${MAX_STREAM_SEC}" \
      -v fixed="${MEM_FIXED_GB}" -v slope="${MEM_GB_PER_FRAME}" \
      -v budget="${MEM_BUDGET_GB}" -v gpus="${GPUS}" 'BEGIN{
        per = fixed + fps * sec * slope;
        n = int(budget / per); if (n > gpus) n = gpus;
        printf "%.0f %d", per, n }')"

  if [ "${NUM_CHUNKS}" -lt 1 ]; then
    echo "=== SKIP ${MODEL}: one worker would peak near ${PER_WORKER_GB} GB on the longest"
    echo "=== video, over the ${MEM_BUDGET_GB} GB budget for the whole node."
    continue
  fi
  if [ "${NUM_CHUNKS}" -lt "${GPUS}" ]; then
    echo "NOTE: ${MODEL} peaks near ${PER_WORKER_GB} GB per worker, so it runs on"
    echo "      ${NUM_CHUNKS} of ${GPUS} GPUs to stay inside ${MEM_BUDGET_GB} GB."
  fi

  echo "=========================================================================="
  echo "=== ${MODEL}: ${N_FRAME_TOKENS} tokens/frame -> n_local=${N_LOCAL}"
  echo "===   (${FRAME_BUDGET} frames, matched to ${REF_N_LOCAL} tokens at ${REF_FRAME_TOKENS} tokens/frame)"
  echo "=========================================================================="

  for DATASET in ${DATASETS}; do
    for ARM in ${ARMS}; do
      METHOD="${ARM%%:*}"
      THR="${ARM##*:}"

      REDUCE_ARGS=""
      TAG=""
      if [ "${METHOD}" != "none" ]; then
          REDUCE_ARGS="--prune_method ${METHOD} --prune_threshold ${THR} \
              --prune_metric ${PRUNE_METRIC} --prune_refresh_every ${PRUNE_REFRESH_EVERY}"
          # Mirrors reduction_tag() in run_eval.py:105-114; %g drops the trailing zero the
          # same way its {:g} does, so this predicts the real directory rather than a
          # near-miss that would defeat the skip check.
          TAG="-${METHOD}$(awk -v t="${THR}" 'BEGIN{printf "%g", t}')${PRUNE_METRIC}"
      fi
      SAVE_DIR="results/${MODEL}/${DATASET}/${RETRIEVE_SIZE}-$(awk -v f="${SAMPLE_FPS}" 'BEGIN{printf "%.1f", f}')${TAG}"

      if [ "${OVERWRITE}" -eq 0 ] && [ -s "${SAVE_DIR}/results.csv" ]; then
        echo "=== SKIP ${MODEL} | ${DATASET} | ${METHOD}@${THR}: $(wc -l < "${SAVE_DIR}/results.csv") rows already in"
        echo "===   ${SAVE_DIR}"
        echo "===   (set OVERWRITE=1 to re-run it)"
        continue
      fi

      echo "=========================================================================="
      echo "=== ${MODEL} | ${DATASET} | ${SAMPLE_FPS} fps | n_local=${N_LOCAL} (${FRAME_BUDGET} frames) | s2=${METHOD}@${THR} ==="
      echo "=== ${NUM_CHUNKS} workers x ~${PER_WORKER_GB} GB host memory at the peak ==="
      echo "=== -> ${SAVE_DIR}"
      echo "=== started $(date) ==="
      python -m video_qa.run_eval \
          --num_chunks ${NUM_CHUNKS} \
          --model ${MODEL} \
          --dataset ${DATASET} \
          --sample_fps ${SAMPLE_FPS} \
          --n_local ${N_LOCAL} \
          --retrieve_size ${RETRIEVE_SIZE} \
          --decode_window ${DECODE_WINDOW} \
          ${REDUCE_ARGS}
      echo "=== finished $(date) ==="
      free -g
    done
  done
done

echo "Job finished at $(date)"

conda deactivate
