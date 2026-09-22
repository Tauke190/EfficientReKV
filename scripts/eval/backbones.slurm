#!/bin/bash
#SBATCH --job-name=rekv_backbones
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_backbones-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --qos=group2-2           # 2 GPUs, up to 190G
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=160G

# Backbone ablation: does rlt_ref (ours) generalise beyond LLaVA-OneVision?
# Same experiment as the llava_ov_7b runs on disk (StreamingBench real-time, 1 fps,
# rlt_ref 0.5 cosine), on the other backbones. Each backbone runs its own unpruned
# baseline too, because budgets differ across backbones -- compare each pruned arm with
# its own baseline, not with llava_ov's.
#
# Per-backbone budgets (n_local in tokens, retrieve_size in blocks = frames):
#     llava_ov_7b      196 tok/frame  Qwen2   32k   15000 / 64   (76.5 frames; on disk already)
#     longva_7b        144 tok/frame  Qwen2  224k   11020 / 64   (same 76.5 frames as llava_ov)
#     video_llava_7b   257 tok/frame  Vicuna   4k    3084 /  8   (loader default; 4k context)
#     flash_vstream_7b  64 tok/frame  Vicuna   4k    3648 / 16   (loader default; 4k context)
# The two Vicuna backbones cannot hold llava_ov's window: load_model asserts on it,
# because over-running a 4k context returns '' silently rather than raising.
#
# Threshold 0.5 is a distance in each backbone's own feature space, so it gives a
# different keep rate on each -- read token_keep_rate from results.csv.
#
# Arms whose results.csv already exists are skipped, so an interrupted job resumes.
# FORCE=1 re-runs them. A failed arm does not stop the others, but the job exits
# non-zero at the end so slurm reports FAILED instead of COMPLETED.
#
# Results: results/<model>/streamingbench_real/<retrieve_size>-1.0[-rlt_ref0.5cosine]
# Usage:
#   sbatch scripts/eval/backbones.slurm                       # all three, one after another
#   MODELS=longva_7b sbatch scripts/eval/backbones.slurm      # one backbone per job
#   ARMS=0.5 sbatch scripts/eval/backbones.slurm              # pruned arm only

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate Rekv

export MPLBACKEND=Agg        # stop matplotlib dumping PNGs into the log
export TQDM_MININTERVAL=60

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/EfficientReKV
# Repo root on sys.path so workers don't import the other ReKV clone
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

num_chunks="${NUM_CHUNKS:-${SLURM_GPUS_ON_NODE:-2}}"   # one process per GPU
models="${MODELS:-longva_7b video_llava_7b flash_vstream_7b}"
arms="${ARMS:-baseline 0.5}"   # baseline = unpruned; numbers are rlt_ref thresholds
dataset=streamingbench_real
fps=1
prune_method=rlt_ref
prune_metric=cosine

budget() {  # model -> "n_local retrieve_size"
  case $1 in
    llava_ov_0.5b|llava_ov_7b) echo "15000 64" ;;
    longva_7b)                 echo "11020 64" ;;
    video_llava_7b)            echo "3084 8" ;;
    flash_vstream_7b)          echo "3648 16" ;;
    *) echo "no budget defined for $1" >&2; return 1 ;;
  esac
}

# Same formatting run_eval uses for the result dir: fps as a float, threshold as %g.
fmt_fps() { python -c "print(float('$1'))"; }
fmt_thr() { python -c "print(f'{float(\"$1\"):g}')"; }

failed=()
for model in ${models}; do
  read -r n_local retrieve_size < <(budget ${model}) || { failed+=("${model}"); continue; }
  for arm in ${arms}; do
    prune_args=""; tag=""
    if [ "${arm}" != "baseline" ]; then
      prune_args="--prune_method ${prune_method} --prune_threshold ${arm} --prune_metric ${prune_metric}"
      tag="-${prune_method}$(fmt_thr ${arm})${prune_metric}"
    fi
    save_dir="results/${model}/${dataset}/${retrieve_size}-$(fmt_fps ${fps})${tag}"

    if [ "${FORCE:-0}" != 1 ] && [ -s "${save_dir}/results.csv" ]; then
      echo "=== skip ${model} | ${arm}: ${save_dir} exists ==="
      continue
    fi

    echo "=== ${model} | ${dataset} | ${fps} fps | ${arm} | n_local ${n_local} | retrieve ${retrieve_size} ==="
    echo "=== started $(date) ==="
    python -m video_qa.run_eval \
        --num_chunks ${num_chunks} \
        --model ${model} \
        --dataset ${dataset} \
        --sample_fps ${fps} \
        --n_local ${n_local} \
        --retrieve_size ${retrieve_size} \
        ${prune_args}
    rc=$?
    # run_eval can return 0 after a worker crashed, leaving an empty results.csv behind
    # (that is how the needle job lost its 2/4 fps arms), so check the output too.
    if [ ${rc} -ne 0 ] || [ ! -s "${save_dir}/results.csv" ]; then
      echo "*** FAILED: ${model} | ${arm} (exit ${rc}, ${save_dir}/results.csv missing or empty) ***"
      failed+=("${model}:${arm}")
    fi
    echo "=== finished $(date) ==="
  done
done

echo "Job finished at $(date)"
if [ ${#failed[@]} -gt 0 ]; then
  echo "*** ${#failed[@]} arm(s) failed: ${failed[*]} ***"
  exit 1
fi
