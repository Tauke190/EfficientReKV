#!/bin/bash
#SBATCH --job-name=rekv_matched
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_matched-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --qos=group2-2           # 2 GPUs, up to 190G
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=160G

# Content-agnostic baselines at matched token rate, @ 1 fps:
#   uniform_frame  evenly spaced whole frames (7% of tokens at 1 fps ~ 1 frame / 14 s)
#   random_token   random tokens in every frame
# Each one is matched to an rlt_ref arm that is already on disk. The rate is that arm's
# token-weighted token_keep_rate, sum(tokens_kept)/sum(tokens_seen), read from its
# results.csv. It is per backbone, because the same threshold keeps a different fraction on
# each. So the pairs are compared at an equal budget, not at an equal threshold.
#
# The matched rlt_ref arm has to exist first (scripts/eval/backbones.slurm). Arms whose
# results.csv exists are skipped; FORCE=1 re-runs them.
#
# Default: llava_ov_7b vs rlt_ref 0.7 (~7-8% kept, ~1 frame / 12-15 s for uniform_frame) on the
# three datasets whose accuracy actually drops at that budget. ovbench/odvbench are flat down to
# ~2% kept, so they cannot separate the arms.
#
# Results: results/<model>/<dataset>/<retrieve_size>-1.0-{uniform_frame<rate>,random_token<rate>-s<seed>}
# Usage:
#   sbatch scripts/eval/matched_baselines.slurm                          # all three datasets, one job
#   DATASETS=streambench sbatch scripts/eval/matched_baselines.slurm     # one dataset per job
#   MODELS="llava_ov_7b longva_7b" THRS="0.5 0.7" sbatch scripts/eval/matched_baselines.slurm
#   METHODS=random_token SEEDS="0 1 2" sbatch scripts/eval/matched_baselines.slurm

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

num_chunks="${NUM_CHUNKS:-${SLURM_GPUS_ON_NODE:-2}}"
models="${MODELS:-llava_ov_7b}"
thrs="${THRS:-0.7}"                         # rlt_ref thresholds to match
methods="${METHODS:-uniform_frame random_token}"
seeds="${SEEDS:-0}"                          # random_token only
datasets="${DATASETS:-streamingbench_real streambench ovobench_realtime}"
fps=1

budget() {  # model -> "n_local retrieve_size"; must match backbones.slurm
  case $1 in
    llava_ov_0.5b|llava_ov_7b) echo "15000 64" ;;
    longva_7b)                 echo "11020 64" ;;
    video_llava_7b)            echo "3084 8" ;;
    flash_vstream_7b)          echo "3648 16" ;;
    *) echo "no budget defined for $1" >&2; return 1 ;;
  esac
}
fmt_fps() { python -c "print(float('$1'))"; }
fmt_thr() { python -c "print(f'{float(\"$1\"):g}')"; }
# Token-weighted keep rate of an rlt_ref run, to 4 decimals (it also names the result dir).
keep_rate() {
  python -c "import pandas as pd; d=pd.read_csv('$1'); print(round(d.tokens_kept.sum()/d.tokens_seen.sum(), 4))"
}

failed=()
for dataset in ${datasets}; do
for model in ${models}; do
  read -r n_local retrieve_size < <(budget ${model}) || { failed+=("${model}"); continue; }
  for thr in ${thrs}; do
    ref="results/${model}/${dataset}/${retrieve_size}-$(fmt_fps ${fps})-rlt_ref$(fmt_thr ${thr})cosine/results.csv"
    if [ ! -s "${ref}" ]; then
      echo "*** ${model} | rlt_ref ${thr}: ${ref} missing, nothing to match ***"
      failed+=("${model}:${dataset}:rlt_ref${thr}"); continue
    fi
    rate=$(keep_rate ${ref})
    echo "=== ${model} | ${dataset} | rlt_ref ${thr} keeps ${rate} of tokens ==="

    for method in ${methods}; do
      for seed in $( [ "${method}" = random_token ] && echo ${seeds} || echo 0 ); do
        tag="-${method}$(fmt_thr ${rate})"
        [ "${method}" = random_token ] && tag="${tag}-s${seed}"
        save_dir="results/${model}/${dataset}/${retrieve_size}-$(fmt_fps ${fps})${tag}"
        if [ "${FORCE:-0}" != 1 ] && [ -s "${save_dir}/results.csv" ]; then
          echo "=== skip ${save_dir} exists ==="; continue
        fi
        echo "=== ${model} | ${dataset} | ${method} ${rate} seed ${seed} | started $(date) ==="
        python -m video_qa.run_eval \
            --num_chunks ${num_chunks} \
            --model ${model} \
            --dataset ${dataset} \
            --sample_fps ${fps} \
            --n_local ${n_local} \
            --retrieve_size ${retrieve_size} \
            --prune_method ${method} --prune_keep_rate ${rate} --prune_seed ${seed}
        rc=$?
        if [ ${rc} -ne 0 ] || [ ! -s "${save_dir}/results.csv" ]; then
          echo "*** FAILED: ${save_dir} (exit ${rc}) ***"
          failed+=("${model}:${dataset}:${method}${rate}-s${seed}")
        fi
        echo "=== finished $(date) ==="
      done
    done
  done
done
done

echo "Job finished at $(date)"
if [ ${#failed[@]} -gt 0 ]; then
  echo "*** ${#failed[@]} arm(s) failed: ${failed[*]} ***"
  exit 1
fi
