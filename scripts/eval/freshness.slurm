#!/bin/bash
#SBATCH --job-name=rekv_fresh
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_fresh-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --qos=group2-2           # 2 GPUs, up to 190G
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=160G

# CAC freshness audit (reviewer: "your method discards the latest frames for real-time
# queries"). Re-runs the pruned arms with the per-question fresh_* columns that
# Abstract_ReKV.freshness_snapshot now records, then prints the audit against the
# existing unpruned run on the same questions.
#
# The baseline is NOT re-run by default: unpruned memory has no staleness (every frame
# keeps its tokens, nothing is pending), so it only serves as the accuracy control, and
# the existing results/<model>/<dataset>/64-1.0/results.csv already is that control.
# Put "baseline" in ARMS if that directory does not exist yet.
#
# Results land in '-fresh' directories (--save_tag), so the original runs are untouched.
#
#   sbatch scripts/eval/freshness.slurm
#   DATASETS="streamingbench_real ovobench_realtime" ARMS="0.5 0.8" sbatch scripts/eval/freshness.slurm
#   MODELS=llava_ov_0.5b sbatch scripts/eval/freshness.slurm

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

num_chunks="${NUM_CHUNKS:-${SLURM_GPUS_ON_NODE:-2}}"   # one process per GPU
models="${MODELS:-llava_ov_7b}"
datasets="${DATASETS:-streamingbench_real}"   # also: ovobench_realtime
arms="${ARMS:-0.5 0.8}"                       # RLT thresholds; add "baseline" if needed
fps="${FPS:-1}"
fps_dir=$(python -c "print(float('${fps}'))")   # run_eval names dirs with the float, e.g. 1.0
n_local=15000
retrieve_size=64
prune_method=rlt_ref
tag=fresh

for model in ${models}; do
  for dataset in ${datasets}; do
    run_dirs=""
    for arm in ${arms}; do
      prune_args=""; suffix="-${tag}"
      if [ "${arm}" != "baseline" ]; then
        prune_args="--prune_method ${prune_method} --prune_threshold ${arm}"
        suffix="-${prune_method}${arm}cosine-${tag}"
      fi

      echo "=== ${model} | ${dataset} | ${fps} fps | ${arm} | $(date +%T) ==="
      python -m video_qa.run_eval \
          --num_chunks ${num_chunks} \
          --model ${model} \
          --dataset ${dataset} \
          --sample_fps ${fps} \
          --n_local ${n_local} \
          --retrieve_size ${retrieve_size} \
          --save_tag ${tag} \
          ${prune_args}
      run_dirs="${run_dirs} results/${model}/${dataset}/${retrieve_size}-${fps_dir}${suffix}/results.csv"
    done

    # Audit. The control is the original unpruned run, or the re-run one if it was asked for.
    base="results/${model}/${dataset}/${retrieve_size}-${fps_dir}/results.csv"
    [ -f "results/${model}/${dataset}/${retrieve_size}-${fps_dir}-${tag}/results.csv" ] && \
        base="results/${model}/${dataset}/${retrieve_size}-${fps_dir}-${tag}/results.csv"
    base_arg=""
    if [ -f "${base}" ]; then base_arg="--baseline ${base}"
    else echo "WARNING: no unpruned run at ${base}; audit runs without the paired control"; fi

    echo "=== freshness audit | ${model} | ${dataset} ==="
    python video_qa/eval/freshness_audit.py ${run_dirs} ${base_arg} --by task \
        --out "results/${model}/${dataset}/freshness_audit.csv"
  done
done

echo "Job finished at $(date)"
