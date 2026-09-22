#!/bin/bash
#SBATCH --job-name=fpsbench_needle
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/fpsbench_needle-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --qos=group2-2           # 2 GPUs, up to 190G
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G               # needle only: <=100 frames per question, KV-Cache is small

# FPS-Bench-Stream, needle only: each question sees only the FPSBench clip, on the same
# exact frame grid as the full-stream run, with the 600 s haystack cut away. Counterpart of
# scripts/eval/eval.slurm's fpsbench_stream sweep; results go to
#   results/needle_only/<model>/fpsbench_stream/64-<fps>[-rlt_ref<thr>cosine]
# Collect with:
#   python scripts/collect_fpsbench_sweep.py --model llava_ov_7b --results_root results/needle_only

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
fps_list="${FPS_LIST:-1 2 4}"
arms="${ARMS:-baseline 0.5 0.9}"   # baseline = unpruned; numbers are RLT thresholds
needle_stop="${NEEDLE_STOP:-needle}"   # 'needle' = whole clip, 'query' = stop at query_time_sec
n_local=15000
retrieve_size=64
prune_method=rlt_ref

for model in ${models}; do
  for fps in ${fps_list}; do
    for arm in ${arms}; do
      prune_args=""
      [ "${arm}" != "baseline" ] && prune_args="--prune_method ${prune_method} --prune_threshold ${arm}"

      echo "=== ${model} | fpsbench_stream needle-only (${needle_stop}) | ${fps} fps | ${arm} ==="
      python -m video_qa.run_eval_fpsbench_needle \
          --num_chunks ${num_chunks} \
          --model ${model} \
          --sample_fps ${fps} \
          --n_local ${n_local} \
          --retrieve_size ${retrieve_size} \
          --needle_stop ${needle_stop} \
          ${prune_args}
    done
  done
done

for model in ${models}; do
  python scripts/collect_fpsbench_sweep.py --model ${model} --results_root results/needle_only \
      --csv results/needle_only/${model}/fpsbench_stream/sweep.csv
done

echo "Job finished at $(date)"
