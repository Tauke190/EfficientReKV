#!/bin/bash
#SBATCH --job-name=fpsbench_rekv
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_eval-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --qos=group2-2           # 2 GPUs, up to 190G
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=160G              # RLT 0.5/0.9 at <=4 fps, 2 workers: 0.5 keeps ~20-27% of tokens (max ~66%/video) -> <=~57 GB/worker worst case

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
# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
#                  flash_vstream_7b
models="${MODELS:-llava_ov_7b}"
arms="${ARMS:-0.5 0.9}"           # space-separated: baseline and/or RLT thresholds, e.g. "baseline 0.6"

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
dataset=fpsbench_stream
fps_list="${FPS_LIST:-1 2 4}"   # space-separated; each rate is its own run and results dir
n_local=15000        # tokens (llava_ov 196/frame, longva 144, video_llava 257, flash_vstream 64)
retrieve_size=64     # blocks (frames) retrieved per query
prune_method=rlt_ref

# Scoring stays on: multiple-choice, and it runs the no-lookahead audit
for model in ${models}; do
  for fps in ${fps_list}; do
    for arm in ${arms}; do
      prune_args=""
      [ "${arm}" != "baseline" ] && prune_args="--prune_method ${prune_method} --prune_threshold ${arm}"

      echo "=== ${model} | ${dataset} | ${fps} fps | ${arm} ==="
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
done
