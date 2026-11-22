#!/bin/bash
#SBATCH --job-name=rekv_isoretention
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_isoretention-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --qos=group2-3           # up to 36 CPUs / 280G per job
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=240G              # same budget as ablation.slurm (StreamBench peaked at ~44G/worker)

# Iso-retention ablation of TTS (rlt_ref): the two ablated rules evaluated at the threshold
# where each keeps the SAME share of tokens as TTS, instead of at the same threshold.
#     rlt_ref   ours         keep token iff dist(feat[t,p], last KEPT feat[p]) > thr
#     rlt_prev  reference    keep token iff dist(feat[t,p], feat[t-1,p])       > thr
#     rlt_frame granularity  keep ALL of frame t iff >50% of its tokens pass the rlt_ref test
#
# Matched thresholds come from the token-retention sweep
# (scripts/efficiency/token_retention_ablation.py, StreamBench full_oe, llava_ov_7b, 1 fps,
# all 275 videos, 0.01 grid). Token-weighted keep rate per arm:
#
#   removal   rlt_ref          rlt_prev          rlt_frame
#   ~74%      0.50 -> 26.02%   0.47 -> 26.63%    0.44 -> 27.07%
#   ~85%      0.60 -> 15.03%   0.57 -> 14.91%    0.49 -> 14.45%
#   ~92%      0.70 -> 7.61%    0.66 -> 7.76%     0.52 -> 8.30%
#
# rlt_frame moves ~2.5 pp per 0.01 of threshold, so 0.01 is as close as it gets; set
# FRAME_ARMS="0.445 0.488 0.525" for interpolated, tighter matches (unverified by the sweep).
# Report the achieved `token_keep_rate` from each results.csv next to its accuracy.
#
# The rlt_ref arms are normally already on disk from eval.slurm and are skipped; they are
# listed so a fresh checkout reproduces the whole table. Arms whose results.csv exists are
# skipped, so an interrupted job resumes. FORCE=1 re-runs them.
#
# StreamBench is open-ended: run_eval scores it with the pinned Meta-Llama-3-8B-Instruct
# judge (--judge_preset streambench), the same judge as every other StreamBench number.
#
# Results: results/<model>/streambench/64-<fps>-<method><thr>cosine
# Usage:
#   sbatch scripts/eval/isoretention.slurm
#   PREV_ARMS="0.475" FRAME_ARMS="0.445" sbatch scripts/eval/isoretention.slurm
#   EXTRA_ARGS="--skip_scoring" sbatch scripts/eval/isoretention.slurm   # QA only

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
model="${MODEL:-llava_ov_7b}"
dataset="${DATASET:-streambench}"
fps="${FPS:-1}"
ref_arms="${REF_ARMS:-0.5 0.6 0.7}"
prev_arms="${PREV_ARMS:-0.47 0.57 0.66}"
frame_arms="${FRAME_ARMS:-0.44 0.49 0.52}"
extra_args="${EXTRA_ARGS:-}"
n_local=15000
retrieve_size=64

# Same formatting run_eval uses for the result dir: fps as a float, threshold as %g.
fmt_fps() { python -c "print(float('$1'))"; }
fmt_thr() { python -c "print(f'{float(\"$1\"):g}')"; }

done_already() {  # method thr
  local d
  for d in results/${model}/${dataset}/${retrieve_size}-$(fmt_fps ${fps})*-$1$(fmt_thr $2)cosine; do
    [ -s "${d}/results.csv" ] && { echo "${d}"; return 0; }
  done
  return 1
}

arms=()
for t in ${ref_arms};   do arms+=("rlt_ref ${t}");   done
for t in ${prev_arms};  do arms+=("rlt_prev ${t}");  done
for t in ${frame_arms}; do arms+=("rlt_frame ${t}"); done

for arm in "${arms[@]}"; do
  set -- ${arm}
  method=$1; thr=$2
  if [ "${FORCE:-0}" != 1 ] && existing=$(done_already ${method} ${thr}); then
    echo "=== skip ${model} | ${dataset} | ${fps} fps | ${method} ${thr}: ${existing} ==="
    continue
  fi
  echo "=== $(date) ${model} | ${dataset} | ${fps} fps | ${method} ${thr} ==="
  python -m video_qa.run_eval \
      --num_chunks ${num_chunks} \
      --model ${model} \
      --dataset ${dataset} \
      --sample_fps ${fps} \
      --n_local ${n_local} \
      --retrieve_size ${retrieve_size} \
      --prune_method ${method} \
      --prune_cache adaptive \
      --prune_threshold ${thr} \
      ${extra_args}
done

echo "Job finished at $(date)"
