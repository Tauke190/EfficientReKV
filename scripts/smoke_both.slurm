#!/bin/bash
#SBATCH --job-name=rekv_smoke
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_smoke-%j.out
#SBATCH -p gpu
#SBATCH -C "ampere&gmem48"
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:45:00
# Throwaway: runs streamingbench_real and ovbench end-to-end on a handful of short videos
# to prove solver -> scorer works. Not a result; the subsets are far too small to read.
module load anaconda3; module load cuda
eval "$(conda shell.bash hook)"; conda activate Rekv
cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export TQDM_MININTERVAL=60
echo "started $(date) on $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader

run () {  # name solver anno outdir scorer
  echo "=============================== SMOKE: $1 ==============================="
  rm -rf "$4"; mkdir -p "$4"
  python "video_qa/$2.py" --model llava_ov_0.5b --sample_fps 1 --n_local 15000 \
      --retrieve_size 64 --save_dir "$4" --anno_path "$3" --debug false \
      --num_chunks 1 --chunk_idx 0 || { echo "SMOKE $1: SOLVER FAILED"; return 1; }
  head -n 1 "$4/1_0.csv" > "$4/results.csv"
  tail -n +2 "$4/1_0.csv" >> "$4/results.csv"
  rm -f "$4/1_0.csv"
  python "video_qa/eval/$5.py" --save_dir "$4" || { echo "SMOKE $1: SCORER FAILED"; return 1; }
  echo "SMOKE $1: OK"
}

run streamingbench_real rekv_streamingbench_vqa \
    data/StreamingBench/smoke_real.json /tmp/rekv_smoke/sb eval_streamingbench
run ovbench rekv_ovobench_vqa \
    data/OVBench/smoke_mc.json /tmp/rekv_smoke/ovb eval_ovbench

echo "finished $(date)"
