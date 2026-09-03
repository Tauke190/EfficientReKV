#!/bin/bash
#SBATCH --job-name=rekv_thrpt
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_thrpt-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C "ampere&gmem32"
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=04:00:00

# Batch wrapper around scripts/measure_throughput.sh -- streaming ingestion FPS per
# reduction arm. Defaults to the half of the 7b sweep that is not on disk yet.
#
# results/throughput/ holds llava_ov_7b at baseline, rlt0.25 and rlt0.5 only; the 0.6-0.9
# arms were run for the 0.5b (results/throughput_sweep/) and never for the 7b. This fills
# that gap in the SAME directory and at the SAME settings as the existing 7b arms -- 400
# frames, 2 fps, encode_chunk_size 1, FPS-Bench-Stream videos 0/1/2 -- so all seven points
# sit on one curve. Do not point this at results/throughput_sweep instead: those runs are
# 1200 frames, and an FPS compared across stream lengths is not a comparison, because
# per-frame cost grows with the cache the frame attends over.
#
# WHY THE NODE IS PINNED. This measures hardware, so the GPU has to be held fixed across
# the whole curve or the sweep measures the scheduler. The existing 7b arms were produced
# on 2026-08-30 15:46-15:55 inside interactive job 360659, which held **c2-2** (ampere,
# gpumem:48G) from 11:54 to 17:58 -- so c2-2 is the reference hardware for this directory.
#
# `-C gmem48` is NOT sufficient to reproduce that and was the first attempt's bug: the
# constraint means *at least* 48 GB, and c4-4/c5-2 advertise both gmem48 and gmem80, so
# job 361940 landed on an A100 80GB. At ~1935 GB/s of HBM2e against c2-2's ~768 GB/s of
# GDDR6, and with streaming prefill bandwidth-bound, those points came out inflated and
# the speedup-vs-baseline ratios would have been reporting the GPU rather than the
# pruning. They are kept, unmixed, in results/throughput_a100/.
#
# This cluster REJECTS `-w` ("Request for specific node ignored"), so c2-2 cannot be
# pinned directly, and a bare submit with no constraint at all lands on whatever is free --
# job 361943 got c1-7, a 12 GB Pascal, which cannot even hold the 7b. The constraint has to
# do the work. `ampere&gmem32` resolves to exactly {c1-8, c2-1, c2-2, c4-2, c4-3}: every
# 48 GB ampere node, and neither 80 GB node, because c4-4/c5-2 advertise gmem48+gmem80 but
# carry no gmem32 feature. That is the tightest expression available here. Residual risk:
# A40 vs A6000 within that set differ ~10% in bandwidth, so PROVENANCE.txt below records
# the host and GPU of every run and should be checked before a curve is quoted.
#
# One GPU is right here: measure_throughput.sh is a sequential loop and the 7b peaks at
# ~20.8 GB, so a 48 GB card has ample headroom. Asking for more would only queue longer.
#
# Re-running is safe -- measure_throughput.sh skips any arm whose CSV already exists, so
# finished runs are not recomputed and an interrupted job resumes by resubmitting.
#
#     sbatch scripts/measure_throughput.slurm
#     sbatch -w c4-3 --export=ALL,MODELS="llava_ov_0.5b" scripts/measure_throughput.slurm

module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate Rekv

LOGDIR=/home/av354855/EfficientReKV/slurmlogs

# A full home quota kills the batch step in ~1s with ExitCode 0:53 and no .out file, before
# this script runs at all. This guard only catches the quota filling later; a job that
# vanishes leaving no log is a quota problem, not a code problem.
if ! touch "${LOGDIR}/.writetest" 2>/dev/null; then
    echo "FATAL: ${LOGDIR} not writable -- home quota is full. Free space and resubmit." >&2
    exit 1
fi
rm -f "${LOGDIR}/.writetest"

export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/EfficientReKV

export MODELS=${MODELS:-"llava_ov_7b"}
export THRESHOLDS=${THRESHOLDS:-"0.25 0.5 0.6 0.7 0.8 0.9"}
export OUT_DIR=${OUT_DIR:-results/throughput}
export VIDEO_IDXS=${VIDEO_IDXS:-"0 1 2"}
export NUM_FRAMES=${NUM_FRAMES:-400}

# Provenance. The CSVs record gpu_peak_bytes but not which GPU produced them, which is how
# an A100 run nearly got merged into a 48 GB curve. Append every contributing host/GPU here
# so a future reader can tell whether a directory is one curve or two.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
mkdir -p "${OUT_DIR}"
echo "$(date -Is) job=${SLURM_JOB_ID} host=$(hostname) gpu=${GPU_NAME} models=${MODELS} thresholds=${THRESHOLDS} frames=${NUM_FRAMES}" \
    >> "${OUT_DIR}/PROVENANCE.txt"

echo "GPU=${GPU_NAME}"
echo "MODELS=${MODELS} THRESHOLDS=${THRESHOLDS} NUM_FRAMES=${NUM_FRAMES} OUT_DIR=${OUT_DIR}"

bash scripts/measure_throughput.sh

echo "Job finished at $(date)"
