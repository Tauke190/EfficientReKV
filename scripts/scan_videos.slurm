#!/bin/bash
#SBATCH --job-name=scan_videos
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/scan_videos-%j.out
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Decode-check every video an annotation file references, so a corrupt file is found in a
# CPU hour here instead of after five GPU-hours of a sweep. CPU only -- no GPU is
# requested, and none is needed.
#
#   sbatch scripts/scan_videos.slurm                                    # StreamingBench real
#   sbatch --export=ALL,ANNO=data/OVBench/full_mc.json scripts/scan_videos.slurm
#
# Writes <anno>.videoscan.json next to the annotation and exits non-zero if anything failed.

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate Rekv

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

ANNO=${ANNO:-data/StreamingBench/real.json}
SAMPLE_FPS=${SAMPLE_FPS:-1.0}

echo "Job started at $(date) on $(hostname)"
echo "scanning ${ANNO} at ${SAMPLE_FPS} fps"

# decord chatters one h264 warning per damaged access unit -- 10k+ lines for a single bad
# file. The scan's own findings go to stdout, so keep stderr out of the log.
python scripts/scan_corrupt_videos.py \
    --anno "${ANNO}" \
    --sample_fps "${SAMPLE_FPS}" \
    --workers 16 \
    2>/dev/null
status=$?

echo "Job finished at $(date) with status ${status}"
exit ${status}
