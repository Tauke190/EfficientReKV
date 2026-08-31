#!/bin/bash
#SBATCH --job-name=rekv_blind_7b
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_blind_7b-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# The language-prior floor for llava_ov_7b on the three wired-up multiple-choice
# benchmarks: ovobench_backward, ovobench_realtime and odvbench.
#
# What this runs. --blind swaps in video_qa/blind_vqa.py, which clears the cache and
# encodes the system prompt exactly as a sighted run does and then answers the question
# against that cache alone -- no frame is ever decoded or handed to the vision tower.
# The number it produces is the floor every sighted 7B number in results/ has to be read
# against: how much of an accuracy is the answer distribution plus the language prior
# rather than the video. On ODV-Bench that is not a formality -- four subtasks totalling
# 3234 questions have a majority answer well above their own chance line (see the header
# of video_qa/blind_vqa.py).
#
# Cost. No decode, no ingestion, one short generate per query: 631 + 837 + 6348 queries,
# so the whole job is dominated by loading the 7B weights three times. Expect well under
# an hour, not the ~4.5 h a sighted 7B pass over OVO-Bench costs.
#
# Directory pairing. Each run writes to the sighted path plus a '-blind' suffix
# (results/llava_ov_7b/<subdir>/64-1.0-blind), so it sits next to 64-1.0 and joins to it
# row-for-row on (video_id, question_id). SAMPLE_FPS is still passed and still lands in
# the path even though no frame is sampled, which is what keeps the pairing a plain
# suffix rather than a lookup; keep it matched to the sighted runs being compared against.

module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate Rekv

export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi
free -g

cd /home/av354855/EfficientReKV
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

MODEL=llava_ov_7b
DATASETS="ovobench_backward ovobench_realtime odvbench"

# Nothing here is sampled, but it names the results directory, so it has to match the
# sighted runs these floors are being compared against (results/llava_ov_7b/*/64-1.0*).
SAMPLE_FPS=1

N_LOCAL=15000
RETRIEVE_SIZE=64

# Must match #SBATCH --gres=gpu:N above. A blind worker holds no KV-Cache and decodes no
# pixels, so unlike the sighted grids there is no host-memory model to cut the worker
# count down: every GPU the job has is usable.
GPUS=2

# Set to 0 to force a re-run of anything already on disk.
SKIP_EXISTING=1

for f in data/ovo_bench/backward.json data/ovo_bench/realtime.json data/odvbench/full_mc.json; do
    if [ ! -f "${f}" ]; then
        echo "FATAL: ${f} missing -- build the annotations first:" >&2
        echo "  python video_qa/convert_ovobench.py" >&2
        echo "  python scripts/setup_odvbench.py" >&2
        exit 1
    fi
done

for DATASET in ${DATASETS}; do
    case "${DATASET}" in
        ovobench_realtime) SUBDIR=ovobench_realtime ;;
        ovobench_backward) SUBDIR=ovobench_backward ;;
        odvbench)          SUBDIR=odvbench ;;
        *) echo "FATAL: unknown dataset ${DATASET}" >&2 ; exit 1 ;;
    esac

    SAVE_DIR="results/${MODEL}/${SUBDIR}/${RETRIEVE_SIZE}-${SAMPLE_FPS}.0-blind"
    if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${SAVE_DIR}/results.csv" ]; then
        echo "=== SKIP (already on disk): ${SAVE_DIR}"
        continue
    fi

    echo "=========================================================================="
    echo "=== ${MODEL} | ${DATASET} | blind (no frames ingested) ==="
    echo "=== -> ${SAVE_DIR} ==="
    echo "=== ${GPUS} workers ==="
    echo "=== started $(date) ==="
    python -m video_qa.run_eval \
        --num_chunks ${GPUS} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --sample_fps ${SAMPLE_FPS} \
        --n_local ${N_LOCAL} \
        --retrieve_size ${RETRIEVE_SIZE} \
        --blind
    echo "=== finished $(date) ==="
    free -g
done

echo "Job finished at $(date)"

conda deactivate
