#!/bin/bash
#SBATCH --job-name=fpsbench_rekv
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_fpsbench-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# FPS-Bench-Stream frame-rate sweep: the same 990-stream (or subset) benchmark run at
# 1/2/4 fps, so accuracy can be read against the sampling rate.
#


# Two ceilings bound the top of the sweep, and both are real:
#
# * **The source rate.** Streams are 24-60 fps (25: 502, 30: 372, 24: 101), and the solver
#   samples on the exact grid, so an arm above the source rate repeats frames at full cost
#   (vision pass + 196 KV tokens). 16 fps is the highest genuinely-16 fps arm; 32 fps is a
#   control point -- 25ish fps with a third more compute -- not more information.
#
# * **CPU RAM.** ReKV holds the whole KV-Cache in pinned host memory for the length of the
#   stream, times one worker per chunk on the node. Cost is a large intercept plus a small
#   per-frame slope (0.5b measured: ~26 GB at 1 fps, ~32 at 4, ~41 at 8), because the
#   pinned pool never returns memory; MEM_* below are those two constants, with 7b scaled
#   by its 4.7x per-token KV -- an estimate. The loop sizes chunks to MEM_BUDGET_GB and
#   skips an arm that cannot fit one worker, so llava_ov_7b runs to ~4 fps here and stops.
#

#

#     1 fps ~0.5 h   2 fps ~1 h   4 fps ~2 h   8 fps ~3.5 h   16 fps ~9 h   32 fps ~28 h
# i.e. ~44 h for the whole sweep, and about twice that with TRIGGER=end. The top two arms
# are superlinear because they also give up GPUs to the memory guard below (16 fps runs on
# 3 workers, 32 fps on 2). Splitting FPS_LIST across several jobs is the normal way to run
# this; the arms are independent and each writes its own directory, so
# FPS_LIST="1 2 4 8" and FPS_LIST="16" and FPS_LIST="32" as three jobs is the same sweep.
#

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
# The repo root has to be on sys.path, not just be the cwd -- see the long note in
# scripts/eval.slurm: without it the workers import the other ReKV clone's code.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

DATASET=fpsbench_stream

#llava_ov_7b is capped by host RAM (see above) and
# will skip its top arms rather than run them.
MODEL=llava_ov_7b

# The sweep. Space-separated, and each value writes to its own results directory
# (results/<model>/fpsbench_stream/<retrieve_size>-<fps>...), so arms never overwrite
# each other and the list can be run in pieces across jobs.
FPS_LIST="1 2 4"

# The query is trigged at the end of evidence window
TRIGGER=query


ANNO=" "

N_LOCAL=15000
RETRIEVE_SIZE=64
DECODE_WINDOW=256

# Reduction arms, as in scripts/eval_odvbench_baseline.slurm: "<method>:<threshold>",
# 'none:0' being the untouched baseline. Every arm here is run at every frame rate, so
# this multiplies the cost of the sweep -- start with the baseline alone.
ARMS="none:0"
PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

# GPUs this job asked for; must match #SBATCH --gres=gpu:N above. The per-fps chunk count
# is capped by it and by memory, never above it.
GPUS=4
# Host RAM all workers together may occupy. The job is allocated --mem-per-cpu x
# --cpus-per-gpu x GPUS = 256G on a 257G node, so this leaves ~56G of headroom for the
# decode windows 
MEM_BUDGET_GB=200

# Peak host memory of one worker: MEM_FIXED_GB + MEM_GB_PER_FRAME x frames. Both measured
# for llava_ov_0.5b (see the CPU RAM note in the header); llava_ov_7b is that scaled by
# its per-token KV, which is 4.7x larger -- an estimate. Re-measure by watching the peak
# RSS of a single worker rather than trusting these if an arm gets tight.
case "${MODEL}" in
    llava_ov_0.5b) MEM_FIXED_GB=24  ; MEM_GB_PER_FRAME=0.0035 ;;
    llava_ov_7b)   MEM_FIXED_GB=32  ; MEM_GB_PER_FRAME=0.0163 ;;
    *) echo "FATAL: no memory figures for ${MODEL}; measure them before sweeping." >&2
       exit 1 ;;
esac

# Longest stream in the benchmark, in seconds. Every stream is 600 s, and under
# TRIGGER=query a worker stops at query_time_sec (max 538 s) -- so this is the worst case
# for both arms, which is what a memory guard wants.
STREAM_SEC=600

if [ ! -f data/fpsbench_stream/test_mc.json ]; then
    echo "FATAL: data/fpsbench_stream/test_mc.json missing -- build it first:" >&2
    echo "  python video_qa/convert_fpsbench_stream.py --out data/fpsbench_stream/test_mc.json" >&2
    exit 1
fi


for FPS in ${FPS_LIST}; do
  # Peak host memory of one worker at this frame rate, and how many of them fit.
  read PER_WORKER_GB CHUNKS <<< "$(awk -v fps="${FPS}" -v sec="${STREAM_SEC}" \
      -v fixed="${MEM_FIXED_GB}" -v slope="${MEM_GB_PER_FRAME}" \
      -v budget="${MEM_BUDGET_GB}" -v gpus="${GPUS}" 'BEGIN{
        per = fixed + fps * sec * slope;
        n = int(budget / per); if (n > gpus) n = gpus;
        printf "%.0f %d", per, n }')"

  if [ "${CHUNKS}" -lt 1 ]; then
    echo "=========================================================================="
    echo "=== SKIP ${MODEL} @ ${FPS} fps: one worker alone would peak near"
    echo "=== ${PER_WORKER_GB} GB of host memory, over the ${MEM_BUDGET_GB} GB budget for the"
    echo "=== whole node. Run this arm on a bigger-memory node, or with a smaller model."
    continue
  fi
  if [ "${CHUNKS}" -lt "${GPUS}" ]; then
    echo "NOTE: ${FPS} fps peaks near ${PER_WORKER_GB} GB per worker, so this arm runs on"
    echo "      ${CHUNKS} of ${GPUS} GPUs to stay inside ${MEM_BUDGET_GB} GB -- and takes"
    echo "      $(awk -v g=${GPUS} -v c=${CHUNKS} 'BEGIN{printf "%.1f", g/c}')x as long as the GPU count alone suggests."
  fi

  for ARM in ${ARMS}; do
    METHOD="${ARM%%:*}"
    THR="${ARM##*:}"

    REDUCE_ARGS=""
    if [ "${METHOD}" != "none" ]; then
        REDUCE_ARGS="--prune_method ${METHOD} --prune_threshold ${THR} \
            --prune_metric ${PRUNE_METRIC} --prune_refresh_every ${PRUNE_REFRESH_EVERY}"
    fi

    echo "=========================================================================="
    echo "=== ${MODEL} | ${DATASET} | ${FPS} fps | trigger=${TRIGGER} | s2=${METHOD}@${THR} ==="
    echo "=== ${CHUNKS} workers x ~${PER_WORKER_GB} GB host memory | anno=${ANNO:-full} ==="
    echo "=== started $(date) ==="
    python -m video_qa.run_eval \
        --num_chunks ${CHUNKS} \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --sample_fps ${FPS} \
        --n_local ${N_LOCAL} \
        --retrieve_size ${RETRIEVE_SIZE} \
        --trigger ${TRIGGER} \
        --decode_window ${DECODE_WINDOW} \
        ${ANNO_ARGS} \
        ${REDUCE_ARGS}
    echo "=== finished $(date) ==="
    free -g
  done
done

echo "Job finished at $(date)"
# run_eval.py tags 'query' runs with -query and leaves 'end' runs untagged, so the filter
# that isolates this sweep is not simply the trigger name. The dashes are left off: they
# would be read as flags.
SWEEP_FILTER=""
[ "${TRIGGER}" = "query" ] && SWEEP_FILTER="query"
[ -n "${ANNO}" ] && SWEEP_FILTER="${SWEEP_FILTER} $(basename ${ANNO} .json)"
echo "Read the sweep with:"
echo "  python scripts/collect_fpsbench_sweep.py --model ${MODEL} --filter ${SWEEP_FILTER}"

conda deactivate
