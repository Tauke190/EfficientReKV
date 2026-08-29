#!/bin/bash
#SBATCH --job-name=rekv_fpsbench
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_fpsbench-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# FPS-Bench-Stream frame-rate sweep: the same 990-stream (or subset) benchmark run at
# 1/2/4/8/16/32 fps, so accuracy can be read against the sampling rate.
#
# Why this benchmark answers a frame-rate question at all: every question carries
# `min_fps`, FPSBench's claim about the rate below which its evidence window is not
# resolvable however good the model is. The distribution over the 990 streams is
#     4 fps: 249   5: 206   6: 153   7: 132   8: 80   9-15: 147   16-30: 23
# so a sweep from 1 to 32 fps crosses every question's threshold: at 1-2 fps no question
# is resolvable, by 8 fps 83% are, and past 16 fps 98% are. The expected
# shape is a rise that flattens, and where it flattens is the measurement.
#
# Two ceilings bound the top of the sweep, and both are real:
#
# * **The source rate.** The assembled streams are 24-60 fps (25: 502, 30: 372, 24: 101,
#   the rest a handful), and the solver samples on the exact grid, taking the most recent
#   source frame for slot k. Above the source rate that repeats frames: at 32 fps on a
#   25 fps stream, 7 slots in every 32 carry no new pixels but still cost a vision-tower
#   pass and 196 KV tokens each. 32 fps is therefore a control point -- it should look
#   like 25ish fps with a third more compute -- not a higher-information arm. 16 fps is
#   under every stream's native rate and is the highest arm that is genuinely 16 fps.
#
# * **CPU RAM.** ReKV offloads the whole KV-Cache to pinned host memory and holds it for
#   the length of the stream, and run_eval.py runs one worker per chunk on one node -- so
#   the node pays that times NUM_CHUNKS. Measured on this node with llava_ov_0.5b (peak
#   RSS of one worker, `scripts/`-adjacent probes, 600 s streams):
#       1 fps (600 frames)   ~26 GB      8 fps (4800)   ~41 GB
#       4 fps (2400)         ~32 GB     32 fps (19200)  ~91 GB (extrapolated)
#   Note the intercept: even a 1 fps worker sits at ~25 GB, far above the ~1.4 GB the KV
#   tensors themselves need, because the pinned-block pool and the host allocator do not
#   give the memory back. It plateaus rather than climbing -- across ten streams a worker
#   stayed at ~23 GB resident / 41 GB peak -- so the model below is an intercept plus a
#   per-frame slope, not a pure per-frame cost. MEM_* are those two constants;
#   llava_ov_7b's are the same measurement scaled by its 4.7x larger per-token KV and are
#   an estimate, not a measurement. The loop sizes each arm's chunk count to fit
#   MEM_BUDGET_GB rather than letting a 16 fps 7b arm take the node down eight hours in,
#   and skips an arm that cannot fit even one worker. This is why the full sweep is a
#   0.5b run: llava_ov_7b fits to about 4 fps on this node and no further.
#
# Decoded pixels used to be a third ceiling -- FrameStream read the whole video up front,
# which at source resolution (a median 4.6 MB/frame here) is 85 GB per worker at 32 fps.
# It now decodes --decode_window slots at a time; the frames are identical, only residency
# changes. Do not set DECODE_WINDOW=0 (whole-video decode) above ~2 fps.
#
# Runtime is the other reason this defaults to a subset: cost is linear in fps, so the six
# arms together cost 63x a single 1 fps arm. Measured on one A6000 with llava_ov_0.5b, the
# solver sustains ~6-7 frames/s end to end (9-10 frames/s of pure encode, the rest decode
# and QA), so on 4 GPUs over the 135-stream subset with TRIGGER=query (which ingests to
# query_time_sec, a mean 305 s of each stream):
#     1 fps ~0.5 h   2 fps ~1 h   4 fps ~2 h   8 fps ~3.5 h   16 fps ~9 h   32 fps ~28 h
# i.e. ~44 h for the whole sweep, and about twice that with TRIGGER=end. The top two arms
# are superlinear because they also give up GPUs to the memory guard below (16 fps runs on
# 3 workers, 32 fps on 2). Splitting FPS_LIST across several jobs is the normal way to run
# this; the arms are independent and each writes its own directory, so
# FPS_LIST="1 2 4 8" and FPS_LIST="16" and FPS_LIST="32" as three jobs is the same sweep.
#
# scripts/eval.slurm is the general-purpose twin; this one exists because the fps loop,
# the per-fps chunk sizing and the subset are all specific to this benchmark.

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

# llava_ov_0.5b sweeps the whole range; llava_ov_7b is capped by host RAM (see above) and
# will skip its top arms rather than run them.
MODEL=llava_ov_0.5b

# The sweep. Space-separated, and each value writes to its own results directory
# (results/<model>/fpsbench_stream/<retrieve_size>-<fps>...), so arms never overwrite
# each other and the list can be run in pieces across jobs.
FPS_LIST="1 2 4 8 16 32"

# Which question each arm asks. 'query' fires at query_time_sec, with the needle still the
# newest thing in the cache: that is the perception measurement, and the one the frame rate
# is actually about -- whether the model saw enough frames to resolve the event. 'end'
# fires after all 600 s, so the needle has to be retrieved back out of memory; a frame-rate
# sweep there measures perception and retrieval together, which is worth having but is not
# the first thing to run. The two write to different directories.
TRIGGER=query

# The subset the sweep runs on. One arm over all 990 streams is 165 hours of footage, so
# the six arms together are 165 x (1+2+4+8+16+32) = ~10,400 hours of decode-and-encode --
# not a thing to start. sub135.json is 135 streams balanced over all 9 question types and
# all 3 needle positions (scripts/make_fpsbench_subset.py), i.e. the same questions in
# every arm, which is what makes the arms comparable. Set ANNO="" to run the full
# benchmark, and expect it to take days.
ANNO=data/fpsbench_stream/sub135.json

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
# decode windows (one --decode_window block of source-resolution pixels per worker, up to
# ~1.6 GB) and for whatever else is resident.
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
if [ -n "${ANNO}" ] && [ ! -f "${ANNO}" ]; then
    echo "=== ${ANNO} missing; building it ==="
    python scripts/make_fpsbench_subset.py --per_cell 5 --out "${ANNO}" || exit 1
fi
ANNO_ARGS=""
[ -n "${ANNO}" ] && ANNO_ARGS="--anno_path ${ANNO}"

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
    # The memory model above is an estimate; this is the ground truth, and it is the line
    # to look at first if a later arm dies.
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
