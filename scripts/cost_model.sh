#!/bin/bash
# The efficiency table: KV-Cache growth, streaming throughput, and GFLOPs/frame, for the
# baseline and every stage-2 threshold, on one FPS-Bench-Stream stream.
#
# Each arm is a real run through video_qa/measure_encoding_fps.py -- the LM included.
# Throughput is the number that forces this: stage 2 drops tokens before the LM prefill,
# which is ~84% of `_encode_video_chunk`, so a measurement that skipped the LM would miss
# the entire effect. Ten arms x 600 frames, ~4 min each on a 7B, so budget ~40 min.
#
# One stream is enough for the shape of the curve and exact for the config-derived
# columns, but the keep rate -- and therefore every column that scales with it -- is a
# point estimate: between-video CV of the keep rate runs 21-42% on the datasets already
# measured here. Raise N_STREAMS once the curve looks right; the collector averages them
# and reports the spread.
#
# Needs a GPU. On a cluster: srun -p gpu --gres=gpu:1 bash scripts/cost_model.sh
#
# Arms already on disk are skipped, so an interrupted run resumes. FORCE=1 re-runs them.

set -euo pipefail

cd "$(dirname "$0")/.."
# The repo root has to be on sys.path, not just be the cwd -- see the note in
# scripts/eval.slurm: without it the workers import the other ReKV clone's code.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

MODEL=${MODEL:-llava_ov_7b}
ANNO=${ANNO:-data/fpsbench_stream/test_mc.json}

# Which stream(s). FPS-Bench-Stream is 990 uniform 600 s streams, so the index only
# selects content, not length -- which is what makes n=1 defensible here and not on a
# dataset with variable-length videos.
VIDEO_IDX=${VIDEO_IDX:-0}
N_STREAMS=${N_STREAMS:-1}

# 600 frames at 1 fps is the whole 600 s stream. Raising FPS raises NUM_FRAMES with it,
# or the run covers only the opening seconds.
#
# These two also decide whether the throughput number means anything. Only chunks encoded
# after the local window filled are steady state -- before that the eviction/offload path
# has not engaged -- and the window fills after n_local / (196 x keep_rate) frames. So the
# harder an arm prunes the longer it takes to get there:
#
#     keep 100% ->   77 frames    keep 25% ->  306      keep 10% ->  765
#     keep  50% ->  153           keep 19% ->  403      keep  5% -> 1531
#
# At 1 fps x 600 frames anything keeping under ~13% never reaches steady state, and the
# collector flags those rows rather than quoting them next to the others. If the aggressive
# thresholds come back flagged, re-run at FPS=2 NUM_FRAMES=1200 (covers >6.4%) or FPS=4
# NUM_FRAMES=2400 (>3.2%). Note the keep rate is itself a function of FPS -- consecutive
# frames are more alike at a higher rate -- so quote the table at the rate the accuracy
# numbers are reported at.
FPS=${FPS:-1}
NUM_FRAMES=${NUM_FRAMES:-600}

# 'baseline' plus each threshold. Order matters only for readability.
THRESHOLDS=${THRESHOLDS:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"}
PRUNE_METRIC=${PRUNE_METRIC:-cosine}

N_LOCAL=${N_LOCAL:-15000}
RETRIEVE_SIZE=${RETRIEVE_SIZE:-64}

# How the encode clock is read.
#   span      -- three CUDA syncs for the whole run (start, local window full, end). The
#                honest streaming figure: no per-frame barrier, so CPU preprocessing
#                overlaps GPU compute the way it does in a live stream.
#   per_chunk -- a sync pair around every chunk. Per-frame resolution, and the protocol
#                every previously recorded number here used, but at chunk size 1 the
#                barriers serialize preprocessing against compute, so it reads low.
# Both write their own file, so running both is a cross-check rather than an overwrite:
# the gap between them is measurement overhead, not system behaviour.
TIMING=${TIMING:-span}

# Steady-state frames measured for GFLOPs, per arm. Measured, not read off the config:
# FlopCounterMode for the dispatcher-visible ops, plus the LM's attention counted from the
# shapes append() is actually called with -- ReKV attends through a Triton kernel, which no
# __torch_dispatch__ counter can see (measured here: a plain counter reports 0 for it).
# The counter is slow, so these frames are excluded from the throughput clock. Frames in
# steady state differ only in keep rate, so 8 is plenty. 0 turns it off and the table falls
# back to the config-derived figure.
FLOPS_FRAMES=${FLOPS_FRAMES:-64}

# ONE frame per forward pass. This is the streaming number and the default (64) is not:
# a live stream has no frame t+1 to batch with frame t, so the eval drives encode_frame
# once per frame. The two are ~2x apart and must never be compared to each other -- see
# the ingestion-granularity note in video_qa/measure_encoding_fps.py.
ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-1}

# QA off. Each FPS-Bench-Stream stream carries exactly one question, so a latency figure
# from this run would be n=1 -- not reportable, and it would add time to every arm.
# Latency has its own protocol (many questions injected mid-stream, --force_answer_length
# on, since answer length is a dependent variable of anything that perturbs the cache).
SKIP_QA=${SKIP_QA:-true}

OUT_DIR=${OUT_DIR:-results/cost_model}
mkdir -p "${OUT_DIR}"

if [ ! -f "${ANNO}" ]; then
    echo "FATAL: ${ANNO} missing -- build it first:" >&2
    echo "  python video_qa/convert_fpsbench_stream.py --out ${ANNO}" >&2
    exit 1
fi

run_arm () {
    local idx="$1" arm="$2" out="$3"
    shift 3
    if [ -f "${out}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "--- skip ${arm} (have ${out}; FORCE=1 to re-run)"
        return
    fi
    echo "=========================================================================="
    echo "=== ${MODEL} | stream ${idx} | ${FPS} fps x ${NUM_FRAMES} frames | ${arm}"
    echo "=== started $(date)"
    python video_qa/measure_encoding_fps.py \
        --model "${MODEL}" \
        --anno_path "${ANNO}" \
        --video_idx "${idx}" \
        --sample_fps "${FPS}" \
        --num_frames "${NUM_FRAMES}" \
        --encode_chunk_size "${ENCODE_CHUNK_SIZE}" \
        --n_local "${N_LOCAL}" \
        --retrieve_size "${RETRIEVE_SIZE}" \
        --skip_qa "${SKIP_QA}" \
        --timing "${TIMING}" \
        --flops_frames "${FLOPS_FRAMES}" \
        --save_path "${out}" \
        "$@"
    echo "=== finished $(date)"
}

for i in $(seq 0 $((N_STREAMS - 1))); do
    IDX=$((VIDEO_IDX + i))
    run_arm "${IDX}" "baseline" "${OUT_DIR}/${MODEL}-fps${FPS}-v${IDX}-baseline-${TIMING}.csv"
    for THR in ${THRESHOLDS}; do
        run_arm "${IDX}" "rlt@${THR}" "${OUT_DIR}/${MODEL}-fps${FPS}-v${IDX}-rlt${THR}-${TIMING}.csv" \
            --prune_method rlt --prune_threshold "${THR}" --prune_metric "${PRUNE_METRIC}"
    done
done

echo
python scripts/collect_cost_model.py --model "${MODEL}" --sample_fps "${FPS}" \
    --n_local "${N_LOCAL}" --dir "${OUT_DIR}" --timing "${TIMING}"
