#!/bin/bash
#SBATCH --job-name=rekv_fpsbench_stream
#SBATCH --output=/home/av354855/EfficientVideoXLPro/slurmlogs/rekv_fpsbench_stream-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -C gmem80
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# FPS-Bench-Stream: the retrievability arm. See scripts/eval_fpsbench_stream.sh for what
# the benchmark measures and why; this is the same sweep on four GPUs.
#
# In one line: each of the 990 streams hides one FPSBench clip (median 9 s) inside 600 s of
# unrelated MLVU footage, and the question is asked at the END of the stream -- a median
# 294 s after the evidence. Only ReKV's retrieval can put that evidence back in front of the
# model, so `layers_hit_frac` (the share of layers whose retrieved top-k held a needle
# frame) is as much the result as qa_acc is. RLT is on trial as the thing that shrinks the
# haystack in memory without shrinking the needle in it.
#
# MEMORY. Offloaded KV is ~6.5 GB per 600 s stream per worker at 1 fps, so four workers is
# ~26 GB of the 256 GB this allocation asks for. At 4 fps that is ~26 GB per worker, which
# is why the dense arms below are pruned-only.
#
# ORDER. Baseline first, then thresholds, then the control, then the dense arms: if the job
# hits its wall clock partway through, what survives is the reference arm plus a prefix of
# the threshold curve, which is readable. The reverse order would leave nothing to compare
# against.
#
# SMOKE FIRST. `SMOKE_STREAMS=20 sbatch scripts/eval_fpsbench_stream.slurm` runs every arm
# over 20 streams into its own results directory. One full arm is 990 x 600 = ~594k frames
# at 1 fps; time the smoke run before committing a wall clock to the full sweep.

module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"
conda activate Rekv

LOGDIR=/home/av354855/EfficientVideoXLPro/slurmlogs

export MPLBACKEND=Agg
export TQDM_MININTERVAL=60

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

cd /home/av354855/EfficientVideoXLPro/ReKV

# Must match #SBATCH --gres=gpu:N above -- run_eval.py spawns one process per chunk, each
# pinned to its own GPU via CUDA_VISIBLE_DEVICES.
NUM_CHUNKS=4
MODEL=llava_ov_7b
RETRIEVE_SIZE=64   # blocks retrieved per question; also the results-dir prefix

# 0 = the full 990 streams. See the smoke note above.
SMOKE=${SMOKE_STREAMS:-0}

# 1 fps is ReKV's default and the rate at which a 600 s stream is affordable unpruned. 4 is
# FPSBench's min_fps floor -- below it no question is resolvable however well retrieval
# works -- and appears pruned-only, for the memory reason above.
SAMPLE_FPS_LIST="1"
DENSE_FPS=4

# Stage-2 RLT thresholds, swept at a fixed frame rate. The short-clip script sweeps
# threshold x fps because there the experiment is about frame rate; here the memory is a
# fixed 600 s and the axis that has to vary is how hard it was compressed.
PRUNE_THRESHOLDS="0.05 0.1 0.2 0.3 0.5"
PRUNE_METRIC=cosine
PRUNE_REFRESH_EVERY=0

# --- annotation ----------------------------------------------------------------------
# Rebuilt every run so a resubmitted job never scores against a stale file. It carries the
# needle window in ASSEMBLED-STREAM seconds, the only clock the solver may use.
ANNO=data/fpsbench_stream/test_mc.json
ANNO_ARG=""
LIMIT_ARG=""
if [ "${SMOKE}" != "0" ]; then
  ANNO=data/fpsbench_stream/smoke${SMOKE}.json
  ANNO_ARG="--anno_path ${ANNO}"
  LIMIT_ARG="--limit ${SMOKE}"
  echo "SMOKE RUN: ${SMOKE} streams -> ${ANNO}"
fi

python video_qa/convert_fpsbench_stream.py \
    --src /home/av354855/FPSBenchStream/fpsbench_stream_v1.jsonl \
    --video_root /home/av354855/FPSBenchStream/videos \
    ${LIMIT_ARG} \
    --out ${ANNO} || exit 1

# run_eval.py builds the results directory from the trigger, the annotation file and
# `reduction_tag(args)`, and --sample_fps is parsed as a float, so a "1" here lands in
# `64-1.0-rlt0.05cosine`. Rebuilt rather than guessed at, so the summary never reports on
# the wrong run.
run_dir () {  # $1 = sample_fps, $2 = "" or "-rlt<thr>cosine", $3 = "" or "-query"
  local subset=""
  [ "${SMOKE}" != "0" ] && subset="-smoke${SMOKE}"
  echo "results/${MODEL}/fpsbench_stream/${RETRIEVE_SIZE}-${1}.0${3}${subset}${2}"
}

prune_tag () {  # $1 = threshold, or "none"
  [ "$1" = "none" ] && echo "" || echo "-rlt$1${PRUNE_METRIC}"
}

# What the arm cost and what retrieval did with it. The scorer prints accuracy; this is the
# half a threshold sweep has to be read against -- a threshold means nothing without the
# keep rate it actually achieved, and an accuracy means nothing here without the hit rate.
summarize_run () {
  python - "$1" <<'PYEOF'
import sys, os, pandas as pd
d = sys.argv[1]; f = os.path.join(d, 'results.csv')
if not os.path.exists(f):
    print(f'  (no results.csv in {d})'); raise SystemExit
df = pd.read_csv(f)
print(f'  rows: {len(df)}   streams: {df.video_id.nunique()}   qa_acc: {df.qa_acc.mean():.1f}%')
if 'token_keep_rate' in df.columns:
    print(f'  tokens kept : mean {100*df.token_keep_rate.mean():5.1f}%   '
          f'min {100*df.token_keep_rate.min():5.1f}%   max {100*df.token_keep_rate.max():5.1f}%')
if 'n_tokens_fed' in df.columns:
    print(f'  tokens fed per stream : mean {df.n_tokens_fed.mean():8.0f}   '
          f'max {df.n_tokens_fed.max():8.0f}   (n_local {int(df.n_local.iloc[0])})')
if 'layers_hit_frac' in df.columns and df.layers_hit_frac.notna().any():
    h = df[df.layers_hit_frac.notna()]
    print(f'  needle in retrieved top-k : {100*h.layers_hit_frac.mean():5.1f}% of layers   '
          f'({int((h.layers_hit_frac==0).sum())}/{len(h)} rows reached by no layer)')
    found, lost = h[h.layers_hit_frac >= 0.5], h[h.layers_hit_frac < 0.5]
    if len(found) and len(lost):
        print(f'  qa_acc | found {found.qa_acc.mean():.1f}%  vs  lost {lost.qa_acc.mean():.1f}%'
              f'   ({len(found)} / {len(lost)} rows)')
elif 'retrieval_fired' in df.columns and not df.retrieval_fired.astype(bool).any():
    print('  retrieval never ran: every stream fit inside n_local, so this arm measures the '
          'backbone over a long context rather than ReKV memory')
PYEOF
}

run_arm () {  # $1 = sample_fps, $2 = threshold or "none", $3 = trigger
  local FPS=$1 THR=$2 TRIGGER=$3
  local PRUNE_ARGS=""
  [ "${THR}" != "none" ] && PRUNE_ARGS="--prune_method rlt --prune_threshold ${THR} \
      --prune_metric ${PRUNE_METRIC} --prune_refresh_every ${PRUNE_REFRESH_EVERY}"
pip install transformers==4.39.2 accelerate==0.29.3 peft==0.10.0 decord==0.6.0 \
            "numpy<2" pillow pyyaml sentencepiece protobuf tqdm huggingface_hub[cli]  echo "=== ${MODEL} on fpsbench_stream @ ${FPS} fps | rlt=${THR} | trigger=${TRIGGER} ==="
  python -m video_qa.run_eval \
      --num_chunks ${NUM_CHUNKS} \
      --model ${MODEL} \
      --dataset fpsbench_stream \
      --sample_fps ${FPS} \
      --n_local 15000 \
      --retrieve_size ${RETRIEVE_SIZE} \
      --max_new_tokens 128 \
      --trigger ${TRIGGER} \
      ${ANNO_ARG} \
      ${PRUNE_ARGS}
  local TAG_TRIGGER=""
  [ "${TRIGGER}" != "end" ] && TAG_TRIGGER="-${TRIGGER}"
  echo "--- summary | ${FPS} fps | rlt=${THR} | trigger=${TRIGGER} ---"
  summarize_run "$(run_dir ${FPS} "$(prune_tag ${THR})" "${TAG_TRIGGER}")"
}

# --- 1. the benchmark: question at the end of the stream ------------------------------
for SAMPLE_FPS in ${SAMPLE_FPS_LIST}; do
  run_arm ${SAMPLE_FPS} none end
  for THR in ${PRUNE_THRESHOLDS}; do
    run_arm ${SAMPLE_FPS} ${THR} end
  done
done

# --- 2. the control: same frames, question asked while the needle is still recent ------
# --trigger query fires at query_time_sec, where retrieval distance is 0 and the needle is
# the newest thing in the cache. The gap against arm 1 at the same threshold is the cost of
# having to retrieve, separated from the model simply not being able to answer.
run_arm 1 none query
run_arm 1 0.2 query

# --- 3. dense sampling, pruned only ---------------------------------------------------
# 4 fps is FPSBench's own resolvability floor. Unpruned it is ~26 GB of offloaded KV per
# worker; these arms are therefore both the accuracy measurement and the argument for RLT
# as an enabler of it rather than as an optimisation.
for THR in 0.2 0.3 0.5; do
  run_arm ${DENSE_FPS} ${THR} end
done

echo "Job finished at $(date)"
conda deactivate
