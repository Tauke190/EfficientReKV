# FPS-Bench-Stream: the retrievability arm. One GPU.
#
# Each of the 990 built streams is a 600 s MLVU video with one FPSBench clip spliced into
# it -- median 9 s of evidence, 1.5% of the stream -- and the question is asked at the END
# of the stream, a median 294 s after the evidence has gone past. At 1 fps that is ~294
# frames of unrelated footage on top of the answer, all of it far past n_local, so the
# needle is reachable only through ReKV's retrieval. That is the whole measurement: this
# benchmark asks whether ReKV can find something it saw five minutes ago, not whether it
# resolved fast motion (which is scripts/eval_fpsbench_stream_small.sh, on the same
# questions and the bare clips).
#
# Two things every arm reports, and the reason a single accuracy number is not enough here:
#
#   qa_acc            -- there IS an answer key in this release, unlike question-only
#                        FPSBench, so this is real.
#   layers_hit_frac   -- the share of the backbone's layers whose retrieved top-k blocks
#                        actually held a needle frame. Accuracy cannot distinguish "never
#                        retrieved it" from "retrieved it and still got it wrong"; this can,
#                        and it is the column a pruning sweep should be read on.
#
# WHAT RLT IS BEING ASKED HERE. Stage-2 pruning drops tokens that repeat what memory
# already holds. The haystack is ordinary MLVU footage and compresses hard; the needle is
# fast motion by construction and does not. So pruning shrinks the pool retrieval searches
# without shrinking the needle inside it, which is a mechanism for pruning to RAISE the hit
# rate rather than merely make the run cheaper. The opposite is equally possible: a block
# spanning a long static stretch plus the needle is a worse retrieval target than a block
# holding the needle alone. The sweep below is what tells those apart, at a measured keep
# rate rather than at a nominal threshold.
#
# THE FRAME RATE IS A CEILING, NOT A KNOB. FPSBench's min_fps floor is 4, so at 1 fps
# essentially no question is resolvable even with perfect retrieval -- the scorer says how
# many. Read the 1 fps arms as a retrieval measurement (layers_hit_frac) and the 4 fps arms
# as the accuracy one. The cost of that is linear and it lands on RAM: at 1 fps a stream is
# ~600 frames = ~118k KV tokens = ~6.5 GB of offloaded cache per worker, and 4 fps is four
# times that. Pruning is what makes the dense arms affordable at all, which is the second
# reason it is on trial here.
#
# NO FRAME CACHE. Decoding is in-process with decord; REKV_FRAME_CACHE is ignored by this
# solver. Pre-extracting one arm would be 990 x 600 = ~594k JPEGs.
#
# SMOKE RUN FIRST. One arm is 594k frames at 1 fps, so measure before committing:
#   SMOKE_STREAMS=20 bash scripts/eval_fpsbench_stream.sh
# builds a 20-stream annotation, runs every arm against it into its own results directory
# (named after the annotation file, so it can never overwrite the full run), and tells you
# what an arm costs.

# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
num_chunks=1

model=llava_ov_7b
retrieve_size=64

# 20 streams is enough to see the pipeline work end to end and to time an arm; it is not
# enough to read an accuracy off. Unset (the default) runs all 990.
smoke=${SMOKE_STREAMS:-0}

# 1 fps is ReKV's default and the rate at which a 600 s stream is affordable unpruned. 4 is
# the lowest rate at which any FPSBench question is resolvable at all (its min_fps floor),
# and is included pruned only -- see the RAM note above.
sample_fps_list="1"
dense_fps=4

# Stage-2 RLT thresholds. Swept at a fixed frame rate here, unlike the short-clip script:
# there the question was frame rate and the threshold had to move with it to keep the token
# budget comparable. Here the question is retrieval over a fixed 600 s memory, so the axis
# that has to vary is how hard that memory was compressed.
prune_thresholds="0.05 0.1 0.2 0.3 0.5"
prune_metric=cosine
prune_refresh_every=0

# --- annotation ----------------------------------------------------------------------
# Rebuilt every run so a resubmitted job never scores against a stale file. It carries the
# needle window in ASSEMBLED-STREAM seconds, which is the only clock the solver may use --
# the release also states the same event in YouTube and MLVU seconds and mixing them is the
# one mistake its README calls out by name.
anno=data/fpsbench_stream/test_mc.json
anno_arg=""
limit_arg=""
if [ "${smoke}" != "0" ]; then
  # A subset gets its own annotation file, and run_eval.py names the results directory
  # after it, so a smoke run can never land on top of the full arm's results.csv.
  anno=data/fpsbench_stream/smoke${smoke}.json
  anno_arg="--anno_path ${anno}"
  limit_arg="--limit ${smoke}"
  echo "SMOKE RUN: ${smoke} streams -> ${anno}"
fi

python video_qa/convert_fpsbench_stream.py \
    --src /home/av354855/FPSBenchStream/fpsbench_stream_v1.jsonl \
    --video_root /home/av354855/FPSBenchStream/videos \
    ${limit_arg} \
    --out ${anno} || exit 1

# run_eval.py builds the results directory from the trigger, the annotation file and
# `reduction_tag(args)`, and --sample_fps is parsed as a float, so a "1" here lands in
# `64-1.0-rlt0.05cosine`. Rebuilt rather than guessed at, so the summary below never
# silently reports on the wrong run.
run_dir () {  # $1 = sample_fps, $2 = "" or "-rlt<thr>cosine", $3 = "" or "-query"
  local subset=""
  [ "${smoke}" != "0" ] && subset="-smoke${smoke}"
  echo "results/${model}/fpsbench_stream/${retrieve_size}-${1}.0${3}${subset}${2}"
}

prune_tag () {  # $1 = threshold, or "none"
  [ "$1" = "none" ] && echo "" || echo "-rlt$1${prune_metric}"
}

# Reduction and retrieval, echoed into the log so the sweep is readable without opening
# five CSVs. The scorer prints accuracy; this prints what it cost and what retrieval did.
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
  local fps=$1 thr=$2 trigger=$3
  local prune_args=""
  [ "${thr}" != "none" ] && prune_args="--prune_method rlt --prune_threshold ${thr} \
      --prune_metric ${prune_metric} --prune_refresh_every ${prune_refresh_every}"
  echo "=== ${model} on fpsbench_stream @ ${fps} fps | rlt=${thr} | trigger=${trigger} ==="
  python -m video_qa.run_eval \
      --num_chunks ${num_chunks} \
      --model ${model} \
      --dataset fpsbench_stream \
      --sample_fps ${fps} \
      --n_local 15000 \
      --retrieve_size ${retrieve_size} \
      --max_new_tokens 128 \
      --trigger ${trigger} \
      ${anno_arg} \
      ${prune_args}
  local tag_trigger=""
  [ "${trigger}" != "end" ] && tag_trigger="-${trigger}"
  echo "--- summary | ${fps} fps | rlt=${thr} | trigger=${trigger} ---"
  summarize_run "$(run_dir ${fps} "$(prune_tag ${thr})" "${tag_trigger}")"
}

# --- 1. the benchmark: question at the end of the stream ------------------------------
# Baseline first: it is the reference every pruned arm is read against, and it is also the
# arm that says whether retrieval reaches the needle at all before compression is involved.
for sample_fps in ${sample_fps_list}; do
  run_arm ${sample_fps} none end
  for thr in ${prune_thresholds}; do
    run_arm ${sample_fps} ${thr} end
  done
done

# --- 2. the control: same frames, question asked while the needle is still recent ------
# query_time_sec is the release's real-time protocol -- the question fires at the end of
# the temporal certificate, where retrieval distance is 0 and the needle is the newest
# thing in the cache. The gap between this and arm 1 at the same threshold is the cost of
# having to retrieve, separated from the cost of the model not being able to answer.
run_arm 1 none query
run_arm 1 0.2 query

# --- 3. dense sampling, pruned only ---------------------------------------------------
# 4 fps is the lowest rate at which FPSBench considers any of these questions resolvable,
# and unpruned it is ~26 GB of offloaded KV per worker for a single stream. These arms say
# whether the accuracy that retrieval unlocks is actually reachable once the frames are
# dense enough to carry the answer -- and, since only pruned arms are affordable, they are
# also the argument for RLT as an enabler rather than as an optimisation.
for thr in 0.2 0.3 0.5; do
  run_arm ${dense_fps} ${thr} end
done
