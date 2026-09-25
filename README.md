# EfficientReKV

Casual Streaming Token reduction for ReKV streaming video QA.


## Eval

```bash
python -m video_qa.run_eval \
    --model llava_ov_0.5b --dataset ovobench_realtime --sample_fps 1 \
    --num_chunks 2 --n_local 15000 --retrieve_size 64 --decode_window 256 \
    --prune_method rlt_ref --prune_threshold 0.5
```

### Datasets

| `--dataset` | Build annotation with |
|---|---|
| `ovobench_realtime`, `ovobench_backward` | `python video_qa/convert_ovobench.py` |
| `odvbench` | `python scripts/dataset_prep/setup_odvbench.py` |
| `fpsbench_stream` | shipped; videos are rebuilt — see [FPS-Bench-Stream](#fps-bench-stream) |
| `streamingbench_real` | `scripts/dataset_prep/prepare_streamingbench.sh` |
| `ovbench` | `scripts/dataset_prep/prepare_ovbench.sh` |
| `streambench` | `scripts/dataset_prep/prepare_streambench.sh` |
| `rvs_ego`, `rvs_movie`, | shipped, free-form |

`ovobench_*`, `odvbench`, `fpsbench_stream`, `streamingbench_*`, `ovbench` and
`streambench` are **streaming**: each question sees only frames up to its own timestamp.


Notes:
- `streambench` is the only benchmark here that is both streaming and open-ended, so it
  needs a judge. `--judge_preset` defaults to `streambench` (upstream's Llama-3-8B-Instruct
  prompt, reproduced byte-for-byte) instead of the repo-wide `prometheus`; override it and
- Free-form (`rvs_*`, `qaego4d`, `activitynet_qa`): pass `--skip_scoring`, then score with
  `scripts/score_llmjudge_gpt.sh` (needs `OPENAI_API_KEY`).
  Score a whole sweep with one judge — mixing judges compares judges, not systems.
- `--blind` answers with no video, giving the language-prior floor. Refuses to run with
  any reduction flag. Multiple-choice datasets use `video_qa/blind_vqa.py`; StreamBench,
  whose answers are free-form, uses `video_qa/blind_stream_vqa.py` and goes through the
  same judge as its sighted arm; `rvs_ego`/`rvs_movie` use `video_qa/blind_rvs_vqa.py`,
  likewise judged like their sighted arms. Read StreamBench's blind number per class, not pooled:
  KG needs no video and should barely move, while LM/SM/OS should collapse.

### Pruning

- **`--prune_method rlt_ref`** — drops tokens before the LM prefill
  shrinks throughput, KV RAM and retrieval together. This is the
  one that matters. Flags: `--prune_threshold`, `--prune_metric cosine|l2`,
  `--prune_refresh_every`.
- **`--prune_method rlt_prev`**: the published RLT rule, which diffs against frame t-1
  instead of the last kept token. It takes the same flags and exists as the ablation for
  `rlt_ref` (`sbatch scripts/eval/ablation.slurm`). Compare the two at matched `token_keep_rate`,
  not at matched threshold.
- **`--prune_method rlt_frame`**: `rlt_ref` with a per-frame decision. A frame is kept whole
  when more than half its tokens are over `--prune_threshold` against the last kept frame,
  and dropped whole otherwise. The ablation for deciding per token.
- **`--prune_cache adaptive|pad`**: how pruned frames enter the KV-Cache. `adaptive` (default,
  CAC) packs the kept tokens into full blocks. `pad` is the no-CAC ablation: each frame is padded
  back to one full block by repeating its last kept token, so the cache layout is exactly
  unpruned ReKV's and pruning saves nothing. Tag: `-rlt_ref0.5cosine-pad`.

**`--prune_threshold` is a cosine distance, not a rate** — a token is kept iff its distance
to the feature its position was last kept with exceeds it. Measured on `llava_ov_0.5b`,
ovobench_realtime @ 1 fps: 0.25 → 72% kept, 0.5 → 22% kept. Thresholds do not transfer across datasets

### Results

`results/<model>/<dataset>/<retrieve_size>-<sample_fps><tag>/results.csv`, where `<tag>` is
`` (baseline), `-blind`, `-rlt_ref0.5cosine`, `-rlt_prev0.5cosine` or `-rlt_frame0.5cosine`. StreamingBench writes one directory per
subset (`streamingbench_real`, …).

Keep rates are already in `results.csv` when a pruner is attached: `tokens_kept`,
`tokens_seen`, `token_keep_rate`, `kv_cache_bytes` and more, per question. **Baseline runs
have none of these columns** — when concatenating arms, fill `token_keep_rate` with 1.0
rather than dropping rows, and aggregate weighted by `tokens_seen`.

## FPS-Bench-Stream

Our long-form streaming benchmark. Each FPS-Bench clip (the *needle*, median 9 s) is
spliced into a 600 s MLVU video (the *haystack*), so the evidence is 1.5% of the stream and
sits at a known timestamp: 990 streams, 165 h. `FPSBenchStream/` holds the annotations and
`FPSBenchStream/README.md` the record layout, the three clocks and the known limitations.
**No video ships with it** — the streams are rebuilt locally.

### Build the videos

Two sources, neither redistributable from here:

1. **Needles** — the FPS-Bench clips, obtained through FPS-Bench's own release, collected
   into one directory (`<clip-dir>` below).
2. **Haystacks** — the 459 MLVU files named in `FPSBenchStream/haystack_files_used.txt`
   (159.4 GB), in one directory (`<mlvu-dir>`).

Then assemble. Needs `ffmpeg`; writes ~197 GB, skips streams already on disk, and shards
with `--shard/--num-shards`:

```bash
cd FPSBenchStream
python scripts/build_stream_dataset.py assemble \
    --plan fpsbench_stream_v1.jsonl \
    --video-dir videos \
    --canvas haystack \
    --clip-dir <clip-dir> \
    --haystack-dir <mlvu-dir> \
    --encoder h264_nvenc              # or libx264, the default
```

`--video-dir` is where the streams are written; `videos` keeps them beside the annotations,
which is what the shipped eval annotation expects.

**Assemble against the shipped plan; do not re-run `plan`.** It redraws haystack
assignments from an RNG that depends on which needle clips are present, so a different set
of clips gives a different — equally valid, but not identical — dataset. `--canvas haystack`
is what the release was cut with. Rebuilds match on frames and timestamps, not bytes: the
encoder settings are not recorded per record, so file hashes will differ.

### Run it

`data/fpsbench_stream/test_mc.json` (990 records pointing into `FPSBenchStream/videos/`) is
already in the repo, so once the videos exist:

```bash
python -m video_qa.run_eval --model llava_ov_0.5b --dataset fpsbench_stream --sample_fps 1 \
    --num_chunks 2 --prune_method rlt_ref --prune_threshold 0.5
```

If you assembled into a different `--video-dir`, regenerate the annotation against it:

```bash
python video_qa/convert_fpsbench_stream.py --src FPSBenchStream/fpsbench_stream_v1.jsonl \
    --video_root <video-dir> --out data/fpsbench_stream/test_mc.json
```

`python -m video_qa.run_eval_fpsbench_needle` runs the needle-only arm — the same questions
against the clip alone — under `results/needle_only/`, which is what isolates retrieval from
the model's ceiling.

## Efficiency

Streaming throughput, KV-Cache growth and GFLOPs/frame, for the baseline and every pruning
threshold, on one FPS-Bench-Stream stream. Needs a GPU.

```bash
scripts/efficiency/cost_model.sh                      # llava_ov_7b, 0.1 .. 0.9
MODEL=llava_ov_0.5b scripts/efficiency/cost_model.sh
THRESHOLDS="0.5 0.9" N_STREAMS=3 scripts/efficiency/cost_model.sh
```

Finished arms are skipped so an interrupted run resumes (`FORCE=1` re-runs them); rebuild
the table any time with `scripts/efficiency/collect_cost_model.py`.

The defaults are what make the numbers comparable — change one and you are measuring
something else: `ENCODE_CHUNK_SIZE=1` (a live stream has no frame t+1 to batch with),
`TIMING=span` (syncs per run, not per chunk), `GPU_PREPROCESS=true` (use `false` for
throughput quoted beside an accuracy number), and a `NUM_FRAMES` past steady state — the
local window fills after `15000 / (196 x keep_rate)` frames, and the collector flags rows
that never got there. Throughput is measured, not derived; the vision tower and
preprocessing are untouched by pruning and cap the speed-up.
