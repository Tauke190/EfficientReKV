# EfficientReKV

Casual Streaming Token reduction for ReKV streaming video QA.


## Eval

```bash
python -m video_qa.run_eval \
    --model llava_ov_0.5b --dataset ovobench_realtime --sample_fps 1 \
    --num_chunks 2 --n_local 15000 --retrieve_size 64 --decode_window 256 \
    --prune_method rlt --prune_threshold 0.5
```

### Datasets

| `--dataset` | Build annotation with |
|---|---|
| `ovobench_realtime`, `ovobench_backward` | `python video_qa/convert_ovobench.py` |
| `odvbench` | `python scripts/dataset_prep/setup_odvbench.py` |
| `fpsbench_stream` | `python video_qa/convert_fpsbench_stream.py` |
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
  any reduction flag.

### Pruning

- **`--prune_method rlt`** — drops tokens before the LM prefill
  shrinks throughput, KV RAM and retrieval together. This is the
  one that matters. Flags: `--prune_threshold`, `--prune_metric cosine|l2`,
  `--prune_refresh_every`.

**`--prune_threshold` is a cosine distance, not a rate** — a token is kept iff its distance
to the feature its position was last kept with exceeds it. Measured on `llava_ov_0.5b`,
ovobench_realtime @ 1 fps: 0.25 → 72% kept, 0.5 → 22% kept. Thresholds do not transfer across datasets

### Results

`results/<model>/<dataset>/<retrieve_size>-<sample_fps><tag>/results.csv`, where `<tag>` is
`` (baseline), `-blind`, or `-rlt0.5cosine`. StreamingBench writes one directory per
subset (`streamingbench_real`, …).

Keep rates are already in `results.csv` when a pruner is attached: `tokens_kept`,
`tokens_seen`, `token_keep_rate`, `kv_cache_bytes` and more, per question. **Baseline runs
have none of these columns** — when concatenating arms, fill `token_keep_rate` with 1.0
rather than dropping rows, and aggregate weighted by `tokens_seen`.

## Efficiency

Streaming throughput, KV-Cache growth and GFLOPs/frame, for the baseline and every pruning
threshold, on one FPS-Bench-Stream stream.

```bash
scripts/efficiency/cost_model.sh                      # llava_ov_7b, 0.1 .. 0.9
MODEL=llava_ov_0.5b scripts/efficiency/cost_model.sh
THRESHOLDS="0.5 0.9" N_STREAMS=3 scripts/efficiency/cost_model.sh
```

Needs a GPU (`srun -p gpu --gres=gpu:1 bash scripts/efficiency/cost_model.sh`). Arms already
on disk are skipped, so an interrupted run resumes; `FORCE=1` re-runs them. The table can be
rebuilt at any time with `scripts/efficiency/collect_cost_model.py`.

Three protocol choices decide whether the numbers are comparable:

- `ENCODE_CHUNK_SIZE=1` - one frame per forward pass, which is what streaming means: a
  live stream has no frame t+1 to batch with frame t. ~2x apart from the batched default.
- `TIMING=span` - three CUDA syncs for the whole run, so CPU preprocessing overlaps GPU
  compute the way it does live. `per_chunk` adds a sync pair per chunk and reads low at
  chunk size 1. Both write their own file, so running both cross-checks rather than
  overwrites.
- `NUM_FRAMES` must reach steady state - only chunks encoded after the local window
  filled run at the sustained rate. The window fills after `15000 / (196 x keep_rate)`
  frames: 77 at baseline, 306 at 25% keep, 1531 at 5%. At 1 fps x 600 frames anything
  keeping under ~13% never gets there, and the collector flags those rows rather than
  quoting them beside the others.

Throughput is measured, not derived: pruning drops tokens before the LM prefill (~84% of
`_encode_video_chunk`), but the vision tower and preprocessing are untouched and cap the
speed-up. GFLOPs/frame is analytic from `config.json` times the measured keep rate.
