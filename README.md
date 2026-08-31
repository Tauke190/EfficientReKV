# EfficientReKV

Casual Streaming Token reduction for ReKV streaming video QA. Two experiments: an **accuracy eval**
(`video_qa/run_eval.py`) and a **speed benchmark** (`video_qa/measure_encoding_fps.py`).


## Eval

```bash
python -m video_qa.run_eval \
    --model llava_ov_0.5b --dataset ovobench_realtime --sample_fps 1 \
    --num_chunks 2 --n_local 15000 --retrieve_size 64 --decode_window 256 \
    --prune_method rlt --prune_threshold 0.5
```

Or edit the variables at the top of `scripts/eval.sh` (direct) / `scripts/eval.slurm`
(sbatch twin), or copy a purpose-built sweep like `scripts/eval_0.5b_pareto.slurm`.

### Datasets

| `--dataset` | Build annotation with |
|---|---|
| `ovobench_realtime`, `ovobench_backward` | `python video_qa/convert_ovobench.py` |
| `odvbench` | `python scripts/setup_odvbench.py` |
| `fpsbench_stream` | `python video_qa/convert_fpsbench_stream.py` |
| `mlvu`, `egoschema`, `cgbench`, `qaego4d` | shipped |
| `rvs_ego`, `rvs_movie`, `activitynet_qa` | shipped, free-form |

`ovobench_*`, `odvbench`, `fpsbench_stream` are **streaming**: each question sees only
frames up to its own timestamp. Notes:

- `odvbench` clips are 5–90 s, so use `--sample_fps 2` or more; no clip nears `n_local`,
  so retrieval never fires there.
- `fpsbench_stream` only: `--trigger query` (default, realtime perception) vs `--trigger
  end` (retrieval, the published protocol). Separate results dirs.
- Free-form (`rvs_*`, `qaego4d`, `activitynet_qa`): pass `--skip_scoring`, then score with
  `scripts/score_open_ended.sh` (local) or `score_llmjudge_gpt.sh` (needs `OPENAI_API_KEY`).
  Score a whole sweep with one judge — local scores aren't comparable to published ones.
- `--blind` answers with no video, giving the language-prior floor. Refuses to run with
  any reduction flag.

### Pruning

Two stages, both off by default (no flags = baseline).

- **`--prune_method rlt`** (stage 2, memory-side) — drops tokens before the LM prefill
  (~84% of encode cost); shrinks throughput, KV RAM and retrieval together. This is the
  one that matters. Flags: `--prune_threshold`, `--prune_metric cosine|l2`,
  `--prune_refresh_every`.
- **`--vision_method rlt`** (stage 1, encoder-side) — attacks the vision tower only
  (~13.6%), KV size unchanged. Usually left off.

**`--prune_threshold` is a cosine distance, not a rate** — a token is kept iff its distance
to the feature its position was last kept with exceeds it. Measured on `llava_ov_0.5b`,
ovobench_realtime @ 1 fps: 0.25 → 72% kept, 0.5 → 22% kept. Thresholds do not transfer
across `sample_fps`, datasets, or between the two stages.

Gotcha: a bare `--prune_threshold` with no `--prune_method` still enables `rlt`, so the
baseline arm must pass *no* reduction flags — not a threshold of 0.

To pick thresholds without running a full eval (keep/drop depends only on projector output,
never the LM):

```bash
python video_qa/analyze_rlt_threshold.py --model llava_ov_0.5b \
    --anno_path data/ovo_bench/realtime.json --sample_fps_list 1 \
    --thresholds 0.25 0.5 0.6 0.7 0.8 0.9 --num_videos 32
```

### Results

`results/<model>/<dataset>/<retrieve_size>-<sample_fps><tag>/results.csv`, where `<tag>` is
`` (baseline), `-blind`, or `-rlt0.5cosine`.

Keep rates are already in `results.csv` when a pruner is attached: `tokens_kept`,
`tokens_seen`, `token_keep_rate`, `kv_cache_bytes` and more, per question. **Baseline runs
have none of these columns** — when concatenating arms, fill `token_keep_rate` with 1.0
rather than dropping rows, and aggregate weighted by `tokens_seen`.

Host RAM, not GPU RAM, is the usual limit — ReKV pins the whole KV-Cache per video, so a
worker's peak is set by its longest video. The slurm scripts size workers as
`fixed + fps × seconds × per_frame` (0.5B: 24 GB + 0.0035/frame; 7B: 32 GB + 0.0163/frame).

## Speed

One 1-hour RVS-Ego video, questions injected mid-stream; reports Video Enc. (FPS) and QA
latency separately.

```bash
scripts/measure_speed.sh                 # both models, baseline + rlt 0.25 + rlt 0.5
scripts/measure_speed.sh llava_ov_7b     # one model
NUM_FRAMES=800 EXTRA="--skip_qa true" scripts/measure_speed.sh llava_ov_0.5b
```

Three protocol choices decide whether the numbers are comparable:

- `--encode_chunk_size 1` — strict frame-by-frame, as the paper describes (~1.6× slower
  than batched).
- `--gpu_preprocess true` — same work on GPU (~2 ms/frame vs 37–57 ms). Absolute FPS sits
  **above** a published CPU-preprocessing figure; never quote a speedup from one protocol
  against a baseline from another.
- `NUM_FRAMES` must reach steady state: the `n_local` window fills after
  `15000 / (196 × keep_rate)` frames — 77 at baseline but 379 at 20% keep. Default 800.

`--skip_qa true` is the fast path for throughput only (QA never mutates the video cache).
