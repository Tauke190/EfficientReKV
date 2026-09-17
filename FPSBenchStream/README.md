# FPS-Bench-Stream v1

Long-form streaming benchmark. Each example is one FPS-Bench clip
(the **needle**) spliced into one long MLVU video (the **haystack**), producing a
600-second stream with exactly two scene cuts.

**This folder contains no video.** It contains the identifiers and timestamps
needed to rebuild every stream locally, in the same spirit as the FPS-Bench
release itself. You need MLVU and the FPS-Bench clips yourself.

## Why it is built this way

FPS-Bench's own source videos are too short to make a long-video benchmark for streaming video LLMs. Therefore it is extended with MLVU padding footage to represent context changes/cuts in long video.

## Files

| file | what it is |
|---|---|
| `fpsbench_stream_v1.jsonl` | canonical annotations, one record per stream |
| `fpsbench_stream_v1.csv` | flattened mirror |
| `fpsbench_stream_v1.schema.json` | JSON Schema (draft 2020-12) |
| `fpsbench_stream_v1_stats.json` | build statistics |
| `haystack_files_used.txt` | the 459 MLVU files to ship alongside these annotations |
| `haystack_manifest.tsv` | duration/resolution/fps of the MLVU pool, as used at plan time |
| `skipped.jsonl` | questions excluded at plan time, with reason |
| `scripts/build_stream_dataset.py` | the planner/assembler that produced this release |

## Record layout

Each record is a **complete FPS-Bench v1 record** with one added `stream` block.
Every original field is preserved verbatim, so this file is a strict superset of
`annotations/fpsbench_v1.jsonl` and works anywhere that file does. (Verified:
diffing all 996 records minus their `stream` block against `fpsbench_v1.jsonl`
gives zero differences, so this file is standalone -- you do not also need the
base annotations.)

```
{ id, source, time, question, temporal_requirements, categories, metadata,   <- unchanged
  stream: { stream_id, target_duration_sec,
            haystack:      { file, window_start_sec, window_duration_sec }   <- SHIPPABLE
            needle:        { clip_file, rebuild_from: {url, start, end} }    <- NOT shippable
            insertion:     { offset_in_window_sec, position_bin }
            timeline:      { needle_start_sec, query_time_sec, ... }         <- use these to evaluate
            normalization: { width, height, fps, audio }                     <- per record, see below
            built, unbuilt_reason } }
```

### Three clocks -- do not mix them

| field | measured in |
|---|---|
| `time.clip_start_sec`, `time.clip_end_sec` | the original **YouTube** video |
| `stream.haystack.window_start_sec` | the original **MLVU** file |
| everything in `stream.timeline` | the **assembled** stream |

The same needle sits at 100 s in its YouTube source and 149.8 s in the assembled
stream. For evaluation you want `stream.timeline` and nothing else; the other two
are build-time provenance.

## Normalization -- the canvas is per stream, not fixed

Each stream is built on **its own haystack's native resolution**, at
`fps = max(haystack_fps, needle_min_fps)`. There is no global 1280x720/30
canvas. `stream.normalization` records the canvas actually used for that stream,
so read it per record rather than assuming one:

- 60 distinct resolutions; the most common are 720x540 (411 streams), 320x240
  (59), 1920x802 (40).
- fps: 25 (502), 30 (378), 24 (101), 28 (6), 50 (4), 60/29 (2 each), 26 (1).

The needle's `min_fps` can raise the canvas above the haystack's rate, because a
needle sampled below `min_fps` would lose the very motion the question asks
about. The haystack's native resolution is kept because it costs less than
upscaling to a fixed canvas and still clears the 384x384 that vision towers
resize each frame to.

Every part -- pre-needle haystack, needle, post-needle haystack -- goes through
one identical filter chain, so the splice leaves no resolution, frame-rate,
SAR, or audio discontinuity:

```
scale=W:H:force_original_aspect_ratio=decrease,pad=W:H:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=F
```

Audio is dropped because the FPS-Bench clips were fetched with `--no-audio`;
keeping the haystack's track would make the cut audible.

## Shipping

The haystack halves are redistributable, the needles are not. Ship:

1. these annotations (including `scripts/` and `haystack_manifest.tsv`), and
2. the 459 MLVU files named in `haystack_files_used.txt` (159.4 GB).

The recipient rebuilds each needle from `stream.needle.rebuild_from` (which
mirrors `source.url` and `time.clip_*_sec`), then runs `assemble`. Nothing in
this folder contains FPS-Bench video.

Integrity is currently only checkable by size: `haystack.file_size_bytes` is
recorded per file, but there are no content hashes and no pinned MLVU release.

## Composition

- **996 streams planned** from 1,000 FPS-Bench questions. 4 were skipped at plan
  time (clip never downloaded; see `skipped.jsonl`), and a further **6 failed to
  assemble** because their YouTube source had gone away, so **990 streams
  actually build**. The 6 carry `stream.built: false` and an
  `unbuilt_reason`; they are kept in the file so ids stay stable.
- **600 s** each. Needle occupies a median of 9.0 s = **1.5%** of the stream
  (range 2-25 s).
- Haystack pool: **459 MLVU videos** of at least 10 minutes. MLVU's own `needle_*`
  subset is excluded because it already has clips spliced into it.
- `count_*` haystacks are never paired with FPS-Bench counting questions
  (`instance_count`, `repetitive_motion`), which would collide.
- Haystack reuse: max 3, mean 2.17. Position is stratified early/middle/late
  (333/331/332 over all 996 records; `fpsbench_stream_v1_stats.json` reports
  329/330/331 because it counts only the 990 that built).
- Assembled footage totals 197.5 GB / 165 h.

## Rebuilding

**Use the shipped plan. Do not re-run `plan`.** `plan` draws haystack
assignments from an RNG whose sequence depends on which needle clips are present
in the clip cache, and it also numbers `stream_id` by position among the
non-skipped questions. A recipient whose clip cache differs by even one file
gets a different -- equally valid, but not identical -- dataset. `--seed` alone
does not pin it.

```bash
python scripts/build_stream_dataset.py assemble \
    --plan fpsbench_stream_v1.jsonl \
    --video-dir videos \
    --canvas haystack \
    --clip-dir ~/.cache/fpsbench/clips/clip \
    --haystack-dir /path/to/mlvu \
    --encoder h264_nvenc          # or libx264, the default
```

`--canvas haystack` is required to reproduce this release; the flag defaults to
`fixed`, which would instead take the canvas from `stream.normalization`.
Those now agree, so either reproduces the shipped geometry -- but `haystack`
is what was actually used and is what re-probes the source. Assemble is
resumable (it skips outputs that already exist) and sharded via
`--shard/--num-shards`.

Reproducibility is at the level of frames and timestamps, not bytes: the
encoder, preset and CRF are not recorded per record, and this release was cut
with `h264_nvenc` while the script defaults to `libx264`. Do not expect
identical file hashes.

To plan a *new* release rather than rebuild this one:

```bash
python scripts/build_stream_dataset.py plan \
    --target-duration 600 --output-dir data/FPSBenchStream \
    --annotations /path/to/fpsbench_v1.jsonl \
    --clip-dir ~/.cache/fpsbench/clips/clip \
    --manifest haystack_manifest.tsv
```

Note that `plan` writes a **flat** layout (`stream_id`, `needle`, `haystack`,
`insertion`, `stream`, `normalization` at top level) and fills `normalization`
from `--width/--height/--fps`. The shipped file is the **merged** layout
described above, with the real per-stream canvas backfilled from the assemble
run. `assemble` accepts both.

## Evaluation protocol

`stream.timeline.query_time_sec` is the moment the question should be asked. It
equals `certificate_end_sec`, the point at which the answer first becomes
determined, so a model answering at that timestamp needs no future frames. Under
a streaming protocol, feed frames up to `query_time_sec` and ask; under an
offline protocol, give the whole stream and accept that the model may look ahead.

## Known limitations

Read these before reporting any number.

- `query_time_sec` fires at `certificate_end`, so retrieval distance is 0: the
  needle is always the most recent content in the cache.
- Needle/haystack pairing is random -- MLVU ships no topic labels -- so both
  topic and aspect-ratio letterboxing differ from the haystack, and a cut
  detector can localise the needle.
- Uniform frame sampling cannot solve this benchmark at any budget below
  ~3600 frames, by construction.
