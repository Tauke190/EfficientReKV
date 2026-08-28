"""Measure ReKV's running speed and memory usage under the paper's streaming protocol.

Reproduces the setup behind the efficiency numbers in the ICLR'25 paper: "Video Enc."
(frames encoded per second), "Latency" (question input to response completion), and the
memory the run costs. Speed is reported as two separate figures with no combined one --
ingestion must keep pace with the live stream whether or not anyone is asking questions,
so folding QA time into throughput would make it depend on question frequency rather than
on the system.

Memory splits the same way, along the axis that matters: the offloaded CPU KV-Cache grows
linearly with video length, so it is reported per hour of video; GPU memory does not grow
with length at all -- it is bounded by the n_local window and the encode batch -- so it is
reported as a flat absolute. That contrast is ReKV's central claim, and averaging the two
into one number would hide it. At the default 1800 frames the stream is exactly one hour,
so the absolutes are already the per-hour figures.

The simulated stream:

  * one 1-hour 1080P video from RVS-Ego, sampled at 0.5 FPS -> 1800 frames, ingested in
    64-frame batches by default -- what the published figures were measured at. Pass
    --encode_chunk_size 1 for the streaming eval's own granularity; see below;
  * 100 questions scattered through the hour at their own annotated timestamps, injected
    **mid-stream** -- so retrieval runs against a partially-filled cache, as it would
    live, instead of against the finished video;
  * every question padded to exactly 64 tokens; answers generated naturally, stopping at
    EOS, capped at 128 tokens. Latency is ~linear in decode steps, so it is reported with
    the mean answer length and a ms/token figure beside it -- a bare mean latency is not
    comparable to anyone else's unless they generated the same number of tokens, and
    published latencies rarely say. --force_answer_length pins every answer to exactly
    --answer_tokens instead, which makes per-question cost constant and removes answer
    length as a variable; use it when the thing being compared could itself change how
    much the model says, since answer length is a dependent variable of anything that
    perturbs the KV-Cache. The ms/token figure is protocol-independent either way.

The two costs ReKV separates, and where they are measured:

  1. Encoding (ingestion) -- `_encode_video_chunk`, timed per frame. Retrieval does NOT
     run here: it is gated behind `set_retrieval()`, which only `question_answering`
     calls. Reported as frames/second.
  2. Question answering -- `question_answering`, timed per question, split into the
     retrieval portion and the prefill+decode portion.
  3. Memory -- `calc_memory_usage()` (model/abstract_rekv.py) for the offloaded CPU
     KV-Cache, and `torch.cuda.max_memory_allocated` for the GPU, with the model's own
     weights measured separately so they are not mistaken for a per-hour cost. Note the
     CPU figure covers the KV-Cache only, not total process RSS.

Excluded from all timers:
  - reading frames off disk (a real stream does not pay this, and the paper pre-extracts
    frames)
  - model loading and the warm-up frames/question

Included in the encode timer: the per-model frame preprocessing inside
`_encode_video_chunk` (resize/normalize), since that method is overridden by each
Video-LLM and timing it as a whole is what stays portable across them. It runs on the CPU
through the HF processor, as the eval does, which costs ~37-57 ms per 1080p frame -- more
than the vision tower. --gpu_preprocess moves it to the GPU (~2.2 ms) and nearly doubles
the reported FPS; that is a faster implementation of the same work, not the configuration
the published numbers come from, so it is off by default.

On ingestion granularity, the two settings answer different questions:

  --encode_chunk_size 1   what the streaming eval actually runs. A live stream has no
                          frame t+1 to batch with frame t, so video_qa/rekv_stream_vqa.py
                          drives `Abstract_ReKV.encode_frame` once per frame and every
                          stage -- preprocessing, vision tower, LM prefill -- sees one
                          frame at a time. Report this as the streaming number.
  --encode_chunk_size 64  the throughput ceiling if the frames were all in hand, as they
                          are for the offline solver. Roughly twice as fast, because
                          batch-1 kernels leave the GPU idle between launches. NOT a
                          streaming number: at 0.5 FPS a frame arrives every 2s, so
                          waiting to fill a 64-frame batch would add ~128s of lag.

The default stays at 64 so previously recorded numbers remain comparable. The two must
not be compared to each other.

Frames are pre-extracted once with ffmpeg into --frame_cache_dir and reused. Decoding the
video directly at this stride costs ~1.9 frames/second -- ~16 minutes per run, inside
untimed calls, which reads as a hang -- against ~12 minutes of extraction amortised over
every model x baseline/reduced run.

Encoding throughput is not flat across a video. Until the KV-Cache fills `n_local` the
local attention window is short and frames are cheap; once it is full, eviction and
RAM/disk offload kick in and per-frame time settles higher. The summary therefore reports
steady-state FPS (frames encoded after the local window is full) alongside the overall
figure. At the default 1800 frames the steady state dominates, which is the point.

Cost, and how to spend less of it: QA dominates wall-clock, because each answer is a
sequence of single-token forward passes against the retrieved blocks. Under
--force_answer_length every question costs the same by construction, so the latency mean
is stable well below 100 questions and `--num_questions 25` gives essentially the same
figure for a quarter of the time. That shortcut does NOT hold for natural-length answers:
the spread in answer length is then the dominant source of variance in the mean, so keep
the full 100 (and read the min/max the summary prints). Neither setting touches the
encoding number at all. `--skip_qa` drops QA entirely;
answering never mutates the video KV-Cache, so Video Enc. is identical either way and the
two metrics can be measured in separate runs.

The retrieval split (`--retrieval_breakdown`) is obtained by wrapping
`ContextManager.get_retrieved_kv`, which every model routes through. It is off by default:
that wrapper synchronises CUDA once per layer per question, which inflates the very
latency the script tells you to report. Turn it on to see where QA time goes, not to
produce the headline number.

Token reduction is loaded through the same helper the eval uses, so the benchmarked model
is configured identically to the one that produced the accuracy numbers. Measured split of
what this script times per frame (RVS-Ego @0.5fps, llava_ov_0.5b, GPU preprocessing):
preprocessing 2.5%, vision tower 13.6%, LM prefill 84.0%. So --vision_method (stage 1)
can only move ~14% of the encoding number and leaves KV and latency untouched, while
--prune_method (stage 2) attacks the 84% and shrinks KV with it. Note that CPU
preprocessing outweighs the vision tower at 1080p, so stage 1's effect is invisible
without --gpu_preprocess true.

Example (the paper's setup is the default):
    python video_qa/measure_encoding_fps.py --model llava_ov_7b
    python video_qa/measure_encoding_fps.py --model llava_ov_7b \
        --prune_method rlt --prune_threshold 0.25
"""

import os
import json
import time
import random
import inspect
import argparse
import warnings

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import logging
from tqdm import tqdm
import logzero
from logzero import logger

from video_qa.base import (MODELS, str2bool, add_reduction_args,
                           pruning_enabled, pruning_load_kwargs,
                           vision_reduction_enabled, vision_reduction_load_kwargs)
# Moved to its own module so the eval solvers can use the same cache and the same frame
# streams; this harness is not the only thing that cannot afford a 22-minute strided
# decode per video, and two copies of the reader would drift apart.
from video_qa.frame_cache import ensure_frame_cache, CachedFrameStream, DecordFrameStream
from model.attention.kv_cache_manager import ContextManager


# Accumulates time spent inside ContextManager.get_retrieved_kv across all
# layers for the question currently being answered. See patch_retrieval_timer.
_RETRIEVAL_SECONDS = 0.0


def patch_retrieval_timer():
    """Wrap `get_retrieved_kv` so retrieval can be separated from generation.

    Every model's `question_answering` reaches retrieval through this method,
    so wrapping it here stays portable instead of forking per-model timing.
    """
    original = ContextManager.get_retrieved_kv

    def timed_get_retrieved_kv(self, query=None):
        global _RETRIEVAL_SECONDS
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original(self, query)
        torch.cuda.synchronize()
        _RETRIEVAL_SECONDS += time.perf_counter() - t0
        return out

    ContextManager.get_retrieved_kv = timed_get_retrieved_kv


def pad_question(tokenizer, question, n_tokens, filler=" and"):
    """Force `question` to tokenize to exactly `n_tokens` tokens.

    The paper pads questions to a fixed length so every question costs the same to
    retrieve with and prefill. Padding is applied to the raw question text, which is what
    `question_answering` tokenizes for the retrieval pass and what `get_prompt` wraps for
    the prefill, so both see a constant budget.

    Done by measurement rather than arithmetic: appending a token to a string does not
    reliably add exactly one token (merges at the boundary), so append and re-count until
    it lands, then assert.
    """
    ids = tokenizer(question, add_special_tokens=False).input_ids
    if len(ids) > n_tokens:
        question = tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)

    guard = 0
    while len(tokenizer(question, add_special_tokens=False).input_ids) < n_tokens:
        question += filler
        guard += 1
        assert guard <= 4 * n_tokens, f"padding failed to converge for: {question[:80]!r}"

    ids = tokenizer(question, add_special_tokens=False).input_ids
    if len(ids) > n_tokens:  # the last filler overshot; trim back down
        question = tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)
    return question


def build_question_schedule(video_sample, args, tokenizer):
    """(frame_idx, input_text) pairs, scattered through the stream by annotated time.

    Questions are injected at their own timestamps -- the same "encode up to the
    question's time, then ask" rule video_qa/rekv_stream_vqa.py:52-60 uses -- so the
    amount of cache each question retrieves against is realistic rather than uniform.
    Sampling is evenly spaced over the time-ordered list, so asking for fewer questions
    still covers the whole hour instead of just its opening minutes.
    """
    convs = sorted(video_sample['conversations'], key=lambda c: c['end_time'])
    in_range = [c for c in convs if int(c['end_time'] * args.sample_fps) <= args.num_frames]
    if not in_range:
        earliest = convs[0]['end_time']
        raise ValueError(
            f"no questions fall within {args.num_frames} frames "
            f"({args.num_frames / args.sample_fps:.0f}s of video); the earliest is at "
            f"{earliest:.0f}s, i.e. frame {int(earliest * args.sample_fps)}. "
            f"Raise --num_frames (the 1800 default covers the whole hour)."
        )
    convs = in_range

    n = min(args.num_questions, len(convs))
    if n < args.num_questions:
        logger.warning(f"video has {len(convs)} questions inside the stream, "
                       f"{args.num_questions} requested; timing {n}")
    step = len(convs) / n
    picked = [convs[int(i * step)] for i in range(n)]

    schedule = []
    for c in picked:
        # Ask only after the frame carrying the answer has been ingested; frame 0 has no
        # cache to retrieve from at all.
        frame_idx = min(max(int(c['end_time'] * args.sample_fps), 1), args.num_frames)
        question = pad_question(tokenizer, c['question'], args.question_tokens)
        n_tok = len(tokenizer(question, add_special_tokens=False).input_ids)
        assert n_tok == args.question_tokens, f"padded to {n_tok}, wanted {args.question_tokens}"
        schedule.append((frame_idx, question))

    schedule.sort(key=lambda x: x[0])
    return schedule


class _Preprocessed:
    """Minimal stand-in for the processor's BatchFeature: only this field is read."""
    __slots__ = ('pixel_values_videos',)

    def __init__(self, pixel_values_videos):
        self.pixel_values_videos = pixel_values_videos


class GPUVideoProcessor:
    """Resize/rescale/normalize on the GPU, as a drop-in for `processor.video_processor`.

    `_encode_video_chunk` is the seam each Video-LLM overrides, and it reaches
    preprocessing through `self.processor.video_processor(...)`. Swapping that attribute
    keeps the timed call path exactly the model's own instead of forking the method here.

    The HF processor does this work single-threaded on the CPU, costing ~37-57 ms per
    1080p frame -- a third of encode time, and more than the vision tower. The same
    operations on the GPU cost ~3.7 ms. Resampling differs slightly from PIL's (mean
    absolute difference ~0.002 on a +-1 tensor), so this is a speed-path choice: pass
    --gpu_preprocess false to measure with byte-identical preprocessing to the eval runs.
    """

    _MODES = {0: 'nearest', 2: 'bilinear', 3: 'bicubic'}

    def __init__(self, ref, device, dtype):
        size = ref.size
        if 'height' in size:
            self.size = (size['height'], size['width'])
        else:  # shortest_edge-style configs resize to a square here anyway
            self.size = (size['shortest_edge'], size['shortest_edge'])
        self.mode = self._MODES.get(int(getattr(ref, 'resample', 3)), 'bicubic')
        self.device, self.dtype = device, dtype
        self.do_resize = getattr(ref, 'do_resize', True)
        self.do_rescale = getattr(ref, 'do_rescale', True)
        self.do_normalize = getattr(ref, 'do_normalize', True)
        self.rescale_factor = getattr(ref, 'rescale_factor', 1 / 255)
        self.mean = torch.tensor(ref.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(ref.image_std, device=device).view(1, 3, 1, 1)

    def __call__(self, video, return_tensors=None):
        x = video if torch.is_tensor(video) else torch.as_tensor(np.asarray(video))
        x = x.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float()
        if self.do_resize:
            if self.mode == 'nearest':
                x = F.interpolate(x, size=self.size, mode='nearest')
            else:
                x = F.interpolate(x, size=self.size, mode=self.mode,
                                  align_corners=False, antialias=True)
            # PIL resamples in the uint8 domain, so overshoot from the bicubic kernel is
            # clipped and rounded away before rescaling. Match that, or the two paths
            # diverge most on exactly the high-contrast edges the vision tower keys on.
            x = x.clamp_(0, 255).round_()
        if self.do_rescale:
            x = x.mul_(self.rescale_factor)
        if self.do_normalize:
            x = x.sub_(self.mean).div_(self.std)
        return _Preprocessed(x.unsqueeze(0).to(self.dtype))


@torch.inference_mode()
def answer_one(model, question, args, breakdown, supports_min_tokens):
    """Time one `question_answering` call, isolating the retrieval portion."""
    global _RETRIEVAL_SECONDS

    input_text = {"question": question, "prompt": model.get_prompt(question)}
    kwargs = {"max_new_tokens": args.answer_tokens}
    if args.force_answer_length and supports_min_tokens:
        # Hold the decode length fixed; otherwise an early EOS makes one question look
        # fast for reasons that have nothing to do with the KV-Cache.
        kwargs["min_new_tokens"] = args.answer_tokens

    _RETRIEVAL_SECONDS = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    answer = model.question_answering(input_text, **kwargs)
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    retrieval_seconds = _RETRIEVAL_SECONDS if breakdown else None
    # Decode steps actually paid for. Under natural-length answers this is the
    # denominator that makes latency comparable across protocols: a mean latency alone
    # conflates per-token cost with how much the model chose to say, and answer length
    # is a dependent variable of anything that changes the KV-Cache (pruning included).
    n_generated = getattr(model, 'last_generated_tokens', None)
    return {
        'latency_seconds': latency,
        'retrieval_seconds': retrieval_seconds,
        'generation_seconds': (latency - retrieval_seconds) if breakdown else None,
        'n_generated_tokens': n_generated,
        'ms_per_token': (1000.0 * latency / n_generated) if n_generated else None,
        'answer_chars': len(answer),
    }


@torch.inference_mode()
def encode_frames(model, stream, start, end, records=None, chunk_size=1, pbar=None):
    """Encode frames [start, end) one chunk at a time, timing each chunk.

    Only the ingestion loop is reproduced here (it is not overridden by any model);
    `_encode_video_chunk` itself is called through, since each Video-LLM overrides it.

    `pbar` is advanced by the frames encoded; updating it is outside every timer.
    """
    i = start
    while i < end:
        n = min(chunk_size, end - i)
        chunk = torch.cat([stream.get(j) for j in range(i, i + n)], dim=0)  # untimed

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model._encode_video_chunk(chunk)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if records is not None:
            # The local window is full once the cached video tokens exceed n_local; from
            # there on the eviction/offload path is active. Read off the model rather
            # than derived from the frame count: with stage-2 pruning a frame no longer
            # contributes a fixed number of tokens.
            records.append({
                'frame_idx': i,
                'num_frames': n,
                'seconds': elapsed,
                'fps': n / elapsed,
                'local_window_full': model._n_tokens_fed > model.n_local,
            })
        i += n

        if pbar is not None:
            # Instantaneous FPS, not the reported figure -- that one comes from the
            # steady-state chunks in summarize().
            pbar.set_postfix_str(f"{n / elapsed:.2f} f/s", refresh=False)
            pbar.update(n)


def summarize(df):
    total_frames = df['num_frames'].sum()
    total_seconds = df['seconds'].sum()
    overall_fps = total_frames / total_seconds

    steady = df[df['local_window_full']]
    steady_fps = steady['num_frames'].sum() / steady['seconds'].sum() if len(steady) else None

    return overall_fps, steady_fps, total_frames, total_seconds, len(steady)


def main():
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_0.5b")
    # The paper's streaming setup is the default: one 1-hour RVS-Ego video at 0.5 FPS
    # (1800 frames), 100 scattered questions, 64-token questions, 128-token answers.
    parser.add_argument("--anno_path", type=str, default="data/rvs/ego/ego4d_oe.json")
    parser.add_argument("--video_idx", type=int, default=0,
                        help="Which video of the annotation to stream.")
    parser.add_argument("--video_id", type=str, default=None,
                        help="Select by id instead of index.")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--num_frames", type=int, default=1800,
                        help="Frames to stream. 1800 = 1 hour at 0.5 FPS.")
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--question_tokens", type=int, default=64,
                        help="Every question is padded/truncated to exactly this length.")
    parser.add_argument("--answer_tokens", type=int, default=128,
                        help="Cap on generated answer length. With --force_answer_length "
                             "it is also the floor, so every answer costs exactly this "
                             "many decode steps.")
    parser.add_argument("--force_answer_length", type=str2bool, nargs='?', const=True, default=False,
                        help="Generate exactly --answer_tokens tokens per answer instead of "
                             "stopping at EOS. Off by default: natural-length answers are what "
                             "a deployment actually pays and what an unspecified published "
                             "latency most likely measured. Turn it on to make per-question "
                             "cost constant, which removes answer length as a variable when "
                             "comparing configurations -- worth doing whenever the thing being "
                             "compared (pruning, retrieval size) could itself change how much "
                             "the model says. Either way, report n_generated_tokens alongside "
                             "the latency; without it the number cannot be compared to anyone "
                             "else's.")
    parser.add_argument("--encode_chunk_size", type=int, default=64,
                        help="Frames per forward pass. 1 is what the streaming eval runs "
                             "(one arriving frame, one pass) and is ~2x slower here, "
                             "because batch-1 kernels leave the GPU idle. The default "
                             "stays at 64 -- encode_video's own default, and what the "
                             "paper's Video Enc. figure was measured at -- so previously "
                             "recorded numbers remain comparable; pass 1 to measure the "
                             "eval path instead.")
    parser.add_argument("--warmup_frames", type=int, default=32,
                        help="Frames encoded, then discarded with the cache, before timing "
                             "starts (CUDA autotune, lazy allocs).")
    parser.add_argument("--frame_cache_dir", type=str,
                        default=os.environ.get('REKV_FRAME_CACHE', 'data/frame_cache'),
                        help="Where to keep frames pre-extracted by ffmpeg, as the paper "
                             "does. Written on first use and reused after. Needs ~400 MB "
                             "for 1800 frames, so put it on a filesystem with room (set "
                             "$REKV_FRAME_CACHE to change the default). 'none' decodes the "
                             "video directly, which is ~30x slower at this stride.")
    parser.add_argument("--gpu_preprocess", type=str2bool, nargs='?', const=True, default=False,
                        help="Resize/normalize frames on the GPU instead of in the HF "
                             "processor (~2.2 vs ~37 ms/frame at 1080p, and nearly 2x the "
                             "reported FPS). Off by default so the measured path is the one "
                             "the eval and the paper actually run; turn it on to see how "
                             "much of the encode cost is CPU preprocessing.")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--skip_qa", type=str2bool, nargs='?', const=True, default=False,
                        help="Measure encoding only, injecting no questions. Answering does "
                             "not touch the video KV-Cache, so Video Enc. is essentially "
                             "the same either way -- use this to get it without paying for "
                             "QA. (Questions do split the encode batch at their timestamp, "
                             "but throughput is flat above ~8 frames/forward, so the effect "
                             "is under 3%%.)")
    parser.add_argument("--retrieval_breakdown", type=str2bool, nargs='?', const=True, default=False,
                        help="Split retrieval out of QA latency. Diagnostic only: it syncs "
                             "CUDA once per layer per question, which inflates the reported "
                             "latency. Off by default so the headline number is clean.")
    # Token reduction, same flags and same defaults as the eval (video_qa/reduction_args.py).
    add_reduction_args(parser)
    parser.add_argument("--progress", type=str2bool, nargs='?', const=True, default=True,
                        help="Show progress bars for the stream and the questions. Purely "
                             "cosmetic: bars are updated outside every timer.")
    parser.add_argument("--progress_interval", type=float, default=1.0,
                        help="Seconds between progress-bar redraws. Raise it when logging to "
                             "a file (e.g. Slurm) to keep the log short.")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    random.seed(2024)

    breakdown = args.retrieval_breakdown
    if breakdown and not args.skip_qa:
        patch_retrieval_timer()

    anno = json.load(open(args.anno_path))
    if args.video_id is not None:
        matches = [v for v in anno if v['video_id'] == args.video_id]
        assert matches, f"video_id {args.video_id} not in {args.anno_path}"
        video_sample = matches[0]
    else:
        video_sample = anno[args.video_idx]

    # Before the model load, not after: extraction needs nothing from the model, takes ~12
    # minutes on a cold cache, and can fail outright on a full disk. Doing it first keeps
    # the GPU free meanwhile and makes that failure cost seconds instead of a model load.
    frame_dir = None
    if args.frame_cache_dir and args.frame_cache_dir.lower() != 'none':
        frame_dir = ensure_frame_cache(video_sample['video_path'], video_sample['video_id'],
                                       args.sample_fps, args.frame_cache_dir)

    model_path = MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    model, _ = load_func(
        model_path=model_path,
        n_local=args.n_local,
        topk=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        **vision_reduction_load_kwargs(args),
        **pruning_load_kwargs(args),
    )

    # The model's own footprint, which does not scale with video length. Separating it out
    # is what makes the per-hour GPU figure mean anything: ReKV's whole claim is that GPU
    # memory stays bounded by the n_local window while the video grows, and that is
    # invisible if the weights are folded into the same number.
    torch.cuda.reset_peak_memory_stats()
    gpu_weights = torch.cuda.memory_allocated()

    # Fixed-length answers need the backend to accept min_new_tokens. Say so rather than
    # silently reporting a latency averaged over variable-length answers.
    supports_min_tokens = 'min_new_tokens' in inspect.signature(model.question_answering).parameters
    if args.force_answer_length and not supports_min_tokens and not args.skip_qa:
        logger.warning(f"{args.model} does not support min_new_tokens: answers stop at EOS, "
                       f"so latency is NOT over fixed {args.answer_tokens}-token answers.")

    tokenizer = model.processor.tokenizer
    schedule = [] if args.skip_qa else build_question_schedule(video_sample, args, tokenizer)

    if frame_dir is not None:
        stream = CachedFrameStream(frame_dir, num_frames=args.num_frames)
        source = f'cached frames ({frame_dir})'
    else:
        stream = DecordFrameStream(video_sample['video_path'], args.sample_fps,
                                   num_frames=args.num_frames)
        source = 'decord (slow: expect ~1.9 frames/s)'

    if args.gpu_preprocess:
        model.processor.video_processor = GPUVideoProcessor(
            model.processor.video_processor, model.device, model.dtype)

    if len(stream) < args.num_frames:
        logger.warning(f"video yields {len(stream)} frames at {args.sample_fps} FPS, "
                       f"{args.num_frames} requested")
    n_frames = len(stream)
    logger.info(f"streaming {video_sample['video_id']}: {n_frames} frames @ {args.sample_fps} FPS, "
                f"{len(schedule)} questions, from {source}")

    def make_bar(total, desc, position, unit):
        return tqdm(total=total, desc=desc, position=position, unit=unit, leave=True,
                    disable=not args.progress, mininterval=args.progress_interval,
                    dynamic_ncols=True)

    # Warm-up, then thrown away: the measured run must start from an empty cache.
    if args.warmup_frames > 0:
        model.clear_cache()
        model.encode_init_prompt()
        n_warmup = min(args.warmup_frames, n_frames)
        with make_bar(n_warmup + bool(schedule), "warm-up", 0, "step") as bar:
            encode_frames(model, stream, 0, n_warmup,
                          chunk_size=args.encode_chunk_size, pbar=bar)
            if schedule:
                bar.set_postfix_str("warm-up question", refresh=True)
                answer_one(model, schedule[0][1], args, breakdown, supports_min_tokens)
                bar.update(1)
        logger.info(f"warm-up done ({args.warmup_frames} frames)")

    model.clear_cache()
    model.encode_init_prompt()

    # Reset the peak *after* the warm-up and its clear_cache(): warm-up allocations and the
    # empty_cache() inside clear_cache would otherwise set a high-water mark that has
    # nothing to do with the measured stream.
    torch.cuda.reset_peak_memory_stats()

    records = []
    qa_records = []
    next_q = 0
    frame = 0
    # Two bars: encoding runs the length of the stream, QA is what actually dominates
    # wall-clock, and the encode bar sits still while a question is being answered.
    enc_bar = make_bar(n_frames, "encode", 0, "frame")
    qa_bar = make_bar(len(schedule), "questions", 1, "q") if schedule else None
    try:
        while frame < n_frames:
            # Encode up to the next question's timestamp, then ask it -- questions arrive
            # mid-stream, so retrieval sees a partially-filled cache.
            stop = schedule[next_q][0] if next_q < len(schedule) else n_frames
            stop = min(stop, n_frames)
            if stop > frame:
                encode_frames(model, stream, frame, stop, records, args.encode_chunk_size,
                              pbar=enc_bar)
                frame = stop

            while next_q < len(schedule) and schedule[next_q][0] <= frame:
                r = answer_one(model, schedule[next_q][1], args, breakdown, supports_min_tokens)
                r['question_idx'] = next_q
                r['frame_idx'] = frame
                qa_records.append(r)
                next_q += 1
                qa_bar.set_postfix_str(f"{r['latency_seconds']:.2f} s/q", refresh=False)
                qa_bar.update(1)
    finally:
        enc_bar.close()
        if qa_bar is not None:
            qa_bar.close()

    keep_counts = None
    if model.token_pruner is not None:
        keep_counts = (model.token_pruner.n_kept, model.token_pruner.n_seen)
    # Stage 1's rate is patches-encoded, a different denominator from stage 2's
    # tokens-stored. Reported separately for that reason: multiplying them would produce
    # a number that describes neither cost.
    v_keep_counts = None
    if model.vision_reducer is not None:
        v_keep_counts = (model.vision_reducer.n_kept, model.vision_reducer.n_seen)

    # Memory, taken once the whole stream is resident. calc_memory_usage covers the KV that
    # was offloaded to host RAM; the n_local window is still on the GPU and shows up in the
    # peak instead.
    gpu_peak = torch.cuda.max_memory_allocated()
    kv_bytes = model.calc_memory_usage()
    gpu_video = max(gpu_peak - gpu_weights, 0)

    df = pd.DataFrame(records)
    df['model'] = args.model
    df['n_local'] = args.n_local
    df['sample_fps'] = args.sample_fps
    df['n_frame_tokens'] = model.n_frame_tokens
    df['prune_method'] = args.prune_method if pruning_enabled(args) else 'none'
    df['prune_threshold'] = args.prune_threshold
    df['tokens_kept'] = keep_counts[0] if keep_counts else None
    df['tokens_seen'] = keep_counts[1] if keep_counts else None
    df['keep_rate'] = (keep_counts[0] / max(keep_counts[1], 1)) if keep_counts else None
    df['vision_method'] = args.vision_method if vision_reduction_enabled(args) else 'none'
    df['vision_threshold'] = args.vision_threshold
    df['vision_patches_kept'] = v_keep_counts[0] if v_keep_counts else None
    df['vision_patches_seen'] = v_keep_counts[1] if v_keep_counts else None
    df['vision_keep_rate'] = (v_keep_counts[0] / max(v_keep_counts[1], 1)) if v_keep_counts else None

    overall_fps, steady_fps, total_frames, total_seconds, n_steady = summarize(df)

    # Per hour of *video*, not of wall-clock: that is the axis memory actually scales on,
    # and it makes runs of different lengths comparable.
    video_hours = total_frames / args.sample_fps / 3600.0
    df['video_hours'] = video_hours
    df['kv_cache_bytes'] = kv_bytes
    df['gpu_weights_bytes'] = gpu_weights
    df['gpu_peak_bytes'] = gpu_peak
    qa_df = pd.DataFrame(qa_records) if qa_records else None

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 70)
    print("ReKV speed -- streaming protocol")
    print("=" * 70)
    print(f"  model             : {args.model}  ({model.n_frame_tokens} tokens/frame)")
    print(f"  gpu               : {gpu}")
    print(f"  video             : {video_sample['video_id']}  ({video_sample.get('duration', 0) / 60:.0f} min)")
    print(f"  stream            : {total_frames} frames @ {args.sample_fps} FPS, "
          f"{args.encode_chunk_size} frame(s)/forward")
    answer_protocol = (f"{args.answer_tokens}-token A (forced)" if args.force_answer_length
                       else f"natural A (cap {args.answer_tokens})")
    print(f"  questions         : {len(qa_records)} injected mid-stream, "
          f"{args.question_tokens}-token Q / {answer_protocol}")
    print(f"  n_local           : {args.n_local}   retrieve_size: {args.retrieve_size}")
    print(f"  frames from       : {source}")
    print(f"  preprocessing     : {'GPU (torch)' if args.gpu_preprocess else 'CPU (HF processor)'}")
    if vision_reduction_enabled(args):
        desc = args.vision_method
        if args.vision_threshold is not None:
            desc += (f"  threshold={args.vision_threshold:g} "
                     f"space={args.vision_mask_space} metric={args.vision_metric}")
            if args.vision_refresh_every:
                desc += f" refresh_every={args.vision_refresh_every}"
        print(f"  stage 1 (encoder) : {desc}")
        if v_keep_counts:
            kept, seen = v_keep_counts
            print(f"    patches encoded : {kept}/{seen} ({100.0 * kept / max(seen, 1):.1f}%)"
                  f"   -- ~14% of encode cost")
    else:
        print("  stage 1 (encoder) : none (baseline)")

    if pruning_enabled(args):
        desc = args.prune_method if args.prune_method not in (None, 'none') else 'rlt'
        if args.prune_threshold is not None:
            desc += f"  threshold={args.prune_threshold:g} metric={args.prune_metric}"
            if args.prune_refresh_every:
                desc += f" refresh_every={args.prune_refresh_every}"
        print(f"  stage 2 (memory)  : {desc}")
        if keep_counts:
            kept, seen = keep_counts
            print(f"    tokens kept     : {kept}/{seen} ({100.0 * kept / max(seen, 1):.1f}%)"
                  f"   -- ~84% of encode cost, plus KV and retrieval")
    else:
        print("  stage 2 (memory)  : none (baseline)")


    print("-" * 70)
    print("  [1] VIDEO ENC. -- ingestion throughput, no retrieval involved")
    print(f"      frames encoded   : {total_frames}")
    print(f"      encode time      : {total_seconds:.2f} s")
    print(f"      overall FPS      : {overall_fps:.2f}")
    if steady_fps is not None:
        print(f"      steady-state FPS : {steady_fps:.2f}   <-- encoding number to report")
        print(f"                         ({n_steady}/{len(df)} chunks, local window full)")
    else:
        print("      steady-state FPS : n/a -- no chunk filled n_local.")
        print("                         Use more frames or a larger --sample_fps.")

    if qa_df is not None:
        print("-" * 70)
        print("  [2] LATENCY -- question input to response completion; retrieval is here")
        print(f"      questions timed  : {len(qa_df)}")
        print(f"      mean latency     : {qa_df['latency_seconds'].mean():.3f} s/question"
              f"   <-- latency number to report")
        print(f"      median latency   : {qa_df['latency_seconds'].median():.3f} s/question")
        # Latency is ~linear in decode steps, so the mean above means nothing without
        # the length it was taken over. Under natural lengths this is also the check on
        # whether the configuration changed how much the model says: if answer length
        # moved between two runs, part of any latency difference is that, not speed.
        if qa_df['n_generated_tokens'].notna().any():
            toks = qa_df['n_generated_tokens']
            print(f"      answer tokens    : {toks.mean():.1f} mean / {toks.median():.0f} median"
                  f"   (min {toks.min():.0f}, max {toks.max():.0f})")
            if not args.force_answer_length:
                capped = int((toks >= args.answer_tokens).sum())
                print(f"                         {capped}/{len(toks)} hit the "
                      f"{args.answer_tokens}-token cap")
            print(f"      per-token cost   : {1000.0 * qa_df['latency_seconds'].sum() / toks.sum():.1f} ms/token"
                  f"   <-- comparable across answer-length protocols")
        if breakdown:
            share = 100.0 * qa_df['retrieval_seconds'].sum() / qa_df['latency_seconds'].sum()
            print(f"        retrieval      : {qa_df['retrieval_seconds'].mean():.3f} s  ({share:.1f}% of latency)")
            print(f"        prefill+decode : {qa_df['generation_seconds'].mean():.3f} s  ({100.0 - share:.1f}%)")
            print("      NOTE: --retrieval_breakdown syncs CUDA per layer per question,")
            print("            inflating the latency above. Drop it for the number to report.")
    else:
        print("-" * 70)
        print("  [2] LATENCY -- skipped (--skip_qa)")

    # Only the offloaded KV scales with video length, so only it gets a per-hour rate.
    # GPU is flat in duration -- it is set by n_local and the encode batch (measured: 2.84 /
    # 5.39 / 13.65 GB at 1, 16, 64 frames per forward, unchanged across 150-300 frames) --
    # and dividing a bounded quantity by hours yields a number that just falls as the video
    # lengthens. At the default 1800 frames the run is exactly 1 hour, so these absolutes
    # are the per-hour figures.
    GB = 1024 ** 3
    print("-" * 70)
    print(f"  [3] MEMORY -- over {video_hours:.2f} h of video "
          f"({total_frames} frames @ {args.sample_fps} FPS)")
    print(f"      KV-Cache (CPU)   : {kv_bytes / GB:6.2f} GB", end='')
    if video_hours > 0:
        print(f"   -> {kv_bytes / GB / video_hours:6.2f} GB/hour   <-- scales with length")
    else:
        print()
    if keep_counts:
        # The memory saving and the token saving are the same claim from two sides;
        # showing the dense equivalent makes that checkable rather than implied.
        kept, seen = keep_counts
        dense = kv_bytes * seen / max(kept, 1)
        print(f"        dense would be: {dense / GB:6.2f} GB", end='')
        if video_hours > 0:
            print(f"   -> {dense / GB / video_hours:6.2f} GB/hour"
                  f"   ({100.0 * kv_bytes / max(dense, 1):.1f}% of dense)")
        else:
            print()
    print(f"      GPU peak         : {gpu_peak / GB:6.2f} GB"
          f"                        <-- flat in video length")
    print(f"        model weights  : {gpu_weights / GB:6.2f} GB")
    print(f"        working set    : {gpu_video / GB:6.2f} GB   (set by n_local={args.n_local} "
          f"and {args.encode_chunk_size} frame(s)/forward)")
    print("=" * 70 + "\n")

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        df.to_csv(args.save_path, index=False)
        logger.info(f"Per-chunk encoding timings written to {args.save_path}")
        if qa_df is not None:
            qa_path = args.save_path.replace('.csv', '_qa.csv')
            if qa_path == args.save_path:
                qa_path = args.save_path + '.qa.csv'
            qa_df['model'] = args.model
            qa_df['n_local'] = args.n_local
            qa_df['retrieve_size'] = args.retrieve_size
            qa_df['answer_tokens_cap'] = args.answer_tokens
            qa_df['force_answer_length'] = args.force_answer_length
            qa_df['prune_method'] = args.prune_method if pruning_enabled(args) else 'none'
            qa_df['vision_method'] = args.vision_method if vision_reduction_enabled(args) else 'none'
            qa_df.to_csv(qa_path, index=False)
            logger.info(f"Per-question QA timings written to {qa_path}")


if __name__ == "__main__":
    main()
