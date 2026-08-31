import warnings
import random
import json
import os
import math
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from decord._ffi.base import DECORDError
from transformers import (
    logging,
    LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
    VideoLlavaForConditionalGeneration, VideoLlavaProcessor
)
import logzero
from logzero import logger

from model import llava_onevision_rekv, video_llava_rekv, longva_rekv
# Re-exported: the flags live in a torch-free module so video_qa/run_eval.py can register
# the same ones without importing every model backend.
from video_qa.reduction_args import add_reduction_args


MODELS = {
    'llava_ov_0.5b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-7b-ov-hf',
    },
    'llava_ov_72b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
    },
    'video_llava_7b': {
        'load_func': video_llava_rekv.load_model,
        'model_class': VideoLlavaForConditionalGeneration,
        'processor_class': VideoLlavaProcessor,
        'model_path': 'model_zoo/Video-LLaVA-7B-hf',
    },
    'longva_7b': {
        'load_func': longva_rekv.load_model,
        'model_path': 'model_zoo/LongVA-7B',
    },
}



class _FrameBatch:
    """decord's batch handle, for the one method every caller here uses."""

    def __init__(self, frames):
        self._frames = frames

    def asnumpy(self):
        return self._frames


class NpyFrameReader:
    """A pre-extracted `(T, H, W, 3)` uint8 frame array behind decord's VideoReader API.

    RVS-Movie (data/rvs/movie/*.npy) ships MovieNet already decoded rather than as a
    container, so decord cannot open those paths at all -- it demuxes the numpy header and
    reports the file as invalid data. Frame count equals the annotation's `duration` in
    seconds, which is where the 1.0 below comes from: the frames are one per second and
    there is no header to read a rate out of.

    Memory-mapped, because these arrays run to ~1 GB and every caller subsamples to a
    stride immediately -- materializing a whole movie to keep every other frame would cost
    more RAM than the sampled frames themselves.
    """

    def __init__(self, video_path):
        self._frames = np.load(video_path, mmap_mode='r')

    def __len__(self):
        return len(self._frames)

    def get_avg_fps(self):
        return 1.0

    def get_batch(self, frame_idx):
        # Fancy-indexing a memmap already returns an in-memory copy, so this is the point
        # where the sampled frames -- and only those -- are read off disk.
        return _FrameBatch(np.asarray(self._frames[list(frame_idx)]))


def decord_num_threads():
    """How many decode threads decord may use.

    decord's default (`num_threads=0`) sizes its thread pool from the machine's hardware
    concurrency, which ignores the CPU affinity mask a scheduler hands the process: 40
    threads on the 8 CPUs `--cpus-per-gpu=8` actually granted. That oversubscription is
    worth avoiding on its own -- under `--num_chunks 4` it is four full-size pools sharing
    one allocation -- but note it is *not* what causes the DECORDError that
    `_RetryingVideoReader` below exists to survive. See that class for the real fault.

    Sizing the pool to the affinity mask costs little: measured on 40 ODV-Bench clips,
    28.1 frames/s at 8 threads against 32.2 at the unbounded default, where forcing a
    single thread would give 13.8.

    `REKV_DECORD_THREADS` overrides. run_eval.py sets it when it fans out, since only the
    parent knows how many workers will be sharing the allocation.
    """
    override = os.environ.get('REKV_DECORD_THREADS')
    if override:
        return max(1, int(override))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        return max(1, os.cpu_count() or 1)


class _RetryingVideoReader:
    """decord VideoReader that drops to single-threaded decode when threading faults.

    decord 0.6.0's multi-threaded decoder is broken on some H.264 streams. Its worker
    calls `avcodec_send_packet`, gets EAGAIN (-11), treats it as fatal and dies; the next
    `Pop()` then fails the liveness check and the whole run aborts with

        DECORDError: threaded_decoder.cc:104: Check failed: run_.load()

    (the more informative "Thread worker: Error sending packet ... (-11 vs. 0)" only
    surfaces on some attempts, which is what makes the bare CHECK so hard to read).

    It is a property of the file, not of load or batch size: `data/rvs/ego/videos/
    9198b9a4-*.mp4` fails at any thread count above one, for a batch of ten frames as
    surely as for the default run's 1800, on an idle node. The same call at
    `num_threads=1` succeeds. Meanwhile 400 ODV-Bench clips decode fine multi-threaded --
    so pinning everything to one thread to satisfy the bad files would pay 2.3x on decode
    for every good one.

    Hence: attempt the threaded decode, and on a DECORDError reopen single-threaded and
    retry once. The downgrade sticks for the life of the reader, so a windowed
    FrameStream that decodes block after block pays the detection cost once rather than
    per block.
    """

    def __init__(self, video_path, num_threads):
        self._path = video_path
        self._num_threads = max(1, num_threads)
        self._vr = VideoReader(video_path, ctx=cpu(0), num_threads=self._num_threads)

    def __len__(self):
        return len(self._vr)

    def get_avg_fps(self):
        return self._vr.get_avg_fps()

    def __getitem__(self, idx):
        return self._vr[idx]

    def get_batch(self, indices):
        try:
            return self._vr.get_batch(indices)
        except DECORDError:
            if self._num_threads == 1:
                raise  # already serial: a real decode failure, not the threading fault
            logger.warning(
                f'{os.path.basename(self._path)}: decord failed at '
                f'num_threads={self._num_threads}; reopening single-threaded. Decode will '
                f'be slower for this video.')
            self._num_threads = 1
            self._vr = VideoReader(self._path, ctx=cpu(0), num_threads=1)
            return self._vr.get_batch(indices)


def open_video_reader(video_path):
    """decord for containers, `NpyFrameReader` for pre-extracted `.npy` frame arrays.

    Both honour the same small slice of the VideoReader API (`len`, `get_avg_fps`,
    `get_batch(...).asnumpy()`), so the sampling grids below are written once and do not
    care which kind of source they are on.
    """
    if video_path.endswith('.npy'):
        return NpyFrameReader(video_path)
    return _RetryingVideoReader(video_path, decord_num_threads())


class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 retrieve_size=64, chunk_size=1, exact_fps=False,
                 decode_window=None) -> None:

        self.sample_fps = sample_fps
        self.exact_fps = exact_fps
        # Slots a FrameStream decodes at a time; None = the whole video up front. Held
        # here rather than in the streaming solvers because `work()` parses it for every
        # solver, and a solver that never opens a stream (blind_vqa) simply ignores it.
        self.decode_window = decode_window

        self.qa_model = qa_model
        self.qa_processor = qa_processor

        # Retrieval Hyperparams
        assert chunk_size <= retrieve_size, f'chunk_size: {chunk_size}, retrieve_size: {retrieve_size}'
        self.retrieve_size = retrieve_size
        self.chunk_size = chunk_size

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno
        self.eval_grounding = 'temporal_windows' in anno[0]['conversations'][0]

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = {(self.retrieve_size, self.chunk_size): []}

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]

    def load_video(self, video_path):
        vr = open_video_reader(video_path)
        fps = round(vr.get_avg_fps())
        if self.exact_fps:
            # Sample on the exact `sample_fps` timestamp grid, each slot taking the most
            # recent source frame, instead of an integer stride. Two consequences, both
            # deliberate:
            #   * below the source rate it removes the stride quantization -- 16 fps off a
            #     30 fps file is 16 fps, not the 30 the stride path delivers;
            #   * above it, the grid has more slots than the file has frames, so frames
            #     repeat. A repeat carries no new evidence (it is bit-identical, so RLT
            #     drops it at any threshold and the KV-Cache is unchanged), which makes
            #     this a cost/throughput condition, not a temporal-resolution one. No
            #     question becomes answerable at 32 fps that was not answerable at 30.
            n_target = max(1, int(round(len(vr) / fps * self.sample_fps)))
            frame_idx = [min(len(vr) - 1, int(t * fps / self.sample_fps))
                         for t in range(n_target)]
        else:
            # Clamped: asking for more frames per second than the file has makes the stride
            # zero and `range` raises. Saturating at every frame is the only thing the request
            # can mean -- there is nothing finer to sample. Note the stride is an integer, so
            # the rate actually delivered is fps/round(fps/sample_fps), not sample_fps: at
            # native 30, requesting 16, 32 and 64 all give the same 30.
            stride = max(1, int(fps / self.sample_fps))
            frame_idx = [i for i in range(0, len(vr), stride)]
        video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f'video shape: {video.shape}')
        return video
    
    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask):
        total_intersection_length = 0.0
    
        for (start_sec, end_sec) in gt_temporal_windows:
            start = math.floor(start_sec)
            end = math.ceil(end_sec)
            for i in range(start, end):
                if i < len(retrieved_mask) and retrieved_mask[i]:
                    intersection_start = max(start_sec, i)
                    intersection_end = min(end_sec, i + 1)
                    total_intersection_length += intersection_end - intersection_start

        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
        retrieved_len = sum(retrieved_mask).item()

        recall = total_intersection_length / gt_len if gt_len > 0 else 0
        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return recall, precision, f1
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s):
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            return s[0]

    def reduction_stats(self):
        """Per-video token-reduction counters, for the results CSV.

        Call after ingesting frames and merge into each record row. Both stages' counters are
        cumulative since their last reset(), and `clear_cache()` resets them at the start
        of every video, so what they hold here is this video's numbers alone.

        Returns {} when neither stage is on, so baseline runs keep their existing columns
        and the two are still concatenable. Otherwise the CSV carries the keep rate the
        run actually achieved, which is the number an accuracy comparison has to be read
        against -- a threshold means nothing without it, since redundancy varies enormously
        between videos, and the aggregate printed at the end of a run hides that spread.
        """
        model = self.qa_model
        stats = {}

        pruner = getattr(model, 'token_pruner', None)
        if pruner is not None:
            stats['tokens_kept'] = pruner.n_kept
            stats['tokens_seen'] = pruner.n_seen
            stats['token_keep_rate'] = round(pruner.keep_rate, 4)

        reducer = getattr(model, 'vision_reducer', None)
        if reducer is not None:
            stats['patches_encoded'] = reducer.n_kept
            stats['patches_seen'] = reducer.n_seen
            stats['patch_keep_rate'] = round(reducer.keep_rate, 4)

        if stats:
            # Recorded alongside the rates because they are what actually determines
            # behaviour: below n_local the whole video fits in the sliding window and
            # retrieval never runs, so a keep rate high enough to trigger that is
            # measuring something different from one that does not.
            stats['n_tokens_fed'] = getattr(model, '_n_tokens_fed', None)
            stats['num_blocks'] = getattr(model, 'num_blocks', None)
            stats['n_local'] = getattr(model, 'n_local', None)
            stats['kv_cache_bytes'] = model.calc_memory_usage()
        return stats

    def video_open_qa(self, question, max_new_tokens=1024):
        pass

    def video_close_qa(self, question, candidates, correct_choice):
        pass

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        pass

    def analyze(self, debug=False):
        video_annos = self.anno[:1] if debug else self.anno
        for video_sample in tqdm(video_annos):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            self.analyze_a_video(video_sample)

        dfs = []
        for (retrieve_size, chunk_size), dict_list in self.record.items():
            df = pd.DataFrame(dict_list)
            df['retrieve_size'] = retrieve_size
            df['chunk_size'] = chunk_size
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False)


def pruning_enabled(args):
    """True when args select a stage-2 method. --prune_threshold alone still means
    'rlt', so scripts written before --prune_method exists keep working."""
    method = getattr(args, 'prune_method', 'none')
    return method not in (None, 'none') or getattr(args, 'prune_threshold', None) is not None


def pruning_load_kwargs(args):
    """Stage-2 kwargs for load_model, empty when it is off.

    Kept empty rather than passing method='none' so backends that never implemented
    pruning keep their original signature and fail loudly if you ask them for it.
    """
    if not pruning_enabled(args):
        return {}
    assert args.model.startswith('llava_ov'), \
        f'token pruning is only implemented for llava_ov_* backends, not {args.model}'
    return dict(
        prune_method=getattr(args, 'prune_method', 'none'),
        prune_threshold=args.prune_threshold,
        prune_metric=args.prune_metric,
        prune_refresh_every=args.prune_refresh_every,
        prune_log_percentiles=getattr(args, 'prune_log_percentiles', False),
    )


def vision_reduction_enabled(args):
    """True when args select a stage-1 (encoder-side) reduction method."""
    return getattr(args, 'vision_method', 'none') not in (None, 'none')


def vision_reduction_load_kwargs(args):
    """Stage-1 kwargs for load_model, empty when it is off.

    Empty rather than method='none' so backends that never implemented it keep their
    original signature and fail loudly if asked for it. Shared with
    video_qa/measure_encoding_fps.py so the benchmark loads the model exactly the way the
    eval does -- a speed number for a differently-configured model is worthless.
    """
    if not vision_reduction_enabled(args):
        return {}
    assert args.model.startswith('llava_ov'), \
        f'vision reduction is only implemented for llava_ov_* backends, not {args.model}'
    return dict(
        vision_method=args.vision_method,
        vision_threshold=args.vision_threshold,
        vision_mask_space=args.vision_mask_space,
        vision_metric=args.vision_metric,
        vision_refresh_every=args.vision_refresh_every,
    )


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS, add_args=None):
    """Run one worker process for QA_CLASS.

    `add_args` lets a solver register extra flags of its own (it is handed the parser)
    and, when given, receives the parsed namespace back as the `args=` keyword on
    QA_CLASS. Solvers that pass nothing get the original signature and the original
    constructor call, so nothing here changes for them. The point of the hook is that a
    variant solver -- blind/blind_stream_vqa.py, say -- reuses this model-loading path
    verbatim instead of copying it, since a copy would drift from the real eval and
    silently compare two differently-configured models.
    """
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--decode_window", type=int, default=256,
                        help="Slots a FrameStream decodes at a time (0 = the whole video "
                             "up front, which is what every solver here did before this "
                             "flag). Frames are held at source resolution, so a whole-video "
                             "decode costs sample_fps x seconds x MB-per-frame per worker: "
                             "a few GB on a short clip, but 85 GB for a 600 s stream at "
                             "32 fps, times --num_chunks workers on one node. Frames arrive "
                             "in order and are never re-read, so windowing decodes each "
                             "block exactly once and only bounds residency -- the frames, "
                             "and the results, are identical either way.")
    parser.add_argument("--exact_fps", type=str2bool, nargs='?', const=True, default=False,
                        help="Sample on the exact --sample_fps grid instead of an integer "
                             "stride, repeating frames when the requested rate exceeds the "
                             "file's own. Required for any rate above the source rate; "
                             "off (default) reproduces every previous run exactly.")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    # Both reduction stages (llava_ov_* only). Every default is off = baseline.
    add_reduction_args(parser)
    parser.add_argument("--prune_log_percentiles", type=str2bool, nargs='?', const=True, default=False)
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # VideoQA model
    model_path = MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    load_kwargs = dict(
        model_path=model_path,
        n_local=args.n_local,
        topk=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
    )
    load_kwargs.update(vision_reduction_load_kwargs(args))
    load_kwargs.update(pruning_load_kwargs(args))
    videoqa_model, videoqa_processor = load_func(**load_kwargs)

    # Load ground truth file
    anno = json.load(open(args.anno_path))

    retrieve_analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        exact_fps=args.exact_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        retrieve_size=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
        decode_window=args.decode_window,
        **({'args': args} if add_args is not None else {}),
    )

    retrieve_analyzer.analyze(debug=args.debug)
