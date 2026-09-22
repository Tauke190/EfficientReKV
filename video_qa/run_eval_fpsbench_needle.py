"""Launch the needle-only FPS-Bench-Stream arm (video_qa/rekv_fpsbench_needle_vqa.py).

Same shape as `eval_fpsbench_stream` in video_qa/run_eval.py -- one worker per GPU, merge,
score with video_qa/eval/eval_fpsbench_stream.py, audit with check_fpsbench_stream.py --
and it imports that file's helpers rather than copying them, so the chunking, merge,
reduction flags and result tags cannot drift from the full-stream arm.

Results land under a separate root, mirroring the full-stream layout:

    results/needle_only/<model>/fpsbench_stream/<retrieve_size>-<fps>[-query][-<subset>][<reduction>]

so the existing sweep collector reads them unchanged:

    python scripts/collect_fpsbench_sweep.py --model llava_ov_7b --results_root results/needle_only

Usage:
    python -m video_qa.run_eval_fpsbench_needle --model llava_ov_7b --sample_fps 2 \
        --num_chunks 2 --prune_method rlt_ref --prune_threshold 0.5
"""

import os
import argparse
import multiprocessing

from video_qa.reduction_args import add_reduction_args
from video_qa.run_eval import (merge_chunks, exec, score, reduction_tag, reduction_args,
                               stream_args)

RESULTS_ROOT = 'results/needle_only'


def run(args):
    stop_tag = "" if args.needle_stop == 'needle' else f"-{args.needle_stop}"
    anno_path = args.anno_path or "data/fpsbench_stream/test_mc.json"
    subset_tag = "" if args.anno_path is None \
        else f"-{os.path.splitext(os.path.basename(anno_path))[0]}"
    save_dir = (f"{RESULTS_ROOT}/{args.model}/fpsbench_stream/{args.retrieve_size}"
                f"-{args.sample_fps}{stop_tag}{subset_tag}{reduction_tag(args)}")

    solver_args = ["--max_new_tokens", str(args.max_new_tokens),
                   "--choice_seed", str(args.choice_seed),
                   "--needle_stop", args.needle_stop]
    if args.no_none_of_above:
        solver_args.append("--no_none_of_above")
    if args.shuffle_choices:
        solver_args.append("--shuffle_choices")
    solver_args += stream_args(args)

    if not args.only_eval:
        processes = []
        for idx in range(args.num_chunks):
            cmd = ["python", "video_qa/rekv_fpsbench_needle_vqa.py",
                   "--model", args.model,
                   "--sample_fps", str(args.sample_fps),
                   "--n_local", str(args.n_local),
                   "--retrieve_size", str(args.retrieve_size),
                   "--save_dir", save_dir,
                   "--anno_path", anno_path,
                   "--debug", args.debug,
                   "--num_chunks", str(args.num_chunks),
                   "--chunk_idx", str(idx)] + reduction_args(args) + solver_args
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        merge_chunks(save_dir, args.num_chunks)

    score(args, f"python video_qa/eval/eval_fpsbench_stream.py --save_dir {save_dir}")
    exec(f"python video_qa/eval/check_fpsbench_stream.py --save_dir {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b",
                        choices=['llava_ov_0.5b', 'llava_ov_7b', 'video_llava_7b',
                                 'longva_7b', 'flash_vstream_7b'])
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--skip_scoring", action="store_true")
    parser.add_argument("--anno_path", type=str, default=None,
                        help="Annotation override, e.g. a --limit subset built with "
                             "video_qa/convert_fpsbench_stream.py. Gets its own directory.")
    parser.add_argument("--needle_stop", type=str, default='needle',
                        choices=['needle', 'query'],
                        help="'needle' (default) feeds the whole needle clip; 'query' "
                             "stops at query_time_sec, like the full-stream query arm.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--no_none_of_above", action="store_true")
    parser.add_argument("--shuffle_choices", action="store_true")
    parser.add_argument("--choice_seed", type=int, default=2024)
    parser.add_argument("--decode_window", type=int, default=64,
                        help="Slots decoded per block. Blocks are aligned to multiples of "
                             "this, so a small window decodes little beyond the needle.")
    add_reduction_args(parser)
    args = parser.parse_args()

    # Same CPU split as run_eval.py: each worker's decord pool gets its share.
    try:
        _cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        _cpus = os.cpu_count() or 1
    os.environ.setdefault('REKV_DECORD_THREADS',
                          str(max(1, _cpus // max(1, args.num_chunks))))

    run(args)
