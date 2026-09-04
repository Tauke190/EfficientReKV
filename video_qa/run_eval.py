import os
import sys
import json
import argparse
import subprocess
import multiprocessing

import pandas as pd

from video_qa.reduction_args import add_reduction_args
from video_qa.eval import judges


def merge_chunks(save_dir, num_chunks):
    """Concatenate the per-chunk CSVs into results.csv, reporting anything missing.

    A chunk file that is absent means that worker died without recording a single video.
    The shell `head`/`tail` pipeline this replaces noted that only as a stderr line from
    `head` before going on to write a results.csv anyway -- empty when every chunk died,
    and, worse, silently half-sized when only one did. Either way the run looked finished.
    So: merge what exists, and say plainly what does not.

    Returns True when the merge is complete, False when videos are missing.
    """
    frames, missing = [], []
    for idx in range(num_chunks):
        path = f'{save_dir}/{num_chunks}_{idx}.csv'
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            missing.append(idx)
            continue
        frames.append(pd.read_csv(path))

    out = f'{save_dir}/results.csv'
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    else:
        open(out, 'w').close()
    for idx in range(num_chunks):
        path = f'{save_dir}/{num_chunks}_{idx}.csv'
        if os.path.exists(path):
            os.remove(path)

    # Written by BaseVQA.analyze for videos whose source would not decode.
    skipped = []
    for idx in range(num_chunks):
        sidecar = f'{save_dir}/skipped_{num_chunks}_{idx}.json'
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                skipped += json.load(f)['skipped']

    if missing or skipped:
        print(f'*** INCOMPLETE RUN: {save_dir}', file=sys.stderr)
        if missing:
            print(f'***   chunk(s) {missing} recorded nothing -- their videos are absent '
                  f'from results.csv', file=sys.stderr)
        if skipped:
            print(f'***   {len(skipped)} video(s) skipped as undecodable: '
                  f'{", ".join(skipped)}', file=sys.stderr)
        print('*** Scores below are computed on a subset. Do not report them as-is.',
              file=sys.stderr)
        return False
    return True


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env)


def score(args, cmd):
    """Run the scoring step unless --skip_scoring was passed.

    Skipping stops after predictions land in results.csv, which is all the scorers
    read -- so scoring can be run later without redoing inference.
    """
    if args.skip_scoring:
        print(f'skip scoring (--skip_scoring): {cmd}')
        return
    exec(cmd)


def reduction_tag(args):
    """Result-directory suffix identifying the reduction config.

    Empty for baseline runs, so their result paths are unchanged. Without this a
    threshold sweep would write every run to the same directory and each would overwrite
    the last. Stage 1 is prefixed 'v' so a stage-1-only, stage-2-only and combined run at
    the same threshold land in three different directories -- three different experiments.
    """
    tag = ""
    if vision_on(args):
        tag += f"-v{args.vision_method}"
        if args.vision_threshold is not None:
            tag += f"{args.vision_threshold:g}{args.vision_metric}"
        if args.vision_mask_space != 'embed':
            tag += f"-{args.vision_mask_space}"
        if args.vision_refresh_every:
            tag += f"-r{args.vision_refresh_every}"
    if pruning_on(args):
        # The method name leads: with more than one method in the registry a bare
        # threshold no longer identifies a run.
        method = args.prune_method if args.prune_method not in (None, 'none') else 'rlt'
        tag += f"-{method}"
        if args.prune_threshold is not None:
            tag += f"{args.prune_threshold:g}{args.prune_metric}"
        if args.prune_refresh_every:
            tag += f"-r{args.prune_refresh_every}"
    return tag


def open_ended_cmd(args, save_dir):
    """Scoring command for free-form-answer datasets (qaego4d/activitynet_qa/rvs_*).

    These have no string-match accuracy -- an LLM has to decide whether the prediction
    means the same thing as the reference. The default judge runs on your own GPU, so a
    sweep scores end-to-end without an API key; its scores are self-consistent across
    runs but not comparable to published gpt-3.5-turbo numbers, which need
    `--judge openai`.
    """
    if args.judge == 'local':
        style, model, suffix = judges.preset(args.judge_preset)
        style = args.judge_style or style
        model = args.judge_model or model
        # Separate output paths so a local-judge run never overwrites API-judge verdicts
        # sitting in the same results dir -- the two are not interchangeable. The
        # per-judge suffix does the same between local judges (see judges.PRESETS).
        return (f"python video_qa/eval/eval_open_ended_local.py "
                f"--pred_path {save_dir}/results.csv "
                f"--output_dir {save_dir}/tmp_local{suffix} "
                f"--output_json {save_dir}/results_local{suffix}.json "
                f"--judge_style {style} "
                f"--judge_model {model} "
                f"--yes_threshold {args.judge_yes_threshold} "
                f"--batch_size {args.judge_batch_size}")
    return (f"python video_qa/eval/eval_open_ended.py "
            f"--pred_path {save_dir}/results.csv "
            f"--output_dir {save_dir}/tmp "
            f"--output_json {save_dir}/results.json")


def vision_on(args):
    return getattr(args, 'vision_method', 'none') not in (None, 'none')


def pruning_on(args):
    """--prune_threshold alone still selects rlt, so pre-existing scripts keep working."""
    return args.prune_method not in (None, 'none') or args.prune_threshold is not None


def reduction_args(args):
    """Flags for both stages, forwarded only when enabled so the baseline command line
    stays byte-for-byte what it was."""
    cmd = []
    if vision_on(args):
        cmd += ["--vision_method", args.vision_method,
                "--vision_mask_space", args.vision_mask_space,
                "--vision_metric", args.vision_metric,
                "--vision_refresh_every", str(args.vision_refresh_every)]
        if args.vision_threshold is not None:
            cmd += ["--vision_threshold", str(args.vision_threshold)]
    if pruning_on(args):
        method = args.prune_method if args.prune_method not in (None, 'none') else 'rlt'
        cmd += ["--prune_method", method]
        if args.prune_threshold is not None:
            cmd += ["--prune_threshold", str(args.prune_threshold),
                    "--prune_metric", args.prune_metric,
                    "--prune_refresh_every", str(args.prune_refresh_every)]
    return cmd


def stream_args(args):
    """Flags for the streaming solvers' frame source.

    Only the streaming datasets take these: the offline solvers read a whole video by
    definition, so windowing has nothing to bound there.
    """
    return ["--decode_window", str(args.decode_window)]


def eval_mlvu(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/mlvu/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/mlvu/dev_debug_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval
    score(args, f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_mlvu_test(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/mlvu_test/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/mlvu/test_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval
    score(args, f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_qaego4d(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/qaego4d/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/qaego4d/test_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval
    score(args, f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_egoschema(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/egoschema/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/egoschema/full.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval
    score(args, f"python video_qa/eval/eval_egoschema.py --save_dir {save_dir}")

def eval_activitynet_qa(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/activitynet_qa/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/activitynet_qa/test.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"rm -rf {save_dir}/tmp")
        merge_chunks(save_dir, num_chunks)
    # eval
    score(args, open_ended_cmd(args, save_dir))

def eval_rvs_ego(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/rvs_ego/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}"
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/ego/ego4d_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"rm -rf {save_dir}/tmp")
        merge_chunks(save_dir, num_chunks)
    # eval
    score(args, open_ended_cmd(args, save_dir))

def eval_rvs_movie(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/rvs_movie/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}"
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/movie/movienet_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"rm -rf {save_dir}/tmp")
        merge_chunks(save_dir, num_chunks)
    # eval
    score(args, open_ended_cmd(args, save_dir))

def eval_ovobench(args, mode):
    """OVO-Bench, one mode at a time (realtime = RVP, backward = BT).

    Streaming solver, unlike every other multiple-choice dataset here: each query carries
    its own `realtime` timestamp and may only see frames up to it, so the video is
    ingested incrementally and questioned in between (video_qa/rekv_ovobench_vqa.py).
    Run video_qa/convert_ovobench.py first to produce the annotation file.

    FAR (REC/SSR/CRR) is not wired up -- those tasks are Yes/No-or-count, not multiple
    choice, and need their own prompts and scorer.

    `--blind` swaps in video_qa/blind_vqa.py, which answers every query with no video at
    all. It works here unchanged because that solver reads `gt_index` and the OVO-Bench
    annotation carries one -- which matters, since OVO-Bench's `answer` is often a
    paraphrase of the option rather than a copy of it, so the index is the only reliable
    route to the gold letter. eval_ovobench.py scores a blind CSV normally: n_frames_seen
    is 0, which satisfies the no-leak invariant rather than bypassing it.
    """
    num_chunks = args.num_chunks
    # Same '-blind' convention as odvbench: the control lives in its own directory so it
    # can never overwrite the sighted run it exists to be compared against, and the
    # pairing stays a plain suffix (64-1.0 <-> 64-1.0-blind) rather than a lookup.
    blind_tag = "-blind" if args.blind else ""
    save_dir = f"results/{args.model}/ovobench_{mode}/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}{blind_tag}"
    solver = "blind_vqa" if args.blind else "rekv_ovobench_vqa"
    anno_path = f"data/ovo_bench/{mode}.json"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args) + stream_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval: OVO-Bench averages per-task accuracies within a mode, so it needs its own
    # scorer rather than eval_multiple_choice.py's pooled mean.
    score(args, f"python video_qa/eval/eval_ovobench.py --save_dir {save_dir}")


def eval_streamingbench(args, subset):
    """StreamingBench, one subset at a time (real / omni / context / sqa).

    Same streaming contract as OVO-Bench -- a query at `time_stamp` t sees only frames in
    [0, t] -- so it runs on the same incremental solver, subclassed only for Sequential
    QA's carried conversation history (video_qa/rekv_streamingbench_vqa.py). Run
    video_qa/convert_streamingbench.py first to produce the annotation file, which in turn
    needs scripts/setup_streamingbench.py to have unpacked the videos.

    Proactive Output is not wired up: it is scored on *when* the model speaks against a
    ground-truth timestamp, not on a letter, and needs its own solver and metric.

    `--blind` swaps in video_qa/blind_vqa.py, which answers every query with no video at
    all -- the floor any real score has to be read against, since StreamingBench's options
    are written by the same annotators who wrote the questions. It works here unchanged
    because that solver reads `gt_index`, which convert_streamingbench.py decides at
    conversion time. Not offered for `sqa`: blind_vqa has no --sqa_context, so a blind SQA
    run would be answering a different question set than the sighted one it is supposed to
    be the control for. eval_streamingbench.py scores a blind CSV normally -- n_frames_seen
    is 0, which satisfies the no-leak invariant rather than bypassing it.
    """
    num_chunks = args.num_chunks
    # Same '-blind' convention as odvbench/ovobench: the control lives in its own directory
    # so it can never overwrite the sighted run it exists to be compared against, and the
    # pairing stays a plain suffix (64-1.0 <-> 64-1.0-blind) rather than a lookup.
    blind_tag = "-blind" if args.blind else ""
    save_dir = f"results/{args.model}/streamingbench_{subset}/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}{blind_tag}"
    solver = "blind_vqa" if args.blind else "rekv_streamingbench_vqa"
    anno_path = f"data/StreamingBench/{subset}.json"
    # SQA's five questions are one conversation: the reference runner prepends the earlier
    # questions and their gold answers. Default it on for that subset alone -- running SQA
    # without it measures a harder, different task -- and let --no_sqa_context say so
    # explicitly rather than by omission.
    # `and not args.blind`: blind_vqa does not register --sqa_context, so passing it
    # would abort the worker on an unrecognised flag. Unreachable while sqa is kept out
    # of BLIND_DATASETS, but the two guards are in different files -- this one keeps the
    # command construction correct on its own.
    sqa_context = (subset == 'sqa') and not args.no_sqa_context and not args.blind
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + (["--sqa_context"] if sqa_context else []) \
                    + reduction_args(args) + stream_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval: StreamingBench's headline is a micro-average over questions, OVO-Bench's is an
    # unweighted mean of per-task accuracies, so they cannot share a scorer.
    score(args, f"python video_qa/eval/eval_streamingbench.py --save_dir {save_dir}")


def eval_streamingbench_real(args):
    eval_streamingbench(args, 'real')


def eval_streamingbench_omni(args):
    eval_streamingbench(args, 'omni')


def eval_streamingbench_context(args):
    eval_streamingbench(args, 'context')


def eval_streamingbench_sqa(args):
    eval_streamingbench(args, 'sqa')


def eval_ovobench_realtime(args):
    eval_ovobench(args, 'realtime')

def eval_ovobench_backward(args):
    eval_ovobench(args, 'backward')

def eval_fpsbench_stream(args):
    """FPS-Bench-Stream: the FPSBench clip hidden in a 600 s haystack.

    Each of 990 built streams is one FPSBench question whose evidence occupies a median
    1.5% of the video (video_qa/rekv_fpsbench_stream_vqa.py). `--trigger` picks which
    question the run asks:

    * `query` (default) asks at query_time_sec, with the needle still the newest thing in
      the cache -- realtime perception, measuring what the model makes of what it just saw;
    * `end` asks after all 600 s are ingested, a median 294 s after the evidence, so
      answering requires retrieving it out of a memory dominated by unrelated footage.
      That is the benchmark's published protocol and the retrieval arm.

    This release ships an answer key, so the run ends at a
    scorer -- accuracy broken down by needle position and by whether retrieval reached the
    needle at all -- rather than at a submission file. Build the annotation first with
    video_qa/convert_fpsbench_stream.py.
    """
    num_chunks = args.num_chunks
    # The tag names the arm, not the default: 'end' keeps the plain directory it has always
    # written to and 'query' keeps its -query suffix, so making 'query' the default did not
    # move anybody's existing results. It does mean the default arm is the tagged one.
    trigger_tag = "" if args.trigger == 'end' else f"-{args.trigger}"
    anno_path = args.anno_path or "data/fpsbench_stream/test_mc.json"
    # A subset run gets its own directory, named after its annotation file. Without this a
    # 20-stream smoke run would land on top of the full arm's results.csv and the two would
    # be indistinguishable afterwards.
    subset_tag = "" if args.anno_path is None \
        else f"-{os.path.splitext(os.path.basename(anno_path))[0]}"
    save_dir = (f"results/{args.model}/fpsbench_stream/{args.retrieve_size}-{args.sample_fps}"
                f"{trigger_tag}{subset_tag}{reduction_tag(args)}")
    solver = "rekv_fpsbench_stream_vqa"
    prompt_args = ["--max_new_tokens", str(args.max_new_tokens),
                   "--choice_seed", str(args.choice_seed),
                   "--trigger", args.trigger]
    if args.no_none_of_above:
        prompt_args.append("--no_none_of_above")
    if args.shuffle_choices:
        prompt_args.append("--shuffle_choices")
    # Latency flags. Recorded columns are unconditional; these two change what the number
    # measures, so they have to reach the worker.
    if args.force_answer_length:
        prompt_args.append("--force_answer_length")
    if args.retrieval_breakdown:
        prompt_args.append("--retrieval_breakdown")
    prompt_args += stream_args(args)
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args) + prompt_args
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # There is an answer key here, so this scores. The streaming audit is the same one the
    # short-clip arm runs -- both write the same no-lookahead columns.
    score(args, f"python video_qa/eval/eval_fpsbench_stream.py --save_dir {save_dir}")
    exec(f"python video_qa/eval/check_fpsbench_stream.py --save_dir {save_dir}")

def eval_cgbench(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/cgbench/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/cgbench/full_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval
    score(args, f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


def eval_odvbench(args):
    """ODV-Bench -- online driving VQA, questioned mid-stream.

    Streaming, like ovobench and unlike every other multiple-choice dataset here: each
    question carries an `end_time` and may only be answered from frames up to it, so the
    video is ingested incrementally and questioned in between. It shares OVO-Bench's
    solver outright (video_qa/rekv_ovobench_vqa.py) -- that file's contract is
    end_time/gt_index/question_id and nothing OVO-specific.

    The time limit is the benchmark. `end_time` sits at a median of ~0.45 of clip
    duration, and 62% of the questions ask what happens after it ("What will the position
    box of the pedestrian be", "Will there be significant traffic risks in the future"),
    so an offline pass answers them from the very frames they are asking the model to
    predict. eval_odvbench.py refuses to score a CSV that shows any sign of it.

    Two call-site consequences. The clips are short -- 5-90 s, median 33 s -- so the
    default sample_fps of 0.5 gives an early question one frame; run this at 2 or more.
    And no clip comes near n_local, so retrieval never fires: this measures the reduction
    stages' effect on perception, not on retrieval.

    `--blind` swaps in video_qa/blind_vqa.py, which answers every question with no video
    at all. Worth running once before trusting any number here: 3234 of the 6348 questions
    sit in subtasks whose majority answer is far above their own chance line (all 123
    Hallucination-detection answers are "Unable to say."), so the language-prior floor is
    high and uneven across subtasks.

    Run scripts/setup_odvbench.py first to produce the annotation file.
    """
    num_chunks = args.num_chunks
    # The '-blind' suffix keeps the control in its own directory, so it can never
    # overwrite the sighted run it exists to be compared against. sample_fps stays in the
    # path even though a blind run has no frames: it makes the pairing a plain suffix
    # (64-2.0 <-> 64-2.0-blind) rather than a lookup.
    blind_tag = "-blind" if args.blind else ""
    save_dir = f"results/{args.model}/odvbench/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}{blind_tag}"
    solver = "blind_vqa" if args.blind else "rekv_ovobench_vqa"
    anno_path = args.anno_path or "data/odvbench/full_mc.json"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args) + stream_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval: the pooled mean in eval_multiple_choice.py would be dominated by Distance
    # Prediction (1488 of 6348) and, more importantly, would not check that the streaming
    # time limit was honoured. eval_odvbench.py does both.
    score(args, f"python video_qa/eval/eval_odvbench.py --save_dir {save_dir}")


def eval_ovbench(args):
    """OVBench -- online video understanding, questioned mid-stream.

    Streaming, like ovobench and odvbench: each question carries a
    `middle_frame_timestamp` and may only be answered from frames up to it, so the video
    is ingested incrementally and questioned in between. It shares OVO-Bench's solver
    outright (video_qa/rekv_ovobench_vqa.py) -- that file's contract is
    end_time/gt_index/question_id/choices and nothing OVO-specific, which
    video_qa/convert_ovbench.py emits directly.

    The time limit is the benchmark. 2797 of the 7090 questions -- all of Past Memory and
    Future Prediction -- ask about something that is not on screen at the timestamp, so an
    offline pass answers Future Prediction from the very frames it is meant to predict.
    eval_ovbench.py refuses to score a CSV that shows any sign of it.

    Two call-site consequences. This is the largest streaming set here by far: 1463 videos
    and 78 hours of ingestion at 1 fps, against odvbench's 6348 questions over short
    clips, so budget accordingly and use --num_chunks. And the sources are heterogeneous
    in a way the others are not -- the three container sources are full-length videos
    where a timestamp can sit 1000 s in (AVA_RAW), while the seven transcoded sources are
    tens of seconds long. sample_fps therefore trades very differently across them; 1 fps
    matches the OVO-Bench and StreamingBench runs and is the value to keep for
    comparability.

    `--blind` swaps in video_qa/blind_vqa.py, which answers every question with no video
    at all. Worth running once before trusting any number here: 1774 questions are binary
    and they concentrate in Temporal Hallucination Verification, so the language-prior
    floor is high and very uneven across sub-tasks (~31.3% pooled chance, but far higher
    for that group).

    Run scripts/prepare_ovbench.sh first to unpack the videos and write the annotation.
    """
    num_chunks = args.num_chunks
    # The '-blind' suffix keeps the control in its own directory, so it can never
    # overwrite the sighted run it exists to be compared against.
    blind_tag = "-blind" if args.blind else ""
    save_dir = f"results/{args.model}/ovbench/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}{blind_tag}"
    solver = "blind_vqa" if args.blind else "rekv_ovobench_vqa"
    anno_path = args.anno_path or "data/OVBench/full_mc.json"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args) + stream_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval: the pooled mean in eval_multiple_choice.py would be dominated by Procedure
    # Recall / Step Verification (2141 of 7090) and, more importantly, would not check
    # that the streaming time limit was honoured. eval_ovbench.py does both.
    score(args, f"python video_qa/eval/eval_ovbench.py --save_dir {save_dir}")


def eval_streambench(args):
    """StreamBench -- streaming video QA with free-form answers, scored by an LLM judge.

    The only benchmark here that is both streaming and open-ended. Each breakpoint carries
    a `time` past which the model may not see (like ovobench/odvbench/ovbench), but the
    references are sentences, so there is no letter to match and an LLM decides whether the
    prediction means the same thing (like rvs_ego/rvs_movie). Solver:
    video_qa/rekv_streambench_vqa.py, a thin subclass of the rvs_* streaming solver that
    adds the question class and the leak-audit columns.

    THE JUDGE IS PINNED TO UPSTREAM'S. StreamChat scores StreamBench with
    eval_ego_streaming_with_llama3.py and meta-llama/Meta-Llama-3-8B-Instruct, and
    judges.StreamBenchLlamaJudge reproduces that prompt byte-for-byte. Judge-to-judge gaps
    dominate the noise in open-ended numbers, so `--judge_preset` defaults to 'streambench'
    here rather than to the repo-wide prometheus default; pass it explicitly to override,
    and expect the result not to be comparable with published StreamBench figures.

    Two things to know when reading the output. KG (Knowledge-based QA) is 298 of the 1838
    questions and is pure world knowledge -- answerable with the video off -- so it lifts
    any pooled figure for reasons unrelated to streaming; eval_streambench.py prints a
    macro-average excluding it. And the six classes are near-evenly balanced by design, so
    the macro-average is the honest headline.

    `--blind` swaps in video_qa/blind_vqa.py. Worth running once: it should leave KG almost
    unchanged while the memory and search classes collapse, and a class that does not
    collapse is one the video was not contributing to.

    Run scripts/prepare_streambench.sh first to write the annotation.
    """
    num_chunks = args.num_chunks
    blind_tag = "-blind" if args.blind else ""
    save_dir = f"results/{args.model}/streambench/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}{blind_tag}"
    solver = "blind_vqa" if args.blind else "rekv_streambench_vqa"
    anno_path = args.anno_path or "data/streambench/full_oe.json"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)] + reduction_args(args) + stream_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        merge_chunks(save_dir, num_chunks)
    # eval, in two steps. First the judge, which grades every row -- that is the only way
    # a free-form answer gets a verdict at all. Then the breakdown, which rejoins those
    # verdicts to the class/source columns and checks the streaming time limit was
    # honoured; neither is something the judge's pooled accuracy can tell you.
    score(args, open_ended_cmd(args, save_dir))
    score(args, f"python video_qa/eval/eval_streambench.py --save_dir {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'video_llava_7b', 'longva_7b'])
    parser.add_argument("--dataset", type=str, default=None, choices=['mlvu', 'mlvu_test', 'qaego4d', 'egoschema', 'activitynet_qa', 'rvs_ego', 'rvs_movie', 'cgbench', 'odvbench', 'ovbench', 'streambench', 'ovobench_realtime', 'ovobench_backward', 'streamingbench_real', 'streamingbench_omni', 'streamingbench_context', 'streamingbench_sqa', 'fpsbench_stream'])
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--blind", action="store_true",
                        help="Blind control: answer every question with no video at all "
                             "(video_qa/blind_vqa.py). Produces the language-prior floor "
                             "a real score has to be read against. Results land in a "
                             "'-blind' directory of their own.")
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--skip_scoring", action="store_true",
                        help="Stop after writing results.csv. For rvs_*/activitynet_qa "
                             "the scorer is an LLM judge; skip it to generate predictions "
                             "now and score later off the same CSV.")
    parser.add_argument("--judge", type=str, default='local', choices=['local', 'openai'],
                        help="Judge for free-form-answer datasets (qaego4d, activitynet_qa, "
                             "rvs_*). Default 'local' runs on your own GPU, needs no API key, "
                             "and writes to results_local<judge>.json (--judge_preset picks "
                             "which); its scores are self-consistent across runs but not "
                             "comparable to published gpt-3.5-turbo ones. Pass --judge openai "
                             "to reproduce those instead (needs OPENAI_API_KEY).")
    parser.add_argument("--judge_preset", type=str, default=judges.DEFAULT_PRESET,
                        help="--judge local only: which local judge to use. Presets: "
                             f"{', '.join(judges.PRESETS)} (or any HF id / local path). "
                             "Each writes to its own results_local*.json, so judges never "
                             "overwrite each other -- but their numbers are not "
                             "interchangeable either: score a whole sweep with one judge.")
    parser.add_argument("--judge_style", type=str, default=None,
                        choices=sorted(judges.STYLES) + ['auto'],
                        help="--judge local only: override the preset's prompt/parser pair "
                             "(see video_qa/eval/judges.py). Rarely needed -- the preset and "
                             "'auto' already pick the right one for known checkpoints.")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="--judge local only: override the preset's checkpoint, keeping "
                             "its prompt style and output paths. For a fine-tune or a local "
                             "copy of the same judge.")
    parser.add_argument("--judge_yes_threshold", type=int, default=4,
                        help="--judge local only, prometheus style only: lowest rubric score "
                             "(1-5) counted as correct.")
    parser.add_argument("--judge_batch_size", type=int, default=16,
                        help="--judge local only: prompts per forward pass.")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--debug", type=str, default='false')
    # FPSBench prompt options, forwarded to video_qa/rekv_fpsbench_stream_vqa.py. Ignored
    # by every other dataset, whose prompts are ReKV's own.
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--no_none_of_above", action="store_true")
    parser.add_argument("--shuffle_choices", action="store_true")
    parser.add_argument("--choice_seed", type=int, default=2024)
    parser.add_argument("--trigger", type=str, default='query', choices=['end', 'query'],
                        help="fpsbench_stream only: when each question fires. 'query' "
                             "(default) asks at query_time_sec, where the needle is still "
                             "the newest thing in the cache -- realtime perception, what "
                             "the model can answer about what it just saw. 'end' asks "
                             "only after the whole 600 s stream has been ingested, so the "
                             "needle must be retrieved back out of memory; that is the "
                             "benchmark's published protocol and the retrieval arm.")
    parser.add_argument("--force_answer_length", action="store_true",
                        help="fpsbench_stream: decode exactly "
                             "--max_new_tokens tokens per question, so QA latency is "
                             "measured over a fixed decode length. Answer length is a "
                             "dependent variable of anything that perturbs the KV-Cache, "
                             "so a latency comparison without this partly measures how "
                             "much each arm chose to say. Changes the answers.")
    parser.add_argument("--retrieval_breakdown", action="store_true",
                        help="fpsbench_stream: split retrieval out "
                             "of QA latency (retrieval_seconds / generation_seconds). "
                             "Costs a CUDA sync per question, which inflates "
                             "latency_seconds, so take the headline latency from a run "
                             "without it.")
    parser.add_argument("--decode_window", type=int, default=256,
                        help="Streaming datasets (fpsbench_stream, ovobench_*, "
                             "streamingbench_*, odvbench): "
                             "slots decoded at a time, 0 = the whole video up front. "
                             "Frames are held at source resolution, so a whole-video decode "
                             "is sample_fps x seconds x MB-per-frame per worker -- fine on "
                             "a short clip, 85 GB per worker for a 600 s stream at 32 fps. "
                             "Windowing decodes each block exactly once, so it changes "
                             "residency, not the frames or the results.")
    parser.add_argument("--no_sqa_context", action="store_true",
                        help="streamingbench_sqa only: ask its five questions "
                             "independently instead of prepending the earlier ones and "
                             "their gold answers, as src/benchmark/StreamingBenchSQA.py "
                             "does. That is a harder, different task -- the questions are "
                             "written as a conversation -- so its numbers are not "
                             "comparable to published Sequential-QA scores.")
    parser.add_argument("--anno_path", type=str, default=None,
                        help="fpsbench_stream: annotation file to "
                             "run against, overriding the dataset default. For a subset "
                             "built with convert_fpsbench_stream.py --limit/--position_bin, "
                             "or for the keyed FPSBench file that local scoring needs.")
    # Both reduction stages (llava_ov_* only), forwarded verbatim to the workers. Every
    # default is off, so a command line without them is the untouched baseline.
    add_reduction_args(parser)
    args = parser.parse_args()

    # Datasets whose eval function knows how to swap in the blind solver. Anything else
    # would silently run sighted and write to a '-blind' directory, which is the one
    # outcome worse than an error.
    # Split the CPU allocation across the workers about to be spawned. Each worker
    # inherits this through os.environ (exec copies it), and video_qa/base.py sizes
    # decord's decode pool from it. Without the division four workers would each open a
    # pool sized to the *whole* allocation and oversubscribe it fourfold, which is the
    # condition that makes decord's threaded decoder time out mid-run. setdefault so an
    # explicit REKV_DECORD_THREADS in the environment still wins.
    try:
        _cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        _cpus = os.cpu_count() or 1
    os.environ.setdefault('REKV_DECORD_THREADS',
                          str(max(1, _cpus // max(1, args.num_chunks))))

    # streamingbench_sqa is deliberately absent: its sighted run prepends the earlier
    # questions and their gold answers (StreamingBenchSQA.py), and blind_vqa does not,
    # so a blind SQA number would not be the floor for the run it sits next to.
    BLIND_DATASETS = {'odvbench', 'ovbench', 'streambench',
                      'ovobench_realtime', 'ovobench_backward',
                      'streamingbench_real', 'streamingbench_omni',
                      'streamingbench_context'}
    if args.blind:
        if args.dataset not in BLIND_DATASETS:
            parser.error(f"--blind is not wired up for {args.dataset!r}; "
                         f"supported: {sorted(BLIND_DATASETS)}")
        # Both stages decide what to drop by comparing a frame against the one before it,
        # so with no frames there is nothing to reduce -- but the reduction tag would
        # still land in the results path, inventing distinct 'blind at threshold 0.2' and
        # 'blind at threshold 0.4' directories holding identical runs. Refuse rather than
        # produce them.
        if vision_on(args) or pruning_on(args):
            parser.error("--blind ingests no frames, so the reduction stages have nothing "
                         "to act on; drop the --vision_*/--prune_* flags.")

    # StreamBench publishes its numbers under a specific judge (Meta-Llama-3-8B-Instruct,
    # see judges.StreamBenchLlamaJudge). Default to it so a run is comparable out of the
    # box, while leaving an explicit --judge_preset in charge.
    if args.dataset == 'streambench' and '--judge_preset' not in sys.argv:
        args.judge_preset = 'streambench'

    func_dic = {
        'mlvu': eval_mlvu,
        'mlvu_test': eval_mlvu_test,
        'qaego4d': eval_qaego4d,
        'egoschema': eval_egoschema,
        'activitynet_qa': eval_activitynet_qa,
        'rvs_ego': eval_rvs_ego,
        'rvs_movie': eval_rvs_movie,
        'cgbench': eval_cgbench,
        'odvbench': eval_odvbench,
        'ovbench': eval_ovbench,
        'streambench': eval_streambench,
        'ovobench_realtime': eval_ovobench_realtime,
        'ovobench_backward': eval_ovobench_backward,
        'streamingbench_real': eval_streamingbench_real,
        'streamingbench_omni': eval_streamingbench_omni,
        'streamingbench_context': eval_streamingbench_context,
        'streamingbench_sqa': eval_streamingbench_sqa,
        'fpsbench_stream': eval_fpsbench_stream,
    }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
