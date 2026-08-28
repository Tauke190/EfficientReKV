import os
import argparse
import subprocess
import multiprocessing

from video_qa.reduction_args import add_reduction_args
from video_qa.eval import judges


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
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
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
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
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
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
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
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
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
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
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
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
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
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
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
    """
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/ovobench_{mode}/{args.retrieve_size}-{args.sample_fps}{reduction_tag(args)}"
    solver = "rekv_ovobench_vqa"
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
                    "--chunk_idx", str(idx)] + reduction_args(args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval: OVO-Bench averages per-task accuracies within a mode, so it needs its own
    # scorer rather than eval_multiple_choice.py's pooled mean.
    score(args, f"python video_qa/eval/eval_ovobench.py --save_dir {save_dir}")


def eval_ovobench_realtime(args):
    eval_ovobench(args, 'realtime')


def eval_ovobench_backward(args):
    eval_ovobench(args, 'backward')


def eval_fpsbench_stream(args):
    """FPS-Bench-Stream: the FPSBench clip hidden in a 600 s haystack, asked at the end.

    The retrieval arm. Each of 990 built streams is one FPSBench question whose evidence
    occupies a median 1.5% of the video and sits a median 294 s before the question, so
    answering requires ReKV to retrieve it out of a memory dominated by unrelated footage
    (video_qa/rekv_fpsbench_stream_vqa.py). `--trigger query` is the control: the same
    frames, the same ingestion, but asked when the needle is still the newest thing in the
    cache.

    Unlike `fpsbench_stream_small`, this release ships an answer key, so the run ends at a
    scorer -- accuracy broken down by needle position and by whether retrieval reached the
    needle at all -- rather than at a submission file. Build the annotation first with
    video_qa/convert_fpsbench_stream.py.
    """
    num_chunks = args.num_chunks
    # Only the control arm is tagged: the end trigger is this benchmark's protocol, so it
    # keeps the plain directory name.
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
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # There is an answer key here, so this scores. The streaming audit is the same one the
    # short-clip arm runs -- both write the same no-lookahead columns.
    score(args, f"python video_qa/eval/eval_fpsbench_stream.py --save_dir {save_dir}")
    exec(f"python video_qa/eval/check_fpsbench_stream.py --save_dir {save_dir}")


def eval_fpsbench_stream_small(args):
    """FPSBench's own 2-25 s clips, streamed, questioned at certificate time.

    "small" is the clip length, not the question count: this is the released FPSBench,
    996 short clips, one question each. The long-video arm built on the same questions is
    `fpsbench_stream` below, where each clip is spliced into a 600 s haystack and the
    question is asked at the end -- that one tests retrieval over a long memory, this one
    tests whether fast motion was resolved as it went past. They share the prompt, the
    exact-fps grid and the audit columns, and nothing else.

    Frames arrive one per forward pass and each question fires at the end of its temporal
    certificate (video_qa/rekv_fpsbench_stream_small_vqa.py). The offline arm was removed:
    ReKV is a streaming model, so answering after the whole clip has been ingested measures
    its ingestion path rather than whether it resolved the motion in time.

    `--sample_fps` is exact here by construction, so there is no `--exact_fps` flag and no
    '-exactfps' directory suffix: the stride path cannot deliver a controlled frame rate on
    clips recorded at 23.98/25/29.97/30 fps.
    """
    num_chunks = args.num_chunks
    trigger_tag = "-fullclip" if args.full_clip else ""
    # A separate directory rather than a column, because an MBA run and a multiple-choice
    # run of the same arm have different row counts and different metrics; merging them
    # into one path would make `results.csv` mean two things.
    mba_tag = "-mba" if args.mba else ""
    save_dir = (f"results/{args.model}/fpsbench_stream_small/{args.retrieve_size}-{args.sample_fps}"
                f"{trigger_tag}{mba_tag}{reduction_tag(args)}")
    solver = "rekv_fpsbench_stream_small_vqa"
    # MBA needs the answer key; multiple choice does not and defaults to the question-only
    # file, which is what keeps the key out of the ordinary path.
    default_anno = ("data/fpsbench/test_mc_keyed.json" if args.mba
                    else "data/fpsbench/test_mc.json")
    anno_path = args.anno_path or default_anno
    prompt_args = ["--max_new_tokens", str(args.max_new_tokens),
                   "--choice_seed", str(args.choice_seed)]
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
    if args.full_clip:
        prompt_args.append("--full_clip")
    if args.mba:
        prompt_args += ["--mba", "--mba_scoring", args.mba_scoring,
                        "--mba_max_new_tokens", str(args.mba_max_new_tokens)]
        if args.mba_include_none:
            prompt_args.append("--mba_include_none")
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
        # Every chunk must have produced its file before anything is merged. The merge is
        # shell redirection with no error checking: a chunk that died leaves `head -n 1`
        # with nothing to read, so `results.csv` ends up with no header row and only the
        # surviving chunks' data -- a file that still loads, still has plausible row
        # counts, and is silently missing a quarter of the benchmark. That is a worse
        # outcome than a crash, so fail here and say which chunks are gone.
        missing = [idx for idx in range(num_chunks)
                   if not os.path.exists(f"{save_dir}/{num_chunks}_{idx}.csv")]
        if missing:
            raise RuntimeError(
                f"{save_dir}: chunks {missing} produced no output -- refusing to merge a "
                f"partial run into results.csv. Check the log above for their traceback; "
                f"the usual cause is num_chunks ({num_chunks}) exceeding the number of "
                f"GPUs the job actually got, which makes the surplus workers fail with "
                f"'No CUDA GPUs are available'.")
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # An MBA run carries the key, so it scores locally. A multiple-choice run against the
    # question-only release does not: it ends at the submission JSONL, and the exporter
    # would in any case make no sense of K binary rows per question.
    if args.mba:
        exec(f"python video_qa/eval/eval_fpsbench_mba.py --save_dir {save_dir}")
    else:
        exec(f"python video_qa/eval/export_fpsbench.py --save_dir {save_dir}")
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
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    score(args, f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'video_llava_7b', 'longva_7b'])
    parser.add_argument("--dataset", type=str, default=None, choices=['mlvu', 'mlvu_test', 'qaego4d', 'egoschema', 'activitynet_qa', 'rvs_ego', 'rvs_movie', 'cgbench', 'ovobench_realtime', 'ovobench_backward', 'fpsbench_stream', 'fpsbench_stream_small'])
    parser.add_argument("--num_chunks", type=int, default=1)
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
    # FPSBench prompt options, forwarded to video_qa/rekv_fpsbench_stream_small_vqa.py. Ignored
    # by every other dataset, whose prompts are ReKV's own.
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--no_none_of_above", action="store_true")
    parser.add_argument("--shuffle_choices", action="store_true")
    parser.add_argument("--choice_seed", type=int, default=2024)
    parser.add_argument("--trigger", type=str, default='end', choices=['end', 'query'],
                        help="fpsbench_stream only: when each question fires. 'end' "
                             "(default) asks after the whole 600 s stream has been "
                             "ingested, so the needle must be retrieved; 'query' asks at "
                             "query_time_sec, where it is still the newest thing in the "
                             "cache. The control arm for how much of any gap is retrieval.")
    parser.add_argument("--force_answer_length", action="store_true",
                        help="fpsbench_stream / fpsbench_stream_small: decode exactly "
                             "--max_new_tokens tokens per question, so QA latency is "
                             "measured over a fixed decode length. Answer length is a "
                             "dependent variable of anything that perturbs the KV-Cache, "
                             "so a latency comparison without this partly measures how "
                             "much each arm chose to say. Changes the answers.")
    parser.add_argument("--retrieval_breakdown", action="store_true",
                        help="fpsbench_stream / fpsbench_stream_small: split retrieval out "
                             "of QA latency (retrieval_seconds / generation_seconds). "
                             "Syncs CUDA once per layer per question and inflates "
                             "latency_seconds, so take the headline latency from a run "
                             "without it.")
    parser.add_argument("--anno_path", type=str, default=None,
                        help="fpsbench_stream / fpsbench_stream_small: annotation file to "
                             "run against, overriding the dataset default. For a subset "
                             "built with convert_fpsbench_stream.py --limit/--position_bin, "
                             "or for the keyed FPSBench file that local scoring needs.")
    parser.add_argument("--mba", action="store_true",
                        help="fpsbench_stream_small only: ask each question as K "
                             "independent Yes/No binaries and score Multiple Binary "
                             "Accuracy (TemporalBench, arXiv 2410.10818) instead of "
                             "five-way multiple choice. Drops the chance floor from 0.200 "
                             "to 1/2**K and sends constant-answer strategies to 0, which "
                             "is what separates a model that understands the video from "
                             "one riding the floor. Implies the keyed annotation file "
                             "(data/fpsbench/test_mc_keyed.json) unless --anno_path says "
                             "otherwise, and ends at video_qa/eval/eval_fpsbench_mba.py "
                             "rather than at the submission exporter.")
    parser.add_argument("--mba_include_none", action="store_true",
                        help="--mba only: keep 'None of the above' among the candidates. "
                             "Off by default -- FPSBench's key never selects it, so its "
                             "binary is 'No' on every question.")
    parser.add_argument("--mba_scoring", type=str, default="logit",
                        choices=["logit", "generate"],
                        help="--mba only: 'logit' decides from the Yes/No logits in one "
                             "forward pass (default); 'generate' decodes and parses text.")
    parser.add_argument("--mba_max_new_tokens", type=int, default=8,
                        help="--mba_scoring generate only: decode budget per binary.")
    parser.add_argument("--full_clip", action="store_true",
                        help="fpsbench_stream_small only: trigger each question at the end of "
                             "the clip instead of the end of its temporal certificate. "
                             "The control arm for 'does answering early cost anything'.")
    # Both reduction stages (llava_ov_* only), forwarded verbatim to the workers. Every
    # default is off, so a command line without them is the untouched baseline.
    add_reduction_args(parser)
    args = parser.parse_args()
    func_dic = {
        'mlvu': eval_mlvu,
        'mlvu_test': eval_mlvu_test,
        'qaego4d': eval_qaego4d,
        'egoschema': eval_egoschema,
        'activitynet_qa': eval_activitynet_qa,
        'rvs_ego': eval_rvs_ego,
        'rvs_movie': eval_rvs_movie,
        'cgbench': eval_cgbench,
        'ovobench_realtime': eval_ovobench_realtime,
        'ovobench_backward': eval_ovobench_backward,
        'fpsbench_stream': eval_fpsbench_stream,
        'fpsbench_stream_small': eval_fpsbench_stream_small,
    }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
