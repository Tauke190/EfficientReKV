"""Open-ended (free-form answer) scoring with a *local* HF judge instead of the API.

Drop-in replacement for eval_open_ended.py: same CLI (--pred_path/--output_dir/
--output_json), same per-item cache files, same combined results.json shape, same
printed Accuracy / Average score. Only the thing producing the yes-no+score verdict
differs -- a local instruct model rather than an OpenAI endpoint.

The judge itself is swappable. Which prompt is sent and how the reply is parsed live in
`video_qa/eval/judges.py`, one class per judge family; this file only drives whichever
one `--judge_style` / `--judge_model` select. Two are available:

  * **prometheus** (default) -- Prometheus 2, an open model *trained to be an evaluator*.
    Graded against an explicit 1-5 rubric; yes/no is derived by thresholding it.
  * **qwen** -- the previous default. A general instruct model asked to emit the same
    `{'pred': ..., 'score': ...}` dict the OpenAI scorer asks for, prompt byte-identical
    to eval_open_ended.py's. Bring it back with `--judge_style qwen`, or
    `JUDGE=qwen scripts/score_open_ended.sh <csv>`.

Differences from the API version that follow from the judge being local:

  * **One process, batched.** The API version forks a 16-way Pool because latency is
    the bottleneck there. Here the weights are the bottleneck, so a single process
    holds one copy and batches prompts through it. Do not wrap this in a Pool.
  * **Greedy decoding.** do_sample=False, so re-running on the same CSV reproduces the
    same scores. The API judge at temperature=0 is only approximately reproducible.
  * **Tolerant parsing.** GPT-3.5 reliably emits the exact dict the prompt asks for;
    local models sometimes wrap it in prose or markdown. Parsing therefore falls back to
    regex, and items that still fail are counted and reported rather than silently
    dropped -- an unparsed verdict is missing data, not a "no".

Scores from a local judge are NOT comparable to published numbers produced by
gpt-3.5-turbo-0613, nor across judges (see judges.py). They are comparable across runs
scored by the same judge, which is what a pruning sweep actually needs. The judge id is
written into every cache file and this run aborts rather than mixing verdicts from two
different judges in one directory.

Usage:
    python video_qa/eval/eval_open_ended_local.py \
        --pred_path   results/llava_ov_0.5b/rvs_ego/64-0.5/results.csv \
        --output_dir  results/llava_ov_0.5b/rvs_ego/64-0.5/tmp_prometheus \
        --output_json results/llava_ov_0.5b/rvs_ego/64-0.5/results_local_prometheus.json
"""

import os
import sys
import json
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# This file is invoked as a script (scripts/score_open_ended.sh), so sys.path[0] is
# video_qa/eval/ and the repo root is not importable. Add it rather than relying on the
# caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from video_qa import answer_types  # noqa: E402
from video_qa.eval import judges  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Open-ended QA scoring with a local LLM judge.")
    parser.add_argument("--pred_path", required=True, help="results.csv written by the eval run.")
    parser.add_argument("--output_dir", required=True, help="Per-item verdict cache. Reused on re-run.")
    parser.add_argument("--output_json", required=True, help="Combined verdicts + final metrics.")
    parser.add_argument("--judge_style", default='auto',
                        choices=['auto'] + sorted(judges.STYLES),
                        help="Which prompt/parser pair to use. 'auto' reads it off "
                             "--judge_model when that names a known family, else falls "
                             f"back to {judges.DEFAULT_STYLE!r} (or the generic dict "
                             "prompt for an unrecognised checkpoint).")
    parser.add_argument("--judge_model", default=None,
                        help="HF id or local path of the judge. Defaults to the style's "
                             "own checkpoint. Smaller judges are harsher on paraphrases, "
                             "which shifts absolute accuracy but applies the same shift "
                             "to every run scored with the same judge.")
    parser.add_argument("--yes_threshold", type=int, default=4,
                        help="--judge_style prometheus only: lowest rubric score counted "
                             "as correct. Prometheus emits 1-5 and no yes/no; 4 is the "
                             "first rubric level that describes a match.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Prompts per forward pass. Lower it if the judge OOMs.")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Defaults to the style's budget: enough for one short dict "
                             "(qwen) or for feedback followed by the score (prometheus). "
                             "Cutting prometheus short loses the [RESULT] tag and every "
                             "item comes back unparsed.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", default="auto",
                        help="Passed to from_pretrained. 'auto' shards across visible GPUs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N items. For smoke-testing the judge.")
    return parser.parse_args()


def build_prediction_set(pred_path, limit=None):
    """CSV rows -> {unique_key: qa_set}, keyed exactly as eval_open_ended.py keys them.

    Videos carry several questions each, so video_id alone is not unique; the API
    version disambiguates by appending an occurrence counter. Same scheme here, so the
    two scorers' cache directories are interchangeable.
    """
    pred_contents = pd.read_csv(pred_path).to_dict(orient="records")

    if "retrieve_size" in pred_contents[0]:  # NOTE: change this line if experiment with different hyper-parameters
        pred_contents = [x for x in pred_contents if x["retrieve_size"] == 64 and x["chunk_size"] == 1]

    video_id_counts = {}
    prediction_set = {}
    order = []
    for sample in pred_contents:
        video_id = sample["video_id"]
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0
        key = f"{video_id}_{video_id_counts[video_id]}"
        prediction_set[key] = {
            "question": str(sample["question"]),
            "answer": str(sample["answer"]),
            # A model that answered with an empty string writes NaN through pandas;
            # str() would make it the literal "nan" and the judge would score that as
            # a real (wrong) answer, which it is -- but keep it explicit.
            "pred_answer": "" if pd.isna(sample["pred_answer"]) else str(sample["pred_answer"]),
        }
        order.append(key)

    if limit is not None:
        order = order[:limit]
    return prediction_set, order


def check_cache_judge(output_dir, judge_id):
    """Refuse to add verdicts to a directory another judge already wrote into.

    Two judges' verdicts averaged together are not a number of anything. Caches written
    before judges were stamped carry no id and are accepted with a warning -- they are
    all qwen, since that was the only judge that existed.
    """
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith('.json'):
            continue
        try:
            with open(os.path.join(output_dir, fname)) as f:
                cached = json.load(f)[0]
        except (ValueError, OSError, IndexError):
            continue
        found = cached.get('judge')
        if found is None:
            print(f"WARNING: {output_dir} holds unstamped verdicts (pre-dating judge "
                  f"selection, i.e. {judges.QwenJudge.default_model}). Assuming they match "
                  f"{judge_id}; delete the directory to re-judge from scratch.")
        elif found != judge_id:
            sys.exit(f"{output_dir} was judged by {found!r}, this run is {judge_id!r}. "
                     f"Mixing verdicts from two judges gives a number that means nothing. "
                     f"Use a separate --output_dir/--output_json, or delete this one.")
        return


@torch.inference_mode()
def run_judge(model, tokenizer, judge, prediction_set, todo, args, prompt_of):
    """Score `todo` keys, writing one cache file per key as soon as it is decoded."""
    unparsed = 0
    max_new_tokens = args.max_new_tokens or judge.default_max_new_tokens
    for start in tqdm(range(0, len(todo), args.batch_size), desc="judging"):
        batch_keys = todo[start:start + args.batch_size]
        prompts = []
        for key in batch_keys:
            qa = prediction_set[key]
            prompts.append(prompt_of(judge.messages(
                qa["question"], qa["answer"], qa["pred_answer"])))

        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=4096).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Left padding means every sequence's continuation starts at the same offset.
        completions = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)

        for key, completion in zip(batch_keys, completions):
            verdict = judge.parse(completion)
            if verdict is None:
                unparsed += 1
                # Cache the raw text too, so a parsing bug can be fixed and replayed
                # without re-running the judge over the whole set.
                verdict = {"pred": None, "score": None, "raw": completion.strip()}
            verdict["judge"] = judge.id
            with open(os.path.join(args.output_dir, f"{key}.json"), "w") as f:
                json.dump([verdict, prediction_set[key]], f)
    return unparsed


def aggregate(output_dir, keys, output_json, judge):
    combined = {}
    for key in keys:
        path = os.path.join(output_dir, f"{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                combined[key] = json.load(f)

    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    missing = 0
    for result in combined.values():
        verdict = result[0]
        pred = verdict.get("pred")
        if pred is None or verdict.get("score") is None:
            missing += 1
            continue
        count += 1
        score_sum += int(verdict["score"])
        if "yes" in str(pred).lower():
            yes_count += 1
        elif "no" in str(pred).lower():
            no_count += 1

    average_score = score_sum / count if count else 0.0
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) else 0.0

    metrics = {
        "judge": judge.id,
        "judge_style": judge.name,
        "judge_model": judge.model,
        "score_range": judge.score_range,
        "num_items": len(combined),
        "num_scored": count,
        "num_unparsed": missing,
        "yes_count": yes_count,
        "no_count": no_count,
        "accuracy": accuracy,
        "average_score": average_score,
    }
    combined["metrics"] = metrics
    # Kept for parity with eval_open_ended.py, whose consumers read these two top-level
    # keys directly.
    combined["accuracy"] = accuracy
    combined["average_score"] = average_score
    with open(output_json, "w") as f:
        json.dump(combined, f)

    print(f"Judge: {judge.id}")
    print("Yes count:", yes_count)
    print("No count:", no_count)
    if missing:
        print(f"Unparsed verdicts (excluded from both metrics): {missing}/{len(combined) - 2}")
    print(f"Accuracy: {accuracy * 100:.1f}%")
    print(f"Average score: {average_score:.2f} (scale {judge.score_range})")

    # Pooled Accuracy above is the ReKV protocol figure and stays exactly as it was. The
    # breakdown is additive: on RVS, half of Ego (50.6%) and a third of Movie (35.7%) are
    # categories the benchmark itself marks (Y/N), which have a majority-class floor near
    # 54% however well the model sees the video -- so the single number is a weighted
    # average of a coin-flip and a real task. The labels come from the official release;
    # ReKV's data conversion drops them, which is why they have to be rejoined here.
    # No-ops on benchmarks that have no answer_type, and on any failure to obtain the
    # labels: a scorer must not stop reporting because an optional annex is unavailable.
    try:
        pairs, correct = [], []
        for key, result in combined.items():
            if key in ("metrics", "accuracy", "average_score"):
                continue
            verdict, meta = result[0], (result[1] if len(result) > 1 else {})
            if verdict.get("pred") is None:
                continue
            pairs.append((meta.get("question", ""), meta.get("answer", "")))
            correct.append("yes" in str(verdict["pred"]).lower())
        for line in answer_types.summarize(pairs, correct):
            print(line)
    except Exception as exc:  # noqa: BLE001
        print(f"(answer_type breakdown unavailable: {exc})")


def make_prompt_formatter(tokenizer, judge):
    """messages -> prompt string, working around templates that reject a system role.

    Mistral-derived templates (Prometheus 2 among them) raise on a system turn. Probe
    once here rather than per item, and fold the system text into the first user turn --
    which is what the official prometheus-eval client does anyway.
    """
    probe = judge.messages("q", "a", "p")

    def render(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        render(probe)
        return render
    except Exception:  # noqa: BLE001 -- templates raise whatever their jinja raises
        def render_merged(messages):
            merged, system = [], None
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                    continue
                if system is not None and msg["role"] == "user":
                    msg = {"role": "user", "content": f'{system}\n\n{msg["content"]}'}
                    system = None
                merged.append(msg)
            return render(merged)

        print(f"note: {judge.model}'s chat template rejects a system role; "
              f"folding the system prompt into the first user turn.")
        return render_merged


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    judge = judges.resolve(args.judge_style, args.judge_model, args.yes_threshold)
    check_cache_judge(args.output_dir, judge.id)

    prediction_set, order = build_prediction_set(args.pred_path, args.limit)
    done = {f[:-5] for f in os.listdir(args.output_dir) if f.endswith(".json")}
    todo = [k for k in order if k not in done]
    print(f"judge: {judge.id}")
    print(f"{len(order)} items, {len(order) - len(todo)} already cached, {len(todo)} to judge")

    if todo:
        if not torch.cuda.is_available():
            # Not fatal, but a 7B on CPU turns a 10-minute job into an overnight one --
            # say so before the download rather than after.
            print("WARNING: no CUDA device visible; the judge will run on CPU and be very slow.")
        tokenizer = AutoTokenizer.from_pretrained(judge.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Decoder-only batched generation requires left padding; with right padding the
        # shorter prompts get pad tokens between the prompt and the continuation.
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            judge.model,
            torch_dtype=getattr(torch, args.dtype),
            device_map=args.device_map,
        )
        model.eval()
        # Instruct checkpoints ship sampling defaults (temperature/top_p/top_k) in
        # generation_config, which warn on every greedy call. Clear them so the config
        # matches how the judge is actually run.
        for field in ("temperature", "top_p", "top_k"):
            if getattr(model.generation_config, field, None) is not None:
                setattr(model.generation_config, field, None)

        prompt_of = make_prompt_formatter(tokenizer, judge)
        unparsed = run_judge(model, tokenizer, judge, prediction_set, todo, args, prompt_of)
        if unparsed:
            print(f"{unparsed}/{len(todo)} completions did not parse; raw text kept in {args.output_dir}")

    aggregate(args.output_dir, order, args.output_json, judge)


if __name__ == "__main__":
    main()
