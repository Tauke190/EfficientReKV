import time
import os
import sys
import argparse
import json
import ast
import pandas as pd
from multiprocessing.pool import Pool
from tqdm import tqdm
import openai

# Invoked as a script, so sys.path[0] is video_qa/eval/ and the repo root is not
# importable. Add it rather than relying on the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from video_qa import answer_types  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--num_tasks", default=16, type=int, help="Number of splits.")
    parser.add_argument("--max_workers", default=8, type=int,
                        help="Concurrent API callers. num_tasks only splits the work list; "
                             "this is what actually bounds request rate.")
    args = parser.parse_args()
    return args

def shrink_string_correctly(text):
    # Split the text into sentences for better analysis
    parts = text.split(" ")
    output = []

    for part in parts:
        # Check if the sentence (or part) has been seen before
        if part not in output:
            output.append(part)
        elif part == output[-1]:
            # Stop adding once a duplicate is found since the example suggests stopping at the first repeat occurrence
            break

    # Reconstruct the string, taking into account the removal of duplicates
    return " ".join(output)

# The paper scored with gpt-3.5-turbo-0613. That snapshot now 404s ("has been
# deprecated"); only the floating gpt-3.5-turbo alias survives, and it currently resolves
# to -0125. We pin -0125 explicitly rather than riding the alias, because the alias is
# free to move to another snapshot and would silently change every score. This is the
# same model family as the paper but NOT the same weights, so numbers produced here are
# self-consistent across runs and still not directly comparable to the published ones.
DEFAULT_JUDGE = "gpt-3.5-turbo-0125"


class GPTService:
    def __init__(self):
        self.model_name = os.environ.get("OPENAI_JUDGE_MODEL", DEFAULT_JUDGE)
        self.max_tokens = 300
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is unset; the open-ended judge cannot run. "
                "Optionally set OPENAI_BASE_URL to use a non-official endpoint."
            )
        self.client = openai.OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            api_key=api_key,
        )

    def _gpt_response(self, user_prompt):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=user_prompt,
            max_tokens=self.max_tokens,
            temperature=0,
        )
        response = json.loads(completion.model_dump_json())
        return response['choices'][0]['message']['content']

    def gpt_with_retry(self, prompt):
        """Retry with exponential backoff, honouring Retry-After on rate limits.

        The original did 10 attempts spaced a flat 1s apart. Under a 429 that burns
        every attempt inside the rate-limit window, so the item fails, gets requeued by
        the loop in main(), and the whole run degrades to a crawl -- the rvs_ego run of
        2026-08-07 did 1297/1465 items in 3.5 minutes and then took two hours for the
        remaining 168. Backing off properly keeps the retries inside one call.
        """
        retry = 10
        for attempt in range(retry):
            try:
                result = self._gpt_response(prompt)
                if result is not None:
                    return result
                delay = min(2 ** attempt, 60)
            except openai.RateLimitError as e:
                # Prefer the server's own hint when it sends one.
                delay = min(2 ** attempt, 60)
                retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
                try:
                    delay = max(delay, float(retry_after.get("retry-after", 0)))
                except (TypeError, ValueError):
                    pass
                print(f"rate limited (attempt {attempt + 1}/{retry}); sleeping {delay:.0f}s")
            except Exception as e:
                print(f"An error occurred: {e}")
                delay = min(2 ** attempt, 60)
            time.sleep(delay)
        return None

def claim_cache_dir(output_dir, judge_model):
    """Bind a verdict cache to one judge, and refuse to let a second judge reuse it.

    Both scorers resume by skipping any item already present in output_dir. That makes
    a re-run with a *different* judge silently a no-op: every item is 'done', nothing is
    re-scored, and the final metrics get stamped with the new judge's name over the old
    judge's verdicts. This has already produced wrong numbers in this repo more than
    once. Cheap marker file, permanent fix.
    """
    marker = os.path.join(output_dir, ".judge_model")
    existing = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    if os.path.exists(marker):
        with open(marker) as f:
            prev = f.read().strip()
        if prev and prev != judge_model:
            raise SystemExit(
                f"\nREFUSING TO RUN: {output_dir} holds {len(existing)} verdicts from "
                f"{prev}, but this run uses {judge_model}.\nCached items are reused "
                f"verbatim, so continuing would report {prev}'s scores under "
                f"{judge_model}'s name.\nMove the directory aside, or point --output_dir "
                f"somewhere else, then rerun.\n"
            )
    elif existing:
        # Predates this check, so its provenance is unknown and cannot be assumed.
        raise SystemExit(
            f"\nREFUSING TO RUN: {output_dir} holds {len(existing)} verdicts with no "
            f"record of which judge produced them.\nReusing them under {judge_model}'s "
            f"name would be a guess.\nIf you know the judge, adopt the directory with:\n"
            f"    echo <judge-model> > {marker}\nOtherwise move it aside and rerun.\n"
        )

    with open(marker, "w") as f:
        f.write(judge_model)


def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    # One client per worker, not one per item: the original rebuilt the OpenAI client
    # inside the loop, so nothing was reused across a few thousand requests.
    gpt_service = GPTService()

    for file in tqdm(caption_files):
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['question']
        answer = qa_set['answer']
        pred = qa_set['pred_answer']

        try:
            # Compute the correctness score
            messages=[
                {
                    "role": "system",
                    "content": 
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                }
            ]
            response_message = gpt_service.gpt_with_retry(messages)
            # Convert response to a Python dictionary.
            # response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def main():
    """
    Main function to control the flow of the program.
    """
    args = parse_args()

    # Fail before the retry loop below: annotate() swallows per-file exceptions, so an
    # unusable judge would otherwise spin forever without completing a single file.
    GPTService()

    pred_contents = pd.read_csv(args.pred_path).to_dict(orient='records')

    if 'retrieve_size' in pred_contents[0]:  # NOTE: change this line if experiment with different hyper-parameters
        pred_contents = [x for x in pred_contents if x['retrieve_size']==64 and x['chunk_size']==1]

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_id']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_id'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['video_id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    claim_cache_dir(output_dir, os.environ.get("OPENAI_JUDGE_MODEL", DEFAULT_JUDGE))

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        video_id = sample['video_id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred_answer']
        qa_set = {"question": question, "answer": answer, "pred_answer": pred}
        prediction_set[video_id] = qa_set

    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    stalled_passes = 0
    prev_remaining = None
    while True:
        try:
            # Files that have not been processed yet.
            # .json only: the dir also holds the .judge_model marker, which would
            # otherwise be counted as a completed item in the progress line.
            completed_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break

            # Give up on items that fail deterministically. annotate() swallows every
            # per-item exception, so without this a permanently-unparseable reply is
            # retried forever and the run never terminates -- it just reprints the same
            # errors. Reporting N unscored items beats hanging.
            if prev_remaining is not None and len(incomplete_files) >= prev_remaining:
                stalled_passes += 1
                if stalled_passes >= 3:
                    print(f"WARNING: {len(incomplete_files)} items failed on 3 consecutive "
                          f"passes with no progress; giving up on them.")
                    break
            else:
                stalled_passes = 0
            prev_remaining = len(incomplete_files)

            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

            # Bounded pool. Bare Pool() defaults to os.cpu_count() workers -- num_tasks
            # only splits the list, it never limited concurrency -- which is what opened
            # the rvs_ego run at ~370 req/min and tripped the rate limit immediately.
            with Pool(processes=min(args.max_workers, len(all_parts))) as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content
    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in combined_contents.items():
        # Computing score
        count += 1
        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score

        # Computing accuracy
        if 'pred' in result[0]:
            pred = result[0]['pred']
        else:
            pred = result[0]['prev']
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    combined_contents["average_score"] = average_score
    combined_contents["accuracy"] = accuracy
    # Provenance, matching eval_open_ended_local.py's metrics block. Without it a
    # results.json is just two floats with no record of what judged them -- which is how
    # the 2026-08-07 rvs_ego run ended up unattributable.
    combined_contents["metrics"] = {
        "judge_model": os.environ.get("OPENAI_JUDGE_MODEL", DEFAULT_JUDGE),
        "num_items": len(combined_contents) - 2,  # minus accuracy + average_score
        "num_scored": count,
        "yes_count": yes_count,
        "no_count": no_count,
        "accuracy": accuracy,
        "average_score": average_score,
    }
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print(f"Accuracy: {accuracy*100:.1f}%")
    print(f"Average score: {average_score:.2f}")

    # Same annex as eval_open_ended_local.py, deliberately identical: the two scorers must
    # not report different things about the same predictions. See video_qa/answer_types.py
    # for why a pooled RVS accuracy needs it. No-ops on non-RVS benchmarks.
    try:
        pairs, correct = [], []
        for key, result in combined_contents.items():
            if key in ("accuracy", "average_score", "metrics"):
                continue
            meta = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
            pred = result[0].get('pred', result[0].get('prev', ''))
            pairs.append((meta.get("question", ""), meta.get("answer", "")))
            correct.append("yes" in str(pred).lower())
        for line in answer_types.summarize(pairs, correct):
            print(line)
    except Exception as exc:  # noqa: BLE001
        print(f"(answer_type breakdown unavailable: {exc})")
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")


if __name__ == "__main__":
    main()

