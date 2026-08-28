"""Prompt construction for model evaluation.

The model only ever sees the question text, the answer choices, and (optionally)
media. It never sees the correct answer, answer_text, min_fps, or the temporal
certificate unless a diagnostic flag explicitly asks for them.

Vendored from FPSBench's own evaluation harness so ReKV asks the questions the way the
benchmark defines them, rather than through ReKV's generic multiple-choice template
(`BaseVQA.format_mcqa_prompt`, which formats options as "(A) text" and prefills the
assistant turn with "Best option: ("). The two differ in system preamble, option
formatting and expected output shape, so predictions from one are not comparable to the
other's. `ANSWER_CHOICES` is the only thing inlined -- upstream it comes from the
package's `__init__`.

Used by both FPSBench solvers: video_qa/rekv_fpsbench_stream_small_vqa.py (the
bare clips) and video_qa/rekv_fpsbench_stream_vqa.py (the same questions inside a 600 s
haystack). They share this prompt so their numbers are comparable.
"""

from __future__ import annotations

import re
import random
from typing import Dict, Iterable, List, Optional, Tuple

# Upstream: `from . import ANSWER_CHOICES`. FPSBench questions carry five options, the
# fifth always "None of the above".
ANSWER_CHOICES = ["A", "B", "C", "D", "E"]

__all__ = ["ANSWER_CHOICES", "DEFAULT_SYSTEM_PROMPT", "BINARY_SYSTEM_PROMPT",
           "build_prompt", "build_binary_prompt", "mba_candidates", "ordered_choices",
           "parse_letter", "parse_yes_no"]

DEFAULT_SYSTEM_PROMPT = (
    "Analyze the video carefully, focusing on rapid motion and fine-grained "
    "temporal details. Answer the multiple-choice question. Start your response "
    "with exactly one option letter from the available choices, then provide a "
    "brief explanation."
)


def ordered_choices(
    choices: Dict[str, str],
    *,
    include_none_of_above: bool = True,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Return choices as an ordered list of ``(letter, text)`` pairs.

    Args:
        choices: mapping like ``{"A": "...", "B": "...", ...}``.
        include_none_of_above: if False, the "None of the above" choice (E) is
            dropped from the presented options.
        shuffle: if True, the *texts* are shuffled across letters (re-lettered
            A, B, C, ...). "None of the above" is always kept last regardless of
            shuffling, matching how human annotators saw it.
        seed: RNG seed for reproducible shuffling.
    """
    items = [(k, v) for k, v in choices.items() if v is not None]
    items.sort(key=lambda kv: ANSWER_CHOICES.index(kv[0]))

    none_items = [kv for kv in items if kv[1] == "None of the above"]
    real_items = [kv for kv in items if kv[1] != "None of the above"]

    if not include_none_of_above:
        none_items = []

    texts = [text for _, text in real_items]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(texts)

    texts += [text for _, text in none_items]
    letters = list(ANSWER_CHOICES)[: len(texts)]
    return list(zip(letters, texts))


def build_prompt(
    example: Dict,
    *,
    include_none_of_above: bool = True,
    shuffle: bool = False,
    seed: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Tuple[str, List[Tuple[str, str]], Dict[str, str]]:
    """Build the user prompt for an example.

    Returns ``(prompt_text, presented_choices, letter_to_text)`` where
    ``presented_choices`` is the ordered ``(letter, text)`` list actually shown
    and ``letter_to_text`` maps the presented letters to their texts (useful for
    re-mapping a shuffled prediction back to the canonical answer).
    """
    q = example["question"]
    presented = ordered_choices(
        q["choices"],
        include_none_of_above=include_none_of_above,
        shuffle=shuffle,
        seed=seed,
    )
    letters = ", ".join(letter for letter, _ in presented)
    options_block = "\n".join(f"{letter}. {text}" for letter, text in presented)

    prompt = (
        f"{system_prompt}\n\n"
        f"Question: {q['text']}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Respond with exactly one of these option letters ({letters}), "
        f"then a brief explanation."
    )
    return prompt, presented, {letter: text for letter, text in presented}


# --- Not part of the upstream module ----------------------------------------------
# The inverse of build_prompt: recover the letter from a response shaped by it. It lives
# here so the solvers (video_qa/rekv_fpsbench_stream*_vqa.py) and the submission exporter
# (video_qa/eval/export_fpsbench.py) parse identically -- two copies of this would drift,
# and a run whose CSV and JSONL disagree about the prediction is worse than either.

# The letter at the very start of the response, as the prompt demands. Tolerates the
# decorations models add anyway: "(A)", "A.", "**A**", "A:".
_LEADING_LETTER_RE = re.compile(r'^\W{0,3}([A-Za-z])\b')
# Fallback for a response that buries the choice ("The answer is C because..."). Anchored
# on the words that introduce a choice rather than on any capital letter, which would
# match the "A" in "A player dribbles" and silently invent a prediction.
_STATED_LETTER_RE = re.compile(r'\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Za-z])\)?\b',
                               re.IGNORECASE)


def parse_letter(response: str, valid: Optional[Iterable[str]] = None) -> str:
    """Option letter from a letter-then-explanation response, '' when there is none.

    Empty rather than a guess: an unparsed response scores as wrong either way, but a
    fabricated letter is indistinguishable from a real prediction downstream, whereas ''
    is counted and reported at export time.
    """
    valid = set(valid) if valid is not None else set(ANSWER_CHOICES)
    text = (response or "").strip()
    for pattern in (_LEADING_LETTER_RE, _STATED_LETTER_RE):
        match = pattern.search(text)
        if match and match.group(1).upper() in valid:
            return match.group(1).upper()
    return ""


# --- Multiple Binary Accuracy (TemporalBench, arXiv 2410.10818) --------------------
# Not part of the upstream FPSBench harness. TemporalBench's finding is that a
# multiple-choice score sits on a chance floor high enough to hide the difference between
# a model that understands the video and one that is guessing: FPSBench's five-way format
# floors at 0.200, and a blind ReKV control measures 0.230 against video arms of
# 0.278-0.295. There is no room left in that gap to read.
#
# MBA re-asks each item as K independent binary decisions -- "is this candidate the
# correct answer?" -- and counts the item correct only when *every* one is right. That
# moves the floor to 1/2**K (0.0625 at K=4) and sends any constant-answer strategy to
# exactly 0: a model that always says Yes fails on the three negatives, one that always
# says No fails on the positive. What survives is consistency, which is the thing a
# guessing model does not have.
#
# The E option is excluded from the candidate set by default. FPSBench's key never selects
# it (0 of 1000 rows in fpsbench_v1.csv), so its binary answer is "No" on every question --
# a free point for any No-biased model, inflating MBA without measuring anything.

BINARY_SYSTEM_PROMPT = (
    "Analyze the video carefully, focusing on rapid motion and fine-grained "
    "temporal details. You are given a question about the video and one candidate "
    "answer. Decide whether that candidate answer is correct for the video. "
    "Respond with exactly one word: Yes or No."
)


def mba_candidates(choices, *, include_none_of_above=False):
    """The (index, text) candidates to ask binaries about, in annotation order.

    Order is not shuffled. Unlike the multiple-choice path, a binary question presents one
    candidate on its own, so there is no option list for position bias to act on and
    nothing for a shuffle to control; keeping annotation order is what lets an MBA row
    join back to its multiple-choice row by candidate index.
    """
    out = []
    for i, text in enumerate(choices):
        if text is None:
            continue
        if not include_none_of_above and str(text).strip() == "None of the above":
            continue
        out.append((i, str(text)))
    return out


def build_binary_prompt(question, candidate, *, system_prompt=BINARY_SYSTEM_PROMPT):
    """The user turn for one binary sub-question.

    The candidate is presented alone and never alongside its siblings: the point of the
    metric is that the K decisions are independent, and a model shown the other options
    could recover the multiple-choice task by elimination -- restoring the very floor MBA
    exists to remove.
    """
    return (
        f"{system_prompt}\n\n"
        f"Question: {question}\n\n"
        f"Candidate answer: {candidate}\n\n"
        f"Is this candidate answer correct for the video? "
        f"Respond with exactly one word, Yes or No."
    )


# Same two-tier shape as the letter parser above. The first rule takes a leading yes/no
# as the verdict, which is what the prompt asked for -- so a response opening "no dribbles
# are visible" reads as No, and that is the right call rather than a misfire. The fallback
# handles a response that buries the verdict ("...so the answer is no"), and is anchored on
# the phrases that introduce one rather than on any occurrence of yes/no, which mid-string
# would match description as often as decision.
_LEADING_YN_RE = re.compile(r'^\W{0,3}(yes|no)\b', re.IGNORECASE)
_STATED_YN_RE = re.compile(
    r'\b(?:answer|response|verdict)\s*(?:is|:)?\s*\(?(yes|no)\)?\b', re.IGNORECASE)


def parse_yes_no(response):
    """True for Yes, False for No, None when the response commits to neither.

    None rather than a default: an unparsed response is scored as wrong by the metric
    either way, but collapsing it to False here would make it indistinguishable from a
    real "No" and hide a model that is refusing or hedging rather than deciding. The
    scorer counts them separately (video_qa/eval/eval_fpsbench_mba.py).
    """
    text = (response or "").strip()
    for pattern in (_LEADING_YN_RE, _STATED_YN_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).lower() == "yes"
    return None
