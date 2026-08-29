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

Used by video_qa/rekv_fpsbench_stream_vqa.py, which asks FPSBench's questions inside a
600 s haystack.
"""

from __future__ import annotations

import re
import random
from typing import Dict, Iterable, List, Optional, Tuple

# Upstream: `from . import ANSWER_CHOICES`. FPSBench questions carry five options, the
# fifth always "None of the above".
ANSWER_CHOICES = ["A", "B", "C", "D", "E"]

__all__ = ["ANSWER_CHOICES", "DEFAULT_SYSTEM_PROMPT", "build_prompt",
           "ordered_choices", "parse_letter"]

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
# beside the prompt that shapes the response, so a change to one is made against the other.

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
