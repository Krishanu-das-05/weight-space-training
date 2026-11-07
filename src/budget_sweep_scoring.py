"""Frozen, inference-free scoring rules for the preregistered budget sweep."""

import math
import re

from probe_metrics import response_metrics

CAPS = (512, 1024, 2048, 4096, 8192)
WILSON_Z = 1.959963984540054


def _tokens(value):
    if (
        not isinstance(value, list) or not 1 <= len(value) <= 8192
        or any(type(token) is not int or not 0 <= token < 2**31 for token in value)
    ):
        raise ValueError("Expected 1-8192 nonnegative integer token IDs.")
    return value


def prefix_matches(short_tokens, long_tokens):
    short, long = _tokens(short_tokens), _tokens(long_tokens)
    if len(short) > 1024:
        raise ValueError("A short validation response cannot exceed 1024 tokens.")
    return short == long[:min(1024, len(long))]


def grade_generation(text, token_ids, expected, *, thinking, eos_ids, terminal_tokens):
    tokens = _tokens(token_ids)
    if not isinstance(text, str) or not 1 <= len(text) <= 200000:
        raise ValueError("Expected bounded decoded response text.")
    if type(thinking) is not bool or not isinstance(expected, str):
        raise ValueError("Invalid grading inputs.")
    if (
        not isinstance(eos_ids, (tuple, list, set, frozenset)) or not eos_ids
        or any(type(token) is not int or not 0 <= token < 2**31 for token in eos_ids)
    ):
        raise ValueError("Invalid EOS IDs.")
    if (
        not isinstance(terminal_tokens, (tuple, list))
        or any(not isinstance(token, str) or not 1 <= len(token) <= 100 for token in terminal_tokens)
    ):
        raise ValueError("Invalid terminal token strings.")
    if not re.fullmatch(r"[+-]?[0-9]{1,20}", expected):
        raise ValueError("Expected a short integer reference.")
    positions = [index + 1 for index, token in enumerate(tokens) if token in eos_ids]
    if positions and positions != [len(tokens)]:
        raise ValueError("A batch-one response cannot contain tokens after its first EOS.")
    metrics = response_metrics(
        text, tokens, expected, thinking=thinking, max_new_tokens=8192,
        eos_ids=set(eos_ids), terminal_tokens=tuple(terminal_tokens),
    )
    return {
        "strict_final_correct": metrics["correct"],
        "final_answer": metrics["final_answer"],
        "natural_eos_index": positions[0] if positions else None,
        "generated_tokens": len(tokens),
    }


def score_at_cap(graded, cap):
    if type(cap) is not int or cap not in CAPS:
        raise ValueError("Unsupported analysis cap.")
    if not isinstance(graded, dict) or type(graded.get("strict_final_correct")) is not bool:
        raise ValueError("Invalid graded outcome.")
    length, eos = graded.get("generated_tokens"), graded.get("natural_eos_index")
    if type(length) is not int or not 1 <= length <= 8192:
        raise ValueError("Invalid generation length.")
    if eos is not None and (type(eos) is not int or eos != length):
        raise ValueError("EOS must be the final generated token.")
    natural = eos is not None and eos <= cap
    return {
        "correct": natural and graded["strict_final_correct"],
        "natural_completion": natural,
        "tokens": min(length, cap),
    }


def wilson(successes, count):
    if type(count) is not int or not 1 <= count <= 100000:
        raise ValueError("Wilson intervals require a positive integer sample size.")
    if type(successes) is not int or not 0 <= successes <= count:
        raise ValueError("Invalid success count.")
    probability, z2 = successes / count, WILSON_Z**2
    denominator = 1 + z2 / count
    center = (probability + z2 / (2 * count)) / denominator
    radius = WILSON_Z * math.sqrt(probability * (1 - probability) / count + z2 / (4 * count**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_comparison(on, off):
    if (
        not isinstance(on, list) or not isinstance(off, list) or len(on) != len(off)
        or not 1 <= len(on) <= 30 or any(type(value) is not bool for value in on + off)
    ):
        raise ValueError("Expected 1-30 paired binary outcomes.")
    wins = sum(a and not b for a, b in zip(on, off))
    losses = sum(b and not a for a, b in zip(on, off))
    discordant = wins + losses
    p_value = (
        min(1.0, 2 * sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1)) / 2**discordant)
        if discordant else 1.0
    )
    return {"on_wins": wins, "off_wins": losses, "ties": len(on) - discordant, "exact_mcnemar_p": p_value}


def holm_adjust(p_values):
    if (
        not isinstance(p_values, list) or not p_values or len(p_values) > 5
        or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in p_values)
    ):
        raise ValueError("Expected up to five finite p-values.")
    result, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        result[index] = running
    return result
