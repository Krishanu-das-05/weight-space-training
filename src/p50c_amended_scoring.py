"""Approved bound exceptions: false correctness, token-only EOS and closure."""

import re

from budget_sweep_scoring import CAPS, _tokens, grade_generation


def amended_grade(text, tokens, expected, terminal_tokens):
    tokens = _tokens(tokens)
    if not isinstance(text, str) or not isinstance(expected, str) or not re.fullmatch(r"[+-]?[0-9]{1,20}", expected):
        raise ValueError("Undecodable text or invalid integer reference; raw IDs remain durable.")
    length = len(text)
    violation = not 1 <= length <= 200000
    if not violation:
        return grade_generation(text, tokens, expected, thinking=True, eos_ids={248044},
                                terminal_tokens=terminal_tokens), False, length
    positions = [i + 1 for i, token in enumerate(tokens) if token == 248044]
    if positions and positions != [len(tokens)]:
        raise ValueError("A batch-one response cannot contain tokens after its first EOS.")
    return {
        "strict_final_correct": False, "final_answer": None,
        "natural_eos_index": positions[0] if positions else None, "generated_tokens": len(tokens),
    }, True, length


def closure_from_tokens(tokens, decode):
    tokens = _tokens(tokens)
    result = {}
    for cap in CAPS:
        text = decode(tokens[:cap])
        if not isinstance(text, str):
            raise ValueError("Undecodable token prefix; raw IDs remain durable.")
        result[str(cap)] = "</think>" in text
    return result
