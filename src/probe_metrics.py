"""Pure response scoring, kept separate from GPU inference."""

import re


def boxed_integer(text):
    """Read only the last simple boxed integer, including malformed-last-box cases."""
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    end = text.find("}", start, start + 101)
    if end == -1:
        return None
    candidate = text[start:end].strip()
    if not re.fullmatch(r"[+-]?\d+", candidate):
        return None
    return str(int(candidate))


def repeated_ngram_fraction(token_ids, n=4):
    """Repeated n-gram occurrences / all occurrences; not a loop classifier."""
    if type(n) is not int or not 1 <= n <= 16:
        raise ValueError("n must be an integer between 1 and 16.")
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def validate_problem(problem):
    if (
        not isinstance(problem, dict)
        or not isinstance(problem.get("id"), str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", problem["id"])
        or not isinstance(problem.get("question"), str)
        or not 1 <= len(problem["question"]) <= 3000
        or not isinstance(problem.get("answer"), str)
        or not 1 <= len(problem["answer"]) <= 200
        or problem.get("grader", "boxed_integer") not in ("boxed_integer", "exact")
    ):
        raise ValueError("Each probe needs a short id/question/answer and a supported grader.")
    if problem.get("grader", "boxed_integer") == "boxed_integer":
        if not re.fullmatch(r"[+-]?\d{1,20}", problem["answer"]):
            raise ValueError("A boxed-integer probe needs an integer answer.")


def _strip_terminal_tokens(text, terminal_tokens):
    result = text.strip()
    tokens = [token for token in terminal_tokens if isinstance(token, str) and token]
    while True:
        match = next((token for token in tokens if result.endswith(token)), None)
        if match is None:
            return result
        result = result[:-len(match)].rstrip()


def response_metrics(
    text, tokens, expected, *, thinking, max_new_tokens, eos_ids,
    grader="boxed_integer", terminal_tokens=(), timed_out=False,
):
    """Distinguish content, reasoning closure, final answer, and natural stop."""
    if type(thinking) is not bool or type(timed_out) is not bool:
        raise ValueError("thinking and timed_out must be booleans.")
    if grader not in ("boxed_integer", "exact"):
        raise ValueError("Unsupported response grader.")
    closed = "</think>" in text
    if thinking:
        final = text.rsplit("</think>", 1)[-1] if closed else ""
        reasoning = text.rsplit("</think>", 1)[0] if closed else text
    else:
        final, reasoning = text, ""
    final = _strip_terminal_tokens(final, terminal_tokens)
    if grader == "boxed_integer":
        expected = str(int(expected))
        answer = boxed_integer(final)
        reasoning_answer = boxed_integer(reasoning)
    else:
        answer = final if final else None
        reasoning_answer = None
    natural_stop = bool(tokens) and tokens[-1] in eos_ids
    cap = len(tokens) >= max_new_tokens and not natural_stop
    stop_reason = (
        "eos" if natural_stop else "token_limit" if cap
        else "time_budget" if timed_out else "other"
    )
    return {
        "boxed_answer": answer if grader == "boxed_integer" else None,
        "final_answer": answer,
        "correct": answer == expected,
        "delivered_correct": answer == expected and natural_stop,
        "reasoning_boxed_answer": reasoning_answer,
        "reasoning_boxed_correct": thinking and reasoning_answer == expected,
        "finished_thinking": closed if thinking else None,
        "thinking": thinking,
        "grader": grader,
        "natural_stop": natural_stop,
        "stop_reason": stop_reason,
        "generated_tokens": len(tokens),
        "hit_token_limit": cap,
        "repeated_4gram_fraction": repeated_ngram_fraction(tokens),
    }
