"""Small paired probes, not a replacement for the article's math benchmark."""

import math
import re
import time

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from probe_metrics import boxed_integer, repeated_ngram_fraction, response_metrics, validate_problem

PROTOCOL_VERSION = "paired-v2"
PROMPT_INSTRUCTIONS = {
    "step_by_step": "Please reason step by step, and put your final answer within \\boxed{}.",
    "final_only": "Please give only your final answer within \\boxed{}, with no explanation.",
}
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
PROBLEMS = (
    {"id": "powers", "question": "What is 2^10 + 24?", "answer": "1048"},
    {
        "id": "discount",
        "question": (
            "An item costs 80 dollars. Its price is discounted by 25%, then "
            "10% sales tax is applied to the discounted price. What is the final price in dollars?"
        ),
        "answer": "66",
    },
    {"id": "linear", "question": "Solve 7*x + 9 = 100. What is x?", "answer": "13"},
    {
        "id": "sum",
        "question": "What is the sum of all positive integers from 1 through 20, inclusive?",
        "answer": "210",
    },
)


class GeneratedPresencePenalty(LogitsProcessor):
    """Subtract once per seen GENERATED token, not once per occurrence/prompt."""

    def __init__(self, prompt_length, penalty):
        if type(prompt_length) is not int or prompt_length < 0:
            raise ValueError("prompt_length must be a nonnegative integer.")
        if (
            type(penalty) not in (int, float)
            or not math.isfinite(penalty)
            or not -2 <= penalty <= 2
        ):
            raise ValueError("penalty must be finite and between -2 and 2.")
        self.prompt_length = prompt_length
        self.penalty = float(penalty)

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty


def build_probe_content(problem, prompt_style="step_by_step"):
    validate_problem(problem)
    if not isinstance(prompt_style, str) or prompt_style not in PROMPT_INSTRUCTIONS:
        raise ValueError("prompt_style must be step_by_step or final_only.")
    if problem.get("grader", "boxed_integer") != "boxed_integer":
        if prompt_style != "step_by_step":
            raise ValueError("Prompt-style controls require boxed-integer probes.")
        return problem["question"]
    return problem["question"] + "\n" + PROMPT_INSTRUCTIONS[prompt_style]


def run_probes(
    model, tokenizer, problems, condition, max_new_tokens=512, seed=0, *,
    thinking=True, max_seconds=None, prompt_style="step_by_step",
):
    """Batch size one; each condition resets the same per-problem random seed."""
    if type(max_new_tokens) is not int or not 32 <= max_new_tokens <= 4096:
        raise ValueError("Use a local smoke-test budget between 32 and 4096 tokens.")
    if type(seed) is not int or not 0 <= seed < 2**32 - 20:
        raise ValueError("seed must be an integer in [0, 2**32 - 20).")
    if type(thinking) is not bool:
        raise ValueError("thinking must be a boolean.")
    if not isinstance(prompt_style, str) or prompt_style not in PROMPT_INSTRUCTIONS:
        raise ValueError("prompt_style must be step_by_step or final_only.")
    if max_seconds is not None and (
        type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
        or not 1 <= max_seconds <= 1800
    ):
        raise ValueError("max_seconds must be between 1 and 1800, or None.")
    if not isinstance(problems, (tuple, list)) or not 1 <= len(problems) <= 20:
        raise ValueError("Supply between 1 and 20 probe problems.")
    for problem in problems:
        build_probe_content(problem, prompt_style)
    if len({problem["id"] for problem in problems}) != len(problems):
        raise ValueError("Probe ids must be unique.")
    if not isinstance(condition, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", condition):
        raise ValueError("condition must be a short identifier.")

    records = []
    device = model.device
    cuda_devices = [device.index] if device.type == "cuda" else []
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    terminal_tokens = tuple(dict.fromkeys(
        [tokenizer.decode([token], skip_special_tokens=False) for token in sorted(eos_ids)]
        + [
            token for token in getattr(tokenizer, "all_special_tokens", ())
            if token in ("<|im_end|>", "<|endoftext|>")
        ]
    ))
    for index, problem in enumerate(problems):
        grader = problem.get("grader", "boxed_integer")
        content = build_probe_content(problem, prompt_style)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]
        processors = LogitsProcessorList([
            GeneratedPresencePenalty(prompt_length, SAMPLING["presence_penalty"])
        ])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            torch.manual_seed(seed + index)
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=SAMPLING["temperature"],
                top_p=SAMPLING["top_p"],
                top_k=SAMPLING["top_k"],
                repetition_penalty=SAMPLING["repetition_penalty"],
                logits_processor=processors,
                **({"max_time": max_seconds} if max_seconds is not None else {}),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        tokens = generated[0, prompt_length:].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        expected = str(int(problem["answer"])) if grader == "boxed_integer" else problem["answer"]
        metrics = response_metrics(
            text, tokens, expected, thinking=thinking, max_new_tokens=max_new_tokens,
            eos_ids=eos_ids, grader=grader, terminal_tokens=terminal_tokens,
            timed_out=max_seconds is not None and elapsed >= max_seconds,
        )
        record = {
            "condition": condition,
            "problem_id": problem["id"],
            "question": problem["question"],
            "expected": expected,
            "seed": seed + index,
            "protocol": PROTOCOL_VERSION,
            "prompt_style": prompt_style,
            "user_message": content,
            "rendered_prompt": prompt,
            "prompt_tokens": prompt_length,
            **metrics,
            "seconds": round(elapsed, 3),
            "text": text,
        }
        records.append(record)
        print(
            f"{condition}/{problem['id']}: {len(tokens)} tokens, "
            f"answer={metrics['final_answer']}, stop={metrics['stop_reason']}, {elapsed:.1f}s",
            flush=True,
        )
    return records
