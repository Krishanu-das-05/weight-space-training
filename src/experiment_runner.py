"""Local-only, budgeted paired experiments with durable per-generation records."""

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList, Qwen3_5ForConditionalGeneration
from transformers.generation.utils import GenerationMixin

from download_model import MODEL_DIR, REVISION, WEIGHT_FILE, WEIGHT_SHA256, WEIGHT_SIZE
from experiment_design import (
    CONDITIONS, DEVELOPMENT, HOLDOUT_CONTROLS, HOLDOUT_MATH,
    holdout_gate, split_fingerprint, summarize,
)
from math_probe import GeneratedPresencePenalty, PROMPT_INSTRUCTIONS, PROTOCOL_VERSION, SAMPLING, run_probes
from probe_metrics import validate_problem
from prompt_control import (
    DESIGN_VERSION, PROMPT_CONDITIONS, PROMPT_PROBLEMS, PROMPT_SEEDS,
    analyze_prompt_controls, validate_prompt_design,
)
from weight_edit import PROJECT_ROOT, UPSTREAM_COMMIT, edited_weights, load_edit, tensor_sha256


def load_local_model():
    weights = MODEL_DIR / WEIGHT_FILE
    if not weights.is_file() or weights.stat().st_size != WEIGHT_SIZE:
        raise FileNotFoundError("The pinned model weights are missing or incomplete.")
    with weights.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != WEIGHT_SHA256:
            raise ValueError("The local model checkpoint failed SHA-256 verification.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")
    free, _ = torch.cuda.mem_get_info()
    if free < 2.5 * 1024**3:
        raise RuntimeError("Free at least 2.5 GiB of GPU memory before loading this model.")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False, use_safetensors=True,
    ).to("cuda").eval()
    verify_base(model)
    return model, tokenizer


def verify_base(model):
    for edit in load_edit():
        if tensor_sha256(model.get_parameter(edit["name"])) != edit["base_sha256"]:
            raise ValueError("A live model tensor is not at the original base checkpoint.")


def sampler_audit():
    config = GenerationConfig(
        do_sample=True, temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
        top_k=SAMPLING["top_k"], repetition_penalty=SAMPLING["repetition_penalty"],
    )
    config._eos_token_tensor = torch.tensor([0])
    processors = GenerationMixin()._get_logits_processor(
        config, input_ids_seq_length=2, device="cpu",
        logits_processor=LogitsProcessorList([GeneratedPresencePenalty(2, SAMPLING["presence_penalty"])]),
    )
    return [type(item).__name__ for item in processors]


def new_run_directory(stage):
    if stage not in ("pilot", "holdout", "anatomy", "prompt_control"):
        raise ValueError("Unknown experiment stage.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = PROJECT_ROOT / "results" / f"{stage}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def validate_plan(problems, conditions, seeds, max_new_tokens, seconds, token_budget, *, stage="pilot"):
    if stage not in ("pilot", "holdout", "prompt_control"):
        raise ValueError("Unknown experiment stage.")
    if type(max_new_tokens) is not int or not 32 <= max_new_tokens <= 4096:
        raise ValueError("Choose a token cap between 32 and 4096.")
    if type(seconds) not in (int, float) or not 1 <= seconds <= 21600:
        raise ValueError("The wall-time budget must be between 1 second and 6 hours.")
    if type(token_budget) is not int or not 32 <= token_budget <= 1_000_000:
        raise ValueError("The generation-token budget must be between 32 and 1,000,000.")
    if (
        not isinstance(seeds, (tuple, list)) or not 1 <= len(seeds) <= 3
        or any(type(seed) is not int or not 0 <= seed < 2**32 - 40 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Use one to three distinct nonnegative integer seeds.")
    if not problems or len(problems) > 30:
        raise ValueError("Use 1-30 uniquely identified problems.")
    for problem in problems:
        validate_problem(problem)
    if len({item["id"] for item in problems}) != len(problems):
        raise ValueError("Problem identifiers must be unique.")
    allowed = PROMPT_CONDITIONS if stage == "prompt_control" else CONDITIONS
    if not conditions or any(condition not in allowed for condition in conditions):
        raise ValueError("Only predeclared experiment conditions are accepted.")
    if len({item.name for item in conditions}) != len(conditions):
        raise ValueError("Experiment conditions must be unique.")
    if stage == "prompt_control":
        validate_prompt_design(problems, conditions, seeds)


def run_suite(
    model, tokenizer, *, stage="pilot", problems=None, conditions=None,
    seeds=None, max_new_tokens=1024, seconds=900, token_budget=None,
):
    problems = tuple(DEVELOPMENT[:2] if problems is None else problems)
    conditions = (
        PROMPT_CONDITIONS if stage == "prompt_control" else CONDITIONS
    ) if conditions is None else conditions
    seeds = (PROMPT_SEEDS if stage == "prompt_control" else (0,)) if seeds is None else seeds
    if token_budget is None:
        token_budget = 16384 if stage == "prompt_control" else 12288
    validate_plan(problems, conditions, seeds, max_new_tokens, seconds, token_budget, stage=stage)
    verify_base(model)
    directory = new_run_directory(stage)
    metadata = {
        "stage": stage, "protocol": PROTOCOL_VERSION, "model_revision": REVISION,
        "edit_commit": UPSTREAM_COMMIT, "sampling": dict(SAMPLING),
        "prompt_instructions": dict(PROMPT_INSTRUCTIONS),
        "prompt_control_design": DESIGN_VERSION if stage == "prompt_control" else None,
        "base_only": all(not item.layers for item in conditions),
        "conditions": [asdict(item) for item in conditions], "seeds": list(seeds),
        "max_new_tokens": max_new_tokens, "wall_budget_seconds": seconds,
        "generation_token_budget": token_budget,
        "problems": problems, "split_fingerprint": split_fingerprint(problems),
        "torch": torch.__version__, "transformers": version("transformers"),
        "sampler_processor_order": sampler_audit(),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("math_probe.py", "probe_metrics.py", "weight_edit.py",
                         "experiment_design.py", "experiment_runner.py", "prompt_control.py")
        },
    }
    _write_json(directory / "plan.json", metadata)
    rows, started, used_tokens = [], time.perf_counter(), 0
    expected = len(problems) * len(conditions) * len(seeds)
    status = "running"
    print("Run directory:", directory, flush=True)
    try:
        with (directory / "records.jsonl").open("x", encoding="utf-8") as journal:
            for seed in seeds:
                for index, problem in enumerate(problems):
                    for condition in conditions:
                        remaining = seconds - (time.perf_counter() - started)
                        if remaining < 1 or token_budget - used_tokens < max_new_tokens:
                            status = "budget_exhausted"
                            break
                        context = (
                            edited_weights(model, layers=condition.layers, strength=condition.strength)
                            if condition.layers else nullcontext([])
                        )
                        with context as changes:
                            row = run_probes(
                                model, tokenizer, [problem], condition.name,
                                max_new_tokens=max_new_tokens, seed=seed + index,
                                thinking=condition.thinking, max_seconds=min(120, remaining),
                                prompt_style=condition.prompt_style,
                            )[0]
                        row.update(
                            seed_group=seed, variant=asdict(condition), changes=changes,
                            split="controls" if problem["id"].startswith("control") else "math",
                        )
                        rows.append(row)
                        used_tokens += row["generated_tokens"]
                        journal.write(json.dumps(row, allow_nan=False) + "\n")
                        journal.flush()
                    if status == "budget_exhausted":
                        break
                if status == "budget_exhausted":
                    break
        if status == "running":
            status = "complete"
    except BaseException:
        status = "interrupted_or_error"
        raise
    finally:
        restored = False
        try:
            verify_base(model)
            restored = True
        finally:
            complete = status == "complete" and len(rows) == expected and restored
            report = {
                **metadata, "status": status, "complete": complete,
                "expected_generations": expected, "completed_generations": len(rows),
                "wall_seconds": time.perf_counter() - started, "used_tokens": used_tokens,
                "restored_base_hashes_verified": restored, "records": rows,
                "summary": summarize(rows),
                "by_split": {
                    split: summarize([row for row in rows if row["split"] == split])
                    for split in sorted({row["split"] for row in rows})
                },
                "decision": (
                    holdout_gate(rows, complete=complete) if stage == "pilot"
                    else analyze_prompt_controls(rows, complete=complete) if stage == "prompt_control"
                    else None
                ),
            }
            _write_json(directory / "summary.json", report)
    print("Saved:", directory / "summary.json", flush=True)
    if report["decision"]:
        print(report["decision"]["reason"], flush=True)
    return directory, report


def frozen_selection(path):
    """Only a successful local pilot can supply a held-out candidate."""
    resolved = Path(path).resolve()
    if (
        not resolved.is_relative_to(PROJECT_ROOT / "results")
        or resolved.name != "summary.json"
        or not resolved.is_file()
        or resolved.stat().st_size > 1_000_000
    ):
        raise ValueError("Select a pilot summary.json inside this project's results directory.")
    report = json.loads(resolved.read_text(encoding="utf-8"))
    decision = report.get("decision")
    if (
        report.get("stage") != "pilot" or report.get("protocol") != PROTOCOL_VERSION
        or report.get("model_revision") != REVISION or report.get("sampling") != SAMPLING
        or report.get("complete") is not True
        or report.get("restored_base_hashes_verified") is not True
        or not isinstance(decision, dict) or decision.get("allow_holdout") is not True
    ):
        raise ValueError("This pilot has not passed the current protocol's held-out gate.")
    candidate = decision.get("selected_condition")
    if isinstance(candidate, dict):
        candidate = {"prompt_style": "step_by_step", **candidate}
    for condition in CONDITIONS[2:]:
        canonical = json.loads(json.dumps(asdict(condition)))
        if candidate == canonical:
            return condition, report["max_new_tokens"]
    raise ValueError("The selected condition is not a predeclared edit variant.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "holdout", "prompt-control"), default="pilot")
    parser.add_argument("--selection", type=Path, help="Passing pilot summary.json for held-out evaluation.")
    parser.add_argument("--problems", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seconds", type=int, help="Soft wall-time budget, at most six hours.")
    parser.add_argument("--token-budget", type=int, help="Total generation-token budget.")
    arguments = parser.parse_args()
    stage = arguments.stage.replace("-", "_")
    if stage == "holdout":
        if arguments.selection is None:
            parser.error("--stage holdout requires --selection.")
        condition, max_tokens = frozen_selection(arguments.selection)
        problems = HOLDOUT_MATH + HOLDOUT_CONTROLS
        conditions, seeds = CONDITIONS[:2] + (condition,), (0, 100)
        seconds = arguments.seconds if arguments.seconds is not None else 14400
        token_budget = arguments.token_budget if arguments.token_budget is not None else 400000
    else:
        if arguments.selection is not None:
            parser.error("--selection is only valid for the holdout stage.")
        if stage == "prompt_control":
            if arguments.problems != 2:
                parser.error("Prompt-control v1 uses exactly two development problems.")
            problems, conditions, seeds = PROMPT_PROBLEMS, PROMPT_CONDITIONS, PROMPT_SEEDS
        else:
            problems, conditions, seeds = DEVELOPMENT[:arguments.problems], CONDITIONS, (0,)
        max_tokens = arguments.max_tokens
        seconds = arguments.seconds if arguments.seconds is not None else 900
        default_tokens = 16384 if stage == "prompt_control" else 12288
        token_budget = arguments.token_budget if arguments.token_budget is not None else default_tokens
    validate_plan(problems, conditions, seeds, max_tokens, seconds, token_budget, stage=stage)
    model, tokenizer = load_local_model()
    run_suite(
        model, tokenizer, stage=stage, problems=problems, conditions=conditions,
        seeds=seeds, max_new_tokens=max_tokens, seconds=seconds, token_budget=token_budget,
    )


if __name__ == "__main__":
    main()
