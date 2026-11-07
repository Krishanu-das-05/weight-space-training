"""Fixed 2x2 prompt/thinking controls and descriptive paired comparisons."""

import json
import math
from dataclasses import asdict
from pathlib import Path

from experiment_design import Condition, DEVELOPMENT, summarize
from math_probe import PROMPT_INSTRUCTIONS, PROTOCOL_VERSION, SAMPLING, build_probe_content
from weight_edit import PROJECT_ROOT

DESIGN_VERSION = "prompt-control-v1"
PROMPT_PROBLEMS = DEVELOPMENT[:2]
PROMPT_SEEDS = (0, 100)
PROMPT_CONDITIONS = (
    Condition("think_on_steps", True),
    Condition("think_on_final", True, prompt_style="final_only"),
    Condition("think_off_steps", False),
    Condition("think_off_final", False, prompt_style="final_only"),
)
COMPARISONS = (
    ("prompt_effect_thinking_on", "think_on_steps", "think_on_final"),
    ("prompt_effect_thinking_off", "think_off_steps", "think_off_final"),
    ("thinking_off_with_steps", "think_on_steps", "think_off_steps"),
    ("thinking_off_with_final", "think_on_final", "think_off_final"),
)


def validate_prompt_design(problems, conditions, seeds):
    if (
        tuple(problems) != PROMPT_PROBLEMS or tuple(conditions) != PROMPT_CONDITIONS
        or tuple(seeds) != PROMPT_SEEDS
    ):
        raise ValueError(
            "Prompt-control v1 requires its four base-only conditions, "
            "the first two development questions, and seed groups (0, 100)."
        )


def _index_records(records, *, require_complete=True):
    if (
        not isinstance(records, list) or len(records) > 16
        or (require_complete and len(records) != 16)
    ):
        return None
    by_name = {item.name: item for item in PROMPT_CONDITIONS}
    problem_indices = {problem["id"]: index for index, problem in enumerate(PROMPT_PROBLEMS)}
    indexed = {}
    for row in records:
        if not isinstance(row, dict):
            return None
        if not isinstance(row.get("condition"), str) or not isinstance(row.get("problem_id"), str):
            return None
        condition = by_name.get(row["condition"])
        problem_id, seed_group = row.get("problem_id"), row.get("seed_group")
        if (
            condition is None or problem_id not in problem_indices
            or type(seed_group) is not int or seed_group not in PROMPT_SEEDS
            or type(row.get("seed")) is not int
            or row["seed"] != seed_group + problem_indices[problem_id]
            or row.get("thinking") is not condition.thinking
            or row.get("prompt_style") != condition.prompt_style
            or any(type(row.get(key)) is not bool for key in
                   ("delivered_correct", "correct", "natural_stop", "hit_token_limit"))
            or type(row.get("generated_tokens")) is not int
            or not 1 <= row["generated_tokens"] <= 4096
            or type(row.get("seconds")) not in (int, float)
            or not math.isfinite(row["seconds"]) or row["seconds"] < 0
            or type(row.get("repeated_4gram_fraction")) not in (int, float)
            or not math.isfinite(row["repeated_4gram_fraction"])
            or not 0 <= row["repeated_4gram_fraction"] <= 1
            or row.get("stop_reason") not in ("eos", "token_limit", "time_budget", "other")
            or row["delivered_correct"] != (row["correct"] and row["natural_stop"])
            or row["natural_stop"] != (row["stop_reason"] == "eos")
            or row["hit_token_limit"] != (row["stop_reason"] == "token_limit")
        ):
            return None
        key = condition.name, problem_id, seed_group
        if key in indexed:
            return None
        indexed[key] = row
    return indexed


def analyze_prompt_controls(records, *, complete):
    if type(complete) is not bool:
        raise ValueError("complete must be a boolean.")
    result = {
        "design_version": DESIGN_VERSION,
        "complete_pairs": False,
        "allow_holdout": False,
        "comparisons": [],
        "by_seed_group": {},
        "reason": "Incomplete, invalid, unpaired, or time-limited run: do not infer prompt effects.",
    }
    indexed = _index_records(records) if complete else None
    if indexed is None or any(row["stop_reason"] == "time_budget" for row in records):
        return result
    pairs = [(problem["id"], seed) for problem in PROMPT_PROBLEMS for seed in PROMPT_SEEDS]
    for label, reference, treatment in COMPARISONS:
        aligned = [
            (indexed[reference, problem, seed], indexed[treatment, problem, seed])
            for problem, seed in pairs
        ]
        wins = sum(not left["delivered_correct"] and right["delivered_correct"] for left, right in aligned)
        losses = sum(left["delivered_correct"] and not right["delivered_correct"] for left, right in aligned)
        result["comparisons"].append({
            "name": label, "reference": reference, "treatment": treatment,
            "pairs": len(pairs), "delivered_correct_wins": wins,
            "delivered_correct_losses": losses, "delivered_correct_ties": len(pairs) - wins - losses,
            "delivered_correct_rate_delta": (wins - losses) / len(pairs),
            "natural_stop_rate_delta": sum(
                int(right["natural_stop"]) - int(left["natural_stop"]) for left, right in aligned
            ) / len(pairs),
            "mean_token_delta": sum(
                right["generated_tokens"] - left["generated_tokens"] for left, right in aligned
            ) / len(pairs),
        })
    result.update(
        complete_pairs=True,
        by_seed_group={
            str(seed): summarize([row for row in records if row["seed_group"] == seed])
            for seed in PROMPT_SEEDS
        },
        reason=(
            "Complete 2x2 development comparison. Effects are descriptive: two questions "
            "and two seed groups are not four independent tasks. No weights were edited; "
            "this experiment does not authorize held-out inference or reopen the weight sweep."
        ),
    )
    return result


def load_prompt_report(path):
    """Read only a bounded, matching local report; recompute its comparisons."""
    if not isinstance(path, (str, Path)):
        raise ValueError("Select a local prompt-control summary path.")
    path = Path(path).resolve()
    if (
        not path.is_relative_to(PROJECT_ROOT / "results")
        or not path.parent.name.startswith("prompt_control_")
        or path.name != "summary.json" or not path.is_file()
        or path.stat().st_size > 5_000_000
    ):
        raise ValueError("Select a prompt_control_*/summary.json inside the project results directory.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Invalid prompt-control report.")
    canonical_conditions = json.loads(json.dumps([asdict(item) for item in PROMPT_CONDITIONS]))
    if (
        report.get("stage") != "prompt_control" or report.get("protocol") != PROTOCOL_VERSION
        or report.get("prompt_control_design") != DESIGN_VERSION
        or report.get("conditions") != canonical_conditions
        or report.get("problems") != list(PROMPT_PROBLEMS)
        or report.get("seeds") != list(PROMPT_SEEDS)
        or report.get("prompt_instructions") != PROMPT_INSTRUCTIONS
        or not isinstance(report.get("sampling"), dict)
        or json.dumps(report["sampling"], sort_keys=True, allow_nan=False)
        != json.dumps(SAMPLING, sort_keys=True, allow_nan=False)
        or type(report.get("complete")) is not bool
        or type(report.get("max_new_tokens")) is not int
        or not 32 <= report["max_new_tokens"] <= 4096
    ):
        raise ValueError("The report does not match prompt-control v1.")
    records = report.get("records")
    if not isinstance(records, list) or len(records) > 16:
        raise ValueError("Invalid prompt-control response records.")
    if _index_records(records, require_complete=False) is None:
        raise ValueError("Invalid prompt-control response metrics or pairing.")
    problems = {item["id"]: item for item in PROMPT_PROBLEMS}
    conditions = {item.name: item for item in PROMPT_CONDITIONS}
    for row in records:
        if not isinstance(row.get("text"), str) or len(row["text"]) > 100_000:
            raise ValueError("Invalid saved response.")
        problem = problems.get(row.get("problem_id"))
        condition = conditions.get(row.get("condition"))
        if (
            problem is None or condition is None
            or row.get("user_message") != build_probe_content(problem, condition.prompt_style)
            or row.get("expected") != problem["answer"]
        ):
            raise ValueError("The response prompt does not match the declared development design.")
    report["decision"] = analyze_prompt_controls(records, complete=report["complete"])
    if (
        report.get("expected_generations") != 16
        or report.get("completed_generations") != len(records)
        or any(row["generated_tokens"] > report["max_new_tokens"] for row in records)
        or (report["complete"] and (
            report.get("status") != "complete" or report.get("restored_base_hashes_verified") is not True
        ))
    ):
        raise ValueError("The report completion state or token counts are inconsistent.")
    if report["complete"] and not report["decision"]["complete_pairs"]:
        raise ValueError("A complete report must contain all sixteen valid paired responses.")
    report["summary"] = summarize(records)
    return report
