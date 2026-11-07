"""Recompute all sweep statistics from raw tokens and the unchanged grader."""

from datetime import datetime, timedelta, timezone
from statistics import mean

from budget_sweep_scoring import CAPS, grade_generation, holm_adjust, paired_comparison, prefix_matches, score_at_cap, wilson

CONDITIONS = ("thinking_on", "thinking_off")


def prefix_status(plan, records):
    indexed = {(row["problem_id"], row["condition"], row["kind"]): row for row in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate response identity.")
    comparisons = []
    for identifier in plan["prefix_validation"]["question_ids"]:
        for condition in CONDITIONS:
            short = indexed.get((identifier, condition, "prefix_validation_short"))
            long = indexed.get((identifier, condition, "main"))
            if short is not None and long is not None:
                comparisons.append({
                    "problem_id": identifier, "condition": condition,
                    "identical": prefix_matches(short["token_ids"], long["token_ids"]),
                    "short_tokens": len(short["token_ids"]), "long_tokens": len(long["token_ids"]),
                })
    failed = any(not item["identical"] for item in comparisons)
    return {"passed": len(comparisons) == 4 and not failed, "failed": failed, "comparisons": comparisons}


def recompute(plan, pool, records):
    expected = {item["id"]: item["answer"] for item in pool}
    indexed, graded = {}, {}
    diagnostics = {condition: {} for condition in CONDITIONS}
    short_wrong = []
    for row in records:
        condition, identifier, kind = row["condition"], row["problem_id"], row["kind"]
        if condition not in CONDITIONS or identifier not in expected or kind not in ("main", "prefix_validation_short"):
            raise ValueError("Unexpected response identity.")
        for category in (kind, "all"):
            values = diagnostics[condition].setdefault(category, {"responses": 0, "responses_with_248046": 0,
                                                                 "token_248046_occurrences": 0, "affected_ids": []})
            occurrences = row["token_ids"].count(248046)
            values["responses"] += 1
            values["responses_with_248046"] += occurrences > 0
            values["token_248046_occurrences"] += occurrences
            if occurrences:
                values["affected_ids"].append({"problem_id": identifier, "kind": kind, "count": occurrences})
        if kind != "main":
            continue
        key = (identifier, condition)
        if key in indexed:
            raise ValueError("Duplicate main response.")
        indexed[key] = row
        outcome = grade_generation(
            row["text"], row["token_ids"], expected[identifier], thinking=condition == "thinking_on",
            eos_ids={248044}, terminal_tokens=plan["stopping"]["terminal_tokens_for_grading"],
        )
        graded[key] = outcome
        eos = outcome["natural_eos_index"]
        if eos is not None and eos <= 1024 and not outcome["strict_final_correct"]:
            short_wrong.append({
                "problem_id": identifier, "condition": condition, "seed": row["seed"],
                "tokens": eos, "final_answer": outcome["final_answer"], "expected": expected[identifier],
                "text": row["text"],
            })
    validation = prefix_status(plan, records)
    complete = len(indexed) == 60 and validation["passed"]
    tables, paired = {}, []
    for condition in CONDITIONS:
        outcomes = [outcome for (identifier, name), outcome in graded.items() if name == condition]
        rows = []
        for cap in CAPS:
            scores = [score_at_cap(item, cap) for item in outcomes]
            n = len(scores)
            correct = sum(item["correct"] for item in scores)
            natural = sum(item["natural_completion"] for item in scores)
            rows.append({
                "cap": cap, "n": n, "correct": correct, "accuracy": correct / n if n else None,
                "accuracy_wilson95": list(wilson(correct, n)) if n else None,
                "natural_completions": natural, "natural_completion_rate": natural / n if n else None,
                "completion_wilson95": list(wilson(natural, n)) if n else None,
                "mean_tokens": mean(item["tokens"] for item in scores) if n else None,
            })
        tables[condition] = rows
    pattern = None
    if complete:
        identifiers = [item["id"] for item in pool]
        for cap in CAPS:
            on = [score_at_cap(graded[(identifier, "thinking_on")], cap)["correct"] for identifier in identifiers]
            off = [score_at_cap(graded[(identifier, "thinking_off")], cap)["correct"] for identifier in identifiers]
            paired.append({"cap": cap, "n": 30, **paired_comparison(on, off)})
        for row, adjusted in zip(paired, holm_adjust([row["exact_mcnemar_p"] for row in paired])):
            row["holm_adjusted_p"] = adjusted
        gains = {name: tables[name][-1]["correct"] - tables[name][1]["correct"] for name in CONDITIONS}
        pattern = {"gain_on": gains["thinking_on"], "gain_off": gains["thinking_off"],
                   "consistent_with_declared_pattern": gains["thinking_on"] >= 8 and gains["thinking_off"] <= 3}
    return {
        "complete_data": complete, "main_responses": len(indexed), "total_responses": len(records),
        "prefix_validation": validation, "tables": tables, "paired_comparisons": paired,
        "hypothesis_pattern": pattern, "token_248046_diagnostics": diagnostics, "short_wrong_answers": short_wrong,
        "inference_note": "Complete-data inference only at 30 questions per condition; partial tables are descriptive.",
    }


def measured_projection(plan, records):
    if not prefix_status(plan, records)["passed"]:
        raise ValueError("Projection requires all four successful prefix comparisons.")
    now = datetime.now(timezone.utc)
    pilot_ids = set(plan["prefix_validation"]["question_ids"])
    pilots = [row for row in records if row["problem_id"] in pilot_ids]
    rates, remaining_seconds, worst_seconds = {}, 0.0, 0.0
    for condition in CONDITIONS:
        long = [row for row in pilots if row["kind"] == "main" and row["condition"] == condition]
        elapsed = sum(row["seconds"] for row in long)
        tokens = sum(len(row["token_ids"]) for row in long)
        if elapsed <= 0 or tokens <= 0:
            raise ValueError("Invalid profiling durations.")
        rate = tokens / elapsed
        done = sum(row["kind"] == "main" and row["condition"] == condition for row in records)
        remaining = 30 - done
        rates[condition] = {"tokens_per_second": rate, "pilot_main_tokens": tokens,
                            "pilot_main_seconds": elapsed, "remaining_main_responses": remaining}
        remaining_seconds += remaining * mean(len(row["token_ids"]) for row in long) / rate
        worst_seconds += remaining * 8192 / rate
    total_seconds = sum(row["seconds"] for row in pilots)
    return {
        "measured_at_utc": now.isoformat(),
        "aggregate_pilot_tokens_per_second": sum(len(row["token_ids"]) for row in pilots) / total_seconds,
        "per_condition": rates,
        "peak_gpu_allocated_bytes": max(row["peak_gpu_allocated_bytes"] for row in pilots),
        "projected_remaining_seconds": remaining_seconds,
        "projected_finish_utc": (now + timedelta(seconds=remaining_seconds)).isoformat(),
        "all_remaining_responses_at_cap_finish_utc": (now + timedelta(seconds=worst_seconds)).isoformat(),
        "note": "Projection from two pilot questions, not a deadline. Rates may change with longer contexts.",
    }
