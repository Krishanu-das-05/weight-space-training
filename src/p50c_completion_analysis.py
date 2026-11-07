"""Frozen paired analysis: reject unverified edit provenance before scoring."""

from budget_sweep_io import project_path, read_json, sha256
from budget_sweep_scoring import CAPS, grade_generation, holm_adjust, paired_comparison, score_at_cap, wilson
from p50c_completion_diagnostic import closure_at_caps
from p50c_completion_io import require_edited_provenance

OUTCOMES = ("natural_completion", "strict_correct_delivery", "thinking_closed")


def original_records(plan):
    rows = []
    for item in plan["baseline"]["selection"]:
        path = project_path(item["file"])
        if sha256(path) != item["sha256"]:
            raise ValueError("A paired original record changed.")
        rows.append(read_json(path))
    return rows


def outcomes(plan, pool, rows, decode, *, edited):
    expected = {row["id"]: row["answer"] for row in pool}
    seeds = {row["id"]: row["seed"] for row in plan["questions_and_seeds"]}
    result = {}
    for row in rows:
        if edited:
            require_edited_provenance(row, plan)
        key = row.get("problem_id")
        if (
            key not in expected or key in result or type(row.get("seed")) is not int or row.get("seed") != seeds[key]
            or row.get("thinking") is not True or row.get("kind") != "main"
            or row.get("condition") != "thinking_on" or row.get("max_new_tokens") != 8192
            or row.get("expected") != expected[key]
        ):
            raise ValueError("Invalid or duplicated paired response identity.")
        tokens = row.get("token_ids")
        graded = grade_generation(
            row.get("text"), tokens, expected[key], thinking=True, eos_ids={248044},
            terminal_tokens=plan["stopping"]["terminal_tokens_for_grading"],
        )
        if graded["natural_eos_index"] is None and len(tokens) != 8192:
            raise ValueError("An unfinished main response must contain exactly 8192 generated tokens.")
        if decode(tokens) != row["text"]:
            raise ValueError("Saved text disagrees with the pinned token decoder.")
        closure = closure_at_caps(tokens, decode)
        result[key] = {
            "graded": graded,
            "caps": {str(k): {
                "natural_completion": score_at_cap(graded, k)["natural_completion"],
                "strict_correct_delivery": score_at_cap(graded, k)["correct"],
                "thinking_closed": closure[str(k)],
                "tokens": score_at_cap(graded, k)["tokens"],
            } for k in CAPS},
        }
    return result


def recompute(plan, pool, edited_records, originals, decode):
    original = outcomes(plan, pool, originals, decode, edited=False)
    edited = outcomes(plan, pool, edited_records, decode, edited=True)
    ids = [row["id"] for row in plan["questions_and_seeds"]]
    if len(ids) != 30 or set(original) != set(ids):
        raise ValueError("The complete 30-question original pairing is required.")
    complete = len(edited) == 30
    tables, paired = {}, {}
    for outcome in OUTCOMES:
        tables[outcome] = {}
        for variant, data in (("original", original), ("edited", edited)):
            rows = []
            for cap in CAPS:
                count = sum(row["caps"][str(cap)][outcome] for row in data.values())
                n = len(data)
                rows.append({
                    "cap": cap, "n": n, "successes": count, "rate": count / n if n else None,
                    "wilson95": wilson(count, n) if n else None,
                    "mean_tokens": sum(row["caps"][str(cap)]["tokens"] for row in data.values()) / n if n else None,
                })
            tables[outcome][variant] = rows
        comparisons = []
        if complete:
            for cap in CAPS:
                effect = paired_comparison(
                    [edited[key]["caps"][str(cap)][outcome] for key in ids],
                    [original[key]["caps"][str(cap)][outcome] for key in ids],
                )
                comparisons.append({
                    "cap": cap, "n": 30, "edited_wins": effect["on_wins"],
                    "original_wins": effect["off_wins"], "ties": effect["ties"],
                    "difference_percentage_points": 100 * (effect["on_wins"] - effect["off_wins"]) / 30,
                    "exact_mcnemar_p": effect["exact_mcnemar_p"],
                })
            for row, adjusted in zip(comparisons, holm_adjust([row["exact_mcnemar_p"] for row in comparisons])):
                row["holm_adjusted_p"] = adjusted
        paired[outcome] = comparisons
    diagnostics = {}
    for variant, records in (("original", originals), ("edited", edited_records)):
        affected = [{"id": row["problem_id"], "occurrences": row["token_ids"].count(248046)}
                    for row in records if 248046 in row["token_ids"]]
        diagnostics[variant] = {"responses_with_248046": len(affected),
                                "occurrences": sum(row["occurrences"] for row in affected),
                                "affected_responses": affected}
    wrong = [
        {"id": row["problem_id"], "seed": row["seed"], "tokens": len(row["token_ids"]),
         "parsed_final_answer": edited[row["problem_id"]]["graded"]["final_answer"],
         "expected": row["expected"], "text": row["text"]}
        for row in edited_records
        if edited[row["problem_id"]]["graded"]["natural_eos_index"] is not None
        and edited[row["problem_id"]]["graded"]["natural_eos_index"] <= 1024
        and not edited[row["problem_id"]]["graded"]["strict_final_correct"]
    ]
    return {
        "complete_data": complete, "edited_main_responses": len(edited), "original_main_responses": 30,
        "missing_edited_ids": [key for key in ids if key not in edited],
        "tables": tables, "paired_comparisons": paired, "token_248046_diagnostics": diagnostics,
        "short_wrong_answers": wrong,
        "per_question": [{"id": key, "seed": plan["questions_and_seeds"][i]["seed"],
                          "original": original[key], "edited": edited.get(key)} for i, key in enumerate(ids)],
    }
