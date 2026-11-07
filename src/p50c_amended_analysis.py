"""Same paired statistics; explicit bound violations are false on correctness only."""

import hashlib

from budget_sweep_io import identity, json_bytes
from budget_sweep_scoring import CAPS, holm_adjust, paired_comparison, score_at_cap, wilson
from p50c_amended_scoring import amended_grade, closure_from_tokens
from p50c_completion_analysis import OUTCOMES, outcomes
from p50c_completion_io import require_edited_provenance


def amended_outcomes(plan, pool, rows, decode, declaration, amendment_hash):
    legacy = declaration["legacy_record_sha256"]
    old = [row for row in rows if identity(row) in legacy]
    for row in old:
        if hashlib.sha256(json_bytes(row)).hexdigest() != legacy[identity(row)]:
            raise ValueError("Pre-amendment response bytes changed.")
    result = outcomes(plan, pool, old, decode, edited=True)
    expected = {row["id"]: row["answer"] for row in pool}
    seeds = {row["id"]: row["seed"] for row in plan["questions_and_seeds"]}
    for row in rows:
        require_edited_provenance(row, plan)
        if identity(row) in legacy:
            continue
        key = row.get("problem_id")
        if (
            key not in declaration["remaining_ids"] or key in result
            or type(row.get("seed")) is not int or row["seed"] != seeds[key]
            or row.get("thinking") is not True or row.get("kind") != "main"
            or row.get("condition") != "thinking_on" or row.get("max_new_tokens") != 8192
            or row.get("expected") != expected[key] or row.get("amendment_sha256") != amendment_hash
        ):
            raise ValueError("Invalid amended response identity.")
        tokens = row["token_ids"]
        text = decode(tokens)
        graded, violation, characters = amended_grade(
            text, tokens, expected[key], plan["stopping"]["terminal_tokens_for_grading"],
        )
        if (
            row.get("text_bound_violation") is not violation
            or type(row.get("text_character_length")) is not int or row["text_character_length"] != characters
            or row.get("text") != (None if violation else text) or row.get("graded") != graded
            or graded["natural_eos_index"] is None and len(tokens) != 8192
        ):
            raise ValueError("Amended response flags/outcomes disagree with token reconstruction.")
        closed = closure_from_tokens(tokens, decode)
        if row.get("thinking_closed_at_cap") != closed:
            raise ValueError("Saved thinking closure disagrees with token prefixes.")
        result[key] = {
            "graded": graded,
            "caps": {str(k): {
                "natural_completion": score_at_cap(graded, k)["natural_completion"],
                "strict_correct_delivery": score_at_cap(graded, k)["correct"],
                "thinking_closed": closed[str(k)], "tokens": score_at_cap(graded, k)["tokens"],
            } for k in CAPS},
        }
    return result


def recompute_amended(plan, pool, records, originals, decode, declaration, amendment_hash):
    original = outcomes(plan, pool, originals, decode, edited=False)
    edited = amended_outcomes(plan, pool, records, decode, declaration, amendment_hash)
    ids = [row["id"] for row in plan["questions_and_seeds"]]
    if len(ids) != 30 or set(original) != set(ids):
        raise ValueError("All 30 fixed original counterparts are required.")
    complete = len(edited) == 30
    tables, paired = {}, {}
    for outcome in OUTCOMES:
        tables[outcome] = {}
        for variant, data in (("original", original), ("edited", edited)):
            values = []
            for cap in CAPS:
                count = sum(row["caps"][str(cap)][outcome] for row in data.values())
                n = len(data)
                values.append({
                    "cap": cap, "n": n, "successes": count, "rate": count / n if n else None,
                    "wilson95": wilson(count, n) if n else None,
                    "mean_tokens": sum(row["caps"][str(cap)]["tokens"] for row in data.values()) / n if n else None,
                })
            tables[outcome][variant] = values
        comparisons = []
        if complete:
            for cap in CAPS:
                effect = paired_comparison(
                    [edited[key]["caps"][str(cap)][outcome] for key in ids],
                    [original[key]["caps"][str(cap)][outcome] for key in ids],
                )
                comparisons.append({
                    "cap": cap, "n": 30, "edited_wins": effect["on_wins"], "original_wins": effect["off_wins"],
                    "ties": effect["ties"], "difference_percentage_points": 100 * (effect["on_wins"] - effect["off_wins"]) / 30,
                    "exact_mcnemar_p": effect["exact_mcnemar_p"],
                })
            for row, adjusted in zip(comparisons, holm_adjust([row["exact_mcnemar_p"] for row in comparisons])):
                row["holm_adjusted_p"] = adjusted
        paired[outcome] = comparisons
    diagnostics = {}
    for variant, rows in (("original", originals), ("edited", records)):
        affected = [{"id": row["problem_id"], "occurrences": row["token_ids"].count(248046)}
                    for row in rows if 248046 in row["token_ids"]]
        diagnostics[variant] = {"responses_with_248046": len(affected),
                                "occurrences": sum(row["occurrences"] for row in affected),
                                "affected_responses": affected}
    wrong, violations = [], []
    for row in records:
        graded = edited[row["problem_id"]]["graded"]
        violation = row.get("text_bound_violation", False)
        if violation:
            violations.append({
                "id": row["problem_id"], "character_length": row["text_character_length"],
                "natural_eos_index": graded["natural_eos_index"],
                "thinking_closed_at_8192": edited[row["problem_id"]]["caps"]["8192"]["thinking_closed"],
                "raw_text_file": row["raw_response"]["text.txt"]["file"],
            })
        if graded["natural_eos_index"] is not None and graded["natural_eos_index"] <= 1024 and not graded["strict_final_correct"]:
            text = row["text"] if not violation else (
                "[Approved text-bound violation: " + str(row["text_character_length"]) +
                " characters. Full raw text retained separately; strict correctness is false by amendment.]"
            )
            wrong.append({"id": row["problem_id"], "seed": row["seed"], "tokens": len(row["token_ids"]),
                          "parsed_final_answer": graded["final_answer"], "expected": row["expected"], "text": text})
    return {
        "complete_data": complete, "edited_main_responses": len(edited), "original_main_responses": 30,
        "missing_edited_ids": [key for key in ids if key not in edited], "tables": tables,
        "paired_comparisons": paired, "token_248046_diagnostics": diagnostics, "short_wrong_answers": wrong,
        "per_question": [{"id": key, "seed": plan["questions_and_seeds"][i]["seed"],
                          "original": original[key], "edited": edited.get(key)} for i, key in enumerate(ids)],
        "text_bound_violations": violations,
    }


def verify_legacy_outcomes(analysis, snapshot):
    current = {row["id"]: row for row in analysis["per_question"]}
    for row in snapshot["per_question"]:
        if row["edited"] is not None and json_bytes(current[row["id"]]) != json_bytes(row):
            raise ValueError("An outcome for one of the original 11 responses changed.")
