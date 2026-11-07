"""Predeclared controls, probe splits, and a gate that limits further compute."""

import hashlib
import json
from dataclasses import asdict, dataclass

from math_probe import PROBLEMS


@dataclass(frozen=True)
class Condition:
    name: str
    thinking: bool
    layers: tuple = ()
    strength: float = 0.0
    prompt_style: str = "step_by_step"


CONDITIONS = (
    Condition("base_thinking", True),
    Condition("base_no_thinking", False),
    Condition("published_both", True, (15, 19), 1.0),
    Condition("layer15_only", True, (15,), 1.0),
    Condition("layer19_only", True, (19,), 1.0),
    Condition("half_strength", True, (15, 19), 0.5),
)
DEVELOPMENT = PROBLEMS
HOLDOUT_MATH = (
    {"id": "math01", "question": "A 200 dollar item is discounted by 20%, then taxed at 10%. What is its final price in dollars?", "answer": "176"},
    {"id": "math02", "question": "Solve 9*x - 7 = 65. What is x?", "answer": "8"},
    {"id": "math03", "question": "What is three eighths of 64, plus 5?", "answer": "29"},
    {"id": "math04", "question": "A vehicle travels at 72 km per hour for 2 hours 30 minutes. How many kilometers does it travel?", "answer": "180"},
    {"id": "math05", "question": "What is the arithmetic mean of 12, 18, 24, and 30?", "answer": "21"},
    {"id": "math06", "question": "A rectangle has sides of length 17 and 9. What is its area?", "answer": "153"},
    {"id": "math07", "question": "What is the perimeter of a square with side length 13?", "answer": "52"},
    {"id": "math08", "question": "What is the remainder when 137 is divided by 9?", "answer": "2"},
    {"id": "math09", "question": "An arithmetic sequence starts 7, 12, 17, 22. What is its tenth term?", "answer": "52"},
    {"id": "math10", "question": "What is the sum of the integers from 1 through 30, inclusive?", "answer": "465"},
    {"id": "math11", "question": "What is 3^5 - 2^4?", "answer": "227"},
    {"id": "math12", "question": "Red and blue counters are in the ratio 2:5. There are 49 counters total. How many are red?", "answer": "14"},
    {"id": "math13", "question": "One tap fills a tank in 6 hours and another in 3 hours. How many hours do they need together?", "answer": "2"},
    {"id": "math14", "question": "How many ordered outcomes of rolling two six-sided dice have a sum of 7?", "answer": "6"},
    {"id": "math15", "question": "What is the simple interest on 600 dollars at 5% per year for 2 years, in dollars?", "answer": "60"},
    {"id": "math16", "question": "Convert 3 meters and 45 centimeters into centimeters.", "answer": "345"},
    {"id": "math17", "question": "What is -8 + 3*(5 - 9)?", "answer": "-20"},
    {"id": "math18", "question": "A triangle has base 14 and perpendicular height 9. What is its area?", "answer": "63"},
    {"id": "math19", "question": "How many positive multiples of 6 are less than or equal to 100?", "answer": "16"},
    {"id": "math20", "question": "What is the sum of the decimal digits of 70839?", "answer": "27"},
)
HOLDOUT_CONTROLS = tuple(
    {"id": f"control{i:02}", "question": question, "answer": answer, "grader": "exact"}
    for i, (question, answer) in enumerate((
        ("Reply with exactly READY and nothing else.", "READY"),
        ("Convert ROBOT to lowercase. Reply with only the converted word.", "robot"),
        ("Copy exactly this string and nothing else: alpha-17", "alpha-17"),
        ("Choose the first word in this list: red, blue. Reply with only that word.", "red"),
        ("Is 4 an even number? Reply with exactly yes or no, in lowercase.", "yes"),
        ('Reply with exactly {"ok":true} and nothing else, with no spaces.', '{"ok":true}'),
        ("Join sun and moon with a hyphen. Reply with only the joined string.", "sun-moon"),
        ("Sort cat, ant, bee alphabetically. Reply with exactly three words separated by comma and space.", "ant, bee, cat"),
        ("What is the capital of France? Reply with only the city's name.", "Paris"),
        ("Give the opposite of hot. Reply with only one lowercase word.", "cold"),
    ), start=1)
)


def split_fingerprint(problems):
    payload = json.dumps(problems, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def summarize(records):
    summaries = {}
    for name in sorted({row["condition"] for row in records}):
        rows = [row for row in records if row["condition"] == name]
        count = len(rows)
        summaries[name] = {
            "count": count,
            "delivered_correct": sum(row["delivered_correct"] for row in rows),
            "delivered_correct_rate": sum(row["delivered_correct"] for row in rows) / count,
            "final_correct_rate": sum(row["correct"] for row in rows) / count,
            "natural_stop_rate": sum(row["natural_stop"] for row in rows) / count,
            "cap_rate": sum(row["hit_token_limit"] for row in rows) / count,
            "time_budget_rate": sum(row["stop_reason"] == "time_budget" for row in rows) / count,
            "mean_tokens": sum(row["generated_tokens"] for row in rows) / count,
            "mean_seconds": sum(row["seconds"] for row in rows) / count,
        }
    return summaries


def holdout_gate(records, *, complete, minimum_gain=0.25):
    """Do not spend holdout compute unless an edit beats BOTH simple baselines."""
    if type(complete) is not bool:
        raise ValueError("complete must be a boolean.")
    if type(minimum_gain) not in (int, float) or not 0 < minimum_gain <= 1:
        raise ValueError("minimum_gain must be in (0, 1].")
    table = summarize(records)
    decision = {
        "allow_holdout": False,
        "selected_condition": None,
        "minimum_gain": minimum_gain,
        "summary": table,
        "reason": "Incomplete or time-limited pilot; no selection is allowed.",
    }
    if not complete or any(row["stop_reason"] == "time_budget" for row in records):
        return decision
    if not {"base_thinking", "base_no_thinking"}.issubset(table):
        return decision
    baseline_keys = {
        (row["problem_id"], row["seed"])
        for row in records if row["condition"] == "base_thinking"
    }
    if not baseline_keys:
        return decision
    for name in (item.name for item in CONDITIONS):
        rows = [row for row in records if row["condition"] == name]
        keys = {(row["problem_id"], row["seed"]) for row in rows}
        if keys != baseline_keys or len(rows) != len(baseline_keys):
            return decision
    best_baseline = max(
        table[name]["delivered_correct_rate"] for name in ("base_thinking", "base_no_thinking")
    )
    candidates = []
    for condition in CONDITIONS[2:]:
        keys = {
            (row["problem_id"], row["seed"])
            for row in records if row["condition"] == condition.name
        }
        if keys != baseline_keys:
            return decision
        score = table[condition.name]
        if (
            score["delivered_correct_rate"] >= best_baseline + minimum_gain
            and score["natural_stop_rate"] >= table["base_thinking"]["natural_stop_rate"]
        ):
            candidates.append(condition)
    if not candidates:
        decision["reason"] = (
            f"No edit improves delivered-correct rate by at least {minimum_gain:.0%} "
            "over the better of thinking-on and thinking-off. Stop this pilot; holdout stays unused."
        )
        return decision
    selected = max(
        candidates,
        key=lambda item: (
            table[item.name]["delivered_correct_rate"],
            table[item.name]["natural_stop_rate"],
            -table[item.name]["mean_tokens"],
        ),
    )
    decision.update(
        allow_holdout=True,
        selected_condition=asdict(selected),
        reason="A predeclared edit passes the development gate. Freeze it before held-out evaluation.",
    )
    return decision
