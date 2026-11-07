"""Read-only raw-record tables and a reproducible, cautiously worded report."""

import argparse
import json
import re
from pathlib import Path

from budget_sweep_analysis import CONDITIONS, recompute
from budget_sweep_io import ROOT, RecordStore, atomic_bytes, read_json
from budget_sweep_plan import load_run

STATUSES = {
    "complete", "running", "initializing", "fatal_error", "interrupted",
    "prefix_mismatch", "token_budget_exhausted", "target_verification_failed",
}


def load_saved_run(directory):
    directory = Path(directory).resolve()
    plan, pool = load_run(directory)
    store = RecordStore(directory, plan, pool)
    if store.journal.stat().st_size > 50_000_000:
        raise ValueError("Journal exceeds its bounded size.")
    records = [json.loads(line) for line in store.journal.read_text(encoding="utf-8").splitlines()]
    if records != store.records():
        raise ValueError("Journal is not synchronized with completed records; do not analyze a live write.")
    summary = read_json(directory / "summary.json")
    if summary.get("plan_sha256") != store.plan_hash:
        raise ValueError("Summary refers to another plan.")
    if summary.get("status") not in STATUSES:
        raise ValueError("Unknown run status.")
    checks = summary.get("target_hash_checks")
    if (
        not isinstance(checks, dict) or set(checks) != {"before", "after", "verified_after"}
        or type(checks.get("verified_after")) is not bool
    ):
        raise ValueError("Invalid target-verification metadata.")
    for stage in ("before", "after"):
        hashes = checks.get(stage)
        if hashes is not None and (
            not isinstance(hashes, dict) or set(hashes) != set(plan["model"]["target_base_sha256"])
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values())
        ):
            raise ValueError("Invalid target hashes.")
    if checks["verified_after"] and not checks["before"] == checks["after"] == plan["model"]["target_base_sha256"]:
        raise ValueError("Inconsistent target-verification claim.")
    runtime = summary.get("runtime")
    expected_runtime = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "set_before_cuda_initialization": True,
        "deterministic_algorithms": True, "deterministic_warn_only": False,
        "torch_threads": 1, "torch_interop_threads": 1, "wall_time_limit": None,
    }
    if not isinstance(runtime, dict) or (runtime and (
        set(runtime) != set(expected_runtime)
        or any(type(runtime[name]) is not type(value) or runtime[name] != value for name, value in expected_runtime.items())
    )):
        raise ValueError("Invalid deterministic runtime metadata.")
    if summary.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS environment metadata mismatch.")
    return plan, pool, records, summary, recompute(plan, pool, records), store.budget(records)


def proportion(value):
    return "NA" if value is None else f"{100 * value:.1f}%"


def interval(values):
    return "NA" if values is None else f"{100 * values[0]:.1f}-{100 * values[1]:.1f}%"


def tables_markdown(derived):
    lines = ["| Thinking | Cap | n | Correct Delivery (95% Wilson) | Natural Completion (95% Wilson) | Mean Tokens |",
             "|---|---:|---:|---|---|---:|"]
    for condition in CONDITIONS:
        for row in derived["tables"][condition]:
            tokens = "NA" if row["mean_tokens"] is None else f"{row['mean_tokens']:.1f}"
            lines.append(
                f"| {condition.removeprefix('thinking_')} | {row['cap']} | {row['n']} | "
                f"{row['correct']}/{row['n']} = {proportion(row['accuracy'])} ({interval(row['accuracy_wilson95'])}) | "
                f"{row['natural_completions']}/{row['n']} = {proportion(row['natural_completion_rate'])} "
                f"({interval(row['completion_wilson95'])}) | {tokens} |"
            )
    return "\n".join(lines)


def paired_markdown(derived):
    if not derived["complete_data"]:
        return "Not performed: the planned complete set of 30 matched questions is unavailable."
    lines = ["| Cap | On Wins | Off Wins | Ties | Exact McNemar p | Holm p |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in derived["paired_comparisons"]:
        lines.append(f"| {row['cap']} | {row['on_wins']} | {row['off_wins']} | {row['ties']} | "
                     f"{row['exact_mcnemar_p']:.6g} | {row['holm_adjusted_p']:.6g} |")
    return "\n".join(lines)


def diagnostics_markdown(derived):
    lines = ["| Thinking | Records | Responses Containing 248046 | Token 248046 Occurrences |",
             "|---|---|---:|---:|"]
    for condition in CONDITIONS:
        for kind in ("main", "prefix_validation_short", "all"):
            values = derived["token_248046_diagnostics"][condition].get(kind, {})
            lines.append(f"| {condition.removeprefix('thinking_')} | {kind} | "
                         f"{values.get('responses_with_248046', 0)}/{values.get('responses', 0)} | "
                         f"{values.get('token_248046_occurrences', 0)} |")
    return "\n".join(lines)


def fenced(text, limit=700):
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", str(text))[:limit]
    longest = max((len(value) for value in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return fence + "text\n" + text + "\n" + fence


def public_summary(derived, status):
    if derived["complete_data"] and status == "complete":
        on, off = derived["tables"]["thinking_on"], derived["tables"]["thinking_off"]
        first = "We compared thinking on and off on 30 GSM8K TRAIN development questions using the same unedited Qwen3.5-0.8B checkpoint and one seed per question."
        second = "One completed trajectory per question and condition was saved with an 8,192-token maximum and counted correct only after natural EOS and strict boxed-integer grading."
        third = (f"At 1,024 and 8,192 tokens, thinking-on accuracy was {proportion(on[1]['accuracy'])} and "
                 f"{proportion(on[-1]['accuracy'])}, compared with {proportion(off[1]['accuracy'])} and "
                 f"{proportion(off[-1]['accuracy'])} with thinking off.")
    else:
        first = "We attempted a fixed token-budget comparison of thinking on and off on 30 GSM8K TRAIN development questions using unedited Qwen3.5-0.8B."
        second = f"The run stopped with status {status.replace('_', ' ')}, before a complete 30-question comparison was available."
        third = f"Only {derived['main_responses']} long responses and {derived['total_responses'] - derived['main_responses']} short validation responses are available, so the planned complete-data hypothesis comparison was not performed."
    fourth = "GSM8K TRAIN is likely to be in Qwen's training data, although per-item inclusion is unverified."
    fifth = "These observations describe this small fixed experiment, not new capability, a benchmark score, or generalization to other models or questions."
    return " ".join((first, second, third, fourth, fifth))


def write_report(directory):
    directory = Path(directory).resolve()
    plan, pool, records, saved, derived, budget = load_saved_run(directory)
    report = [
        "# Accuracy Vs. Token Budget: Unedited Qwen3.5-0.8B\n",
        public_summary(derived, saved["status"]), "\n## Question\n",
        "Under a fixed step-by-step instruction, does thinking-on correct delivery rise substantially with budget while thinking-off stays approximately flat?",
        "\n## Fixed Design\n",
        f"- Model: `{plan['model']['repo']}`, revision `{plan['model']['revision']}`, BF16, eager attention, CUDA, batch 1.",
        f"- Data: 30 integer-answer GSM8K `main` TRAIN items, MIT, revision `{plan['dataset']['revision']}`, sampling seed 20261106; seeds 0-29 are paired across conditions.",
        f"- Data SHA-256: `{plan['dataset']['sha256']}`; curated pool SHA-256: `{plan['dataset']['dev_sha256']}`.",
        "- Unchanged step-by-step prompt and sampling: temperature 0.6, top-p 0.95, top-k 20, generated-token presence penalty 1.5, repetition penalty 1.0.",
        "- One 8192-token maximum generation per question/condition; all five caps are derived afterwards. Four real 1024-token prefix runs reuse their four long counterparts in the main set.",
        "- Strict natural EOS is 248044 only. Token 248046 is diagnostic only. A box in unfinished reasoning is not correct delivery.",
        "- Accuracy/completion have 95% pointwise Wilson intervals; mean tokens = mean(min(length,cap)), without a Wilson interval. Exact paired McNemar plus Holm adjustment covers five cap comparisons.",
        "- No time-based generation stop; fixed threads 1/1 and strict deterministic algorithms with no warn-only fallback.",
        f"- CUBLAS_WORKSPACE_CONFIG: `{saved['CUBLAS_WORKSPACE_CONFIG']}`; before-CUDA setup and runtime policy: `{saved.get('runtime', {})}`.",
        "- No held-out question was generated or scored; no weight edits, model saves, package installation, or new data/model downloads.",
        f"- Run: `{directory.relative_to(ROOT).as_posix()}`. Plan and implementation source hashes are saved with explicit approval.",
        "\n## Results\n", f"Run status: **{saved['status']}**; main responses: **{derived['main_responses']}/60**.",
        tables_markdown(derived),
        f"\n![Accuracy and natural completion vs token budget]({directory.relative_to(ROOT).as_posix()}/accuracy_vs_token_budget.png)",
        "\nMean tokens is counterfactual consumption at each cap, not additional compute spent after the single long generation.",
        "\n## Paired Effects\n", paired_markdown(derived),
    ]
    pattern = derived["hypothesis_pattern"]
    if pattern is not None:
        report.append(
            f"\nFrom 1024 to 8192, thinking on gains {pattern['gain_on']}/30 correct answers and thinking off gains "
            f"{pattern['gain_off']}/30. The predeclared on>=8/30 and off<=3/30 pattern is "
            f"**{'consistent' if pattern['consistent_with_declared_pattern'] else 'not consistent'}** with these data."
        )
    report += ["\n## Truncation And Runtime\n", fenced(json.dumps(derived["prefix_validation"], indent=2), 12000)]
    if "truncation_milestone" in saved:
        report += [fenced(json.dumps(saved["truncation_milestone"], indent=2), 12000)]
    if records:
        longest = max(records, key=lambda row: row["seconds"])
        report.append(
            f"\nLongest recorded generation wall time: {longest['seconds']:.2f} seconds, "
            f"`{longest['problem_id']}` / `{longest['condition']}` / `{longest['kind']}`, "
            f"{len(longest['token_ids'])} tokens. Wall time includes any host pauses or delays; "
            "their cause is not diagnosed by these logs. All recorded durations are retained. "
            "An initial ETA does not account for later pauses and is not a deadline."
        )
    report += [
        "\nThroughput estimates use measured generation wall time; GPU memory is peak PyTorch allocated memory, not total device usage. ETA is a projection, never a stop condition.",
        "\n## Token 248046 Diagnostics\n", diagnostics_markdown(derived),
        "\nEvery affected response is identified in summary.json and carries an occurrence count in records.jsonl. Main and validation counts are separate to avoid double-counting the main comparison. No stop or scoring rule changed.",
    ]
    for condition in CONDITIONS:
        affected = derived["token_248046_diagnostics"][condition].get("all", {}).get("affected_ids", [])
        if affected:
            report.append("\nAffected responses (" + condition + "):")
            report += [f"- `{item['problem_id']}` / `{item['kind']}`: {item['count']} occurrences." for item in affected]
    report.append("\n## Short Wrong Answers\n")
    if not derived["short_wrong_answers"]:
        report.append("None among the completed main responses under the predeclared natural-EOS <=1024 rule.")
    for row in derived["short_wrong_answers"]:
        report += [f"\n### {row['problem_id']} / {row['condition']}\n",
                   f"Seed {row['seed']}; {row['tokens']} tokens; parsed final `{row['final_answer']}`; expected `{row['expected']}`.",
                   fenced(next(item["question"] for item in pool if item["id"] == row["problem_id"]), 5000),
                   "Response excerpt (first 700 characters; full raw text remains in records.jsonl):",
                   fenced(row["text"])]
    report += ["\n## Limitations\n"]
    report += ["- " + value for value in plan["analysis"]["limitations"]]
    report += ["- " + plan["analysis"]["power"],
               "- The deterministic prefix check covers two questions in both conditions, not a proof for every input.",
               "\n## Stop\n", f"Status: **{saved['status']}**.",
               f"Final summary timestamp: `{saved['updated_utc']}`. Known completed generation time: "
               f"{sum(row['seconds'] for row in records) / 3600:.2f} hours (excludes setup and postprocessing).",
               f"Budget: {budget['completed_actual_tokens']:,} known completed tokens + "
               f"{budget['unreconciled_reserved_tokens']:,} unreconciled reserved tokens = "
               f"**{budget['accounted_tokens']:,}/{budget['limit']:,}** conservatively accounted.",
               f"Original target tensor checks: {json.dumps(saved['target_hash_checks'], sort_keys=True)}.",
               fenced(json.dumps(saved["error"])) if saved.get("error") else "No runtime error recorded.",
               "\nNo further generation or experiment extension is scheduled."]
    text = "\n\n".join(report) + "\n"
    atomic_bytes(ROOT / "BUDGET_SWEEP_REPORT.md", text.encode("ascii", errors="backslashreplace"), replace=True)
    archived = text.replace(directory.relative_to(ROOT).as_posix() + "/accuracy_vs_token_budget.png",
                            "accuracy_vs_token_budget.png")
    atomic_bytes(directory / "BUDGET_SWEEP_REPORT.md", archived.encode("ascii", errors="backslashreplace"), replace=True)
    return derived


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    write_report(args.run)
