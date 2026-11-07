"""Document the frozen guard stop without changing inference or completion gates."""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from budget_sweep_io import ROOT, RunLock, atomic_bytes, atomic_json, json_bytes, project_path, read_json, sha256
from p50c_completion_diagnostic import BASELINE_DIAGNOSTIC, protected_snapshot
from p50c_completion_io import RUN
from p50c_completion_outputs import (
    diagnostic_markdown, load_saved_analysis, paired_markdown,
    short_wrong_markdown, tables_markdown,
)
from p50c_restore_audit import load_restore_audit, restoration_failure_report, write_restoration_failure


def render_partial_report(plan, records, summary, analysis, budget, restore_audit):
    if not restore_audit.get("passed"):
        return restoration_failure_report(plan, restore_audit)
    if analysis["complete_data"] or summary["status"] != "fatal_error" or not 0 < len(records) < 30:
        raise ValueError("This report is only for an incomplete fatal-error stop, not a completed comparison.")
    if summary.get("error") != {"type": "ValueError", "message": "Expected bounded decoded response text."}:
        raise ValueError("This report must not mislabel a different failure as the decoded-text guard.")
    missing = summary["missing_edited_ids"]
    missing_text = ", ".join("`" + item + "`" for item in missing)
    endpoint = {outcome: analysis["tables"][outcome]["edited"][-1]
                for outcome in ("natural_completion", "strict_correct_delivery", "thinking_closed")}
    paragraphs = [
        f"The approved p50c edited thinking-on sweep stopped with {len(records)} of 30 responses durably saved.",
        "The next scheduled item failed a frozen decoded-text validation guard, and that attempt's raw output was not persisted.",
        "No complete-sample comparison or planned paired tests are reported.",
        "Both post-run verification artifacts independently confirm restoration to the planned original tensor hashes.",
        "GSM8K TRAIN is likely in Qwen's training data (per-item inclusion is unverified), so this partial run is not a capability or benchmark claim.",
    ]
    return f"""# P50c Completion Report - Partial Guard Stop

{' '.join(paragraphs)}

## Question And Fixed Design

The intended comparison is natural EOS (primary), strict correct delivery
(secondary), and literal `</think>` closure (diagnostic) for the full published
p50c edit versus saved unedited thinking-on responses on the same 30 questions.
The study is **not complete** and does not resolve that comparison.

The approved plan is unchanged: pinned Qwen3.5-0.8B, the same 30 GSM8K `main`
TRAIN items, seeds 0-29, thinking on, the fixed step-by-step instruction,
temperature 0.6, top-p 0.95, top-k 20, generated-token presence penalty 1.5,
batch size 1, a single 8,192-token call per completed item, and post-hoc caps
512/1024/2048/4096/8192. EOS is 248044 only; 248046 remains diagnostic.
CUBLAS_WORKSPACE_CONFIG=:4096:8 precedes CUDA; strict deterministic algorithms
remain enabled with warn_only=False and torch threads 1/1. No wall-clock limit
stopped a generation. Both published edited tensor hashes were verified before
each of the {len(records)} accepted responses. No weights were saved.

Published edit attribution: Rikiya Takehi et al., `RikiyaT/lm_understand-logs`,
commit `{plan['edit']['revision']}`, layers 15 and 19, strength 1.0, CC BY 4.0.
The edit and sampling were not retuned after seeing these records.

## Partial Descriptive Records

Only completed, validated records appear here. At 8,192 tokens, the edited
records contain {endpoint['natural_completion']['successes']}/{len(records)}
natural completions, {endpoint['strict_correct_delivery']['successes']}/{len(records)}
strict correct deliveries, and {endpoint['thinking_closed']['successes']}/{len(records)}
thinking closures. These are **not 30-question estimates**.

**Denominators differ:** the original reference contains all 30 saved questions;
the edited column contains only the first {len(records)} successfully committed
questions. Do not interpret the columns as a completed matched comparison.
Stopping at a specific failed item can make this prefix unrepresentative.
Wilson intervals describe the observed records and do not repair that selection.
The failed output is missing, not imputed as correct, incorrect or completed.

{tables_markdown(analysis)}

## Planned Paired Effects

{paired_markdown(analysis)}

The planned n=30 exact McNemar/Holm comparisons are withheld. No replacement
subset test, missing-output imputation, article-estimate validation, or success
claim is substituted. Natural EOS remains primary, distinct from thinking closure.

## Short Wrong Answers

Within the saved edited records only:

{short_wrong_markdown(analysis)}

## Token 248046 Diagnostic

{diagnostic_markdown(analysis)}

## Failure And Accounting

The first uncompleted scheduled item is `{missing[0]}` (item
{len(records) + 1}, seed {len(records)}). The worker recorded
`ValueError: Expected bounded decoded response text.` The unchanged grader
requires a string with 1-200,000 decoded characters. Saved logs do not identify
which part of this guard failed; **the failed raw text and token IDs were not
persisted**, because validation ran before the generation record was returned.
No empty-versus-oversized-output diagnosis or model mechanism is inferred.

There are {len(records)} durable completed responses, one unpersisted failed
attempt, and {29 - len(records)} items never started. The {len(missing)} currently uncompleted
IDs (including that failed item) are: {missing_text}.

The ledger accounts for **{budget['accounted_tokens']:,}/250,000 tokens**:
{budget['completed_actual_tokens']:,} observed completed tokens plus
{budget['unreconciled_reserved_tokens']:,} conservatively charged reserved tokens
for the failed attempt. Its actual length is unavailable; the reservation is
not refunded. Completed work is not repeated and no further generation starts.

## Stop

**Execution stopped on a frozen validation guard; the 30-response sweep is
incomplete.** Worker status is `fatal_error`, not a completed-study claim.
The frozen grader, text bound, EOS, determinism, sampling, question order and
token accounting are not changed, and the deterministic failed item is not
silently retried.

**Restoration PASSED independently in both files.** In `summary.json` and
`{restore_audit['latest_history_file'].split('/')[-1]}`, `verified_after_restore`
is true and `original_after_restore` exactly matches the planned base hashes
for both targets. This conclusion uses the flags and full hash maps, not any
status field. Full evidence is in `restoration_audit.json`. Missing/mismatched
evidence would instead be reported as RESTORATION NOT VERIFIED regardless of status.

The full-study finalizer correctly refuses partial data. The completed-study
figure, notebook 07 and paired inference are therefore **not finalized**.
Their completion gates remain strict. Any raw-token-first diagnostic capture,
guard/failure-policy amendment or further generation requires fresh explicit
approval; none occurs here.

No held-out generation/scoring, new downloads or installations, model/tensor
saves, commits or pushes occur. Original checkpoint bytes and protected prior
reports, records, notebooks and scorers remain unchanged.
"""


def main():
    with RunLock(RUN / "worker.lock"):
        plan, _, records, summary, analysis, budget = load_saved_analysis(RUN, require_complete=False)
        restore_audit = load_restore_audit(RUN, plan)
        if not restore_audit["passed"]:
            write_restoration_failure(RUN, plan, restore_audit)
            raise RuntimeError("Restoration failed; the Stop section reports it independently of status.")
        if budget != summary["budget"]:
            raise RuntimeError("Partial accounting differs from its durable ledger.")
        for key, value in analysis.items():
            if json_bytes({"value": value}) != json_bytes({"value": summary[key]}):
                raise RuntimeError("Partial summary differs from frozen raw recomputation.")
        expected = read_json(BASELINE_DIAGNOSTIC)["protected_artifact_sha256"]
        if protected_snapshot() != expected:
            raise RuntimeError("A protected baseline artifact changed.")
        for relative, digest in (
            (plan["model"]["weights_file"], plan["model"]["checkpoint_sha256"]),
            (plan["dataset"]["source_file"], plan["dataset"]["sha256"]),
            (plan["dataset"]["metadata_file"], plan["dataset"]["metadata_sha256"]),
        ):
            if sha256(project_path(relative)) != digest:
                raise RuntimeError("A pinned original cached input changed.")
        text = render_partial_report(plan, records, summary, analysis, budget, restore_audit).encode("utf-8")
        targets = (ROOT / "P50C_COMPLETION_REPORT.md", RUN / "report.md",
                   RUN / "analysis_recomputed.json", RUN / "restoration_audit.json", RUN / "partial_validation.json")
        if any(path.exists() for path in targets):
            raise FileExistsError("Refusing to replace an existing partial artifact.")
        environment = os.environ.copy()
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                           HF_HUB_DISABLE_IMPLICIT_TOKEN="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", UV_OFFLINE="1")
        tests = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=ROOT, env=environment,
                               capture_output=True, text=True)
        if tests.returncode:
            print(tests.stdout, tests.stderr)
            raise RuntimeError("Offline regression gates failed.")
        count = re.search(r"Ran ([0-9]+) tests", tests.stderr)
        if not count:
            raise RuntimeError("Test count could not be confirmed.")
        subprocess.run(["uv", "pip", "check", "--python", sys.executable], cwd=ROOT, env=environment, check=True)
        if protected_snapshot() != expected:
            raise RuntimeError("A protected prior artifact changed during the gates.")
        if "torch" in sys.modules or "transformers" in sys.modules:
            raise RuntimeError("Partial reporting imported the model runtime.")
        atomic_bytes(targets[0], text)
        atomic_bytes(targets[1], text)
        atomic_json(targets[2], analysis)
        atomic_json(targets[3], restore_audit)
        atomic_json(targets[4], {
            "validated_utc": datetime.now(timezone.utc).isoformat(), "status": summary["status"],
            "complete_comparison": False, "edited_records": len(records), "budget": budget,
            "summary_and_latest_history_independently_confirm_restore": True,
            "latest_weight_verification_file": restore_audit["latest_history_file"],
            "all_saved_record_hashes_and_prompts_verified": True,
            "partial_summary_matches_frozen_raw_recomputation": True, "planned_paired_inference_withheld": True,
            "failed_raw_response_not_persisted": True, "failed_reservation_not_refunded": True,
            "original_checkpoint_and_dataset_unchanged": True, "all_protected_baseline_artifacts_unchanged": True,
            "all_offline_tests_pass": True, "offline_test_count": int(count.group(1)),
            "installed_dependencies_compatible": True, "figure_notebook_not_finalized": True,
            "inference_performed_by_reporter": False,
            "artifact_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in targets[:4]},
        })
        print(json.dumps({"status": "PARTIAL_GUARD_STOP", "records": len(records), "budget": budget,
                          "restoration_confirmed_in_both_files": True, "tests": int(count.group(1)),
                          "report": str(targets[0]), "inference_performed": False}, indent=2))


if __name__ == "__main__":
    main()
