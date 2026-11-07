"""Independent post-run restoration evidence; status is never proof of cleanup."""

import json
import re
from pathlib import Path

from budget_sweep_io import ROOT, atomic_bytes, project_path, read_json, sha256
from budget_sweep_outputs import fenced


def compare_restore_evidence(plan, summary, history):
    expected = plan["model"]["target_base_sha256"]

    def evidence(document):
        checks = document.get("target_hash_checks", {}) if isinstance(document, dict) else {}
        if not isinstance(checks, dict):
            checks = {}
        actual = checks.get("original_after_restore")
        matches = isinstance(actual, dict) and actual == expected
        safe_actual = {}
        for name in expected:
            value = actual.get(name) if isinstance(actual, dict) else None
            safe_actual[name] = value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else "MISSING_OR_INVALID_SHA256"
        return {
            "verified_after_restore_is_true": checks.get("verified_after_restore") is True,
            "original_after_restore_matches_planned_base": matches,
            "observed_original_after_restore": safe_actual,
        }

    result = {"planned_base_hashes": dict(expected), "summary": evidence(summary), "latest_history": evidence(history)}
    result["passed"] = all(
        row["verified_after_restore_is_true"] and row["original_after_restore_matches_planned_base"]
        for row in (result["summary"], result["latest_history"])
    )
    return result


def load_restore_audit(directory, plan):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("Restoration evidence must be project-local.")
    paths = sorted(directory.glob("weight_verification_*.json"))
    summary_path = directory / "summary.json"
    summary, history = {}, {}
    issues = []
    latest = paths[-1] if paths else None
    for label, path in (("summary", summary_path), ("latest_history", latest)):
        try:
            if path is None:
                raise ValueError("Missing verification history.")
            safe = project_path(path.relative_to(ROOT).as_posix())
            value = read_json(safe)
            if label == "summary":
                summary = value
            else:
                history = value
        except (OSError, ValueError):
            issues.append(label + "_missing_or_invalid")
    result = compare_restore_evidence(plan, summary, history)
    result.update(
        summary_file=summary_path.relative_to(ROOT).as_posix(),
        latest_history_file=latest.relative_to(ROOT).as_posix() if latest else None,
        issues=issues, status_used_as_restoration_proof=False,
    )
    expected_plan = sha256(directory / "plan.json")
    if summary.get("plan_sha256") != expected_plan or history.get("plan_sha256") != expected_plan:
        result["issues"].append("verification_evidence_plan_identity_mismatch")
    result["passed"] = result["passed"] and not result["issues"]
    return result


def restoration_failure_report(plan, audit):
    return (
        "# P50c Thinking-On Completion Report\n\n"
        "This run cannot be presented as a fully verified comparison because post-run restoration was not confirmed. "
        "The summary and latest weight-verification record were checked independently against the planned base hashes. "
        "A status value of complete does not establish that restoration succeeded. "
        "No further generation is started by this report. "
        "GSM8K TRAIN is likely in Qwen's training data, so no capability or benchmark claim is made.\n\n"
        "## Stop\n\n**RESTORATION NOT VERIFIED.** At least one required artifact is missing/invalid, "
        "has verified_after_restore other than true, or has original_after_restore different from "
        "the planned base hashes. This failure takes precedence over any status field. "
        "Do not claim successful restoration or a fully verified run. No weights are saved by this report.\n\n"
        + fenced(json.dumps(audit, indent=2, sort_keys=True)) + "\n"
    )


def write_restoration_failure(directory, plan, audit):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("Failure report must be project-local.")
    text = restoration_failure_report(plan, audit).encode("utf-8")
    metadata_path = directory / "finalization.json"
    owned = read_json(project_path(metadata_path.relative_to(ROOT).as_posix())).get("artifact_sha256", {}) if metadata_path.exists() else {}
    amendment = directory / "amendments" / "text_bound_01" / "amendment.json"
    seal = amendment.parent / "runtime_source_manifest.json"
    if amendment.exists() and seal.exists():
        source = read_json(project_path(seal.relative_to(ROOT).as_posix()))
        if source.get("amendment_sha256") == sha256(project_path(amendment.relative_to(ROOT).as_posix())):
            owned.update(source.get("pre_generation_report_sha256", {}))
    for path in (ROOT / "P50C_COMPLETION_REPORT.md", directory / "report.md"):
        if not path.exists():
            atomic_bytes(path, text)
        elif owned.get(path.relative_to(ROOT).as_posix()) == sha256(path):
            atomic_bytes(path, text, replace=True)
        else:
            alternative = directory / "restoration_failure_report.md"
            if not alternative.exists():
                atomic_bytes(alternative, text)
