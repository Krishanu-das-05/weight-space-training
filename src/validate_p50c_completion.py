"""Final offline gates for the saved edited run and its read-only deliverables."""

import ast
import json
import os
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone

import nbformat

from budget_sweep_io import ROOT, RunLock, atomic_json, project_path, read_json, sha256
from p50c_completion_diagnostic import BASELINE_DIAGNOSTIC, protected_snapshot
from p50c_completion_io import RUN, load_approved
from p50c_completion_outputs import load_saved_analysis, make_notebook, public_summary
from p50c_restore_audit import load_restore_audit, write_restoration_failure
from p50c_amendment import DECLARATION, MANIFEST, load_amended


def _validate_locked():
    plan, _ = load_approved(RUN)
    restore_audit = load_restore_audit(RUN, plan)
    if not restore_audit["passed"]:
        write_restoration_failure(RUN, plan, restore_audit)
        atomic_json(RUN / "restoration_audit.json", restore_audit, replace=True)
        raise RuntimeError("Restoration verification failed independently of status; recorded in the report's Stop section.")
    plan, _, records, summary, analysis, budget = load_saved_analysis(RUN)
    diagnostic = read_json(BASELINE_DIAGNOSTIC)
    if protected_snapshot() != diagnostic["protected_artifact_sha256"]:
        raise RuntimeError("A protected baseline artifact changed.")
    for relative, expected in (
        (plan["model"]["weights_file"], plan["model"]["checkpoint_sha256"]),
        (plan["dataset"]["source_file"], plan["dataset"]["sha256"]),
        (plan["dataset"]["metadata_file"], plan["dataset"]["metadata_sha256"]),
    ):
        if sha256(project_path(relative)) != expected:
            raise RuntimeError("A pinned cached original input changed.")
    runtime = summary["runtime"]
    if (
        runtime.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or runtime.get("set_before_cuda_initialization") is not True
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("deterministic_warn_only") is not False
        or runtime.get("torch_threads") != 1 or runtime.get("torch_interop_threads") != 1
        or summary["stopping_eos_ids"] != [248044] or budget["accounted_tokens"] > 250000
    ):
        raise RuntimeError("Observed execution does not satisfy the frozen policy.")
    if sum(len(row["token_ids"]) for row in records) != budget["completed_actual_tokens"]:
        raise RuntimeError("Completed-token accounting mismatch.")
    for row in records:
        if (
            row["verified_edited_tensor_sha256"] != plan["edit"]["target_edited_sha256"]
            or row["edit_verified_before_generation"] is not True
            or row["contains_token_248046"] is not (248046 in row["token_ids"])
            or row["token_248046_count"] != row["token_ids"].count(248046)
        ):
            raise RuntimeError("Invalid response provenance/diagnostic metadata.")
    per_question = {row["id"]: row for row in analysis["per_question"]}
    for row in records:
        fresh = per_question[row["problem_id"]]["edited"]
        if row["graded"] != fresh["graded"] or row["thinking_closed_at_cap"] != {
            str(cap): fresh["caps"][str(cap)]["thinking_closed"] for cap in plan["analysis"]["caps"]
        }:
            raise RuntimeError("Recorded outcomes differ from frozen raw recomputation.")
    histories = [read_json(path) for path in sorted(RUN.glob("weight_verification_*.json"))]
    if not histories or histories[-1]["status"] != "complete":
        raise RuntimeError("Final weight-verification history is missing.")
    final_checks = histories[-1]["target_hash_checks"]
    if (
        final_checks["original_before"] != plan["model"]["target_base_sha256"]
        or final_checks["edited_after_apply"] != plan["edit"]["target_edited_sha256"]
        or final_checks["original_after_restore"] != plan["model"]["target_base_sha256"]
        or final_checks["verified_after_restore"] is not True
    ):
        raise RuntimeError("Original/edit/restore history failed.")
    notebook_path = RUN / "07_p50c_completion.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    expected = make_notebook()
    if [(cell.cell_type, cell.source) for cell in notebook.cells] != [(cell.cell_type, cell.source) for cell in expected.cells]:
        raise RuntimeError("Read-only notebook template changed.")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None or any(out.output_type == "error" for out in cell.outputs) for cell in code_cells):
        raise RuntimeError("Notebook has unexecuted/error cells.")
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    finalization = read_json(RUN / "finalization.json")
    if DECLARATION.exists():
        _, _, declaration, amendment_hash = load_amended(sealed=True)
        retained = declaration["retained_unreconciled_attempt_id"]
        if (
            any(row["attempt_id"] == retained for row in records)
            or budget["unreconciled_reserved_tokens"] < 8192
            or finalization.get("amendment_sha256") != amendment_hash
            or finalization.get("amended_runtime_seal_sha256") != sha256(MANIFEST)
            or summary.get("legacy_11_outcomes_unchanged") is not True
        ):
            raise RuntimeError("Amendment provenance, legacy equivalence or original failed charge is invalid.")
    if finalization.get("notebook_executed_without_inference") is not True:
        raise RuntimeError("Notebook read-only execution has not passed.")
    for relative, digest in finalization["artifact_sha256"].items():
        if sha256(project_path(relative)) != digest:
            raise RuntimeError("A finalized artifact changed.")
    for relative, digest in finalization["postprocessing_source_sha256"].items():
        if sha256(project_path(relative)) != digest:
            raise RuntimeError("A postprocessing implementation changed after finalization.")
    png = RUN / "p50c_vs_original_token_budget.png"
    with png.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) != 24:
        raise RuntimeError("Invalid PNG.")
    dimensions = list(struct.unpack(">II", header[16:24]))
    if dimensions[0] < 2000 or dimensions[1] < 900:
        raise RuntimeError("Figure too small for readable sharing.")
    svg = (RUN / "p50c_vs_original_token_budget.svg").read_text(encoding="utf-8")
    if "<svg" not in svg or "<script" in svg.lower() or "javascript:" in svg.lower() or re.search(r"(?:href|src)=[\"'](?!#)", svg):
        raise RuntimeError("SVG contains unexpected executable/external content.")
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", public_summary(analysis, summary))
    if len(sentences) != 5 or "not a capability or benchmark claim" not in sentences[-1]:
        raise RuntimeError("The public summary does not satisfy its frozen caveats.")
    environment = os.environ.copy()
    environment.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", UV_OFFLINE="1",
    )
    tests = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=ROOT, env=environment,
                           capture_output=True, text=True)
    if tests.returncode:
        print(tests.stdout, tests.stderr)
        raise RuntimeError("Offline regression tests failed.")
    matched = re.search(r"Ran ([0-9]+) tests", tests.stderr)
    if not matched:
        raise RuntimeError("Cannot confirm the executed test count.")
    subprocess.run(["uv", "pip", "check", "--python", sys.executable], cwd=ROOT, env=environment, check=True)
    if protected_snapshot() != diagnostic["protected_artifact_sha256"]:
        raise RuntimeError("A protected baseline artifact changed during final checks.")
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("Read-only validation imported the subject model runtime.")
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    audit = {
        "validated_utc": datetime.now(timezone.utc).isoformat(), "status": summary["status"],
        "plan_sha256": sha256(RUN / "plan.json"), "edited_records": len(records),
        "every_record_has_verified_planned_edited_hashes": True, "all_prompts_match_original": True,
        "paired_summary_matches_frozen_raw_recomputation": True, "budget": budget,
        "original_targets_restored_verified": True, "original_disk_checkpoint_and_dataset_unchanged": True,
        "summary_and_latest_history_independently_confirm_restore": True,
        "amendment_sha256": summary.get("amendment_sha256"),
        "legacy_11_record_bytes_and_outcomes_unchanged": summary.get("legacy_11_outcomes_unchanged"),
        "original_failed_reservation_retained_tokens": 8192 if DECLARATION.exists() else 0,
        "text_bound_violations": analysis.get("text_bound_violations", []),
        "latest_weight_verification_file": restore_audit["latest_history_file"],
        "all_protected_baseline_artifacts_unchanged": True,
        "notebook07_executed_without_inference": True, "notebook_code_cells_executed": len(code_cells),
        "png_dimensions": dimensions, "svg_no_executable_or_external_content": True,
        "all_offline_tests_pass": True, "offline_test_count": int(matched.group(1)),
        "installed_dependency_compatibility_pass": True, "staged_file_count": len(staged),
        "artifact_sha256": finalization["artifact_sha256"],
        "model_runtime_imported_by_validator": False,
    }
    atomic_json(RUN / "validation.json", audit)
    print(json.dumps(audit, indent=2))


def validate():
    with RunLock(RUN / "worker.lock"):
        return _validate_locked()


if __name__ == "__main__":
    validate()
