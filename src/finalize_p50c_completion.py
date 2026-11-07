"""Finalize saved edited records; no model runtime or inference is imported."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import json
import subprocess
import sys
from datetime import datetime, timezone

import nbformat

from budget_sweep_io import ROOT, RunLock, atomic_bytes, atomic_json, read_json, sha256
from p50c_completion_diagnostic import protected_snapshot
from p50c_completion_io import RUN, load_approved
from p50c_completion_outputs import load_saved_analysis, make_notebook, render_report
from p50c_restore_audit import load_restore_audit, write_restoration_failure
from p50c_amendment import DECLARATION, MANIFEST, load_amended

STEM = "p50c_vs_original_token_budget"


def _finalize_locked():
    before = protected_snapshot()
    plan, _ = load_approved(RUN)
    restore_audit = load_restore_audit(RUN, plan)
    if not restore_audit["passed"]:
        write_restoration_failure(RUN, plan, restore_audit)
        atomic_json(RUN / "restoration_audit.json", restore_audit, replace=True)
        raise RuntimeError("Restoration not verified; the report's Stop section records the failure regardless of status.")
    plan, _, records, summary, analysis, budget = load_saved_analysis(RUN)
    root_report = ROOT / "P50C_COMPLETION_REPORT.md"
    archive_report = RUN / "report.md"
    png, svg = RUN / (STEM + ".png"), RUN / (STEM + ".svg")
    notebook = RUN / "07_p50c_completion.ipynb"
    replace = {}
    declaration, amendment_hash, report_hashes = None, None, {}
    if DECLARATION.exists():
        _, _, declaration, amendment_hash = load_amended(sealed=True)
        report_hashes = read_json(MANIFEST)["pre_generation_report_sha256"]
    for path in (root_report, archive_report, png, svg, notebook, RUN / "analysis_recomputed.json",
                 RUN / "plot_renderer.json", RUN / "finalization.json", RUN / "restoration_audit.json"):
        replace[path] = False
        if path.exists():
            relative = path.relative_to(ROOT).as_posix()
            expected = report_hashes.get(relative)
            if declaration and expected is None:
                expected = declaration["artifact_snapshots"].get(relative, {}).get("sha256")
            if expected is None or sha256(path) != expected:
                raise FileExistsError("Refusing to replace an unrecognized or modified artifact: " + path.name)
            replace[path] = True
    if sha256(ROOT / plan["model"]["weights_file"]) != plan["model"]["checkpoint_sha256"]:
        raise ValueError("The original disk checkpoint changed.")
    atomic_json(RUN / "analysis_recomputed.json", analysis, replace=replace[RUN / "analysis_recomputed.json"])
    environment = os.environ.copy()
    environment.update(MPLCONFIGDIR=str(ROOT / ".cache" / "matplotlib"), MPLBACKEND="Agg")
    subprocess.run(
        [getattr(sys, "_base_executable", sys.executable), str(ROOT / "plot_p50c_completion.py"),
         str(RUN / "analysis_recomputed.json")],
        cwd=ROOT, env=environment, check=True,
    )
    relative_image = png.relative_to(ROOT).as_posix()
    atomic_bytes(root_report, render_report(plan, records, summary, analysis, image_relative=relative_image,
                                           restore_audit=restore_audit).encode("utf-8"), replace=replace[root_report])
    atomic_bytes(archive_report, render_report(plan, records, summary, analysis, image_relative=png.name,
                                              restore_audit=restore_audit).encode("utf-8"), replace=replace[archive_report])
    atomic_json(RUN / "restoration_audit.json", restore_audit, replace=replace[RUN / "restoration_audit.json"])
    atomic_bytes(notebook, nbformat.writes(make_notebook()).encode("utf-8"))
    if protected_snapshot() != before:
        raise RuntimeError("A protected completed artifact changed during finalization.")
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("Read-only finalization must not import the model runtime.")
    atomic_json(RUN / "finalization.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha256(RUN / "plan.json"),
        "status": summary["status"], "edited_records": len(records), "budget": budget,
        "all_verified_record_hashes_match": True, "all_actual_prompts_match_original": True,
        "original_targets_restored_verified": summary["target_hash_checks"]["verified_after_restore"],
        "original_checkpoint_sha256_verified": True, "protected_artifacts_unchanged": True,
        "model_runtime_imported": False, "inference_performed": False,
        "independent_summary_and_latest_history_restoration_pass": True,
        "amendment_sha256": amendment_hash,
        "amended_runtime_seal_sha256": sha256(MANIFEST) if amendment_hash else None,
        "legacy_11_record_bytes_and_outcomes_unchanged": True if amendment_hash else None,
        "original_failed_reservation_retained_tokens": 8192 if amendment_hash else 0,
        "postprocessing_source_sha256": {name: sha256(ROOT / name) for name in (
            "p50c_completion_outputs.py", "p50c_restore_audit.py", "plot_p50c_completion.py",
            "finalize_p50c_completion.py", "execute_p50c_notebook.py", "validate_p50c_completion.py",
            "finish_p50c_when_ready.py",
        )},
        "notebook_location": notebook.relative_to(ROOT).as_posix(),
        "artifact_sha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                            for path in (root_report, archive_report, png, svg, notebook,
                                         RUN / "analysis_recomputed.json", RUN / "restoration_audit.json",
                                         RUN / "plot_renderer.json")},
    })
    print(json.dumps({"report": str(root_report), "notebook": str(notebook), "figure": str(png),
                      "budget": budget, "status": summary["status"]}, indent=2))


def finalize():
    with RunLock(RUN / "worker.lock"):
        return _finalize_locked()


if __name__ == "__main__":
    finalize()
