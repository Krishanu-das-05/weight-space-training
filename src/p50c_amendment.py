"""Append-only approval and provenance for the pre-remaining-19 amendment."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from budget_sweep_io import ROOT, RunLock, atomic_bytes, atomic_json, identity, project_path, read_json, sha256
from p50c_completion_analysis import original_records, recompute
from p50c_completion_io import RUN, EditedRecordStore, load_approved, make_decoder
from p50c_amended_analysis import recompute_amended
from p50c_amended_store import AmendedRecordStore

DIRECTORY = RUN / "amendments" / "text_bound_01"
DECLARATION = DIRECTORY / "amendment.json"
MANIFEST = DIRECTORY / "runtime_source_manifest.json"
FILES = (
    "p50c_amendment.py", "p50c_amended_scoring.py", "p50c_amended_store.py",
    "p50c_amended_generate.py", "p50c_amended_analysis.py", "p50c_amended_worker.py",
    "test_p50c_amendment.py",
)
AUTHORIZATION = (
    "Approved amendment, declared before generating the remaining 19 items: "
    "Persist raw token IDs (and raw text if decodable) BEFORE any validation. "
    "If decoded text fails the 1-200,000 character bound, do not crash: score strict-correct=false, "
    "determine natural completion and </think> closure from token IDs only, flag text_bound_violation "
    "with character length. Document amendment after 11 saved records, before the remaining 19; "
    "no outcome change for those 11. Resume the same run, regenerate gsm8k_train_02851 and the "
    "18 never-started items, retain the original reservation charge, produce full deliverables, then stop."
)


def prepare():
    with RunLock(RUN / "worker.lock"):
        plan, pool = load_approved()
        store = EditedRecordStore(RUN, plan, pool)
        records = store.recover_journal()
        if len(records) != 11 or records[-1]["seed"] != 10:
            raise ValueError("The approved amendment must be declared after exactly the original 11 records.")
        if DIRECTORY.exists():
            raise FileExistsError("Amendment already exists; do not redeclare or replace it.")
        (RUN / "amendments").mkdir(exist_ok=True)
        DIRECTORY.mkdir()
        archive = DIRECTORY / "before"
        archive.mkdir()
        snapshots = {}
        paths = [
            ROOT / "P50C_COMPLETION_REPORT.md", RUN / "summary.json", RUN / "report.md",
            RUN / "analysis_recomputed.json", RUN / "restoration_audit.json", RUN / "partial_validation.json",
            RUN / "records.jsonl", RUN / "runtime_source_manifest.json",
        ]
        for path in paths:
            target = archive / ("ROOT_" + path.name if path.parent == ROOT else path.name)
            atomic_bytes(target, path.read_bytes())
            snapshots[path.relative_to(ROOT).as_posix()] = {
                "snapshot": target.relative_to(ROOT).as_posix(), "sha256": sha256(target),
            }
        analysis = recompute(plan, pool, records, original_records(plan), make_decoder(plan))
        atomic_json(DIRECTORY / "legacy_analysis.json", analysis)
        ledger = {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted((RUN / "budget_ledger").glob("*.json"))}
        completed = {row["attempt_id"] for row in records}
        orphans = [read_json(path) for path in (RUN / "budget_ledger").glob("*.json") if path.stem not in completed]
        budget = store.budget(records)
        if budget != {"completed_actual_tokens": 78048, "unreconciled_reserved_tokens": 8192,
                      "accounted_tokens": 86240, "limit": 250000} or len(orphans) != 1:
            raise ValueError("Unexpected pre-amendment accounting.")
        declaration = {
            "schema_version": 1, "amendment_id": "text_bound_01", "approved": True,
            "declared_utc": datetime.now(timezone.utc).isoformat(), "user_authorization": AUTHORIZATION,
            "plan_sha256": store.plan_hash, "after_saved_records": 11, "before_remaining_generations": 19,
            "remaining_ids": [row["id"] for row in pool[11:]], "first_regenerated_id": "gsm8k_train_02851",
            "generation_settings_unchanged": True, "text_min_characters": 1, "text_max_characters": 200000,
            "persist_raw_tokens_before_response_validation": True, "persist_decodable_text_before_response_validation": True,
            "bound_violation_strict_correct": False, "natural_eos_from_token_ids": True,
            "thinking_closure_from_decoded_token_prefixes_without_text_bound": True,
            "undecodable_response_policy": "Keep raw IDs durable and stop; no new decoding-error fallback.",
            "legacy_record_sha256": {identity(row): sha256(RUN / "durable_records" / (identity(row) + ".json"))
                                     for row in records},
            "legacy_analysis_sha256": sha256(DIRECTORY / "legacy_analysis.json"),
            "legacy_journal_bytes": (RUN / "records.jsonl").stat().st_size,
            "legacy_journal_sha256": sha256(RUN / "records.jsonl"),
            "pre_amendment_budget": budget, "pre_amendment_ledger_sha256": ledger,
            "retained_unreconciled_attempt_id": orphans[0]["attempt_id"], "retained_reservation_tokens": 8192,
            "maximum_remaining_allocation": 19 * 8192,
            "worst_case_total_with_original_failed_charge": budget["accounted_tokens"] + 19 * 8192,
            "artifact_snapshots": snapshots, "original_runtime_manifest_sha256": sha256(RUN / "runtime_source_manifest.json"),
            "original_plan_and_23_sources_unchanged": True,
        }
        atomic_json(DECLARATION, declaration)
        atomic_bytes(DIRECTORY / "AMENDMENT.md", (
            "# Approved Text-Bound Amendment\n\n" + AUTHORIZATION + "\n\n"
            "Declared after 11 saved records and before any of the remaining 19 generations. "
            "The original failed 8,192-token reservation is retained. Worst case including the "
            "19 remaining full caps is 241,888/250,000 accounted tokens. "
            "Original plan, source manifests, baseline, 11 canonical records and prior artifacts are preserved. "
            "Only response capture and the explicitly approved bound-violation policy change; "
            "sampling, prompts, seeds, EOS, determinism, edit verification and restoration do not. "
            "Raw capture is passive persistence, not acceptance of unchecked input; structural validation "
            "still occurs before scoring/commit. The amended runtime is sealed before generation.\n"
        ).encode("utf-8"))
        print(json.dumps({"amendment": str(DECLARATION), "saved": 11, "remaining": 19,
                          "worst_case_accounted": declaration["worst_case_total_with_original_failed_charge"]}, indent=2))


def load_amended(*, sealed=True):
    plan, pool = load_approved()
    declaration = read_json(DECLARATION)
    digest = sha256(DECLARATION)
    manifest = read_json(MANIFEST) if sealed else None
    if sealed and (
        manifest.get("amendment_sha256") != digest or manifest.get("plan_sha256") != sha256(RUN / "plan.json")
        or set(manifest.get("source_sha256", {})) != set(FILES)
        or manifest.get("legacy_outcomes_unchanged") is not True
    ):
        raise ValueError("Missing amended runtime seal or legacy equivalence proof.")
    if (
        declaration.get("approved") is not True or declaration.get("plan_sha256") != sha256(RUN / "plan.json")
        or declaration.get("after_saved_records") != 11 or declaration.get("before_remaining_generations") != 19
        or declaration.get("remaining_ids") != [row["id"] for row in pool[11:]]
        or declaration.get("persist_raw_tokens_before_response_validation") is not True
        or declaration.get("persist_decodable_text_before_response_validation") is not True
        or declaration.get("bound_violation_strict_correct") is not False
        or declaration.get("text_min_characters") != 1 or declaration.get("text_max_characters") != 200000
        or declaration.get("generation_settings_unchanged") is not True
        or declaration.get("natural_eos_from_token_ids") is not True
        or declaration.get("thinking_closure_from_decoded_token_prefixes_without_text_bound") is not True
    ):
        raise ValueError("The approved amendment identity/policy changed.")
    legacy = declaration.get("legacy_record_sha256")
    if (
        not isinstance(legacy, dict) or set(legacy) != {identity(job) for job in plan["schedule"][:11]}
        or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in legacy.values())
        or type(declaration.get("legacy_journal_bytes")) is not int
        or not 1 <= declaration["legacy_journal_bytes"] <= 50_000_000
        or declaration.get("original_runtime_manifest_sha256") != sha256(RUN / "runtime_source_manifest.json")
    ):
        raise ValueError("Invalid original-record snapshot or journal boundary.")
    if sha256(DIRECTORY / "legacy_analysis.json") != declaration["legacy_analysis_sha256"]:
        raise ValueError("The preserved 11-record analysis changed.")
    for key, expected in declaration["legacy_record_sha256"].items():
        if sha256(RUN / "durable_records" / (key + ".json")) != expected:
            raise ValueError("An original saved response changed.")
    with (RUN / "records.jsonl").open("rb") as handle:
        prefix = handle.read(declaration["legacy_journal_bytes"])
    if hashlib.sha256(prefix).hexdigest() != declaration["legacy_journal_sha256"]:
        raise ValueError("The original 11 journal records changed.")
    for relative, expected in declaration["pre_amendment_ledger_sha256"].items():
        if sha256(project_path(relative)) != expected:
            raise ValueError("An original token reservation changed.")
    for snapshot in declaration["artifact_snapshots"].values():
        path = project_path(snapshot["snapshot"])
        if not path.is_relative_to(DIRECTORY) or sha256(path) != snapshot["sha256"]:
            raise ValueError("A pre-amendment artifact snapshot changed.")
    if sealed:
        for name, expected in manifest["source_sha256"].items():
            if sha256(project_path(name)) != expected:
                raise ValueError("An amended runtime source changed: " + name)
    return plan, pool, declaration, digest


def seal():
    with RunLock(RUN / "worker.lock"):
        plan, pool, declaration, digest = load_amended(sealed=False)
        if MANIFEST.exists():
            raise FileExistsError("Do not replace an already sealed amended runtime.")
        decode = make_decoder(plan)
        store = AmendedRecordStore(RUN, plan, pool, declaration, digest, decode)
        rows = store.records()
        if len(rows) != 11 or list(store.raw_directory.glob("*/tokens.json")):
            raise ValueError("The seal must precede every remaining generation.")
        derived = recompute_amended(plan, pool, rows, original_records(plan), decode, declaration, digest)
        saved = read_json(DIRECTORY / "legacy_analysis.json")
        for key, value in saved.items():
            if json.dumps(derived[key], sort_keys=True) != json.dumps(value, sort_keys=True):
                raise ValueError("Amendment changes an outcome/table from the original 11.")
        environment = os.environ.copy()
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                           HF_HUB_DISABLE_IMPLICIT_TOKEN="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        tests = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=ROOT, env=environment,
                               capture_output=True, text=True)
        if tests.returncode:
            print(tests.stdout, tests.stderr)
            raise RuntimeError("Amendment regression gates failed; no seal or generation.")
        count = re.search(r"Ran ([0-9]+) tests", tests.stderr)
        if not count:
            raise RuntimeError("Cannot confirm the amendment test count.")
        atomic_json(MANIFEST, {
            "created_utc": datetime.now(timezone.utc).isoformat(), "amendment_sha256": digest,
            "plan_sha256": declaration["plan_sha256"], "legacy_outcomes_unchanged": True,
            "legacy_record_bytes_unchanged": True, "sampling_call_and_rng_ast_identical": True,
            "offline_tests_passed_before_remaining_generation": int(count.group(1)),
            "source_sha256": {name: sha256(ROOT / name) for name in FILES},
            "pre_generation_report_sha256": {
                path.relative_to(ROOT).as_posix(): sha256(path)
                for path in (ROOT / "P50C_COMPLETION_REPORT.md", RUN / "report.md")
            },
            "policy": "Append-only amendment seal; source drift stops start/resume.",
        })
        load_amended(sealed=True)
        print(json.dumps({"sealed": str(MANIFEST), "tests": int(count.group(1)),
                          "legacy_11_unchanged": True, "remaining": 19}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    seal() if args.seal else prepare()
