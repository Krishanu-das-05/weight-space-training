"""Resume the same run under the approved, sealed raw-first amendment."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import json
from datetime import datetime, timezone

from budget_sweep_generate import configure_determinism, load_model, verify_targets
from budget_sweep_io import RunLock, atomic_json, identity, read_json
from p50c_amendment import DIRECTORY, load_amended
from p50c_amended_analysis import recompute_amended, verify_legacy_outcomes
from p50c_amended_generate import RawFirstEditedGenerator
from p50c_amended_store import AmendedRecordStore
from p50c_completion_analysis import original_records
from p50c_completion_io import RUN, make_decoder
from weight_edit import edited_weights


def save_summary(store, status, runtime, checks, *, error=None):
    rows = store.records()
    analysis = recompute_amended(
        store.plan, list(store.pool.values()), rows, original_records(store.plan), store.decode,
        store.declaration, store.amendment_hash,
    )
    verify_legacy_outcomes(analysis, read_json(DIRECTORY / "legacy_analysis.json"))
    summary = {
        "status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": store.plan_hash, "amendment_sha256": store.amendment_hash,
        "amendment_after_saved_records": 11, "legacy_11_outcomes_unchanged": True,
        "runtime": runtime, "target_hash_checks": checks, "budget": store.budget(rows),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "stopping_eos_ids": [248044], "diagnostic_token_only": 248046, "error": error, **analysis,
    }
    atomic_json(store.directory / "summary.json", summary, replace=True)
    return summary


def run_schedule(store, gate, runtime, checks):
    gate.assert_ready()
    rows = store.recover_raw()
    completed = {identity(row) for row in rows}
    for index, job in enumerate(store.plan["schedule"]):
        if identity(job) in completed:
            continue
        gate.assert_ready()
        attempt = store.reserve(job)
        if attempt is None:
            return "token_budget_exhausted"
        print(f"START {index + 1}/30 {job['problem_id']} seed={job['seed']} raw-first p50c cap=8192", flush=True)
        row = gate(store.pool[job["problem_id"]], job, attempt)
        store.commit(row)
        completed.add(identity(job))
        print(
            f"DONE {index + 1}/30 tokens={len(row['token_ids'])} EOS={row['graded']['natural_eos_index']} "
            f"correct={row['graded']['strict_final_correct']} thinking_closed={row['thinking_closed_at_cap']['8192']} "
            f"text_bound_violation={row['text_bound_violation']} characters={row['text_character_length']} "
            f"seconds={row['seconds']:.2f} tokens/s={len(row['token_ids']) / row['seconds']:.2f}", flush=True,
        )
        save_summary(store, "running", runtime, checks)
    return "complete"


def execute():
    plan, pool, declaration, digest = load_amended(sealed=True)
    with RunLock(RUN / "worker.lock"):
        decode = make_decoder(plan)
        store = AmendedRecordStore(RUN, plan, pool, declaration, digest, decode)
        store.recover_journal()
        runtime = {}
        checks = {"original_before": None, "edited_after_apply": None,
                  "original_after_restore": None, "verified_after_restore": False}
        model, status, error = None, "initializing", None
        started = datetime.now(timezone.utc)
        try:
            runtime = configure_determinism()
            save_summary(store, status, runtime, checks)
            model, tokenizer = load_model(plan)
            runtime["planned_hardware_and_software_verified"] = True
            checks["original_before"] = verify_targets(model, plan)
            with edited_weights(model, layers=(15, 19), strength=1.0):
                gate = RawFirstEditedGenerator(model, tokenizer, plan, store.plan_hash, store)
                checks["edited_after_apply"] = gate.verify_on_start_or_resume()
                print("FRESH AMENDED START/RESUME: P50C EDITED HASHES VERIFIED", flush=True)
                print(json.dumps(checks["edited_after_apply"], indent=2), flush=True)
                save_summary(store, "edit_verified", runtime, checks)
                status = run_schedule(store, gate, runtime, checks)
        except BaseException as exception:
            status = "interrupted" if isinstance(exception, KeyboardInterrupt) else "fatal_error"
            error = {"type": type(exception).__name__, "message": str(exception)[:1000]}
            print("STOP:", status, json.dumps(error), flush=True)
        finally:
            if model is not None:
                try:
                    checks["original_after_restore"] = verify_targets(model, plan)
                    checks["verified_after_restore"] = checks["original_after_restore"] == plan["model"]["target_base_sha256"]
                except BaseException as exception:
                    status = "restoration_verification_failed"
                    error = {"type": type(exception).__name__, "message": str(exception)[:1000]}
            atomic_json(RUN / started.strftime("weight_verification_%Y%m%dT%H%M%S_%fZ.json"), {
                "start_utc": started.isoformat(), "end_utc": datetime.now(timezone.utc).isoformat(),
                "status": status, "plan_sha256": store.plan_hash, "amendment_sha256": digest,
                "runtime": runtime, "target_hash_checks": checks, "error": error,
            })
            summary = save_summary(store, status, runtime, checks, error=error)
        print("FINAL STATUS:", summary["status"], "edited responses:", summary["edited_main_responses"], flush=True)
        return 0 if status == "complete" and checks["verified_after_restore"] else 1


if __name__ == "__main__":
    raise SystemExit(execute())
