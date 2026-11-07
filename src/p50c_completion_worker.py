"""Run/resume only after fresh, actual in-memory p50c hash verification."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from budget_sweep_generate import configure_determinism, load_model, verify_targets
from budget_sweep_io import RunLock, atomic_json, identity
from p50c_completion_analysis import original_records, recompute
from p50c_completion_diagnostic import closure_at_caps
from p50c_completion_generate import VerifiedEditedGenerator
from p50c_completion_io import RUN, EditedRecordStore, load_approved, make_decoder
from weight_edit import edited_weights


def save_summary(store, status, runtime, checks, decode, *, error=None):
    rows = store.records()
    summary = {
        "status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": store.plan_hash, "runtime": runtime, "target_hash_checks": checks,
        "budget": store.budget(rows), "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "stopping_eos_ids": [248044], "diagnostic_token_only": 248046, "error": error,
        **recompute(store.plan, list(store.pool.values()), rows, original_records(store.plan), decode),
    }
    atomic_json(store.directory / "summary.json", summary, replace=True)
    return summary


def profile(store, rows):
    path = store.directory / "profile_milestone.json"
    if len(rows) < 2 or path.exists():
        return
    seconds = sum(row["seconds"] for row in rows)
    rate = sum(len(row["token_ids"]) for row in rows) / seconds
    remaining_seconds = (30 - len(rows)) * 8192 / rate
    now = datetime.now(timezone.utc)
    milestone = {
        "measured_from_main_records": len(rows), "tokens_per_second": rate,
        "peak_gpu_allocated_bytes": max(row["peak_gpu_allocated_bytes"] for row in rows),
        "peak_gpu_reserved_bytes": max(row["peak_gpu_reserved_bytes"] for row in rows),
        "projected_finish_utc": (now + timedelta(seconds=remaining_seconds)).isoformat(),
        "projection_is_not_a_stop_condition": True,
        "caveat": "Early-sample extrapolation at remaining full caps; host delays and natural EOS change the ETA.",
    }
    atomic_json(path, milestone)
    print("MEASURED PROFILE; CONTINUING AUTOMATICALLY", flush=True)
    print(json.dumps(milestone, indent=2), flush=True)


def run_schedule(store, gate, runtime, checks, decode):
    gate.assert_ready()
    rows = store.recover_journal()
    completed = {identity(row) for row in rows}
    for index, job in enumerate(store.plan["schedule"]):
        if identity(job) in completed:
            continue
        gate.assert_ready()
        attempt = store.reserve(job)
        if attempt is None:
            return "token_budget_exhausted"
        print(f"START {index + 1}/30 {job['problem_id']} seed={job['seed']} p50c cap=8192", flush=True)
        row = gate(store.pool[job["problem_id"]], job, attempt)
        if decode(row["token_ids"]) != row["text"]:
            raise ValueError("Pinned generated-token decoding mismatch.")
        row["thinking_closed_at_cap"] = closure_at_caps(row["token_ids"], decode)
        store.commit(row)
        rows.append(row)
        completed.add(identity(job))
        print(
            f"DONE {index + 1}/30 tokens={len(row['token_ids'])} "
            f"EOS={row['graded']['natural_eos_index']} correct={row['graded']['strict_final_correct']} "
            f"thinking_closed={row['thinking_closed_at_cap']['8192']} "
            f"seconds={row['seconds']:.2f} tokens/s={len(row['token_ids']) / row['seconds']:.2f} "
            f"token_248046_count={row['token_248046_count']}", flush=True,
        )
        save_summary(store, "running", runtime, checks, decode)
        profile(store, rows)
    return "complete"


def execute(directory=RUN):
    plan, pool = load_approved(directory)
    directory = Path(directory).resolve()
    with RunLock(directory / "worker.lock"):
        store = EditedRecordStore(directory, plan, pool)
        store.recover_journal()
        decode = make_decoder(plan)
        runtime, checks = {}, {"original_before": None, "edited_after_apply": None,
                               "original_after_restore": None, "verified_after_restore": False}
        model, status, error = None, "initializing", None
        started = datetime.now(timezone.utc)
        try:
            runtime = configure_determinism()
            save_summary(store, status, runtime, checks, decode)
            model, tokenizer = load_model(plan)
            runtime["planned_hardware_and_software_verified"] = True
            checks["original_before"] = verify_targets(model, plan)
            with edited_weights(model, layers=(15, 19), strength=1.0):
                gate = VerifiedEditedGenerator(model, tokenizer, plan, store.plan_hash)
                checks["edited_after_apply"] = gate.verify_on_start_or_resume()
                print("FRESH P50C APPLICATION AND EDITED HASH VERIFICATION PASSED", flush=True)
                print(json.dumps(checks["edited_after_apply"], indent=2), flush=True)
                save_summary(store, "edit_verified", runtime, checks, decode)
                status = run_schedule(store, gate, runtime, checks, decode)
        except BaseException as exception:
            status = "interrupted" if isinstance(exception, KeyboardInterrupt) else "fatal_error"
            error = {"type": type(exception).__name__, "message": str(exception)[:1000]}
            print("STOP:", status, json.dumps(error), flush=True)
        finally:
            if model is not None:
                try:
                    checks["original_after_restore"] = verify_targets(model, plan)
                    checks["verified_after_restore"] = (
                        checks["original_after_restore"] == plan["model"]["target_base_sha256"]
                    )
                except BaseException as exception:
                    status = "restoration_verification_failed"
                    error = {"type": type(exception).__name__, "message": str(exception)[:1000]}
            history = {
                "start_utc": started.isoformat(), "end_utc": datetime.now(timezone.utc).isoformat(),
                "status": status, "plan_sha256": store.plan_hash, "runtime": runtime,
                "target_hash_checks": checks, "error": error,
            }
            atomic_json(directory / started.strftime("weight_verification_%Y%m%dT%H%M%S_%fZ.json"), history)
            summary = save_summary(store, status, runtime, checks, decode, error=error)
        print("FINAL STATUS:", summary["status"], "edited responses:", summary["edited_main_responses"], flush=True)
        return 0 if status == "complete" and checks["verified_after_restore"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN, help="The approved run; used identically for start/resume.")
    arguments = parser.parse_args()
    try:
        return execute(arguments.run)
    except Exception as exception:
        print("PRE-GENERATION GUARD FAILED:", type(exception).__name__, str(exception)[:1000], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
