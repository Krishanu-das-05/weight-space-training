"""Run/resume the user-approved sweep; strict policy failures are fatal."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from budget_sweep_analysis import measured_projection, prefix_status, recompute
from budget_sweep_generate import configure_determinism, generate_record, load_model, verify_targets
from budget_sweep_io import RecordStore, RunLock, atomic_json, identity
from budget_sweep_plan import create_approved_run, load_run


def save_summary(store, status, runtime, checks, *, error=None):
    records = store.records()
    statistics = recompute(store.plan, list(store.pool.values()), records)
    summary = {
        "status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": store.plan_hash, "runtime": runtime, "target_hash_checks": checks,
        "budget": store.budget(records), **statistics,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "stopping_eos_ids": [248044], "diagnostic_token_only": 248046,
        "error": error,
    }
    milestone = store.directory / "truncation_milestone.json"
    if milestone.is_file():
        from budget_sweep_io import read_json
        summary["truncation_milestone"] = read_json(milestone)
    atomic_json(store.directory / "summary.json", summary, replace=True)
    return summary


def run_schedule(store, model, tokenizer, runtime, checks, generator=generate_record):
    records = store.recover_journal()
    completed = {identity(row) for row in records}
    for index, job in enumerate(store.plan["schedule"]):
        validation = prefix_status(store.plan, records)
        if validation["failed"]:
            return "prefix_mismatch"
        if index >= 8 and not validation["passed"]:
            raise RuntimeError("The main sweep cannot precede successful truncation validation.")
        if validation["passed"] and not (store.directory / "truncation_milestone.json").exists():
            milestone = measured_projection(store.plan, records)
            atomic_json(store.directory / "truncation_milestone.json", milestone)
            print("TRUNCATION CHECK PASSED; CONTINUING AUTOMATICALLY", flush=True)
            print(json.dumps(milestone, indent=2), flush=True)
        if identity(job) in completed:
            continue
        attempt = store.reserve(job)
        if attempt is None:
            return "token_budget_exhausted"
        print(f"START {index + 1}/64 {job['problem_id']} {job['condition']} {job['kind']} cap={job['max_new_tokens']}", flush=True)
        row = generator(model, tokenizer, store.plan, store.pool[job["problem_id"]], job, attempt, store.plan_hash)
        store.commit(row)
        records.append(row)
        completed.add(identity(job))
        print(
            f"DONE {index + 1}/64 tokens={len(row['token_ids'])} "
            f"EOS={row['graded']['natural_eos_index']} correct={row['graded']['strict_final_correct']} "
            f"seconds={row['seconds']:.2f} tokens/s={len(row['token_ids']) / row['seconds']:.2f} "
            f"token_248046_count={row['token_248046_count']}", flush=True,
        )
        save_summary(store, "running", runtime, checks)
    if prefix_status(store.plan, records)["failed"]:
        return "prefix_mismatch"
    return "complete"


def execute(directory):
    plan, pool = load_run(directory)
    with RunLock(Path(directory) / "worker.lock"):
        store = RecordStore(directory, plan, pool)
        store.recover_journal()
        runtime, checks = {}, {"before": None, "after": None, "verified_after": False}
        model, status, error = None, "initializing", None
        try:
            runtime = configure_determinism()
            save_summary(store, status, runtime, checks)
            model, tokenizer = load_model(plan)
            checks["before"] = verify_targets(model, plan)
            status = run_schedule(store, model, tokenizer, runtime, checks)
        except BaseException as exception:
            status = "interrupted" if isinstance(exception, KeyboardInterrupt) else "fatal_error"
            error = {"type": type(exception).__name__, "message": str(exception)[:2000]}
            print("STOP:", status, json.dumps(error), flush=True)
        finally:
            if model is not None:
                try:
                    checks["after"] = verify_targets(model, plan)
                    checks["verified_after"] = checks["after"] == checks["before"]
                except BaseException as exception:
                    status = "target_verification_failed"
                    error = {"type": type(exception).__name__, "message": str(exception)[:2000]}
            summary = save_summary(store, status, runtime, checks, error=error)
        print("FINAL STATUS:", summary["status"], "main responses:", summary["main_responses"], flush=True)
        return 0 if status == "complete" and checks["verified_after"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--start-approved", action="store_true")
    choice.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    directory = create_approved_run() if arguments.start_approved else arguments.resume
    print("RUN DIRECTORY:", directory, flush=True)
    return execute(directory)


if __name__ == "__main__":
    raise SystemExit(main())
