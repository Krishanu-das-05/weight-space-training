"""Bind the explicit user approval and three additions to an immutable run."""

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from budget_sweep_io import ROOT, atomic_bytes, atomic_json, project_path, read_json, sha256

BASE_PLAN = ROOT / "results" / "budget_sweep_20261106T185132_004320Z" / "plan.json"
BASE_SHA256 = "6885859678c83c84252ee9d108d402fcabc7cb7dd938c61dfd29c6aed4626d72"
FROZEN_FILES = {
    "probe_metrics.py", "math_probe.py", "budget_sweep_scoring.py", "gsm8k_source.py",
    "decode_gsm8k_train.py", "prepare_budget_plan.py", "test_budget_sweep_plan.py", "test_gsm8k_download.py",
}
RUNTIME_FILES = {"budget_sweep_io.py", "budget_sweep_analysis.py", "budget_sweep_plan.py",
                 "budget_sweep_generate.py", "budget_sweep_worker.py"}


def check_sources(plan):
    if set(plan["source_sha256"]) != FROZEN_FILES:
        raise ValueError("Unexpected frozen source list.")
    for name in FROZEN_FILES:
        if sha256(ROOT / name) != plan["source_sha256"][name]:
            raise ValueError("Frozen source changed: " + name)
    implementation = plan.get("implementation_source_sha256")
    if implementation is not None:
        if set(implementation) != RUNTIME_FILES:
            raise ValueError("Unexpected runtime source list.")
        for name in RUNTIME_FILES:
            if sha256(ROOT / name) != implementation[name]:
                raise ValueError("Runtime source changed: " + name)


def create_approved_run():
    if sha256(BASE_PLAN) != BASE_SHA256:
        raise ValueError("The user-approved base plan changed.")
    base = read_json(BASE_PLAN)
    check_sources(base)
    plan = copy.deepcopy(base)
    now = datetime.now(timezone.utc)
    directory = ROOT / "results" / now.strftime("budget_sweep_%Y%m%dT%H%M%S_%fZ")
    directory.mkdir()
    plan.update(
        status="APPROVED_WITH_USER_ADDITIONS", generation_authorized=True,
        created_utc=now.isoformat(),
        approved_base={"path": str(BASE_PLAN.relative_to(ROOT)).replace("\\", "/"), "sha256": BASE_SHA256},
        implementation_source_sha256={name: sha256(ROOT / name) for name in sorted(RUNTIME_FILES)},
        approved_additions={
            "cublas": "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Torch import/CUDA initialization; deterministic enablement failure is fatal. Record observed settings in summary.json.",
            "token_248046": "Log every containing response and report response/occurrence counts per condition, separated into main/short/all. EOS remains 248044 only; scoring unchanged.",
            "profiling": "After all four prefix checks pass, report measured tokens/s, peak allocated GPU memory and projected finish time, then continue automatically without a wall-time cutoff.",
        },
    )
    plan["execution"]["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    plan["execution"]["cublas_must_be_set_before_cuda_initialization"] = True
    plan["execution"]["determinism_failure_policy"] = "STOP; never disable deterministic algorithms or use warn_only."
    atomic_json(directory / "plan.json", plan)
    approval = {
        "approved": True, "plan_sha256": sha256(directory / "plan.json"), "approved_base_sha256": BASE_SHA256,
        "source": "User explicitly approved the plan with three additions in the conversation.",
        "approved_additions": plan["approved_additions"], "recorded_utc": now.isoformat(),
    }
    atomic_json(directory / "approval.json", approval)
    markdown = (BASE_PLAN.parent / "PLAN.md").read_text(encoding="utf-8")
    markdown = markdown.replace("Status: **AWAITING YOUR APPROVAL. No generation has run.**",
                                "Status: **APPROVED with the three user-specified additions below.**")
    markdown = markdown.replace("**Approval is for the plan only at this point. No inference starts until you explicitly approve.**",
                                "**User approval is bound to plan.json by approval.json.**")
    markdown += "\n## Approved Additions\n"
    for text in plan["approved_additions"].values():
        markdown += "- " + text + "\n"
    atomic_bytes(directory / "PLAN.md", markdown.encode("utf-8"))
    return directory


def load_run(directory):
    directory = Path(directory).resolve()
    if directory.parent != ROOT / "results" or not re.fullmatch(r"budget_sweep_[0-9]{8}T[0-9]{6}_[0-9]{6}Z", directory.name):
        raise ValueError("Not a project budget-sweep directory.")
    plan, approval = read_json(directory / "plan.json"), read_json(directory / "approval.json")
    if approval.get("approved") is not True or approval.get("plan_sha256") != sha256(directory / "plan.json"):
        raise ValueError("Missing or mismatched explicit approval.")
    if approval.get("approved_base_sha256") != BASE_SHA256 or plan.get("generation_authorized") is not True:
        raise ValueError("Unapproved experiment.")
    check_sources(plan)
    pool_path = project_path(plan["dataset"]["dev_file"])
    if pool_path != ROOT / "dev_pool_gsm8k_30.jsonl" or sha256(pool_path) != plan["dataset"]["dev_sha256"]:
        raise ValueError("The frozen development pool changed.")
    pool = [json.loads(line) for line in pool_path.read_text(encoding="utf-8").splitlines()]
    if len(pool) != 30 or [item["id"] for item in pool] != [item["id"] for item in plan["questions_and_seeds"]]:
        raise ValueError("Question identity mismatch.")
    if [item["seed"] for item in plan["questions_and_seeds"]] != list(range(30)):
        raise ValueError("Unexpected per-question seeds.")
    return plan, pool
