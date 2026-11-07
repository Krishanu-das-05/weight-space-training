"""Approval, provenance and durable storage for the edited-only comparison."""

import importlib.metadata
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

from budget_sweep_io import ROOT, RecordStore, project_path, read_json, sha256
from p50c_completion_diagnostic import BASELINE_DIAGNOSTIC, protected_snapshot

RUN = ROOT / "results" / "p50c_completion_20261107T094340_563457Z"
PLAN_SHA256 = "e3904e1c337631c8114ff49e5c7ddfef8cd31906277457ca21702bf5bf807bfb"
RUNTIME_FILES = (
    "p50c_completion_io.py", "p50c_completion_generate.py", "p50c_completion_analysis.py",
    "p50c_completion_worker.py", "test_p50c_completion_runtime.py",
)


def require_edited_provenance(row, plan):
    if not isinstance(row, dict):
        raise ValueError("Expected an edited response object.")
    actual = row.get("verified_edited_tensor_sha256")
    expected = plan["edit"]["target_edited_sha256"]
    if (
        not isinstance(actual, dict) or set(actual) != set(expected)
        or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in actual.values())
        or actual != expected or row.get("edit_verified_before_generation") is not True
        or row.get("model_variant") != "published_p50c"
    ):
        raise ValueError("Response lacks matching, verified edited-tensor hashes; analysis/generation rejected.")


class EditedRecordStore(RecordStore):
    def check_record(self, row):
        require_edited_provenance(row, self.plan)
        return super().check_record(row)


def load_approved(directory=RUN):
    directory = Path(directory).resolve()
    if directory != RUN or sha256(directory / "plan.json") != PLAN_SHA256:
        raise ValueError("Not the explicitly approved edited-run plan.")
    plan = read_json(directory / "plan.json")
    approval = read_json(directory / "approval.json")
    if (
        approval.get("approved") is not True or approval.get("plan_sha256") != PLAN_SHA256
        or approval.get("verify_edit_on_every_start_and_resume") is not True
        or approval.get("per_response_verified_hashes_required") is not True
        or approval.get("analysis_rejects_unverified_hashes") is not True
    ):
        raise ValueError("Missing explicit approval or mandatory user additions.")
    for name, digest in {**plan["source_sha256"], **plan["additional_source_sha256"]}.items():
        if sha256(project_path(name)) != digest:
            raise ValueError("A planned source changed: " + name)
    runtime = read_json(directory / "runtime_source_manifest.json")
    if (
        runtime.get("plan_sha256") != PLAN_SHA256
        or runtime.get("approval_sha256") != sha256(directory / "approval.json")
        or set(runtime.get("source_sha256", {})) != set(RUNTIME_FILES)
    ):
        raise ValueError("Runtime source manifest is missing or invalid.")
    for name, digest in runtime["source_sha256"].items():
        if sha256(project_path(name)) != digest:
            raise ValueError("Runtime source changed: " + name)
    if sha256(project_path(plan["study"]["predeclaration_file"])) != plan["study"]["predeclaration_sha256"]:
        raise ValueError("The predeclaration changed.")
    if sha256(BASELINE_DIAGNOSTIC) != plan["baseline"]["closure_sha256"]:
        raise ValueError("The closure baseline changed.")
    diagnostic = read_json(BASELINE_DIAGNOSTIC)
    if protected_snapshot() != diagnostic["protected_artifact_sha256"]:
        raise ValueError("A protected completed baseline artifact changed.")
    for item in plan["baseline"]["selection"]:
        if sha256(project_path(item["file"])) != item["sha256"]:
            raise ValueError("A paired original record changed.")
    pool_path = project_path(plan["dataset"]["dev_file"])
    if sha256(pool_path) != plan["dataset"]["dev_sha256"]:
        raise ValueError("The fixed development pool changed.")
    pool = [json.loads(line) for line in pool_path.read_text(encoding="utf-8").splitlines()]
    if [row["id"] for row in pool] != [row["id"] for row in plan["questions_and_seeds"]]:
        raise ValueError("The paired question identity changed.")
    for artifact in plan["edit"]["cached_artifacts"].values():
        path = project_path(artifact["path"])
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise ValueError("A cached published edit artifact changed.")
    return plan, pool


def make_decoder(plan):
    if importlib.metadata.version("tokenizers") != plan["outcomes"]["diagnostic"]["decoder"]["tokenizers_version"]:
        raise ValueError("The pinned token-decoder software version changed.")
    path = ROOT / "models" / "Qwen3.5-0.8B" / "tokenizer.json"
    if sha256(path) != plan["model"]["cached_config_sha256"][path.relative_to(ROOT).as_posix()]:
        raise ValueError("The pinned tokenizer changed.")
    backend = Tokenizer.from_file(str(path))
    return lambda ids: backend.decode(ids, skip_special_tokens=False)
