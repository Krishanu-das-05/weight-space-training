"""Prepare an unapproved p50c comparison plan using cached data only."""

import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from datetime import datetime, timezone

from budget_sweep_io import ROOT, atomic_bytes, atomic_json, json_bytes, project_path, read_json, sha256
from budget_sweep_outputs import load_saved_run
from p50c_completion_diagnostic import BASELINE, BASELINE_DIAGNOSTIC, baseline_diagnostic, protected_snapshot
from p50c_completion_plan_markdown import render_plan

DECLARATION = ROOT / "P50C_COMPLETION_PREDECLARATION.md"
EDIT_COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
EDIT_ARTIFACTS = {
    "manifest.json": (1326, "83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225"),
    "edit.npz": (387619, "e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7"),
}
DETERMINISM_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG", "cublas_must_be_set_before_cuda_initialization",
    "deterministic_algorithms", "determinism_failure_policy", "float32_matmul_precision",
    "tf32", "cudnn_benchmark", "torch_num_threads", "torch_num_interop_threads",
    "model_eval", "torch_inference_mode", "attention_implementation", "compile",
)


def cached_edit(base):
    artifacts = {}
    directory = ROOT / ".cache" / "published-edit"
    for name, (size, digest) in EDIT_ARTIFACTS.items():
        path = directory / name
        if path.stat().st_size != size or sha256(path) != digest:
            raise ValueError("A cached published-edit artifact changed: " + name)
        artifacts[name] = {"path": path.relative_to(ROOT).as_posix(), "bytes": size, "sha256": digest}
    manifest = read_json(directory / "manifest.json")
    specs = manifest.get("tensors")
    if (
        not isinstance(specs, list) or len(specs) != 2 or manifest.get("changed_tensors") != 2
        or manifest.get("changed_entries_total") != 130188
        or {spec["name"]: spec["base_sha256"] for spec in specs} != base["model"]["target_base_sha256"]
        or any(not re.fullmatch(r"[0-9a-f]{64}", spec["edited_sha256"]) for spec in specs)
    ):
        raise ValueError("Unexpected pinned p50c edit metadata.")
    return {
        "variant": "published_p50c", "upstream_repo": "RikiyaT/lm_understand-logs",
        "revision": EDIT_COMMIT, "upstream_directory": "exp8_math/runs/p50c/edit", "license": "CC BY 4.0",
        "cached_artifacts": artifacts, "layers": [15, 19], "strength": 1.0,
        "changed_entries": 130188, "path": "weight_edit.edited_weights",
        "call": "edited_weights(model, layers=(15, 19), directory=EDIT_DIR, strength=1.0)",
        "target_original_sha256": {spec["name"]: spec["base_sha256"] for spec in specs},
        "target_edited_sha256": {spec["name"]: spec["edited_sha256"] for spec in specs},
        "lifecycle": [
            "Fresh process: load the pinned original BF16 checkpoint locally; verify both original target hashes.",
            "Enter the existing verified context with both layers and strength 1.0. Verify both applied hashes equal the published edited hashes before any generation.",
            "Generate only pending scheduled records, serially. Record the active variant and verified edited hashes; never generate an unedited control again.",
            "In finally, exit the context, restore both original tensors, and independently verify both original hashes. Do this after success, budget stop, exception, and handled KeyboardInterrupt.",
            "A hash/cleanup failure is fatal and must be reported, never claimed as successful restoration. Do not continue inference or save weights.",
            "A hard kill/power loss cannot run Python cleanup. Edited tensors exist only in the terminated process; the next process must reload the unchanged original disk checkpoint and verify it before reapplying the edit.",
        ],
        "model_or_tensor_saves_allowed": False,
    }


def build_plan():
    before = protected_snapshot()
    base, pool, records, saved, _, _ = load_saved_run(BASELINE)
    diagnostic = read_json(BASELINE_DIAGNOSTIC)
    if json_bytes(diagnostic) != json_bytes(baseline_diagnostic()):
        raise ValueError("The saved closure baseline does not match raw recomputation.")
    if (
        platform.python_version() != base["execution"]["python"]
        or importlib.metadata.version("torch") != base["execution"]["torch"]
        or importlib.metadata.version("transformers") != base["execution"]["transformers"]
    ):
        raise ValueError("The baseline Python/package versions changed.")
    weights = project_path(base["model"]["weights_file"])
    if weights.parent != ROOT / "models" / "Qwen3.5-0.8B" or sha256(weights) != base["model"]["checkpoint_sha256"]:
        raise ValueError("The cached original checkpoint changed.")
    for relative, digest in base["model"]["cached_config_sha256"].items():
        path = project_path(relative)
        if path.parent != weights.parent or sha256(path) != digest:
            raise ValueError("A pinned cached model/tokenizer configuration changed.")
    source = project_path(base["dataset"]["source_file"])
    if source != ROOT / ".cache" / "gsm8k-main-train" / "train.parquet" or sha256(source) != base["dataset"]["sha256"]:
        raise ValueError("The cached approved TRAIN artifact changed.")
    edit = cached_edit(base)
    model = copy.deepcopy(base["model"])
    model.update(
        active_variant="published_p50c", target_edited_sha256=edit["target_edited_sha256"],
        planning_cached_checkpoint_sha256_verified=True,
        required_runtime_checks="Verify original -> apply exact published edit -> verify edited -> generate pending records -> restore and verify originals in finally. Never save weights.",
    )
    execution = copy.deepcopy(base["execution"])
    execution.update(
        worker="New p50c_completion_worker.py adapter after approval; fresh .venv/Scripts/python.exe process.",
        seed_rule="Fork CPU/CUDA RNG state and reset both to the recorded seed before each call, exactly as the original generate_record function.",
        existing_short_cap_runner_must_not_be_used="Use the unchanged budget_sweep_generate core; do not use experiment_runner/run_probes.",
    )
    selected = {row["problem_id"]: row for row in records if row["kind"] == "main" and row["condition"] == "thinking_on"}
    references = []
    for item in base["questions_and_seeds"]:
        row = selected[item["id"]]
        path = BASELINE / "durable_records" / (item["id"] + "__thinking_on__main.json")
        if sha256(path) != hashlib.sha256(json_bytes(row)).hexdigest():
            raise ValueError("Baseline canonical record mismatch.")
        references.append({
            **item, "file": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
            "user_message": row["user_message"], "rendered_prompt": row["rendered_prompt"],
        })
    inherited_sources = {**base["source_sha256"], **base["implementation_source_sha256"]}
    added_sources = ("weight_edit.py", "p50c_completion_diagnostic.py", "prepare_p50c_completion_plan.py",
                     "p50c_completion_plan_markdown.py", "test_p50c_completion_plan.py")
    plan = {
        "design_version": "p50c-completion-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AWAITING_EXPLICIT_USER_APPROVAL", "generation_authorized": False, "generated_responses": 0,
        "approval": {
            "record_file": "approval.json", "must_reference_sha256_of_this_plan": True,
            "policy": "No approval record or inference before explicit user approval of this exact plan. Bind approval to its SHA-256 without rewriting plan.json.",
            "implementation": "After approval, implement and test the adapter; fingerprint all added runtime sources before the first generation. Resume requires the same fingerprints. Scientific changes require a revised plan and fresh approval.",
        },
        "study": {
            "question": "Does the full published p50c edit improve thinking-on natural completion versus the saved original model, at fixed token caps?",
            "primary_endpoint_cap": 8192, "baseline_already_observed": True,
            "new_edited_gsm8k_responses_observed": 0, "earlier_local_toy_pilots_already_known": True,
            "predeclaration_file": DECLARATION.relative_to(ROOT).as_posix(), "predeclaration_sha256": sha256(DECLARATION),
            "expectation": {"source": "User estimate from article/public logs, not verified response-level data",
                            "endpoint": "thinking closure, NOT natural EOS", "edited_estimate": 0.30,
                            "edited_informal_range": [0.28, 0.40], "original_estimate": 0.09,
                            "success_threshold": None},
        },
        "model": model, "edit": edit, "dataset": copy.deepcopy(base["dataset"]),
        "questions_and_seeds": copy.deepcopy(base["questions_and_seeds"]),
        "conditions": [{"name": "thinking_on", "enable_thinking": True, "model_variant": "published_p50c"}],
        "batch_size": 1, "prompt": {**base["prompt"], "must_equal_baseline_user_and_rendered_prompt": True},
        "sampling": copy.deepcopy(base["sampling"]), "execution": execution, "hardware": copy.deepcopy(base["hardware"]),
        "hardware_observation": "Copied from the verified baseline; re-check exact software/OS/GPU/driver identity at launch, before loading/applying/generating. Drift => STOP.",
        "stopping": {**base["stopping"], "valid_return": "First EOS must be final token, or exactly 8192 tokens with no EOS; any other return is fatal."},
        "baseline": {
            "directory": BASELINE.relative_to(ROOT).as_posix(), "plan_sha256": sha256(BASELINE / "plan.json"),
            "records_sha256": sha256(BASELINE / "records.jsonl"), "summary_sha256": sha256(BASELINE / "summary.json"),
            "selected_condition": "thinking_on", "selected_kind": "main", "selected_count": 30,
            "selection": references, "reuse_without_regeneration": True,
            "closure_file": BASELINE_DIAGNOSTIC.relative_to(ROOT).as_posix(), "closure_sha256": sha256(BASELINE_DIAGNOSTIC),
            "known_at_8192": {"n": 30, "natural_completion": 2, "strict_correct_delivery": 1, "thinking_closed": 2},
        },
        "schedule": [{**{"problem_id": item["id"], "seed": item["seed"]}, "condition": "thinking_on",
                      "model_variant": "published_p50c", "kind": "main", "max_new_tokens": 8192}
                     for item in base["questions_and_seeds"]],
        "token_budget": {"limit": 250000, "scheduled_main_responses": 30, "main_worst_case": 245760,
                         "validation_short_worst_case": 0, "total_worst_case_no_crash": 245760, "headroom": 4240,
                         "scope": "This edited run only; all generated tokens, including interrupted attempts. No transfer of unused prior-run allocation.",
                         "reservation_rule": base["token_budget"]["reservation_rule"],
                         "crash_caveat": "An unresolved 8192-token attempt exceeds the 4240-token worst-case headroom; savings from natural EOS may fund a retry, otherwise stop partial without exceeding 250000."},
        "prefix_validation": {
            "new_check_scheduled": False, "new_short_generations": 0,
            "baseline_all_four_passed": saved["prefix_validation"]["passed"],
            "baseline_comparisons": saved["prefix_validation"]["comparisons"],
            "identical_determinism_signature": {key: base["execution"][key] for key in DETERMINISM_KEYS},
            "generation_core": "budget_sweep_generate.generate_record", "core_sha256": sha256(ROOT / "budget_sweep_generate.py"),
            "rationale": "Same pinned model architecture/tokenizer, input prompts, batch/threads, seeded sampling, strict deterministic policy, eager BF16 forward, dynamic cache and unchanged generation core. Only the two weight tensors change; max_new_tokens remains a stopping bound. Reuse the baseline's exact-prefix validation, not a claim of newly verified edited prefixes on every input.",
            "on_config_or_core_drift": "STOP and request a revised plan/approval (including a new prefix check if needed); never silently relax determinism or add generations.",
        },
        "source_sha256": inherited_sources,
        "additional_source_sha256": {name: sha256(ROOT / name) for name in added_sources},
    }
    plan.update(reporting_and_reliability(base, diagnostic))
    validate_plan(plan, base)
    if protected_snapshot() != before:
        raise RuntimeError("Completed baseline artifacts changed while planning.")
    return plan, diagnostic, before


def reporting_and_reliability(base, diagnostic):
    return {
        "outcomes": {
            "primary": {"name": "natural_completion", "definition": "First natural EOS 248044 occurs at a 1-based generated-token index <= k; correctness and thinking closure are not required."},
            "secondary": {"name": "strict_correct_delivery", "definition": "Natural EOS index <= k AND the unchanged strict final boxed-integer grader is correct.",
                          "implementation": "budget_sweep_scoring.grade_generation + score_at_cap"},
            "diagnostic": {"name": "thinking_closed", "thinking_on_only": True,
                           "definition": diagnostic["definition"], "implementation": "p50c_completion_diagnostic.closure_at_caps",
                           "decoder": diagnostic["decoder"], "requires_eos": False, "requires_correctness": False,
                           "count_unit": "One boolean per response/cap, not the number of closing tags."},
        },
        "analysis": {
            "caps": copy.deepcopy(base["analysis"]["caps"]), "n": 30, "primary_endpoint_cap": 8192,
            "binary_intervals": copy.deepcopy(base["analysis"]["binary_intervals"]),
            "paired": "For EACH outcome and cap, pair edited and original by identical question ID and seed. Report edited wins/original wins/ties, signed rate difference in percentage points, and two-sided exact binomial McNemar p (zero discordance => p=1).",
            "multiplicity": "Report raw p plus Holm-adjusted p across the five caps SEPARATELY for each outcome (three explicitly distinct families). Familywise control is within a family, not across all 15 tests. Primary natural completion retains priority; secondary/diagnostic results do not replace it.",
            "expectation_interpretation": "The estimated 28-40% closure range is not a confidence interval, significance threshold, or pass/fail gate. No post-result changes of endpoint, cap, grader, question subset or threshold.",
            "mean_tokens_at_k": base["analysis"]["mean_tokens_at_k"],
            "mean_tokens_interval": None, "mean_tokens_interval_note": base["analysis"]["mean_tokens_interval_note"],
            "incomplete": "Preserve partial records, report completed counts/missing IDs and reason; descriptive partial rates use their actual denominator. No planned n=30 paired inferential comparison until all 30 edited records exist and baseline integrity/weight checks pass.",
            "short_wrong": "List every edited main response with natural EOS <=1024 and strict correct delivery false, including ID, seed, tokens, parsed final answer, gold and excerpt. Do not regrade after inspection.",
            "power": base["analysis"]["power"],
            "limitations": [
                "Thirty GSM8K TRAIN items, one paired seed per item, one pinned 0.8B model; baseline already observed and original/edited were run at different times.",
                "GSM8K TRAIN is likely in Qwen's training data; per-item inclusion is unverified. Absolute accuracy is not a capability or benchmark claim.",
                "Thinking closure and natural EOS are distinct. The article-based closure estimate is unverified and its questions/runtime may differ; this is like-for-like in endpoint only, not a reproduction of the article's rate.",
                "One matched fixed-weight intervention, not training, a general capability claim, a variant search or a held-out evaluation.",
                "Existing prefix evidence is inherited under the identical deterministic generation implementation, not a new empirical edited-prefix check.",
                "Wilson bands are pointwise; n=30 has limited power and non-significance does not demonstrate equivalence.",
                "Durations include host pauses/delays. Preserve all timings; an ETA is not a deadline or a controlled performance benchmark.",
            ],
        },
        "durability_and_resume": {
            "lock": "One exclusive worker lock per run; never start concurrent generators.",
            "records": "Reuse the unchanged RecordStore's fsynced atomic per-response canonical JSON plus records.jsonl journal and durable reservation ledger. Flush a completed record before starting the next call; recover a journal from canonical records without changing token data.",
            "identity": "Bind every row to this approved plan SHA, question ID/seed, thinking_on, main, published_p50c variant and maximum8192. Record actual token IDs, decoded text, prompts, expected integer, EOS index, frozen grading, closure per cap, wall seconds and GPU peaks.",
            "edit_provenance": "Record verified original-before, edited-active and restored-original hashes. A record cannot silently be relabelled from original to edited; verify the edit is active before every generation.",
            "resume": "Verify all pinned sources, baseline/pool/model/edit hashes, matching approval and software/hardware first. Skip every durable completed record. Freshly load original weights, verify, apply and verify edit before any still-pending call.",
            "budget": base["token_budget"]["reservation_rule"],
            "retries": "Only an unfinished item may restart with the same seed under a newly funded reservation. Historical unresolved reservations stay fully charged even after the eventual response is completed.",
            "errors": "Determinism, hash, malformed return, decoding or restoration errors STOP. Preserve raw records and reservations; do not disable guards, silently drop records or continue with originals.",
        },
        "runtime_reporting": {
            "before_cuda": "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Torch import/CUDA initialization. Enable strict deterministic algorithms with warn_only=False; any failure stops, no fallback.",
            "time_policy": "No wall-time cutoff, subprocess kill timer, max_time or time-based stopping criterion may stop a generation. Log wall time only.",
            "token_248046": "Log every containing response with occurrence count; report response and occurrence totals separately for original and edited. EOS remains 248044 only.",
            "profile": "After two edited main records, report measured tokens/s, peak allocated/reserved GPU memory and a clearly provisional finish projection, then continue automatically. No extra warmup/probe generations.",
        },
        "deliverables_after_approval": [
            "This run directory: unchanged plan.json/PLAN.md, approval.json bound to the plan SHA, runtime source manifest, records.jsonl, canonical records, reservation ledger, summary.json and weight-verification/error history.",
            "P50C_COMPLETION_REPORT.md with a five-sentence public summary, fixed design, all three outcome tables/Wilson intervals, paired counts/exact p/Holm, short wrong answers, diagnostic248046, timing caveats, limitations and stop.",
            "One clearly labelled PNG+SVG figure with primary natural completion, secondary strict correct delivery and diagnostic thinking closure panels versus log2 token budget; original and edited lines with n=30 and pointwise Wilson bands.",
            "Read-only 07_p50c_completion.ipynb, RUN_EXPERIMENT=False, recomputing from saved original/edited records; no inference by default.",
            "All existing and new offline tests pass; verify prior baseline reports/scorers/notebooks/records are unchanged, original weights restored and cached checkpoint unchanged; save validation.json.",
        ],
        "prohibitions": [
            "No generation, edit application or approval record before explicit approval.",
            "No new downloads, installations, held-out use, new questions, resampling, baseline regeneration, extra seeds/conditions/strengths/layer ablations or model saves.",
            "No changes to completed sweep reports/scoring/records/notebooks; no commits or pushes.",
            "No adaptive extension after results or spending leftover budget on unscheduled generations. After reporting this one fixed comparison, STOP.",
        ],
    }


def validate_plan(plan, base):
    if plan["generation_authorized"] is not False or plan["generated_responses"] != 0:
        raise ValueError("Planning cannot authorize or perform generation.")
    if (
        plan["questions_and_seeds"] != base["questions_and_seeds"] or len(plan["schedule"]) != 30
        or [job["seed"] for job in plan["schedule"]] != list(range(30))
        or [job["problem_id"] for job in plan["schedule"]] != [item["id"] for item in base["questions_and_seeds"]]
        or any(job["condition"] != "thinking_on" or job["model_variant"] != "published_p50c"
               or job["kind"] != "main" or job["max_new_tokens"] != 8192 for job in plan["schedule"])
    ):
        raise ValueError("The fixed thirty edited thinking-on jobs changed.")
    if (
        plan["sampling"] != base["sampling"] or plan["batch_size"] != 1
        or any(plan["prompt"][key] != value for key, value in base["prompt"].items())
        or any(plan["execution"][key] != base["execution"][key] for key in DETERMINISM_KEYS)
        or any(plan["model"][key] != base["model"][key] for key in (
            "repo", "revision", "weights_file", "checkpoint_sha256", "dtype", "device",
            "class", "quantization", "cached_config_sha256", "target_base_sha256",
        ))
        or plan["hardware"] != base["hardware"]
        or plan["prefix_validation"]["identical_determinism_signature"] != {key: base["execution"][key] for key in DETERMINISM_KEYS}
    ):
        raise ValueError("Generation policy differs from the baseline; new plan/check/approval required.")
    if (
        plan["token_budget"]["limit"] != 250000 or plan["token_budget"]["total_worst_case_no_crash"] != 30 * 8192
        or plan["token_budget"]["headroom"] != 4240 or plan["prefix_validation"]["new_short_generations"] != 0
        or plan["prefix_validation"]["new_check_scheduled"] is not False
        or not plan["prefix_validation"]["baseline_all_four_passed"]
        or len(plan["prefix_validation"]["baseline_comparisons"]) != 4
        or not all(item["identical"] for item in plan["prefix_validation"]["baseline_comparisons"])
        or plan["analysis"]["caps"] != [512, 1024, 2048, 4096, 8192]
    ):
        raise ValueError("Budget, inherited validation or cap policy changed.")
    if (
        plan["stopping"]["eos_token_id"] != 248044 or plan["stopping"]["forced_eos_token_id"] is not None
        or plan["stopping"]["max_time"] is not None or plan["stopping"]["wall_time_generation_limit"] is not None
        or plan["edit"]["layers"] != [15, 19] or plan["edit"]["strength"] != 1.0
        or plan["edit"]["model_or_tensor_saves_allowed"] is not False
        or plan["outcomes"]["primary"]["name"] != "natural_completion"
        or plan["outcomes"]["diagnostic"]["requires_eos"] is not False
    ):
        raise ValueError("Outcome, stopping or verified edit policy changed.")


def main():
    plan, diagnostic, before = build_plan()
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("Plan preparation must not import the model runtime.")
    now = datetime.fromisoformat(plan["created_utc"])
    directory = ROOT / "results" / now.strftime("p50c_completion_%Y%m%dT%H%M%S_%fZ")
    directory.mkdir()
    atomic_json(directory / "plan.json", plan)
    digest = sha256(directory / "plan.json")
    atomic_bytes(directory / "PLAN.md", render_plan(plan, digest).encode("utf-8"))
    atomic_json(directory / "baseline_closure.json", diagnostic)
    after = protected_snapshot()
    if after != before:
        raise RuntimeError("A protected completed artifact changed.")
    atomic_json(directory / "planning_validation.json", {
        "plan_sha256": digest, "generated_responses": 0, "inference_performed": False,
        "model_runtime_imported": False, "weights_loaded_or_applied": False,
        "baseline_diagnostic_matches_raw_recomputation": True, "protected_artifacts_unchanged": True,
        "protected_artifact_sha256": before, "approval_file_present": (directory / "approval.json").exists(),
    })
    print(json.dumps({"plan_directory": str(directory), "plan_sha256": digest, "status": plan["status"]}, indent=2))


if __name__ == "__main__":
    main()
