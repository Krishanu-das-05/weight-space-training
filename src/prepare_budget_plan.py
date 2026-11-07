"""Create an offline preregistration only. This module cannot run generation."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from budget_sweep_scoring import CAPS, WILSON_Z
from download_model import MODEL_DIR, MODEL_ID, REVISION, WEIGHT_FILE, WEIGHT_SHA256
from gsm8k_source import CACHE, ROOT, file_sha256, write_json
from math_probe import PROMPT_INSTRUCTIONS, SAMPLING
from probe_metrics import validate_problem


TRAINING_CONTAMINATION_CAVEAT = (
    "GSM8K TRAIN is likely to be present in Qwen's training data; per-item "
    "inclusion has not been verified. Absolute accuracy on this development "
    "sample is not a capability claim."
)


def windows_json(command):
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True, capture_output=True, text=True, cwd=ROOT,
    )
    return json.loads(completed.stdout)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = json.loads((CACHE / "receipt.json").read_text(encoding="utf-8"))
    selection = json.loads((CACHE / "selection.json").read_text(encoding="utf-8"))
    decoder = json.loads((CACHE / "decoder.json").read_text(encoding="utf-8"))
    pool_path = ROOT / "dev_pool_gsm8k_30.jsonl"
    pool = [json.loads(line) for line in pool_path.read_text(encoding="utf-8").splitlines()]
    if len(pool) != 30 or file_sha256(pool_path) != selection["dev_pool_sha256"]:
        raise ValueError("The frozen 30-question selection changed.")
    for item in pool:
        validate_problem(item)
    if file_sha256(CACHE / "train.parquet") != receipt["sha256"]:
        raise ValueError("The approved TRAIN artifact changed.")
    if file_sha256(CACHE / "api_dataset_info.json") != receipt["metadata_sha256"]:
        raise ValueError("The approved dataset metadata changed.")
    if file_sha256(CACHE / "train.jsonl") != decoder["decoded_jsonl_sha256"]:
        raise ValueError("The decoded TRAIN data changed.")
    weights_path = MODEL_DIR / WEIGHT_FILE
    if file_sha256(weights_path) != WEIGHT_SHA256:
        raise ValueError("The cached checkpoint is not the pinned original.")
    manifest = json.loads((ROOT / ".cache" / "published-edit" / "manifest.json").read_text(encoding="utf-8"))
    target_hashes = {}
    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        for spec in manifest["tensors"]:
            tensor = checkpoint.get_tensor(spec["name"])
            digest = hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
            if digest != spec["base_sha256"]:
                raise ValueError("Original target tensor verification failed.")
            target_hashes[spec["name"]] = digest
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    generation = GenerationConfig.from_model_config(config)
    if generation.eos_token_id != 248044 or tokenizer.pad_token_id != 248044:
        raise ValueError("The prior run's EOS/padding configuration changed.")
    cpu = windows_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress")
    ram = windows_json("Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum | Select-Object Sum | ConvertTo-Json -Compress")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip()
    properties = torch.cuda.get_device_properties(0)
    questions = [{"id": item["id"], "seed": index} for index, item in enumerate(pool)]
    conditions = [{"name": "thinking_on", "enable_thinking": True}, {"name": "thinking_off", "enable_thinking": False}]
    schedule = []
    for index, item in enumerate(questions):
        for condition in conditions:
            base = {"problem_id": item["id"], "seed": item["seed"], "condition": condition["name"]}
            if index < 2:
                schedule.append({**base, "kind": "prefix_validation_short", "max_new_tokens": 1024})
            schedule.append({**base, "kind": "main", "max_new_tokens": 8192})
    source_names = (
        "probe_metrics.py", "math_probe.py", "budget_sweep_scoring.py",
        "gsm8k_source.py", "decode_gsm8k_train.py", "prepare_budget_plan.py",
        "test_budget_sweep_plan.py", "test_gsm8k_download.py",
    )
    created = datetime.now(timezone.utc)
    plan = {
        "design_version": "budget-sweep-v1", "status": "AWAITING_USER_APPROVAL",
        "generation_authorized": False, "generated_responses": 0, "created_utc": created.isoformat(),
        "approval": {
            "record_file": "approval.json", "must_reference_sha256_of_this_plan": True,
            "policy": "A separate approval record may be written only after the user explicitly approves this exact plan. The frozen plan itself is not rewritten to authorize execution.",
        },
        "hypothesis": "Under fixed step-by-step instructions, thinking-on correct delivery rises substantially with token budget while thinking-off stays approximately flat.",
        "model": {
            "repo": MODEL_ID, "revision": REVISION, "weights_file": str(weights_path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": WEIGHT_SHA256, "dtype": "bfloat16", "device": "cuda:0",
            "class": "Qwen3_5ForConditionalGeneration", "quantization": None,
            "local_files_only": True, "trust_remote_code": False,
            "target_base_sha256": target_hashes, "planning_cached_target_hashes_verified": True,
            "required_runtime_checks": "Verify loaded target hashes before and after, including error/interrupt cleanup; hash mismatch stops the run. Never edit, restore, or save weights; never enter any weight-edit context.",
            "cached_config_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
                for path in sorted(MODEL_DIR.iterdir()) if path.suffix in (".json", ".jinja")
            },
        },
        "dataset": {
            **receipt, "source_file": ".cache/gsm8k-main-train/train.parquet",
            "metadata_file": ".cache/gsm8k-main-train/api_dataset_info.json",
            "offline_decoder": decoder, "sampling": selection,
            "dev_file": "dev_pool_gsm8k_30.jsonl", "dev_sha256": file_sha256(pool_path),
            "answer_verification_scope": "Exact integer parsing of the source TRAIN annotations after ####, plus file/row identity checks; not independent mathematical proof of every annotation.",
            "overlap_policy": "Exact and case-sensitive Unicode-whitespace-collapsed question comparison against both local held-out pools. Collision => STOP, never silently resample. Held-out answers are not used.",
        },
        "smoke_tests": {"file": "smoke_pool_handmade_4.jsonl", "sha256": selection["smoke_pool_sha256"], "count": 4, "scheduled_generations": 0, "included_in_analysis": False},
        "questions_and_seeds": questions, "conditions": conditions, "batch_size": 1,
        "prompt": {"style": "step_by_step", "content": "question + newline + instruction", "instruction": PROMPT_INSTRUCTIONS["step_by_step"], "include_gold_in_prompt": False},
        "sampling": {**SAMPLING, "do_sample": True, "num_beams": 1, "use_cache": True, "cache_implementation": "dynamic", "min_new_tokens": 0},
        "stopping": {
            "main_max_new_tokens": 8192, "eos_token_id": 248044, "pad_token_id": 248044,
            "tokenizer_eos_token_id_not_used_as_extra_stop": 248046,
            "forced_eos_token_id": None, "max_time": None, "wall_time_generation_limit": None,
            "other_stopping_criteria": [], "stop_on_box_or_think_tag": False,
            "valid_return": "Main responses must end at their first EOS or contain exactly 8192 tokens. Non-EOS early returns, tokens after EOS, or over-budget returns are fatal; short validation uses the equivalent 1024 rule.",
            "eos_index": "1-based generated-token index including EOS; null if no EOS. Match prior model generation config, not tokenizer EOS alias.",
            "terminal_tokens_for_grading": tokenizer.all_special_tokens,
        },
        "execution": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda_runtime": torch.version.cuda, "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "seed_rule": "For each generation, fork CPU/CUDA RNG state and reset both to the recorded per-question seed; same seed for short/long and both conditions.",
            "attention_implementation": "eager", "compile": False, "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8", "cudnn_benchmark": False,
            "float32_matmul_precision": "highest", "tf32": False,
            "model_eval": True, "torch_inference_mode": True,
            "worker": ".venv/Scripts/python.exe; fresh process; no subprocess timeout or time-based stopping criteria",
            "existing_short_cap_runner_must_not_be_used": "experiment_runner/run_probes currently have incompatible cap/time safeguards. A separate approved-spec worker must be implemented and tested after plan approval.",
        },
        "hardware": {
            "os": platform.platform(), "cpu": cpu, "ram_bytes": ram["Sum"],
            "gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
            "nvidia_driver": driver,
        },
        "prefix_validation": {
            "question_ids": [item["id"] for item in questions[:2]], "both_conditions": True,
            "short_runs": 4, "long_runs_reused_in_main": 4,
            "rule": "short_tokens == long_tokens[:min(1024, len(long_tokens))], including early EOS; compare integer token IDs, never decoded strings.",
            "order": "For each of the first two questions and both conditions: short 1024 then long 8192, flush both, compare immediately. Only if all four pass, schedule remaining 56 main responses.",
            "on_mismatch": "STOP and preserve/report all completed records; no main continuation, retuning, or extra generations.",
        },
        "schedule": schedule,
        "token_budget": {
            "limit": 500000, "main_worst_case": 491520, "validation_short_worst_case": 4096,
            "total_worst_case_no_crash": 495616, "headroom": 4384,
            "includes": "Every generated token in main, short validation, and interrupted attempts.",
            "reservation_rule": "Durably reserve the full call cap BEFORE generation. Reconcile to actual tokens only after its completed record is durable. Unreconciled crash reservations remain charged at the full cap. A retry needs a new reservation. Never start an unfunded call; stop cleanly as incomplete.",
        },
        "durability_and_resume": {
            "exclusive_worker": True,
            "record_commit": "Persist an immutable per-generation record atomically, flush+fsync; append and flush+fsync records.jsonl before the next call.",
            "identity": "(design hash, problem ID, condition, kind); validate seeds and cap too. Skip every already completed identity, including validation responses.",
            "recovery": "Reconcile committed records and reservation ledger. Preserve original crash-damaged journal bytes in an immutable backup before rebuilding a canonical journal from committed records. Never discard completed records or silently ignore corruption.",
            "unfinished_calls": "May restart from their fixed seed, but their unreconciled reservation is still charged. Completed calls are never repeated.",
            "drift": "Changed plan/data/grader/scorer/model/software/hardware settings => stop and obtain amended approval, not silent continuation.",
        },
        "analysis": {
            "caps": list(CAPS), "sample_size": 30,
            "correct_at_k": "Natural EOS index <= k AND unchanged strict final boxed-integer grader is correct; a box in unfinished reasoning is never delivered.",
            "completion_at_k": "Natural EOS index <= k regardless of correctness.",
            "mean_tokens_at_k": "Mean(min(actual generated length, k)), including EOS when earlier; counterfactual tokens consumed at cap, NOT actual compute already spent producing the 8192 run.",
            "binary_intervals": {"method": "two-sided Wilson score", "confidence": 0.95, "z": WILSON_Z},
            "mean_tokens_interval": None,
            "mean_tokens_interval_note": "Wilson is binomial and invalid for a continuous mean. Mean tokens is a point estimate only; no unrequested post-hoc interval.",
            "paired": "Per question at each cap: thinking-on wins / thinking-off wins / ties, two-sided exact binomial McNemar, discordance=0 => p=1.",
            "multiplicity": "Report raw exact p-values and Holm-adjusted p-values for the family of five cap comparisons; no significance fishing.",
            "power": "With n=30 only large effects (~25+ percentage points) are detectable; exact power depends on baseline and paired discordance. Non-significance does not prove equivalence.",
            "descriptive_hypothesis_pattern": "Compare 8192 vs 1024: on gains >=8/30 (26.7 points) AND off gains <=3/30 (10 points). Label only consistent/not consistent with budget-sensitivity pattern, not causal proof or equivalence.",
            "short_wrong": "List ALL main responses with natural EOS <=1024 and strict final answer incorrect; report question ID, condition, seed, tokens, parsed final answer and gold. Show response excerpts without silently correcting grading.",
            "incomplete": "Label partial, state completed counts/missing IDs; do not claim n=30 or run the planned complete-data inferential comparison. Save stop reason and budget accounting.",
            "limitations": [
                "Thirty selected TRAIN development items, one seed per item, one 0.8B model; not a benchmark or generalization claim.",
                TRAINING_CONTAMINATION_CAVEAT,
                "No claim about model training or new capability; weights unchanged.",
                "Instruction is held fixed, and the prior prompt-control pool differs: this cannot isolate cap rather than instruction as the primary cause.",
                "Strict integer/closure/EOS rubric differs from broader mathematical-equivalence grading.",
                "Prefix identity is checked on two questions per condition, not proved for all inputs.",
                "Source annotation correctness, near-duplicate overlap, and pretraining contamination are not established by integer parsing or exact whitespace checks.",
            ],
        },
        "source_sha256": {name: file_sha256(ROOT / name) for name in source_names},
        "deliverables_after_approval": {
            "run_files": ["plan.json", "PLAN.md", "records.jsonl", "summary.json", "durable_records/", "budget_ledger/"],
            "report": "BUDGET_SWEEP_REPORT.md with exactly five plain-language summary sentences at top, then question/design/results/paired effects/limitations/stop",
            "required_report_caveat": TRAINING_CONTAMINATION_CAVEAT,
            "public_summary_must_include_training_contamination_caveat": True,
            "figure": {
                "files": ["accuracy_vs_budget.png", "accuracy_vs_budget.svg"],
                "panels": ["accuracy with Wilson bands", "natural completion with Wilson bands"],
                "x_scale": "log2", "x_ticks": list(CAPS), "x_limits": [512, 8192], "y_limits": [0, 1],
                "labels": "Qwen3.5-0.8B, GSM8K TRAIN development sample; explicit token budget and rate axes; n=30 only if complete",
                "lines": "one per condition, markers at measured caps; no smoothing",
                "colors": {"thinking_on": "#145D73", "thinking_off": "#BC6C25"},
                "size_inches": [12, 5.2], "png_dpi": 200, "confidence_alpha": 0.15,
                "renderer": "already-installed system Matplotlib 3.11.0, Agg, bundled DejaVu Sans; MPLCONFIGDIR inside project .cache",
            },
            "notebook": "06_budget_sweep.ipynb; RUN_EXPERIMENT=False; recompute all tables from raw saved records and frozen scorer, not summary.json claims",
        },
        "prohibitions": ["generation before explicit approval", "extra downloads", "package installs", "weight edit contexts or saves", "held-out generation/scoring/tuning", "changing notebooks 01-05 or prior runs", "commits", "pushes"],
    }
    run_dir = ROOT / "results" / created.strftime("budget_sweep_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir()
    write_json(run_dir / "plan.json", plan)
    markdown = f"""# Budget Sweep: Frozen Plan

Status: **AWAITING YOUR APPROVAL. No generation has run.**

## Question
Under the same step-by-step request, does thinking-on correct delivery rise
substantially as the cap grows, while thinking-off stays approximately flat?

## Data And Model
- Original Qwen3.5-0.8B revision `{REVISION}`, BF16, CUDA, batch 1.
- Cached checkpoint and both original target tensor hashes verified during planning; repeat loaded-tensor checks before and after execution. No edits, edit contexts, restores, or model saves.
- GSM8K `main` TRAIN only, MIT, commit `{receipt['revision']}`.
- **Training-data caveat:** {TRAINING_CONTAMINATION_CAVEAT} Include this caveat in the eventual report and its public summary.
- One TRAIN download, SHA-256 `{receipt['sha256']}`; no TEST data or other downloads.
- 30 items in `dev_pool_gsm8k_30.jsonl`, sampling seed **20261106**, deterministic full-TRAIN shuffle then integer-final filtering. Zero items skipped in this sample.
- Exact and whitespace-normalized collision tests pass against the locally handmade held-out pools. Held-out use stops at provenance/collision checks.
- All 30 IDs and per-question seeds **0-29** are fixed in `plan.json`. The original four handmade smoke items stay separate and unscheduled.

## Generation And Safety
- Thinking on/off via chat template; both use the **unchanged step-by-step instruction** and sampling 0.6 / top-p 0.95 / top-k 20 / presence penalty 1.5 / repetition penalty 1.0.
- Generate each of the **60 main responses once at 8192**, never separately per analysis cap. Record all token IDs, raw text, seed, prompts, timing and the 1-based natural EOS index.
- EOS is **248044**, matching the prior model generation config; tokenizer alias 248046 is not an added stop. No forced EOS or stop at a box/think tag.
- First two selected questions, both conditions: **four real short 1024 runs**, each followed by its 8192 run; exact token-prefix comparison including early EOS. Long runs count toward the 60 main responses. Any mismatch stops the experiment before the remaining 56.
- Fixed threads **1 intra-op / 1 inter-op**, batch 1, deterministic algorithms, eager attention, BF16, no compilation; versions and hardware fixed in JSON.
- **No wall-time limit** may stop a generation. Time is logged only. The old short-cap/time-limited runner is not suitable and will not be reused unchanged.
- Worst case **491,520 + 4,096 = 495,616 tokens**, below the **500,000** limit. All validation/interruption tokens count. Durable pre-call reservations prevent budget overshoot even after a crash.
- Flush and fsync every completed record before the next call. Resume skips completed identities; incomplete calls may restart only with an additional funded reservation. Preserve crash artifacts. Stop cleanly if unfunded.

## Frozen Analysis
- Caps: **512, 1024, 2048, 4096, 8192**, derived from each single long response.
- Correct at k means **natural EOS at/before k AND the unchanged strict boxed-integer final-answer grade is correct**. Unfinished thinking never counts.
- Accuracy and natural-completion proportions get **95% Wilson intervals**. Mean tokens is `mean(min(length,k))`, a point estimate: Wilson intervals are not valid for a continuous mean.
- Per-cap paired thinking-on/off wins/losses/ties and **exact two-sided McNemar**; report raw and Holm-adjusted p-values for the five comparisons.
- With **n=30**, only large effects (~25+ points) are detectable; non-significance is not equivalence.
- Descriptive pattern: on gains at least **8/30** from 1024 to 8192, off gains at most **3/30**. This cannot prove that cap rather than instruction is the main cause: instruction is fixed, and the previous experiment used different questions.
- List every naturally completed, wrong main answer of at most **1024 tokens**. Freeze grader/scorer source hashes now; do not revise after seeing results.
- If incomplete, label the data partial and do not claim the planned n=30 inference.

## Deliverables After Approval
`records.jsonl`, `summary.json`, a report with a five-sentence public summary,
a labelled log-x two-panel PNG+SVG (accuracy/completion with Wilson bands),
and read-only-by-default `06_budget_sweep.ipynb` recomputing from raw records.
No benchmark, model-improvement, or broader-generalization claims.

**Approval is for the plan only at this point. No inference starts until you explicitly approve.**
"""
    with (run_dir / "PLAN.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown)
    print("PLAN:", run_dir.relative_to(ROOT))
    print("PLAN SHA-256:", file_sha256(run_dir / "plan.json"))
    print("Main generations:", sum(item["kind"] == "main" for item in schedule))
    print("Short validation generations:", sum(item["kind"] != "main" for item in schedule))
    print("Worst-case generated tokens:", sum(item["max_new_tokens"] for item in schedule))
    print("Generation is NOT authorized or implemented by this planner.")


if __name__ == "__main__":
    main()
