"""Inference-free thinking-closure diagnostics from saved generated token IDs."""

import importlib.metadata
import json
import sys

from tokenizers import Tokenizer

from budget_sweep_io import ROOT, atomic_json, read_json, sha256
from budget_sweep_outputs import load_saved_run
from budget_sweep_scoring import CAPS, score_at_cap, wilson

BASELINE = ROOT / "results" / "budget_sweep_20261106T191821_788114Z"
BASELINE_DIAGNOSTIC = ROOT / "p50c_unedited_closure_baseline.json"
CLOSING_TEXT = "</think>"


def closure_at_caps(token_ids, decode):
    if (
        not isinstance(token_ids, (list, tuple)) or not 1 <= len(token_ids) <= 8192
        or any(type(token) is not int or not 0 <= token < 2**31 for token in token_ids)
        or not callable(decode)
    ):
        raise ValueError("Expected 1-8192 integer token IDs and a local decoder.")
    result = {}
    for cap in CAPS:
        text = decode(list(token_ids[:cap]))
        if not isinstance(text, str) or len(text) > 200000:
            raise ValueError("Invalid decoded generated prefix.")
        result[str(cap)] = CLOSING_TEXT in text
    return result


def protected_snapshot():
    paths = {path for path in BASELINE.rglob("*") if path.is_file()}
    paths.update(ROOT.glob("*.ipynb"))
    paths.update(ROOT / name for name in (
        "BUDGET_SWEEP_REPORT.md", "PROMPT_CONTROL_REPORT.md", "EXPERIMENT_REPORT.md",
        "probe_metrics.py", "math_probe.py", "budget_sweep_scoring.py",
        "budget_sweep_io.py", "budget_sweep_analysis.py", "budget_sweep_generate.py",
        "budget_sweep_plan.py", "budget_sweep_worker.py", "weight_edit.py",
    ))
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(paths)}


def baseline_diagnostic():
    before = protected_snapshot()
    plan, pool, records, saved, derived, _ = load_saved_run(BASELINE)
    if saved["status"] != "complete" or not derived["complete_data"] or not saved["target_hash_checks"]["verified_after"]:
        raise ValueError("A complete verified unedited baseline is required.")
    model_dir = ROOT / "models" / "Qwen3.5-0.8B"
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = model_dir / name
        if sha256(path) != plan["model"]["cached_config_sha256"][path.relative_to(ROOT).as_posix()]:
            raise ValueError("The pinned local tokenizer changed.")
    config = read_json(model_dir / "tokenizer_config.json")
    if config.get("clean_up_tokenization_spaces") is not False:
        raise ValueError("The baseline decoder's cleanup setting changed.")
    closing = [
        int(token) for token, metadata in config["added_tokens_decoder"].items()
        if metadata.get("content") == CLOSING_TEXT
    ]
    if closing != [248069]:
        raise ValueError("Unexpected closing-delimiter token.")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False)

    main = {row["problem_id"]: row for row in records if row["kind"] == "main" and row["condition"] == "thinking_on"}
    if len(main) != 30 or set(main) != {item["id"] for item in pool}:
        raise ValueError("Expected exactly the 30 unedited thinking-on main records.")
    per_question = []
    for item in plan["questions_and_seeds"]:
        row = main[item["id"]]
        if row["seed"] != item["seed"] or row["thinking"] is not True or decode(row["token_ids"]) != row["text"]:
            raise ValueError("Baseline identity or decoded response mismatch.")
        outcomes = closure_at_caps(row["token_ids"], decode)
        positions = [index + 1 for index, token in enumerate(row["token_ids"]) if token == closing[0]]
        if len(positions) != row["text"].count(CLOSING_TEXT):
            raise ValueError("Baseline literal/canonical closing-delimiter counts disagree.")
        first = positions[0] if positions else None
        if first is not None and (
            CLOSING_TEXT not in decode(row["token_ids"][:first])
            or CLOSING_TEXT in decode(row["token_ids"][:first - 1])
        ):
            raise ValueError("Baseline first closing-token boundary is inconsistent.")
        per_question.append({
            "id": item["id"], "seed": item["seed"],
            "generated_tokens": len(row["token_ids"]), "thinking_closed_at_cap": outcomes,
            "first_canonical_closing_token_index": first, "closing_delimiter_occurrences": len(positions),
            "natural_eos_index": row["graded"]["natural_eos_index"],
            "natural_completion_at_cap": {str(k): score_at_cap(row["graded"], k)["natural_completion"] for k in CAPS},
            "strict_correct_delivery_at_cap": {str(k): score_at_cap(row["graded"], k)["correct"] for k in CAPS},
        })
    tables = []
    for cap in CAPS:
        closed = sum(row["thinking_closed_at_cap"][str(cap)] for row in per_question)
        tables.append({"cap": cap, "n": 30, "thinking_closed": closed, "rate": closed / 30, "wilson95": wilson(closed, 30)})
    if protected_snapshot() != before:
        raise RuntimeError("A protected completed artifact changed during this read-only calculation.")
    return {
        "schema_version": 1, "scope": "Completed unedited thinking-on main responses only; excludes short validation and thinking-off.",
        "inference_performed": False, "weights_loaded_or_applied": False,
        "baseline_directory": BASELINE.relative_to(ROOT).as_posix(),
        "baseline_plan_sha256": sha256(BASELINE / "plan.json"),
        "baseline_records_sha256": sha256(BASELINE / "records.jsonl"),
        "baseline_summary_sha256": sha256(BASELINE / "summary.json"),
        "definition": "Exact literal </think> occurs wholly within decoded generated token IDs [:k]. No EOS or correctness requirement; do not count the prompt.",
        "decoder": {"implementation": "tokenizers.Tokenizer.decode", "tokenizers_version": importlib.metadata.version("tokenizers"),
                    "skip_special_tokens": False, "clean_up_tokenization_spaces": False,
                    "tokenizer_sha256": sha256(model_dir / "tokenizer.json"), "canonical_closing_token_id": 248069,
                    "all_full_decodes_equal_saved_text": True},
        "tables": tables, "per_question": per_question,
        "protected_artifact_sha256": before, "protected_artifacts_unchanged": True,
    }


if __name__ == "__main__":
    artifact = baseline_diagnostic()
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("This diagnostic must not import the model runtime.")
    atomic_json(BASELINE_DIAGNOSTIC, artifact)
    print(json.dumps({"output": str(BASELINE_DIAGNOSTIC), "tables": artifact["tables"]}, indent=2))
