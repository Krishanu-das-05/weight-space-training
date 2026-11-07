"""Finalize saved data only: report, figure, read-only notebook and offline gates."""

import argparse
import ast
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from budget_sweep_io import ROOT, atomic_bytes, atomic_json, sha256
from budget_sweep_outputs import load_saved_run, write_report

ORIGINAL_NOTEBOOKS = {
    "01_inspect_model.ipynb": "4a1c3bf3bbb75edb0f767325a1d22f6bdc59af76792dcca6596ff63720479b20",
    "02_replay_weight_edit.ipynb": "304a54dc6c8649d60d3fd6d92a131ab949f6d213e392b80979d2acee201417c0",
    "03_edit_anatomy.ipynb": "fe08153a70b8156945f58a62fe5ab8556c329a4b673499b86e99f07d88cfd76b",
    "04_controlled_experiments.ipynb": "36e77c5fbf5df504966f05ca77208dbc19e4a8e30fe670ce00c1b181f158dabb",
    "05_prompt_controls.ipynb": "0c6d73e1374f764e9ae1d138ada998d46937320920b170b794185fda67b92df1",
}


def finalize(directory):
    directory = Path(directory).resolve()
    plan, pool, records, saved, derived, budget = load_saved_run(directory)
    if saved["status"] in ("running", "initializing"):
        raise RuntimeError("The worker has not recorded a terminal status; do not finalize a live run.")
    for name, expected in ORIGINAL_NOTEBOOKS.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError("An original notebook differs from its preserved snapshot: " + name)
    if sha256(ROOT / plan["model"]["weights_file"]) != plan["model"]["checkpoint_sha256"]:
        raise RuntimeError("The cached original checkpoint changed.")
    if sha256(ROOT / plan["dataset"]["source_file"]) != plan["dataset"]["sha256"]:
        raise RuntimeError("The cached approved TRAIN artifact changed.")
    for key in ("tables", "paired_comparisons", "hypothesis_pattern", "token_248046_diagnostics",
                "short_wrong_answers", "prefix_validation"):
        if saved[key] != derived[key]:
            raise RuntimeError("Saved summary differs from raw recomputation: " + key)
    if saved["budget"] != budget:
        raise RuntimeError("Saved budget differs from its durable ledger.")
    if saved["status"] == "complete" and (
        not derived["complete_data"] or len(records) != 64 or not saved["target_hash_checks"]["verified_after"]
    ):
        raise RuntimeError("The complete-run claim is not supported by all integrity checks.")
    write_report(directory)
    environment = os.environ.copy()
    environment.update(
        MPLCONFIGDIR=str(ROOT / ".cache" / "matplotlib"), MPLBACKEND="Agg",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        JUPYTER_RUNTIME_DIR=str(ROOT / ".cache" / "jupyter-runtime"),
    )
    renderer = getattr(sys, "_base_executable", sys.executable)
    subprocess.run([renderer, str(ROOT / "plot_budget_sweep.py"), str(directory)],
                   cwd=ROOT, env=environment, check=True)
    notebook_path = ROOT / "06_budget_sweep.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    assignments = [
        node for cell in notebook.cells if cell.cell_type == "code"
        for node in ast.walk(ast.parse(cell.source))
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "RUN_EXPERIMENT"
                                               for target in node.targets)
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Constant) or assignments[0].value.value is not False:
        raise RuntimeError("Notebook default must explicitly disable inference.")
    old_environment = {name: os.environ.get(name) for name in environment if name in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUBLAS_WORKSPACE_CONFIG", "JUPYTER_RUNTIME_DIR",
    )}
    try:
        os.environ.update({name: environment[name] for name in old_environment})
        NotebookClient(
            notebook, timeout=120, kernel_name="weight-space",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
    finally:
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise RuntimeError("The read-only notebook contains execution errors.")
    atomic_bytes(notebook_path, nbformat.writes(notebook).encode("utf-8"), replace=True)
    with (directory / "accuracy_vs_token_budget.png").open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("Invalid PNG header.")
    dimensions = list(struct.unpack(">II", header[16:24]))
    if dimensions[0] < 2000 or dimensions[1] < 900:
        raise RuntimeError("Figure dimensions are too small for sharing.")
    svg = (directory / "accuracy_vs_token_budget.svg").read_text(encoding="utf-8")
    if "<svg" not in svg or "<script" in svg.lower() or re.search(r"(?:href|src)=[\"'](?!#)", svg):
        raise RuntimeError("Unexpected executable/external SVG content.")
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=ROOT, env=environment, check=True)
    subprocess.run(["uv", "pip", "check", "--python", sys.executable], cwd=ROOT, env=environment, check=True)
    for name, expected in ORIGINAL_NOTEBOOKS.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError("An original notebook changed during finalization.")
    stages = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    audit = {
        "raw_recomputation_matches_summary": True, "cached_checkpoint_sha256_verified": True,
        "cached_dataset_sha256_verified": True, "original_notebook_sha256": ORIGINAL_NOTEBOOKS,
        "notebook06_executed_without_inference_or_errors": True, "png_dimensions": dimensions,
        "svg_has_no_executable_or_external_content": True,
        "all_unittests_pass": True, "installed_dependency_compatibility_pass": True,
        "staged_file_count": len(stages), "status": saved["status"],
        "artifacts": {name: sha256(directory / name) for name in (
            "plan.json", "records.jsonl", "summary.json", "BUDGET_SWEEP_REPORT.md",
            "accuracy_vs_token_budget.png", "accuracy_vs_token_budget.svg",
        )},
    }
    atomic_json(directory / "validation.json", audit, replace=True)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    finalize(parser.parse_args().run)
