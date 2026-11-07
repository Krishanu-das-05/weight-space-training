"""Execute only the generated read-only notebook using project-local runtime files."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import json
import sys
import uuid

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from budget_sweep_io import atomic_bytes, atomic_json, read_json, sha256
from p50c_completion_io import RUN
from p50c_completion_outputs import make_notebook


def local_kernel(cache):
    cache = Path(cache).resolve()
    if not cache.is_relative_to(ROOT / ".cache"):
        raise ValueError("Notebook runtime files must remain inside the project's cache.")
    cache.mkdir(exist_ok=True)
    kernels = cache / "kernels"
    kernels.mkdir(exist_ok=True)
    spec_directory = kernels / "weight-space"
    spec_directory.mkdir(exist_ok=True)
    spec = {
        "argv": [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (weight-space)", "language": "python",
        "metadata": {"supported_encryption": ["curve"]},
    }
    spec_path = spec_directory / "kernel.json"
    if spec_path.exists():
        if read_json(spec_path) != spec:
            raise ValueError("The project-local kernel spec differs; do not replace it automatically.")
    else:
        atomic_json(spec_path, spec)
    environment = os.environ.copy()
    environment.update(
        IPYTHONDIR=str(cache / "ipython"), JUPYTER_CONFIG_DIR=str(cache / "config"),
        JUPYTER_DATA_DIR=str(cache / "data"), JUPYTER_RUNTIME_DIR=str(cache),
        MPLCONFIGDIR=str(ROOT / ".cache" / "matplotlib"), PYTHONUTF8="1",
    )
    specs = KernelSpecManager(kernel_dirs=[str(kernels)], ensure_native_kernel=False)
    manager = KernelManager(
        kernel_name="weight-space", kernel_spec_manager=specs,
        connection_file=str(cache / ("kernel-" + uuid.uuid4().hex + ".json")),
        ip="127.0.0.1",
        transport_encryption="required",
    )
    return manager, environment


def execute():
    path = RUN / "07_p50c_completion.ipynb"
    audit = read_json(RUN / "finalization.json")
    relative = path.relative_to(ROOT).as_posix()
    if sha256(path) != audit["artifact_sha256"][relative]:
        raise ValueError("Notebook differs from the generated artifact; refusing to execute changed code.")
    notebook = nbformat.read(path, as_version=4)
    expected = make_notebook()
    if [(cell.cell_type, cell.source) for cell in notebook.cells] != [(cell.cell_type, cell.source) for cell in expected.cells]:
        raise ValueError("Notebook is not the exact generated read-only template.")
    manager, environment = local_kernel(ROOT / ".cache" / "p50c-notebook-execution")
    client = NotebookClient(
        notebook, km=manager, timeout=240, allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute(env=environment, cwd=str(ROOT))
    finally:
        if client.kc is not None:
            client.kc.stop_channels()
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise RuntimeError("The read-only notebook has execution errors.")
    atomic_bytes(path, nbformat.writes(notebook).encode("utf-8"), replace=True)
    audit["artifact_sha256"][relative] = sha256(path)
    audit.update(notebook_executed_without_inference=True, notebook_error_count=0,
                 notebook_runtime_files_project_local=True, notebook_transport_encryption="CurveZMQ required")
    atomic_json(RUN / "finalization.json", audit, replace=True)
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("Notebook execution controller imported the model runtime.")
    print(json.dumps({"notebook": str(path), "errors": 0, "inference": False}, indent=2))


if __name__ == "__main__":
    execute()
