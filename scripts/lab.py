"""Local-only research commands; no automatic installs, downloads or inference."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.workbench import ARCHIVE, WORKBENCH, prepare

VERIFY_CODE = """
import json
import sys
from p50c_completion_outputs import load_saved_analysis
from p50c_restore_audit import load_restore_audit
from p50c_completion_io import RUN
plan, pool, records, summary, analysis, budget = load_saved_analysis()
audit = load_restore_audit(RUN, plan)
assert audit['passed'] is True
assert summary['legacy_11_outcomes_unchanged'] is True
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print(json.dumps({
    'edited_responses': len(records),
    'complete_data': analysis['complete_data'],
    'both_restoration_artifacts_verified': audit['passed'],
    'legacy_11_unchanged': True,
    'budget': budget,
    'primary_at_8192': analysis['paired_comparisons']['natural_completion'][-1],
    'inference_performed': False,
    'model_runtime_imported': False,
}, indent=2))
"""


def python_executable():
    relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    path = ROOT / ".venv" / relative
    if not path.is_file():
        raise RuntimeError("The pinned .venv is missing; see docs/SETUP.md. Nothing will be installed automatically.")
    return str(path)


def offline_environment():
    return {
        **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUTF8": "1", "UV_OFFLINE": "1",
    }


def require_historical_data():
    if not (ARCHIVE / "snapshot.json").is_file():
        raise RuntimeError(
            "The historical audit pack is local-only and absent. "
            "Run the portable repository checks instead; see docs/REPRODUCIBILITY.md."
        )
    expected = WORKBENCH / "results" / "p50c_completion_20261107T094340_563457Z" / "summary.json"
    if not expected.is_file():
        raise RuntimeError("Saved local runs are absent; prepare with --with-local-data. No downloads will be attempted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "test", "verify", "notebook"))
    parser.add_argument("--with-local-data", action="store_true", help="Mirror existing local inputs; never fetch missing data.")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(with_local_data=args.with_local_data), indent=2))
        return 0
    if args.command == "notebook":
        prepare(with_local_data=args.with_local_data)
        environment = offline_environment()
        environment.update(
            JUPYTER_CONFIG_DIR=str(ROOT / ".jupyter"), JUPYTER_DATA_DIR=str(ROOT / ".jupyter" / "data"),
            IPYTHONDIR=str(ROOT / ".ipython"),
        )
        return subprocess.run([
            python_executable(), "-m", "jupyterlab", "--ip=127.0.0.1",
            "--ServerApp.root_dir=" + str(ROOT), "--ServerApp.default_url=/lab/tree/notebooks/00_results_overview.ipynb",
        ], cwd=ROOT, env=environment).returncode
    prepare(with_local_data=True)
    require_historical_data()
    command = [python_executable(), "-m", "unittest", "-q"] if args.command == "test" else [
        python_executable(), "-c", VERIFY_CODE,
    ]
    return subprocess.run(command, cwd=WORKBENCH, env=offline_environment()).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
