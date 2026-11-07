"""Start an authenticated, loopback-only Jupyter server for this project."""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
        raise SystemExit("Run this with .venv\\Scripts\\python.exe.")

    environment = os.environ.copy()
    for name, relative_path in {
        "HF_HOME": ".cache/huggingface",
        "JUPYTER_CONFIG_DIR": ".jupyter/config",
        "JUPYTER_DATA_DIR": ".jupyter/data",
        "JUPYTER_RUNTIME_DIR": ".jupyter/runtime",
        "IPYTHONDIR": ".ipython",
    }.items():
        environment[name] = str(PROJECT_ROOT / relative_path)
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyterlab",
            str(PROJECT_ROOT / "01_inspect_model.ipynb"),
            f"--ServerApp.root_dir={PROJECT_ROOT}",
            "--ServerApp.ip=127.0.0.1",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
