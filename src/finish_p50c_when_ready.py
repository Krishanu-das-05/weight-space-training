"""Wait without a deadline for the worker, then run read-only deliverable gates."""

import errno
import os
import subprocess
import sys
import time

from budget_sweep_io import ROOT, RunLock, read_json
from p50c_completion_io import RUN


def worker_lock_is_free():
    probe = RunLock(RUN / "worker.lock")
    try:
        with probe:
            return True
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EAGAIN):
            raise
        return False
    finally:
        handle = getattr(probe, "handle", None)
        if handle is not None and not handle.closed:
            handle.close()


def main():
    last_count = None
    while not worker_lock_is_free():
        summary = read_json(RUN / "summary.json")
        count = summary.get("edited_main_responses")
        if type(count) is int and 0 <= count <= 30 and count != last_count:
            print(f"WAITING FOR EXISTING WORKER: {count}/30 edited records saved", flush=True)
            last_count = count
        time.sleep(30)
    print("WORKER LOCK RELEASED; CHECKING RESTORATION INDEPENDENTLY OF STATUS", flush=True)
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                       HF_HUB_DISABLE_IMPLICIT_TOKEN="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", UV_OFFLINE="1")
    for script in ("finalize_p50c_completion.py", "execute_p50c_notebook.py", "validate_p50c_completion.py"):
        print("READ-ONLY FINALIZATION:", script, flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
