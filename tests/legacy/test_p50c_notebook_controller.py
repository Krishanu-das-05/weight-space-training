"""Project-local read-only kernel execution smoke tests; no model calls."""

import tempfile
import unittest
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from budget_sweep_io import ROOT
from execute_p50c_notebook import local_kernel


class NotebookControllerTests(unittest.TestCase):
    def test_local_kernel_executes_without_subject_model_imports(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            manager, environment = local_kernel(Path(temporary))
            notebook = nbformat.v4.new_notebook(cells=[
                nbformat.v4.new_code_cell(
                    "import sys\nassert 'torch' not in sys.modules\n"
                    "assert 'transformers' not in sys.modules\nprint('FAKE-READ-ONLY-KERNEL-CHECK')"
                ),
            ])
            client = NotebookClient(notebook, km=manager, timeout=60, allow_errors=False,
                                    resources={"metadata": {"path": str(ROOT)}})
            try:
                client.execute(env=environment, cwd=str(ROOT))
            finally:
                if client.kc is not None:
                    client.kc.stop_channels()
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
            self.assertEqual(notebook.cells[0].execution_count, 1)
            self.assertEqual(notebook.cells[0].outputs[0]["text"].strip(), "FAKE-READ-ONLY-KERNEL-CHECK")
            self.assertTrue(Path(manager.connection_file).is_relative_to(ROOT / ".cache"))
            self.assertEqual(manager.transport_encryption, "required")
            self.assertTrue(manager.curve_publickey)

    def test_kernel_runtime_cannot_escape_project_cache(self):
        with self.assertRaises(ValueError):
            local_kernel(ROOT / "FAKE-NOT-A-CACHE-PATH")


if __name__ == "__main__":
    unittest.main()
