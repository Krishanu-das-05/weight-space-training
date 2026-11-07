"""Lossless local preparation and input-boundary regression tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import workbench


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        (workbench.ROOT / ".local").mkdir(exist_ok=True)

    def fixture(self, parent):
        root = parent / "project"
        source = root / "src" / "example.py"
        source.parent.mkdir(parents=True)
        source.write_text("VALUE = 7\n", encoding="utf-8")
        value = {"schema_version": 1, "files": {"example.py": {
            "file": "src/example.py", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "bytes": source.stat().st_size,
        }}}
        (root / "repository-layout.json").write_text(json.dumps(value), encoding="utf-8")
        return root, source

    def test_source_only_preparation_is_exact_idempotent_and_never_starts_a_process(self):
        with tempfile.TemporaryDirectory(dir=workbench.ROOT / ".local") as temporary:
            root, source = self.fixture(Path(temporary))
            destination = root / ".local" / "workbench"
            with (
                patch.object(workbench, "ROOT", root),
                patch.object(workbench, "WORKBENCH", destination),
                patch.object(workbench, "ARCHIVE", root / ".local" / "original-layout"),
                patch.object(workbench.subprocess, "run", side_effect=AssertionError("No process needed")),
            ):
                result = workbench.prepare()
                self.assertEqual((destination / source.name).read_bytes(), source.read_bytes())
                self.assertEqual(result, workbench.prepare())
                self.assertFalse(result["inference_performed"])
                self.assertFalse(result["model_downloaded"])
                self.assertEqual(result["local_data_files"], 0)

    def test_existing_workbench_edits_are_preserved_and_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=workbench.ROOT / ".local") as temporary:
            root, source = self.fixture(Path(temporary))
            destination = root / ".local" / "workbench"
            destination.mkdir(parents=True)
            edited = destination / source.name
            edited.write_text("USER_EDIT = 9\n", encoding="utf-8")
            with patch.object(workbench, "ROOT", root), patch.object(workbench, "WORKBENCH", destination):
                with self.assertRaises(FileExistsError):
                    workbench._copy_exact(source, edited)
            self.assertEqual(edited.read_text(encoding="utf-8"), "USER_EDIT = 9\n")

    def test_changed_frozen_source_stops_before_preparation(self):
        with tempfile.TemporaryDirectory(dir=workbench.ROOT / ".local") as temporary:
            root, source = self.fixture(Path(temporary))
            source.write_text("CHANGED = True\n", encoding="utf-8")
            with patch.object(workbench, "ROOT", root):
                with self.assertRaises(ValueError):
                    workbench.layout_sources()

    def test_copy_rejects_destinations_outside_the_workbench(self):
        with tempfile.TemporaryDirectory(dir=workbench.ROOT / ".local") as temporary:
            root, source = self.fixture(Path(temporary))
            destination = root / ".local" / "workbench"
            with patch.object(workbench, "ROOT", root), patch.object(workbench, "WORKBENCH", destination):
                outside = root / "outside.py"
                with self.assertRaises(ValueError):
                    workbench._copy_exact(source, outside)
                self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()
