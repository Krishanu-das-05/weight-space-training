"""Portable repository gates; no model packages, caches or network needed."""

import ast
import json
import unittest
from pathlib import Path

from tools.check_repo import (
    HOME_PATH, ROOT, notebook_issues, public_path, raw_fields, secret_findings, svg_is_safe,
)
from tools.workbench import layout_sources


class PublicationTests(unittest.TestCase):
    def test_all_sixty_research_sources_and_tests_keep_their_exact_bytes(self):
        self.assertEqual(len(layout_sources()), 60)

    def test_original_140_research_tests_are_preserved_not_skipped(self):
        count = 0
        for path in (ROOT / "tests" / "legacy").glob("test_*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            count += sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_") for node in ast.walk(tree))
        self.assertEqual(count, 140)

    def test_public_scope_accepts_only_expected_source_and_documentation(self):
        for path in ("src/weight_edit.py", "tests/legacy/test_runner.py", "notebooks/00_results_overview.ipynb",
                     "docs/results/p50c_completion.json", "README.md"):
            self.assertTrue(public_path(path))

    def test_public_scope_rejects_private_inputs_and_path_traversal(self):
        for path in ("", "../outside.py", "/tmp/outside.py", ".env", ".local/snapshot.json",
                     "models/model.safetensors", "results/records.jsonl", "src/private.key",
                     "src/__pycache__/cache.pyc", "docs/data.parquet"):
            self.assertFalse(public_path(path))

    def test_secret_scan_detects_fake_credential_shapes_without_returning_values(self):
        fake_value = "ghp_" + "FAKE" * 8
        findings = secret_findings("value = '" + fake_value + "'")
        self.assertTrue(any(row["rule"] == "github_credential" for row in findings))
        self.assertNotIn(fake_value, json.dumps(findings))

    def test_secret_scan_flags_literal_credentials_but_accepts_explicit_fake_placeholders(self):
        fake_example = "password=" + repr("EXAMPLE-NOT-A-REAL-CREDENTIAL")
        self.assertTrue(secret_findings(fake_example))
        self.assertFalse(secret_findings("password='FAKE-PASSWORD-FOR-TESTING'"))

    def test_home_path_detection_is_unchanged_by_splitting_its_source_literal(self):
        for text in ("C:/" + "Users/" + "FAKE-USER/project", "/" + "home/" + "FAKE-USER/project",
                     "/" + "Users/" + "FAKE-USER/project"):
            self.assertIsNotNone(HOME_PATH.search(text))

    def test_notebook_outputs_and_machine_metadata_are_rejected(self):
        value = {"nbformat": 4, "metadata": {"widgets": {}}, "cells": [
            {"cell_type": "code", "outputs": [{"output_type": "stream"}], "execution_count": 1},
        ]}
        self.assertIn("saved_notebook_execution", notebook_issues(value))
        self.assertIn("unreviewed_notebook_metadata", notebook_issues(value))

    def test_all_eight_public_notebooks_are_clean_and_parseable(self):
        paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook_issues(value), [], path.name)
            for cell in value["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]), filename=path.name)

    def test_curated_exports_never_contain_raw_responses_or_questions(self):
        paths = list((ROOT / "docs" / "results").glob("*.json"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            self.assertFalse(raw_fields(json.loads(path.read_text(encoding="utf-8"))))
        self.assertTrue(raw_fields({"nested": [{"token_ids": [1, 2]}]}))

    def test_svg_validation_rejects_scripts_and_external_references(self):
        self.assertTrue(svg_is_safe('<svg><use href="#local"/></svg>'))
        self.assertFalse(svg_is_safe('<svg><script>bad()</script></svg>'))
        self.assertFalse(svg_is_safe('<svg><image href="https://example.invalid/pixel"/></svg>'))

    def test_public_aggregate_preserves_primary_multiplicity_caveat_and_failed_charge(self):
        value = json.loads((ROOT / "docs" / "results" / "p50c_completion.json").read_text(encoding="utf-8"))
        self.assertFalse(value["primary_at_8192_clears_holm_005"])
        self.assertEqual(value["budget"]["unreconciled_reserved_tokens"], 8192)
        self.assertTrue(value["legacy_11_outcomes_unchanged"])
        self.assertEqual(len(value["text_bound_violations"]), 1)
        self.assertEqual(value["text_bound_violations"][0]["character_length"], 669118)


if __name__ == "__main__":
    unittest.main()
