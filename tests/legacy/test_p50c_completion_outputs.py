"""Synthetic presentation/read-only notebook checks; no subject inference."""

import ast
import copy
import json
import re
import subprocess
import sys
import unittest

import nbformat

from budget_sweep_io import ROOT
from p50c_completion_analysis import recompute
from p50c_completion_outputs import (
    make_notebook, paired_markdown, public_summary,
    render_report, short_wrong_markdown, tables_markdown,
)
from test_p50c_completion_runtime import fake_decode, fixture, row_for


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.pool = fixture()
        cls.plan["model"]["revision"] = "FAKE-MODEL-REVISION"
        cls.plan["edit"]["revision"] = "FAKE-EDIT-REVISION"
        cls.plan["dataset"] = {"revision": "FAKE-DATASET-REVISION"}
        cls.plan["prompt"] = {"instruction": "FAKE-STEP-BY-STEP-INSTRUCTION"}
        cls.records = [row_for(cls.plan, i) for i in range(30)]
        cls.originals = [row_for(cls.plan, i, edited=False) for i in range(30)]
        cls.analysis = recompute(cls.plan, cls.pool, cls.records, cls.originals, fake_decode)
        total = sum(len(row["token_ids"]) for row in cls.records)
        cls.summary = {"status": "complete",
                       "target_hash_checks": {"verified_after_restore": True,
                                              "original_after_restore": cls.plan["model"]["target_base_sha256"]},
                       "budget": {"accounted_tokens": total, "completed_actual_tokens": total,
                                  "unreconciled_reserved_tokens": 0, "limit": 250000}}
        cls.restore_audit = {"passed": True, "latest_history_file": "FAKE-weight_verification.json"}

    def test_public_summary_has_five_sentences_and_keeps_primary_distinct(self):
        text = public_summary(self.analysis, self.summary)
        self.assertEqual(len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)), 5)
        self.assertIn("9/30 naturally", text)
        self.assertIn("7/30 edited", text)
        self.assertIn("10/30 edited", text)
        self.assertIn("not the same event as natural EOS", text)
        self.assertIn("not a capability or benchmark claim", text)
        self.assertNotIn("got smarter", text)

    def test_tables_and_tests_label_pointwise_intervals_and_separate_families(self):
        table = tables_markdown(self.analysis)
        paired = paired_markdown(self.analysis)
        self.assertIn("95% pointwise Wilson", table)
        self.assertIn("no Wilson interval applies", table)
        self.assertIn("not all 15 tests jointly", paired)
        self.assertIn("0.015625", paired)
        self.assertIn("0.078125", paired)

    def test_three_panel_figure_has_log_caps_labels_and_two_confidence_bands_each(self):
        code = (
            "import json,sys; from plot_p50c_completion import plot_comparison; "
            "data=json.load(sys.stdin); figure=plot_comparison(data); assert len(figure.axes)==3\n"
            "for axis in figure.axes:\n"
            " assert axis.get_xscale()=='log'\n"
            " assert len(axis.lines)==2 and len(axis.collections)==2\n"
            " assert list(axis.lines[0].get_xdata())==[512,1024,2048,4096,8192]\n"
            " assert all('n=30' in t.get_text() for t in axis.get_legend().get_texts())\n"
            "figure.clear()\n"
        )
        subprocess.run(
            [getattr(sys, "_base_executable", sys.executable), "-c", code],
            input=json.dumps(self.analysis), text=True, cwd=ROOT, check=True,
        )

    def test_notebook_is_valid_parseable_and_has_no_inference_path(self):
        notebook = make_notebook()
        nbformat.validate(notebook)
        code = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        tree = ast.parse(code)
        self.assertIn("RUN_EXPERIMENT = False", code)
        self.assertIn("read-only notebook", code)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(node.func.attr, "generate")
        self.assertFalse(imported & {"torch", "transformers", "weight_edit", "p50c_completion_worker"})

    def test_short_wrong_excerpts_are_fenced_and_partial_data_has_no_full_inference(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["short_wrong_answers"][0]["text"] = "FAKE-TEXT\n```\n<script>FAKE</script>\x1b"
        text = short_wrong_markdown(analysis)
        self.assertIn("````text", text)
        self.assertNotIn("\x1b", text)
        partial = recompute(self.plan, self.pool, self.records[:1], self.originals, fake_decode)
        self.assertIn("Not performed", paired_markdown(partial))
        summary = public_summary(partial, {"status": "running"})
        self.assertIn("1 of 30", summary)
        self.assertNotIn("9/30", summary)

    def test_final_report_has_raw_pairing_closure_and_restoration_without_changing_scoring(self):
        report = render_report(self.plan, self.records, self.summary, self.analysis, image_relative="FAKE-FIGURE.png",
                               restore_audit=self.restore_audit)
        for expected in ("natural EOS", "Thinking closure", "Exact two-sided McNemar", "verified_edited_tensor_sha256",
                         "8,192", "Short Wrong Answers", "No secondary/diagnostic result replaces the primary",
                         "per-item inclusion is unverified", "original target weights"):
            self.assertIn(expected.lower(), report.lower())
        with self.assertRaises(ValueError):
            render_report(self.plan, self.records, {**self.summary, "status": "running"},
                          self.analysis, image_relative="FAKE.png", restore_audit=self.restore_audit)
        failure = render_report(self.plan, self.records, self.summary, self.analysis,
                                image_relative="FAKE.png", restore_audit={"passed": False})
        self.assertIn("RESTORATION NOT VERIFIED", failure)
        self.assertIn("## Stop", failure)


if __name__ == "__main__":
    unittest.main()
