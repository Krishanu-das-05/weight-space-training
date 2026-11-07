"""Partial guard-stop reporting cannot promote missing data to completed evidence."""

import copy
import re
import unittest

from p50c_completion_analysis import recompute
from test_p50c_completion_runtime import fake_decode, fixture, row_for
from write_partial_p50c_report import render_partial_report


class PartialReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.pool = fixture()
        cls.plan["edit"]["revision"] = "FAKE-EDIT-REVISION"
        cls.records = [row_for(cls.plan, i) for i in range(11)]
        cls.originals = [row_for(cls.plan, i, edited=False) for i in range(30)]
        cls.analysis = recompute(cls.plan, cls.pool, cls.records, cls.originals, fake_decode)
        cls.summary = {
            "status": "fatal_error",
            "error": {"type": "ValueError", "message": "Expected bounded decoded response text."},
            "missing_edited_ids": [row["id"] for row in cls.pool[11:]],
        }
        completed = sum(len(row["token_ids"]) for row in cls.records)
        cls.budget = {"completed_actual_tokens": completed, "unreconciled_reserved_tokens": 8192,
                      "accounted_tokens": completed + 8192}
        cls.audit = {"passed": True, "latest_history_file": "FAKE-weight_verification.json"}

    def test_partial_report_labels_denominators_and_lost_output_and_withholds_paired_tests(self):
        report = render_partial_report(self.plan, self.records, self.summary, self.analysis, self.budget, self.audit)
        self.assertIn("11 of 30", report)
        self.assertIn("**Denominators differ:**", report)
        self.assertIn("raw text and token IDs were not\npersisted", report)
        self.assertIn("planned n=30 exact McNemar/Holm comparisons are withheld", report)
        self.assertIn("figure, notebook 07 and paired inference", report)
        self.assertIn("## Stop", report)
        self.assertIn("Restoration PASSED independently in both files", report)
        self.assertEqual(len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", report.split("\n\n")[1])), 5)
        self.assertTrue(all(not rows for rows in self.analysis["paired_comparisons"].values()))

    def test_incomplete_report_rejects_complete_or_running_or_different_failure(self):
        for change in ({"status": "complete"}, {"status": "running"},
                       {"error": {"type": "ValueError", "message": "FAKE-DIFFERENT-ERROR"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                render_partial_report(self.plan, self.records, {**self.summary, **change},
                                      self.analysis, self.budget, self.audit)
        complete = copy.deepcopy(self.analysis)
        complete["complete_data"] = True
        with self.assertRaises(ValueError):
            render_partial_report(self.plan, self.records, self.summary, complete, self.budget, self.audit)

    def test_restore_failure_takes_precedence_even_with_complete_status(self):
        report = render_partial_report(self.plan, self.records, {**self.summary, "status": "complete"},
                                       self.analysis, self.budget, {"passed": False})
        self.assertIn("RESTORATION NOT VERIFIED", report)
        self.assertIn("## Stop", report)


if __name__ == "__main__":
    unittest.main()
