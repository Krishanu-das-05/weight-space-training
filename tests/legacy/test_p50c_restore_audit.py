"""Regression checks: status cannot mask missing or mismatched restoration."""

import copy
import unittest

from p50c_restore_audit import compare_restore_evidence, restoration_failure_report


class RestoreEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"model": {"target_base_sha256": {"FAKE-LAYER-15": "3" * 64, "FAKE-LAYER-19": "4" * 64}}}
        self.good = {"status": "complete", "target_hash_checks": {
            "verified_after_restore": True,
            "original_after_restore": dict(self.plan["model"]["target_base_sha256"]),
        }}

    def test_both_artifacts_must_confirm_true_and_exact_base_hashes(self):
        audit = compare_restore_evidence(self.plan, self.good, self.good)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["summary"]["original_after_restore_matches_planned_base"])
        self.assertTrue(audit["latest_history"]["verified_after_restore_is_true"])

    def test_complete_status_cannot_hide_false_missing_or_nonboolean_flags(self):
        for value in (False, None, 1, "true"):
            bad = copy.deepcopy(self.good)
            bad["target_hash_checks"]["verified_after_restore"] = value
            with self.subTest(flag=value):
                audit = compare_restore_evidence(self.plan, self.good, bad)
                self.assertFalse(audit["passed"])
                report = restoration_failure_report(self.plan, audit)
                self.assertIn("## Stop", report)
                self.assertIn("RESTORATION NOT VERIFIED", report)
                self.assertIn("takes precedence over any status field", report)

    def test_hash_mismatch_in_either_artifact_is_failure_even_when_both_flags_true(self):
        bad = copy.deepcopy(self.good)
        bad["target_hash_checks"]["original_after_restore"]["FAKE-LAYER-15"] = "0" * 64
        self.assertFalse(compare_restore_evidence(self.plan, bad, self.good)["passed"])
        self.assertFalse(compare_restore_evidence(self.plan, self.good, bad)["passed"])

    def test_status_is_not_used_as_evidence_in_either_direction(self):
        another_status = copy.deepcopy(self.good)
        another_status["status"] = "fatal_error"
        self.assertTrue(compare_restore_evidence(self.plan, self.good, another_status)["passed"])
        self.assertFalse(compare_restore_evidence(self.plan, self.good, {"status": "complete"})["passed"])


if __name__ == "__main__":
    unittest.main()
