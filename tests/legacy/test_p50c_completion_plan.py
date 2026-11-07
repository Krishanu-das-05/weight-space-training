"""Offline plan/closure regression tests; no subject-model inference."""

import copy
import subprocess
import sys
import unittest

from budget_sweep_io import ROOT, json_bytes, read_json
from budget_sweep_outputs import load_saved_run
from budget_sweep_scoring import grade_generation, score_at_cap
from p50c_completion_diagnostic import BASELINE, BASELINE_DIAGNOSTIC, baseline_diagnostic, closure_at_caps, protected_snapshot
from p50c_completion_plan_markdown import render_plan
from prepare_p50c_completion_plan import build_plan, validate_plan


def fake_decode(tokens):
    pieces = {1: "</thi", 2: "nk>", 9: "x", 248069: "</think>", 248044: "<|endoftext|>"}
    return "".join(pieces[token] for token in tokens)


class ClosureTests(unittest.TestCase):
    def test_complete_closing_delimiter_at_cap_boundary_counts(self):
        result = closure_at_caps([9] * 511 + [248069], fake_decode)
        self.assertTrue(all(result.values()))

    def test_delimiter_split_across_cap_boundary_waits_for_last_piece(self):
        result = closure_at_caps([9] * 511 + [1, 2], fake_decode)
        self.assertFalse(result["512"])
        self.assertTrue(result["1024"])

    def test_closure_does_not_require_eos_and_is_not_correct_delivery(self):
        tokens = [248069] + [9] * 8191
        self.assertTrue(closure_at_caps(tokens, fake_decode)["8192"])
        graded = grade_generation("</think>\n" + r"\boxed{7}", tokens, "7", thinking=True,
                                  eos_ids={248044}, terminal_tokens=[])
        self.assertFalse(score_at_cap(graded, 8192)["natural_completion"])
        self.assertFalse(score_at_cap(graded, 8192)["correct"])

    def test_natural_eos_does_not_imply_closure(self):
        tokens = [9, 248044]
        self.assertFalse(closure_at_caps(tokens, fake_decode)["8192"])
        graded = grade_generation(r"\boxed{7}<|endoftext|>", tokens, "7", thinking=True,
                                  eos_ids={248044}, terminal_tokens=["<|endoftext|>"])
        self.assertTrue(score_at_cap(graded, 8192)["natural_completion"])

    def test_closure_is_boolean_not_number_of_tags(self):
        result = closure_at_caps([248069, 248069, 248044], fake_decode)
        self.assertIs(result["512"], True)

    def test_match_is_exact_and_prompt_is_not_part_of_generated_prefix(self):
        for text in ("</THINK>", "</think >", "<think>open only", "no generated closing tag"):
            with self.subTest(text=text):
                self.assertFalse(any(closure_at_caps([9], lambda _: text).values()))

    def test_invalid_inputs_fail_closed(self):
        for tokens in ([], [True], [-1], [3.5], [9] * 8193):
            with self.subTest(size=len(tokens)):
                with self.assertRaises(ValueError):
                    closure_at_caps(tokens, fake_decode)
        with self.assertRaises(ValueError):
            closure_at_caps([9], lambda _: None)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = protected_snapshot()
        cls.plan, cls.diagnostic, _ = build_plan()
        cls.base = load_saved_run(BASELINE)[0]

    @classmethod
    def tearDownClass(cls):
        if protected_snapshot() != cls.before:
            raise AssertionError("A protected completed artifact changed during tests.")

    def test_raw_closure_baseline_is_two_of_thirty_and_preserves_old_artifacts(self):
        saved = read_json(BASELINE_DIAGNOSTIC)
        self.assertEqual(json_bytes(saved), json_bytes(self.diagnostic))
        self.assertEqual(json_bytes(saved), json_bytes(baseline_diagnostic()))
        self.assertEqual([row["thinking_closed"] for row in saved["tables"]], [0, 0, 0, 1, 2])
        events = {row["id"]: row["first_canonical_closing_token_index"] for row in saved["per_question"]
                  if row["first_canonical_closing_token_index"] is not None}
        self.assertEqual(events, {"gsm8k_train_06229": 5355, "gsm8k_train_01026": 2400})

    def test_plan_is_unapproved_and_only_schedules_thirty_full_edits(self):
        plan = self.plan
        self.assertFalse(plan["generation_authorized"])
        self.assertEqual(plan["generated_responses"], 0)
        self.assertEqual(len(plan["schedule"]), 30)
        self.assertEqual([row["seed"] for row in plan["schedule"]], list(range(30)))
        self.assertTrue(all(row["condition"] == "thinking_on" and row["max_new_tokens"] == 8192
                            and row["model_variant"] == "published_p50c" for row in plan["schedule"]))
        self.assertEqual(plan["edit"]["layers"], [15, 19])
        self.assertEqual(plan["edit"]["strength"], 1.0)
        self.assertNotEqual(plan["edit"]["target_original_sha256"], plan["edit"]["target_edited_sha256"])
        self.assertFalse(plan["edit"]["model_or_tensor_saves_allowed"])
        self.assertTrue(plan["baseline"]["reuse_without_regeneration"])

    def test_budget_and_inherited_prefix_policy_are_fixed(self):
        self.assertEqual(self.plan["token_budget"]["total_worst_case_no_crash"], 245760)
        self.assertEqual(self.plan["token_budget"]["headroom"], 4240)
        self.assertEqual(self.plan["token_budget"]["limit"], 250000)
        self.assertFalse(self.plan["prefix_validation"]["new_check_scheduled"])
        self.assertTrue(self.plan["prefix_validation"]["baseline_all_four_passed"])
        self.assertEqual(len(self.plan["baseline"]["selection"]), 30)

    def test_runtime_scoring_or_schedule_drift_requires_a_new_plan(self):
        mutations = (
            lambda p: p["execution"].update(deterministic_algorithms=False),
            lambda p: p["model"].update(dtype="float16"),
            lambda p: p["hardware"].update(nvidia_driver="FAKE-DRIVER-FOR-TEST"),
            lambda p: p["sampling"].update(temperature=0.7),
            lambda p: p["stopping"].update(eos_token_id=248046),
            lambda p: p["schedule"][0].update(max_new_tokens=1024),
            lambda p: p["edit"].update(strength=0.5),
            lambda p: p["token_budget"].update(limit=300000),
            lambda p: p["outcomes"]["primary"].update(name="thinking_closed"),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(self.plan)
            mutate(candidate)
            with self.assertRaises(ValueError):
                validate_plan(candidate, self.base)

    def test_plan_render_has_approval_gate_and_endpoint_distinction(self):
        text = render_plan(self.plan, "a" * 64)
        for expected in ("AWAITING EXPLICIT APPROVAL", "2/30 = 6.7%", "250,000", "245,760",
                         "28-40%", "distinct events", "No new truncation check", "restore", "STOP"):
            self.assertIn(expected, text)
        self.assertFalse(self.plan["outcomes"]["diagnostic"]["requires_eos"])
        self.assertFalse(self.plan["outcomes"]["diagnostic"]["requires_correctness"])

    def test_plan_modules_do_not_import_torch_or_transformers(self):
        code = (
            "import sys; import prepare_p50c_completion_plan; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
