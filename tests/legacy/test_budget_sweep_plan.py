"""CPU-only checks; no model generation, downloads, or package installation."""

import json
import unittest

from budget_sweep_scoring import CAPS, grade_generation, holm_adjust, paired_comparison, prefix_matches, score_at_cap, wilson
from experiment_design import HOLDOUT_CONTROLS, HOLDOUT_MATH
from gsm8k_source import CACHE, ROOT, file_sha256, integer_final_answer, overlap_questions, select_development


class SourceTests(unittest.TestCase):
    def test_integer_parser(self):
        for value, expected in (
            ("explanation\n#### 1,234", "1234"), ("#### -02", "-2"),
            ("#### +4", "4"), ("#### 2.0", None), ("no delimiter", None),
            ("#### 2/3", None), ("#### __import__('os')", None), (None, None),
        ):
            with self.subTest(value=value):
                self.assertEqual(integer_final_answer(value), expected)

    def test_actual_30_are_the_frozen_train_sample(self):
        rows = [json.loads(line) for line in (CACHE / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        selected, details = select_development(rows)
        actual = [json.loads(line) for line in (ROOT / "dev_pool_gsm8k_30.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(selected, actual)
        self.assertEqual(len(actual), 30)
        self.assertEqual(details["sampling_seed"], 20261106)
        self.assertEqual(file_sha256(ROOT / "dev_pool_gsm8k_30.jsonl"), "f457da0d4bc3acc6d6b9d8b1e98e363786c8497aa195b1f229f7897565c4777d")

    def test_actual_dev_has_no_exact_or_whitespace_heldout_overlap(self):
        selected = [json.loads(line) for line in (ROOT / "dev_pool_gsm8k_30.jsonl").read_text(encoding="utf-8").splitlines()]
        questions = [item["question"] for item in HOLDOUT_MATH + HOLDOUT_CONTROLS]
        self.assertEqual(overlap_questions(selected, questions), [])
        self.assertEqual(overlap_questions([{"id": "FAKE", "question": "FAKE   TEST\nTEXT"}], ["FAKE TEST TEXT"]), ["FAKE"])
        self.assertEqual(overlap_questions([{"id": "FAKE", "question": "SAME"}], ["SAME"]), ["FAKE"])

    def test_integer_failures_advance_in_the_same_seeded_order(self):
        rows = [{"question": f"FAKE-ITEM-{index}", "answer": "#### 1"} for index in range(40)]
        first, details = select_development(rows, seed=7, count=3)
        rows[details["selection"][0]["row_index"]]["answer"] = "#### 1.5"
        next_items, next_details = select_development(rows, seed=7, count=3)
        self.assertEqual(next_items[:2], first[1:])
        self.assertEqual(len(next_details["skipped"]), 1)


class CapScoringTests(unittest.TestCase):
    def grade(self, tokens, text=r"\boxed{7}", thinking=False):
        return grade_generation(text, tokens, "7", thinking=thinking, eos_ids={9}, terminal_tokens=("<EOS>",))

    def test_exact_prefix_including_early_eos(self):
        long = [3] * 8192
        self.assertTrue(prefix_matches(long[:1024], long))
        self.assertFalse(prefix_matches([4] + long[1:1024], long))
        early = [3] * 30 + [9]
        self.assertTrue(prefix_matches(early, early))
        self.assertFalse(prefix_matches(early, early + [3] * 2000))

    def test_eos_at_cap_is_counted_but_later_eos_is_not(self):
        graded = self.grade([3] * 1023 + [9])
        self.assertFalse(score_at_cap(graded, 512)["correct"])
        self.assertTrue(score_at_cap(graded, 1024)["correct"])
        self.assertEqual(score_at_cap(graded, 512)["tokens"], 512)
        self.assertEqual(score_at_cap(graded, 8192)["tokens"], 1024)
        self.assertTrue(score_at_cap(self.grade([3] * 8191 + [9]), 8192)["correct"])

    def test_correct_box_without_eos_or_without_thinking_closure_fails(self):
        for tokens, text, thinking in (
            ([3] * 8192, r"\boxed{7}", False),
            ([3, 9], r"\boxed{7}", True),
            ([3, 9], r"\boxed{8}", False),
            ([3, 9], r"\boxed{7.0}", False),
        ):
            self.assertFalse(score_at_cap(self.grade(tokens, text, thinking), 8192)["correct"])
        self.assertTrue(score_at_cap(self.grade([3, 9], r"</think>\boxed{7}", True), 512)["correct"])

    def test_tokens_after_eos_and_invalid_inputs_are_rejected(self):
        for tokens in ([3, 9, 3], [True], [], [-1]):
            with self.assertRaises(ValueError):
                self.grade(tokens)
        with self.assertRaises(ValueError):
            score_at_cap(self.grade([9]), 1000)

    def test_statistics_match_known_values(self):
        low, high = wilson(0, 30)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(high, 0.1135133931739688)
        self.assertAlmostEqual(wilson(30, 30)[0], 1 - high)
        self.assertEqual(paired_comparison([True] * 8, [False] * 8)["exact_mcnemar_p"], 0.0078125)
        self.assertEqual(paired_comparison([True], [True])["exact_mcnemar_p"], 1)
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
