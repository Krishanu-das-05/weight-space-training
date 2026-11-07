"""Synthetic statistical/output checks; no model generation or downloads."""

import re
import unittest

from budget_sweep_analysis import recompute
from budget_sweep_outputs import diagnostics_markdown, fenced, paired_markdown, public_summary, tables_markdown


def synthetic_run():
    pool = [{"id": f"FAKE_ITEM_{index:02d}", "answer": "7"} for index in range(30)]
    plan = {"prefix_validation": {"question_ids": [item["id"] for item in pool[:2]]},
            "stopping": {"terminal_tokens_for_grading": ["<EOS>"]}}
    records = []
    for index, item in enumerate(pool):
        for condition in ("thinking_on", "thinking_off"):
            thinking = condition == "thinking_on"
            long_answer = thinking and 2 <= index < 10
            tokens = [3] * 1024 + [248044] if long_answer else [248046, 3, 248044]
            text = ("</think>" if thinking else "") + (r"\boxed{7}" if index < 10 else r"\boxed{8}") + "<EOS>"
            row = {
                "problem_id": item["id"], "condition": condition, "kind": "main", "seed": index,
                "token_ids": tokens, "text": text,
                "graded": {"FAKE-MEMO-DO-NOT-TRUST": True},
            }
            records.append(row)
            if index < 2:
                records.append({**row, "kind": "prefix_validation_short"})
    return plan, pool, records


class OutputTests(unittest.TestCase):
    def test_full_sample_recomputes_known_cap_effects_and_pairing_from_raw_records(self):
        plan, pool, records = synthetic_run()
        result = recompute(plan, pool, records)
        self.assertTrue(result["complete_data"])
        on = result["tables"]["thinking_on"]
        self.assertEqual(on[1]["correct"], 2)
        self.assertEqual(on[2]["correct"], 10)
        self.assertEqual(result["tables"]["thinking_off"][1]["correct"], 10)
        self.assertEqual(result["paired_comparisons"][1]["off_wins"], 8)
        self.assertEqual(result["paired_comparisons"][1]["exact_mcnemar_p"], 0.0078125)
        self.assertEqual(result["hypothesis_pattern"]["gain_on"], 8)
        self.assertTrue(result["hypothesis_pattern"]["consistent_with_declared_pattern"])
        self.assertEqual(len(result["short_wrong_answers"]), 40)

    def test_incomplete_data_has_no_planned_n30_inference_or_pattern(self):
        plan, pool, records = synthetic_run()
        result = recompute(plan, pool, records[:8])
        self.assertFalse(result["complete_data"])
        self.assertEqual(result["paired_comparisons"], [])
        self.assertIsNone(result["hypothesis_pattern"])
        self.assertIn("Not performed", paired_markdown(result))
        self.assertIn("before a complete", public_summary(result, "prefix_mismatch"))

    def test_diagnostic_counts_do_not_double_count_validation_as_main(self):
        plan, pool, records = synthetic_run()
        result = recompute(plan, pool, records)
        off = result["token_248046_diagnostics"]["thinking_off"]
        self.assertEqual(off["main"]["responses_with_248046"], 30)
        self.assertEqual(off["prefix_validation_short"]["responses_with_248046"], 2)
        self.assertEqual(off["all"]["responses_with_248046"], 32)
        self.assertIn("248046", diagnostics_markdown(result))
        self.assertIn("95% Wilson", tables_markdown(result))

    def test_public_summary_includes_caveats_without_capability_improvement_claim(self):
        plan, pool, records = synthetic_run()
        result = recompute(plan, pool, records)
        text = public_summary(result, "complete")
        self.assertIn("likely to be in Qwen's training data", text)
        self.assertIn("not new capability", text)
        self.assertNotIn("got smarter", text)
        self.assertEqual(len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)), 5)

    def test_model_excerpt_cannot_break_out_of_its_code_fence(self):
        value = fenced("FAKE-TEXT\n```\n<script>FAKE</script>\n\x1b")
        self.assertTrue(value.startswith("````text\n"))
        self.assertTrue(value.endswith("\n````"))
        self.assertNotIn("\x1b", value)


if __name__ == "__main__":
    unittest.main()
