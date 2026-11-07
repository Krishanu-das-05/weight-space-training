"""Deterministic harness tests: no downloads, GPU, or model generation."""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch
from transformers import GenerationConfig, LogitsProcessorList
from transformers.generation.utils import GenerationMixin

from experiment_design import (
    CONDITIONS, DEVELOPMENT, HOLDOUT_CONTROLS, HOLDOUT_MATH, holdout_gate, split_fingerprint,
)
from math_probe import GeneratedPresencePenalty, run_probes
from probe_metrics import response_metrics, validate_problem


class MetricsTests(unittest.TestCase):
    def metrics(self, text, tokens=(1, 9), **kwargs):
        options = dict(thinking=True, max_new_tokens=32, eos_ids={9}, terminal_tokens=("<EOS>",))
        options.update(kwargs)
        return response_metrics(text, list(tokens), "13", **options)

    def test_reasoning_answer_does_not_count_as_delivered(self):
        row = self.metrics(r"I think \boxed{13}", tokens=[1] * 32)
        self.assertTrue(row["reasoning_boxed_correct"])
        self.assertFalse(row["correct"])
        self.assertFalse(row["delivered_correct"])
        self.assertEqual(row["stop_reason"], "token_limit")

    def test_final_answer_and_eos_are_separate(self):
        row = self.metrics(r"\boxed{8}</think>\boxed{13}", tokens=[1] * 32)
        self.assertTrue(row["correct"])
        self.assertFalse(row["delivered_correct"])
        self.assertEqual(row["reasoning_boxed_answer"], "8")

    def test_eos_on_last_budget_token_is_not_a_cap_hit(self):
        row = self.metrics(r"</think>\boxed{13}<EOS>", tokens=[1] * 31 + [9])
        self.assertTrue(row["delivered_correct"])
        self.assertFalse(row["hit_token_limit"])
        self.assertEqual(row["stop_reason"], "eos")

    def test_non_thinking_answers_do_not_require_closing_think_tag(self):
        row = self.metrics(r"\boxed{13}<EOS>", thinking=False)
        self.assertTrue(row["delivered_correct"])
        self.assertIsNone(row["finished_thinking"])

    def test_exact_grader_only_strips_terminal_special_tokens(self):
        row = response_metrics(
            "READY<EOS>", [1, 9], "READY", thinking=False, max_new_tokens=32,
            eos_ids={9}, grader="exact", terminal_tokens=("<EOS>",),
        )
        self.assertTrue(row["delivered_correct"])
        self.assertIsNone(row["boxed_answer"])
        row = response_metrics(
            "READY extra<EOS>", [1, 9], "READY", thinking=False, max_new_tokens=32,
            eos_ids={9}, grader="exact", terminal_tokens=("<EOS>",),
        )
        self.assertFalse(row["correct"])

    def test_time_budget_is_not_mislabeled_as_a_natural_stop(self):
        row = self.metrics(r"</think>\boxed{13}", tokens=[1] * 10, timed_out=True)
        self.assertEqual(row["stop_reason"], "time_budget")
        self.assertFalse(row["natural_stop"])


class SamplerTests(unittest.TestCase):
    def test_installed_transformers_orders_presence_before_sampling_warpers(self):
        config = GenerationConfig(
            do_sample=True, temperature=0.6, top_k=20, top_p=0.95, repetition_penalty=1.0,
        )
        config._eos_token_tensor = torch.tensor([9])
        processors = GenerationMixin()._get_logits_processor(
            config, input_ids_seq_length=2, device="cpu",
            logits_processor=LogitsProcessorList([GeneratedPresencePenalty(2, 1.5)]),
        )
        self.assertEqual(
            [type(item).__name__ for item in processors],
            ["GeneratedPresencePenalty", "TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"],
        )

    def test_non_thinking_inference_path_and_rng_preservation(self):
        class Batch(dict):
            def to(self, device):
                return self

        class Tokenizer:
            eos_token_id = 9
            all_special_tokens = ("<|im_end|>", "<EOS>")
            thinking = None

            def apply_chat_template(self, messages, **kwargs):
                self.thinking = kwargs["enable_thinking"]
                return messages[0]["content"]

            def __call__(self, prompt, **kwargs):
                return Batch(input_ids=torch.tensor([[1, 2]]))

            def decode(self, ids, **kwargs):
                return "<EOS>" if ids == [9] else "READY<|im_end|><EOS>"

        class Model:
            device = torch.device("cpu")
            generation_config = SimpleNamespace(eos_token_id=9)

            def generate(self, **kwargs):
                torch.rand(1)
                return torch.tensor([[1, 2, 3, 9]])

        tokenizer = Tokenizer()
        state = torch.get_rng_state().clone()
        with redirect_stdout(io.StringIO()):
            rows = run_probes(
                Model(), tokenizer,
                [{"id": "fixture", "question": "Reply READY.", "answer": "READY", "grader": "exact"}],
                "fixture", max_new_tokens=32, thinking=False,
            )
        self.assertFalse(tokenizer.thinking)
        self.assertTrue(rows[0]["delivered_correct"])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


class DesignTests(unittest.TestCase):
    def test_split_ids_are_disjoint_and_every_probe_has_a_valid_schema(self):
        groups = (DEVELOPMENT, HOLDOUT_MATH, HOLDOUT_CONTROLS)
        seen = set()
        for group in groups:
            for problem in group:
                validate_problem(problem)
                self.assertNotIn(problem["id"], seen)
                seen.add(problem["id"])
        self.assertEqual(len(HOLDOUT_MATH), 20)
        self.assertEqual(len(HOLDOUT_CONTROLS), 10)
        self.assertEqual(split_fingerprint(DEVELOPMENT), split_fingerprint(list(DEVELOPMENT)))

    def rows(self, scores=None):
        scores = scores or {}
        rows = []
        for condition in CONDITIONS:
            for index in range(2):
                correct = index < scores.get(condition.name, 0)
                rows.append(dict(
                    condition=condition.name, problem_id=f"fixture{index}", seed=index,
                    delivered_correct=correct, correct=correct, natural_stop=True,
                    hit_token_limit=False, stop_reason="eos", generated_tokens=10, seconds=1.0,
                ))
        return rows

    def test_no_holdout_when_thinking_off_already_matches_best_edit(self):
        decision = holdout_gate(self.rows({"base_no_thinking": 2, "published_both": 2}), complete=True)
        self.assertFalse(decision["allow_holdout"])

    def test_a_clear_paired_gain_selects_and_freezes_a_candidate(self):
        decision = holdout_gate(self.rows({"published_both": 2}), complete=True)
        self.assertTrue(decision["allow_holdout"])
        self.assertEqual(decision["selected_condition"]["name"], "published_both")

    def test_partial_or_time_limited_runs_cannot_select(self):
        rows = self.rows({"published_both": 2})
        self.assertFalse(holdout_gate(rows, complete=False)["allow_holdout"])
        rows[0]["stop_reason"] = "time_budget"
        self.assertFalse(holdout_gate(rows, complete=True)["allow_holdout"])

    def test_unpaired_or_duplicate_rows_cannot_select(self):
        rows = self.rows({"published_both": 2})
        rows[2]["problem_id"] = "unpaired"
        self.assertFalse(holdout_gate(rows, complete=True)["allow_holdout"])
        self.assertFalse(holdout_gate(self.rows() + self.rows()[:1], complete=True)["allow_holdout"])


if __name__ == "__main__":
    unittest.main()
