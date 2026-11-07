"""Prompt construction, paired effects, and base-only runner regression tests."""

import copy
import io
import itertools
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import experiment_runner as runner
from experiment_design import HOLDOUT_MATH
from math_probe import PROMPT_INSTRUCTIONS, PROTOCOL_VERSION, build_probe_content, run_probes
from probe_metrics import response_metrics
from prompt_control import (
    PROMPT_CONDITIONS, PROMPT_PROBLEMS, PROMPT_SEEDS,
    analyze_prompt_controls, load_prompt_report, validate_prompt_design,
)


def fixture_probe(model, tokenizer, problems, condition, **options):
    problem = problems[0]
    correct = condition == "think_off_final"
    answer = problem["answer"] if correct else "-999"
    text = ("</think>" if options["thinking"] else "") + rf"\boxed{{{answer}}}<EOS>"
    count = 64 if options["prompt_style"] == "step_by_step" else 16
    return [{
        "condition": condition, "problem_id": problem["id"], "seed": options["seed"],
        "expected": problem["answer"], "protocol": PROTOCOL_VERSION,
        "prompt_style": options["prompt_style"],
        "user_message": build_probe_content(problem, options["prompt_style"]),
        "text": text, "seconds": 0.01,
        **response_metrics(
            text, [3] * (count - 1) + [9], problem["answer"],
            thinking=options["thinking"], max_new_tokens=options["max_new_tokens"],
            eos_ids={9}, terminal_tokens=("<EOS>",),
        ),
    }]


def fixture_records():
    rows = []
    for group in PROMPT_SEEDS:
        for index, problem in enumerate(PROMPT_PROBLEMS):
            for condition in PROMPT_CONDITIONS:
                row = fixture_probe(
                    None, None, [problem], condition.name, seed=group + index,
                    thinking=condition.thinking, prompt_style=condition.prompt_style, max_new_tokens=64,
                )[0]
                row["seed_group"] = group
                rows.append(row)
    return rows


class PromptTests(unittest.TestCase):
    def test_legacy_default_is_byte_for_byte_unchanged(self):
        problem = PROMPT_PROBLEMS[0]
        self.assertEqual(
            build_probe_content(problem),
            problem["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        )

    def test_final_only_has_no_step_instruction_or_answer_leak(self):
        problem = PROMPT_PROBLEMS[0]
        content = build_probe_content(problem, "final_only")
        self.assertEqual(content, problem["question"] + "\n" + PROMPT_INSTRUCTIONS["final_only"])
        self.assertNotIn("step by step", content)
        self.assertNotIn(problem["answer"], content)

    def test_invalid_style_fails_before_inference(self):
        for style in (None, True, [], "unknown"):
            with self.subTest(style=style):
                with self.assertRaises(ValueError):
                    run_probes(None, None, PROMPT_PROBLEMS, "fixture", prompt_style=style)

    def test_exact_controls_keep_their_original_instruction(self):
        problem = dict(id="fixture", question="Reply READY.", answer="READY", grader="exact")
        self.assertEqual(build_probe_content(problem), problem["question"])
        with self.assertRaises(ValueError):
            build_probe_content(problem, "final_only")

    def test_generation_receives_and_records_the_declared_prompt(self):
        class Batch(dict):
            def to(self, device):
                return self

        class Tokenizer:
            eos_token_id = 9
            all_special_tokens = ("<|im_end|>", "<EOS>")

            def apply_chat_template(self, messages, **options):
                self.message = messages[0]["content"]
                self.thinking = options["enable_thinking"]
                return "<USER>" + self.message + "<ASSISTANT>"

            def __call__(self, text, **options):
                return Batch(input_ids=torch.tensor([[1, 2]]))

            def decode(self, ids, **options):
                return "<EOS>" if ids == [9] else r"\boxed{1048}<|im_end|><EOS>"

        class Model:
            device = torch.device("cpu")
            generation_config = SimpleNamespace(eos_token_id=9)

            def generate(self, **options):
                self.options = options
                torch.rand(1)
                return torch.tensor([[1, 2, 3, 9]])

        saved_state = torch.get_rng_state().clone()
        for style in PROMPT_INSTRUCTIONS:
            with self.subTest(style=style), redirect_stdout(io.StringIO()):
                model, tokenizer = Model(), Tokenizer()
                row = run_probes(
                    model, tokenizer, [PROMPT_PROBLEMS[0]], "fixture",
                    max_new_tokens=32, seed=7, thinking=False, prompt_style=style,
                )[0]
                self.assertEqual(row["user_message"], tokenizer.message)
                self.assertEqual(row["rendered_prompt"], "<USER>" + tokenizer.message + "<ASSISTANT>")
                self.assertEqual(row["prompt_style"], style)
                self.assertTrue(row["delivered_correct"])
                self.assertFalse(tokenizer.thinking)
                self.assertEqual(model.options["temperature"], 0.6)
                self.assertEqual(model.options["top_k"], 20)
                self.assertEqual(model.options["top_p"], 0.95)
                self.assertEqual(model.options["logits_processor"][0].penalty, 1.5)
        self.assertTrue(torch.equal(saved_state, torch.get_rng_state()))


class PairedAnalysisTests(unittest.TestCase):
    def test_four_predeclared_comparisons_use_all_four_matching_pairs(self):
        analysis = analyze_prompt_controls(fixture_records(), complete=True)
        self.assertTrue(analysis["complete_pairs"])
        self.assertFalse(analysis["allow_holdout"])
        self.assertEqual(len(analysis["comparisons"]), 4)
        effect = next(item for item in analysis["comparisons"] if item["name"] == "prompt_effect_thinking_off")
        self.assertEqual(effect["pairs"], 4)
        self.assertEqual(effect["delivered_correct_wins"], 4)
        self.assertEqual(effect["delivered_correct_losses"], 0)
        self.assertEqual(effect["delivered_correct_rate_delta"], 1.0)
        self.assertEqual(effect["mean_token_delta"], -48)
        self.assertEqual(set(analysis["by_seed_group"]), {"0", "100"})

    def test_duplicates_wrong_seeds_and_bad_metadata_are_not_interpreted(self):
        for field, value in (("seed", 123), ("thinking", 1), ("condition", []), ("correct", 1)):
            rows = fixture_records()
            rows[0][field] = value
            with self.subTest(field=field):
                self.assertFalse(analyze_prompt_controls(rows, complete=True)["complete_pairs"])
        rows = fixture_records()
        rows[-1] = copy.deepcopy(rows[0])
        self.assertFalse(analyze_prompt_controls(rows, complete=True)["complete_pairs"])

    def test_partial_or_time_limited_runs_have_no_effect_estimates(self):
        rows = fixture_records()
        self.assertEqual(analyze_prompt_controls(rows, complete=False)["comparisons"], [])
        rows[0].update(stop_reason="time_budget", natural_stop=False, delivered_correct=False)
        self.assertFalse(analyze_prompt_controls(rows, complete=True)["complete_pairs"])

    def test_scope_cannot_expand_to_holdout_or_weight_edits(self):
        self.assertTrue(all(not item.layers and item.strength == 0 for item in PROMPT_CONDITIONS))
        for problems, conditions, seeds in (
            (HOLDOUT_MATH[:2], PROMPT_CONDITIONS, PROMPT_SEEDS),
            (PROMPT_PROBLEMS, runner.CONDITIONS, PROMPT_SEEDS),
            (PROMPT_PROBLEMS, PROMPT_CONDITIONS, (0,)),
        ):
            with self.assertRaises(ValueError):
                validate_prompt_design(problems, conditions, seeds)


class PromptRunnerTests(unittest.TestCase):
    def execute(self, directory, *, token_budget=None, failure=False, expired=False):
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "new_run_directory", return_value=directory))
            verified = stack.enter_context(patch.object(runner, "verify_base"))
            edits = stack.enter_context(patch.object(runner, "edited_weights"))
            stack.enter_context(patch.object(runner, "sampler_audit", return_value=[]))
            stack.enter_context(patch.object(
                runner, "run_probes",
                side_effect=RuntimeError("FAKE-PROMPT-RUN-ERROR") if failure else fixture_probe,
            ))
            if expired:
                clock = itertools.chain((0.0,), itertools.repeat(1000.0))
                stack.enter_context(patch.object(runner.time, "perf_counter", side_effect=lambda: next(clock)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if failure:
                with self.assertRaisesRegex(RuntimeError, "FAKE-PROMPT"):
                    runner.run_suite(None, None, stage="prompt_control", max_new_tokens=64)
            else:
                runner.run_suite(
                    None, None, stage="prompt_control", max_new_tokens=64, token_budget=token_budget,
                )
            edits.assert_not_called()
            self.assertGreaterEqual(verified.call_count, 2)
        return load_prompt_report(directory / "summary.json")

    def test_complete_run_is_paired_base_only_and_never_opens_holdout(self):
        with tempfile.TemporaryDirectory(prefix="prompt_control_fixture_", dir=runner.PROJECT_ROOT / "results") as temp:
            directory = Path(temp)
            report = self.execute(directory)
            self.assertTrue(report["base_only"])
            self.assertTrue(report["complete"])
            self.assertEqual(report["completed_generations"], 16)
            self.assertEqual(report["generation_token_budget"], 16384)
            self.assertTrue(report["decision"]["complete_pairs"])
            self.assertFalse(report["decision"]["allow_holdout"])
            self.assertTrue(all(row["changes"] == [] for row in report["records"]))
            self.assertEqual(len((directory / "records.jsonl").read_text().splitlines()), 16)
            with self.assertRaisesRegex(ValueError, "held-out gate"):
                runner.frozen_selection(directory / "summary.json")

    def test_token_and_wall_limits_preserve_partial_data_without_conclusions(self):
        for budget, expired, count in ((64, False, 1), (None, True, 0)):
            with self.subTest(budget=budget, expired=expired):
                with tempfile.TemporaryDirectory(prefix="prompt_control_fixture_", dir=runner.PROJECT_ROOT / "results") as temp:
                    report = self.execute(Path(temp), token_budget=budget, expired=expired)
                    self.assertFalse(report["complete"])
                    self.assertEqual(report["completed_generations"], count)
                    self.assertEqual(report["decision"]["comparisons"], [])

    def test_error_propagates_with_a_saved_partial_report(self):
        with tempfile.TemporaryDirectory(prefix="prompt_control_fixture_", dir=runner.PROJECT_ROOT / "results") as temp:
            report = self.execute(Path(temp), failure=True)
            self.assertEqual(report["status"], "interrupted_or_error")
            self.assertFalse(report["complete"])

    def test_reader_does_not_trust_saved_comparison_or_summary_values(self):
        with tempfile.TemporaryDirectory(prefix="prompt_control_fixture_", dir=runner.PROJECT_ROOT / "results") as temp:
            directory = Path(temp)
            report = self.execute(directory)
            report["summary"]["think_off_final"]["delivered_correct"] = 999
            report["decision"]["comparisons"] = []
            (directory / "summary.json").write_text(json.dumps(report), encoding="utf-8")
            reread = load_prompt_report(directory / "summary.json")
            self.assertEqual(reread["summary"]["think_off_final"]["delivered_correct"], 4)
            self.assertEqual(len(reread["decision"]["comparisons"]), 4)

    def test_reader_rejects_external_paths_and_invalid_parameter_types(self):
        for path in (runner.PROJECT_ROOT / "README.md", 123):
            with self.assertRaises(ValueError):
                load_prompt_report(path)


if __name__ == "__main__":
    unittest.main()
