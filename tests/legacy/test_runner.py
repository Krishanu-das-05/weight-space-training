"""Test budgets and durable reports without running an LLM."""

import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import experiment_runner as runner
from experiment_design import CONDITIONS, DEVELOPMENT


class RunnerTests(unittest.TestCase):
    def fake_probe(self, model, tokenizer, problems, condition, **options):
        correct = condition == "base_no_thinking"
        return [dict(
            condition=condition, problem_id=problems[0]["id"], seed=options["seed"],
            delivered_correct=correct, correct=correct, natural_stop=True,
            hit_token_limit=False, stop_reason="eos", generated_tokens=32, seconds=0.01,
        )]

    def execute(self, directory, *, budget=1000, failure=False):
        with (
            patch.object(runner, "new_run_directory", return_value=directory),
            patch.object(runner, "verify_base") as restored,
            patch.object(runner, "edited_weights", side_effect=lambda *args, **kwargs: nullcontext([])),
            patch.object(
                runner, "run_probes",
                side_effect=RuntimeError("FAKE-INFERENCE-ERROR") if failure else self.fake_probe,
            ),
            redirect_stdout(io.StringIO()),
        ):
            if failure:
                with self.assertRaisesRegex(RuntimeError, "FAKE-INFERENCE"):
                    runner.run_suite(None, None, max_new_tokens=32, seconds=20, token_budget=budget)
            else:
                runner.run_suite(None, None, max_new_tokens=32, seconds=20, token_budget=budget)
            self.assertGreaterEqual(restored.call_count, 2)
        return json.loads((directory / "summary.json").read_text(encoding="utf-8"))

    def test_complete_paired_run_records_all_conditions_and_does_not_select(self):
        with tempfile.TemporaryDirectory(dir=runner.PROJECT_ROOT / ".cache") as temp:
            directory = Path(temp)
            report = self.execute(directory)
            self.assertTrue(report["complete"])
            self.assertEqual(report["completed_generations"], 12)
            self.assertFalse(report["decision"]["allow_holdout"])
            self.assertEqual(len((directory / "records.jsonl").read_text().splitlines()), 12)
            for condition in CONDITIONS:
                rows = [row for row in report["records"] if row["condition"] == condition.name]
                self.assertEqual([row["seed"] for row in rows], [0, 1])

    def test_token_budget_stops_before_starting_an_unfunded_generation(self):
        with tempfile.TemporaryDirectory(dir=runner.PROJECT_ROOT / ".cache") as temp:
            report = self.execute(Path(temp), budget=32)
            self.assertFalse(report["complete"])
            self.assertEqual(report["completed_generations"], 1)
            self.assertEqual(report["status"], "budget_exhausted")
            self.assertFalse(report["decision"]["allow_holdout"])

    def test_error_is_propagated_but_partial_report_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=runner.PROJECT_ROOT / ".cache") as temp:
            report = self.execute(Path(temp), failure=True)
            self.assertEqual(report["status"], "interrupted_or_error")
            self.assertFalse(report["complete"])
            self.assertTrue(report["restored_base_hashes_verified"])
            self.assertFalse(report["decision"]["allow_holdout"])

    def test_invalid_plan_fails_before_model_work(self):
        for tokens, seconds, budget in ((0, 20, 1000), (32, -1, 1000), (32, 20, 0)):
            with self.subTest(tokens=tokens, seconds=seconds, budget=budget):
                with self.assertRaises(ValueError):
                    runner.validate_plan(DEVELOPMENT[:2], CONDITIONS, (0,), tokens, seconds, budget)

    def test_selection_cannot_read_outside_the_results_directory(self):
        with self.assertRaises(ValueError):
            runner.frozen_selection(runner.PROJECT_ROOT / "README.md")

    def test_failed_pilot_cannot_be_promoted_to_holdout(self):
        with tempfile.TemporaryDirectory(dir=runner.PROJECT_ROOT / "results") as temp:
            directory = Path(temp)
            self.execute(directory)
            with self.assertRaisesRegex(ValueError, "held-out gate"):
                runner.frozen_selection(directory / "summary.json")

    def test_legacy_selection_defaults_to_unchanged_step_prompt(self):
        candidate = asdict(CONDITIONS[2])
        candidate.pop("prompt_style")
        fixture = dict(
            stage="pilot", protocol=runner.PROTOCOL_VERSION, model_revision=runner.REVISION,
            sampling=runner.SAMPLING, complete=True, restored_base_hashes_verified=True,
            max_new_tokens=1024, decision=dict(allow_holdout=True, selected_condition=candidate),
        )
        with tempfile.TemporaryDirectory(dir=runner.PROJECT_ROOT / "results") as temp:
            path = Path(temp) / "summary.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            selected, budget = runner.frozen_selection(path)
            self.assertEqual(selected, CONDITIONS[2])
            self.assertEqual(selected.prompt_style, "step_by_step")
            self.assertEqual(budget, 1024)

    def test_prompt_cli_rejects_scope_changes_before_model_loading(self):
        with (
            patch("sys.argv", ["experiment_runner.py", "--stage", "prompt-control", "--problems", "4"]),
            patch.object(runner, "load_local_model") as loader,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                runner.main()
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
