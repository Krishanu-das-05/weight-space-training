"""Synthetic-only durability, deterministic-policy and worker-gating regressions."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import budget_sweep_generate as generation
from budget_sweep_analysis import prefix_status, recompute
from budget_sweep_io import ROOT, RecordStore, atomic_json, identity
from budget_sweep_scoring import grade_generation
from budget_sweep_worker import run_schedule


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="budget_runtime_test_", dir=ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.pool = [{"id": f"FAKE_{i}", "question": "FAKE-TEST-ITEM", "answer": "7"} for i in range(2)]
        self.plan = {
            "schedule": [], "token_budget": {"limit": 500000},
            "prefix_validation": {"question_ids": [item["id"] for item in self.pool]},
            "stopping": {"terminal_tokens_for_grading": ["<EOS>"]},
        }
        for index, problem in enumerate(self.pool):
            for condition in ("thinking_on", "thinking_off"):
                for kind, cap in (("prefix_validation_short", 1024), ("main", 8192)):
                    self.plan["schedule"].append({
                        "problem_id": problem["id"], "condition": condition, "seed": index,
                        "kind": kind, "max_new_tokens": cap,
                    })
        atomic_json(self.directory / "plan.json", self.plan)
        self.store = RecordStore(self.directory, self.plan, self.pool)
        self.store.recover_journal()

    def fake_generation(self, model, tokenizer, plan, problem, job, attempt, plan_hash):
        thinking = job["condition"] == "thinking_on"
        tokens = [248046, 3, 248044]
        text = ("</think>" if thinking else "") + r"\boxed{7}<EOS>"
        return {
            **job, "attempt_id": attempt, "plan_sha256": plan_hash,
            "expected": "7", "thinking": thinking, "token_ids": tokens, "text": text,
            "graded": grade_generation(text, tokens, "7", thinking=thinking, eos_ids={248044}, terminal_tokens=["<EOS>"]),
            "seconds": 1.0, "peak_gpu_allocated_bytes": 123,
            "token_248046_count": 1,
        }

    def commit_first(self):
        job = self.plan["schedule"][0]
        attempt = self.store.reserve(job)
        row = self.fake_generation(None, None, self.plan, self.pool[0], job, attempt, self.store.plan_hash)
        self.store.commit(row)
        return row

    def test_unfinished_reservations_remain_charged_and_unfunded_work_stops(self):
        job = self.plan["schedule"][0]
        self.store.reserve(job)
        self.assertEqual(self.store.budget()["unreconciled_reserved_tokens"], 1024)
        self.plan["token_budget"]["limit"] = 1024
        self.assertIsNone(self.store.reserve(job))
        self.assertEqual(len(self.store.records()), 0)

    def test_completed_generations_are_not_repeated_on_resume(self):
        first = self.commit_first()
        callback = Mock(side_effect=self.fake_generation)
        with redirect_stdout(io.StringIO()):
            status = run_schedule(self.store, None, None, {}, {}, callback)
        self.assertEqual(status, "complete")
        self.assertEqual(callback.call_count, 7)
        self.assertEqual(self.store.records()[0], first)
        with redirect_stdout(io.StringIO()):
            run_schedule(self.store, None, None, {}, {}, callback)
        self.assertEqual(callback.call_count, 7)
        self.assertEqual(self.store.budget()["completed_actual_tokens"], 24)

    def test_record_committed_before_journal_append_is_recovered_without_generation(self):
        row = self.commit_first()
        snapshot = self.directory / "records.before-recovery.jsonl"
        self.store.journal.rename(snapshot)
        self.assertEqual(self.store.recover_journal(), [row])
        self.assertEqual(json.loads(self.store.journal.read_text()), row)

    def test_partial_journal_tail_is_backed_up_not_discarded(self):
        self.commit_first()
        with self.store.journal.open("ab") as handle:
            handle.write(b'{"FAKE-CRASH-FRAGMENT":')
        original = self.store.journal.read_bytes()
        self.store.recover_journal()
        backups = list((self.directory / "recovery").glob("*.jsonl"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertEqual(len(self.store.journal.read_text().splitlines()), 1)

    def test_unknown_journal_corruption_stops(self):
        self.commit_first()
        with self.store.journal.open("ab") as handle:
            handle.write(b'{"FAKE-BAD":}\n')
        with self.assertRaisesRegex(ValueError, "corruption"):
            self.store.recover_journal()

    def test_prefix_mismatch_stops_after_the_first_pair(self):
        def differing(*args):
            row = self.fake_generation(*args)
            if row["kind"] == "main":
                row["token_ids"][1] = 4
            return row
        callback = Mock(side_effect=differing)
        with redirect_stdout(io.StringIO()):
            status = run_schedule(self.store, None, None, {}, {}, callback)
        self.assertEqual(status, "prefix_mismatch")
        self.assertEqual(callback.call_count, 2)
        self.assertTrue(prefix_status(self.plan, self.store.records())["failed"])

    def test_248046_is_counted_without_becoming_a_stopping_token(self):
        self.commit_first()
        summary = recompute(self.plan, self.pool, self.store.records())
        self.assertEqual(summary["token_248046_diagnostics"]["thinking_on"]["all"]["responses_with_248046"], 1)
        row = self.store.records()[0]
        self.assertEqual(row["graded"]["natural_eos_index"], 3)
        self.assertEqual(row["token_ids"][0], 248046)


class DeterminismTests(unittest.TestCase):
    def test_determinism_failure_is_fatal_and_never_disabled(self):
        fake_torch = Mock()
        fake_torch.cuda.is_initialized.return_value = False
        fake_torch.use_deterministic_algorithms.side_effect = RuntimeError("FAKE-NONDETERMINISTIC")
        with patch.object(generation, "torch", fake_torch):
            with self.assertRaisesRegex(RuntimeError, "without a fallback"):
                generation.configure_determinism()
        fake_torch.use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)

    def test_cuda_already_initialized_is_rejected(self):
        with patch.object(generation.torch.cuda, "is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before CUDA"):
                generation.configure_determinism()
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")


if __name__ == "__main__":
    unittest.main()
