"""Synthetic resume/provenance/paired-analysis checks; no subject inference."""

import copy
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from budget_sweep_io import ROOT, atomic_json
from budget_sweep_scoring import grade_generation
from p50c_completion_analysis import outcomes, recompute
from p50c_completion_generate import VerifiedEditedGenerator
from p50c_completion_io import EditedRecordStore, require_edited_provenance
from p50c_completion_worker import execute, run_schedule

FAKE_HASHES = {"FAKE-LAYER-15": "1" * 64, "FAKE-LAYER-19": "2" * 64}
FAKE_ORIGINALS = {"FAKE-LAYER-15": "3" * 64, "FAKE-LAYER-19": "4" * 64}


class FakeModel:
    def __init__(self, values):
        self.values = dict(values)

    def get_parameter(self, name):
        return self.values[name]


def fake_decode(tokens):
    pieces = {9: "x", 7: r"\boxed{7}", 8: r"\boxed{8}", 248069: "</think>", 248044: "<|endoftext|>"}
    return "".join(pieces[token] for token in tokens)


def fixture():
    pool = [{"id": f"FAKE-ITEM-{i:02d}", "answer": "7"} for i in range(30)]
    jobs = [{"problem_id": item["id"], "seed": i, "condition": "thinking_on", "kind": "main",
             "model_variant": "published_p50c", "max_new_tokens": 8192} for i, item in enumerate(pool)]
    plan = {
        "schedule": jobs, "questions_and_seeds": [{"id": item["id"], "seed": i} for i, item in enumerate(pool)],
        "edit": {"target_edited_sha256": dict(FAKE_HASHES)},
        "model": {"target_base_sha256": dict(FAKE_ORIGINALS)},
        "stopping": {"terminal_tokens_for_grading": ["<|endoftext|>"]},
        "token_budget": {"limit": 250000},
    }
    return plan, pool


def row_for(plan, index, *, edited=True):
    job = plan["schedule"][index]
    natural = index < (9 if edited else 2)
    correct = index < (7 if edited else 1)
    tokens = [248069, 7 if correct else 8, 248044] if natural else [9] * 8192
    if edited and index == 10:
        tokens[0] = 248069
    text = fake_decode(tokens)
    return {
        **job, "thinking": True, "expected": "7", "token_ids": tokens, "text": text,
        "graded": grade_generation(text, tokens, "7", thinking=True, eos_ids={248044},
                                   terminal_tokens=["<|endoftext|>"]),
        "verified_edited_tensor_sha256": dict(FAKE_HASHES), "edit_verified_before_generation": True,
        "token_248046_count": 0, "contains_token_248046": False,
        "seconds": 1.0, "peak_gpu_allocated_bytes": 1, "peak_gpu_reserved_bytes": 1,
    }


class GenerationGuardTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.pool = fixture()
        self.raw = Mock(return_value={"model_variant": "published_p50c"})
        self.model = FakeModel(FAKE_HASHES)
        self.gate = VerifiedEditedGenerator(self.model, None, self.plan, "FAKE-PLAN",
                                            generator=self.raw, hasher=lambda value: value)

    def test_resumed_run_without_verified_edit_cannot_generate_or_reserve(self):
        # A durable prior response is not authorization for a fresh process.
        store = SimpleNamespace(recover_journal=Mock(return_value=[row_for(self.plan, 0)]), reserve=Mock())
        with self.assertRaisesRegex(RuntimeError, "start/resume"):
            run_schedule(store, self.gate, {}, {}, fake_decode)
        self.raw.assert_not_called()
        store.reserve.assert_not_called()
        store.recover_journal.assert_not_called()

    def test_correct_hashes_still_require_explicit_fresh_start_verification(self):
        with self.assertRaises(RuntimeError):
            self.gate(self.pool[0], self.plan["schedule"][0], "FAKE-ATTEMPT")
        self.raw.assert_not_called()
        self.assertEqual(self.gate.verify_on_start_or_resume(), FAKE_HASHES)
        row = self.gate(self.pool[0], self.plan["schedule"][0], "FAKE-ATTEMPT")
        self.assertEqual(row["verified_edited_tensor_sha256"], FAKE_HASHES)
        self.assertIs(row["edit_verified_before_generation"], True)
        self.raw.assert_called_once()

    def test_original_weights_cannot_arm_gate(self):
        self.model.values = dict(FAKE_ORIGINALS)
        with self.assertRaises(RuntimeError):
            self.gate.verify_on_start_or_resume()
        with self.assertRaises(RuntimeError):
            self.gate(self.pool[0], self.plan["schedule"][0], "FAKE-ATTEMPT")
        self.raw.assert_not_called()

    def test_changed_weights_after_verification_stop_before_generator(self):
        self.gate.verify_on_start_or_resume()
        self.model.values = dict(FAKE_ORIGINALS)
        with self.assertRaises(RuntimeError):
            self.gate(self.pool[0], self.plan["schedule"][0], "FAKE-ATTEMPT")
        self.raw.assert_not_called()
        self.model.values = dict(FAKE_HASHES)
        with self.assertRaises(RuntimeError):
            self.gate(self.pool[0], self.plan["schedule"][0], "FAKE-ATTEMPT")


class AnalysisAndStorageTests(unittest.TestCase):
    def test_analysis_rejects_missing_or_mismatched_hashes_before_decoding(self):
        plan, pool = fixture()
        for field in ("missing", "wrong", "flag", "variant"):
            row = row_for(plan, 0)
            if field == "missing":
                row.pop("verified_edited_tensor_sha256")
            elif field == "wrong":
                row["verified_edited_tensor_sha256"]["FAKE-LAYER-15"] = "0" * 64
            elif field == "flag":
                row["edit_verified_before_generation"] = False
            else:
                row["model_variant"] = "original"
            decode = Mock()
            with self.subTest(field=field), self.assertRaises(ValueError):
                outcomes(plan, pool, [row], decode, edited=True)
            decode.assert_not_called()

    def test_paired_counts_and_exact_tests_keep_all_three_outcomes_distinct(self):
        plan, pool = fixture()
        original = [row_for(plan, i, edited=False) for i in range(30)]
        edited = [row_for(plan, i) for i in range(30)]
        result = recompute(plan, pool, edited, original, fake_decode)
        self.assertTrue(result["complete_data"])
        self.assertEqual(result["tables"]["natural_completion"]["edited"][-1]["successes"], 9)
        self.assertEqual(result["tables"]["strict_correct_delivery"]["edited"][-1]["successes"], 7)
        self.assertEqual(result["tables"]["thinking_closed"]["edited"][-1]["successes"], 10)
        effect = result["paired_comparisons"]["natural_completion"][-1]
        self.assertEqual((effect["edited_wins"], effect["original_wins"], effect["ties"]), (7, 0, 23))
        self.assertEqual(effect["exact_mcnemar_p"], 0.015625)
        self.assertEqual(effect["holm_adjusted_p"], 0.078125)
        self.assertEqual(len(result["short_wrong_answers"]), 2)

    def test_partial_analysis_does_not_report_planned_complete_data_p_values(self):
        plan, pool = fixture()
        result = recompute(plan, pool, [row_for(plan, 0)], [row_for(plan, i, edited=False) for i in range(30)], fake_decode)
        self.assertFalse(result["complete_data"])
        self.assertTrue(all(not values for values in result["paired_comparisons"].values()))
        self.assertEqual(result["tables"]["thinking_closed"]["edited"][-1]["n"], 1)

    def test_storage_resume_preserves_provenance_completed_rows_and_crash_charge(self):
        plan, pool = fixture()
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            directory = Path(temporary)
            atomic_json(directory / "plan.json", plan)
            store = EditedRecordStore(directory, plan, pool)
            attempt = store.reserve(plan["schedule"][0])
            row = {**row_for(plan, 0), "attempt_id": attempt, "plan_sha256": store.plan_hash}
            store.commit(row)
            orphan = store.reserve(plan["schedule"][1])
            self.assertIsNotNone(orphan)
            resumed = EditedRecordStore(directory, plan, pool)
            self.assertEqual(resumed.recover_journal(), [row])
            self.assertEqual(resumed.budget()["accounted_tokens"], 8195)
            with self.assertRaises(ValueError):
                resumed.reserve(plan["schedule"][0])
            corrupted = copy.deepcopy(row)
            corrupted["verified_edited_tensor_sha256"]["FAKE-LAYER-19"] = "0" * 64
            with self.assertRaises(ValueError):
                resumed.check_record(corrupted)
            require_edited_provenance(row, plan)


class WorkerLifecycleTests(unittest.TestCase):
    def exercise(self, *, interrupt=False, bad_edit=False, starts=1):
        plan, pool = fixture()
        events = []
        model = FakeModel(FAKE_ORIGINALS)
        store = SimpleNamespace(plan_hash="FAKE-PLAN", recover_journal=Mock(return_value=[]))

        def load(_):
            events.append("load_original")
            model.values = dict(FAKE_ORIGINALS)
            return model, None

        def verify_original(actual_model, _):
            self.assertEqual(actual_model.values, FAKE_ORIGINALS)
            events.append("verify_original")
            return dict(FAKE_ORIGINALS)

        @contextmanager
        def context(actual_model, **kwargs):
            self.assertEqual(kwargs, {"layers": (15, 19), "strength": 1.0})
            events.append("apply_edit")
            actual_model.values = dict(FAKE_HASHES)
            if bad_edit:
                actual_model.values["FAKE-LAYER-15"] = "0" * 64
            try:
                yield []
            finally:
                events.append("restore")
                actual_model.values = dict(FAKE_ORIGINALS)

        def gate_factory(actual_model, tokenizer, actual_plan, digest):
            events.append("new_unverified_gate")
            return VerifiedEditedGenerator(actual_model, tokenizer, actual_plan, digest,
                                           generator=Mock(), hasher=lambda value: value)

        def schedule(_, gate, runtime, checks, decode):
            gate.assert_ready()
            self.assertEqual(checks["edited_after_apply"], FAKE_HASHES)
            events.append("verified_work")
            if interrupt:
                raise KeyboardInterrupt()
            return "complete"

        def summary(_, status, *args, **kwargs):
            return {"status": status, "edited_main_responses": 0}

        with (
            patch("p50c_completion_worker.load_approved", return_value=(plan, pool)),
            patch("p50c_completion_worker.RunLock", side_effect=lambda _: nullcontext()),
            patch("p50c_completion_worker.EditedRecordStore", return_value=store),
            patch("p50c_completion_worker.make_decoder", return_value=fake_decode),
            patch("p50c_completion_worker.configure_determinism", return_value={}),
            patch("p50c_completion_worker.save_summary", side_effect=summary),
            patch("p50c_completion_worker.load_model", side_effect=load),
            patch("p50c_completion_worker.verify_targets", side_effect=verify_original),
            patch("p50c_completion_worker.edited_weights", side_effect=context),
            patch("p50c_completion_worker.VerifiedEditedGenerator", side_effect=gate_factory),
            patch("p50c_completion_worker.run_schedule", side_effect=schedule) as scheduled,
            patch("p50c_completion_worker.atomic_json") as written,
            patch("builtins.print"),
        ):
            codes = [execute(ROOT / "results" / "FAKE-RUN-NOT-WRITTEN") for _ in range(starts)]
        self.assertEqual(model.values, FAKE_ORIGINALS)
        return events, codes, scheduled, written

    def test_every_start_and_resume_reapplies_and_verifies_before_work(self):
        events, codes, scheduled, _ = self.exercise(starts=2)
        expected_cycle = ["load_original", "verify_original", "apply_edit", "new_unverified_gate",
                          "verified_work", "restore", "verify_original"]
        self.assertEqual(events, expected_cycle * 2)
        self.assertEqual(codes, [0, 0])
        self.assertEqual(scheduled.call_count, 2)

    def test_interrupt_restores_and_independently_verifies_originals(self):
        events, codes, _, written = self.exercise(interrupt=True)
        self.assertEqual(codes, [1])
        self.assertEqual(events[-2:], ["restore", "verify_original"])
        history = written.call_args.args[1]
        self.assertEqual(history["status"], "interrupted")
        self.assertTrue(history["target_hash_checks"]["verified_after_restore"])

    def test_failed_edit_verification_stops_before_schedule_and_restores(self):
        events, codes, scheduled, written = self.exercise(bad_edit=True)
        self.assertEqual(codes, [1])
        scheduled.assert_not_called()
        self.assertEqual(events[-2:], ["restore", "verify_original"])
        self.assertEqual(written.call_args.args[1]["status"], "fatal_error")


if __name__ == "__main__":
    unittest.main()
