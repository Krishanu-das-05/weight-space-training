"""Synthetic raw-first/durability tests plus inference-free legacy equivalence."""

import ast
import copy
import hashlib
import inspect
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from budget_sweep_generate import generate_record
from budget_sweep_io import ROOT, atomic_json, identity, json_bytes, read_json
from budget_sweep_scoring import grade_generation
from p50c_amendment import DIRECTORY, load_amended
from p50c_amended_analysis import recompute_amended, verify_legacy_outcomes
from p50c_amended_generate import RawFirstEditedGenerator, generate_raw_record
from p50c_amended_scoring import amended_grade, closure_from_tokens
from p50c_amended_store import AmendedRecordStore
from p50c_amended_worker import execute, run_schedule
from p50c_completion_analysis import original_records
from p50c_completion_io import RUN, make_decoder
from test_p50c_completion_runtime import FAKE_HASHES, FAKE_ORIGINALS, FakeModel, fake_decode, fixture


class AmendmentTests(unittest.TestCase):
    def make_store(self, directory, decode=fake_decode):
        plan, pool = fixture()
        plan["sampling"] = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        plan["baseline"] = {"selection": [{"id": row["id"], "user_message": "FAKE-CONTENT",
                                          "rendered_prompt": "FAKE-PROMPT"} for row in pool]}
        atomic_json(directory / "plan.json", plan)
        declaration = {"legacy_record_sha256": {}, "remaining_ids": [row["id"] for row in pool]}
        return AmendedRecordStore(directory, plan, pool, declaration, "FAKE-AMENDMENT", decode)

    def raw(self, store, attempt, tokens, decode=fake_decode, *, complete=True):
        job = store.plan["schedule"][0]
        capture = store.capture(attempt)
        capture.tokens({
            **job, "attempt_id": attempt, "plan_sha256": store.plan_hash,
            "amendment_sha256": store.amendment_hash, "expected": "7", "thinking": True,
            "verified_edited_tensor_sha256": dict(FAKE_HASHES), "edit_verified_before_generation": True,
            "raw_captured_before_response_validation": True, "seconds": 1.0,
            "peak_gpu_allocated_bytes": 1, "peak_gpu_reserved_bytes": 1,
            "user_message": "FAKE-CONTENT", "rendered_prompt": "FAKE-PROMPT", "prompt_tokens": 1,
        }, tokens)
        if complete:
            capture.text(decode(tokens))
        return capture

    def test_bound_violation_is_false_correct_but_eos_and_closure_survive(self):
        pieces = {9: "x" * 200001, 248069: "</think>", 7: r"\boxed{7}", 248044: "<|endoftext|>"}
        decode = lambda tokens: "".join(pieces[token] for token in tokens)
        tokens = [9, 248069, 7, 248044]
        text = decode(tokens)
        with self.assertRaises(ValueError):
            grade_generation(text, tokens, "7", thinking=True, eos_ids={248044}, terminal_tokens=["<|endoftext|>"])
        graded, violation, characters = amended_grade(text, tokens, "7", ["<|endoftext|>"])
        self.assertTrue(violation)
        self.assertEqual(characters, len(text))
        self.assertFalse(graded["strict_final_correct"])
        self.assertEqual(graded["natural_eos_index"], 4)
        self.assertTrue(all(closure_from_tokens(tokens, decode).values()))

    def test_empty_text_still_uses_token_eos_and_token_248046_is_not_eos(self):
        graded, violation, characters = amended_grade("", [248044], "7", ["<|endoftext|>"])
        self.assertEqual((graded["natural_eos_index"], violation, characters), (1, True, 0))
        self.assertFalse(graded["strict_final_correct"])
        graded, _, _ = amended_grade("", [248046] * 8192, "7", ["<|endoftext|>"])
        self.assertIsNone(graded["natural_eos_index"])
        with self.assertRaises(ValueError):
            amended_grade("", [248044, 9], "7", ["<|endoftext|>"])

    def test_closure_handles_a_literal_split_across_token_ids_and_cap_boundary(self):
        tokens = [9] * 510 + [4, 5, 6] + [9] * 100
        pieces = {9: "x", 4: "</", 5: "think", 6: ">"}
        result = closure_from_tokens(tokens, lambda ids: "".join(pieces[token] for token in ids))
        self.assertFalse(result["512"])
        self.assertTrue(result["1024"])

    def test_raw_text_and_ids_survive_bound_violation_with_exact_character_length(self):
        decode = lambda tokens: " " * 200001 + "</think>" + r"\boxed{7}" + "<|endoftext|>"
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            store = self.make_store(Path(temporary), decode)
            attempt = store.reserve(store.plan["schedule"][0])
            capture = self.raw(store, attempt, [9, 248044], decode)
            row = store.build_record(attempt)
            self.assertTrue(row["text_bound_violation"])
            self.assertEqual(row["text_character_length"], len(decode([])))
            self.assertIsNone(row["text"])
            self.assertTrue((capture.directory / "text.txt").exists())
            store.commit(row)
            self.assertEqual(store.records(), [row])
            self.assertEqual(store.budget()["accounted_tokens"], 2)
            corrupted = copy.deepcopy(row)
            corrupted["verified_edited_tensor_sha256"] = {}
            with self.assertRaises(ValueError):
                store.check_record(corrupted)

    def test_crash_after_raw_ids_recovers_without_generation_and_retains_old_charge(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            store = self.make_store(Path(temporary))
            original_failed = store.reserve(store.plan["schedule"][0])
            retry = store.reserve(store.plan["schedule"][0])
            self.raw(store, retry, [248069, 7, 248044], complete=False)
            rows = store.recover_raw()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["attempt_id"], retry)
            self.assertEqual(store.budget()["accounted_tokens"], 8195)
            self.assertTrue((store.directory / "budget_ledger" / (original_failed + ".json")).exists())
            self.assertEqual(store.recover_raw(), rows)
            with self.assertRaises(ValueError):
                store.reserve(store.plan["schedule"][0])

    def test_resumed_amended_run_without_fresh_verified_edit_cannot_generate_or_reserve(self):
        plan, pool = fixture()
        raw = Mock()
        store = SimpleNamespace(recover_raw=Mock(), reserve=Mock())
        gate = RawFirstEditedGenerator(FakeModel(FAKE_HASHES), None, plan, "FAKE-PLAN", store,
                                       generator=raw, hasher=lambda value: value)
        with self.assertRaisesRegex(RuntimeError, "start/resume"):
            run_schedule(store, gate, {}, {})
        raw.assert_not_called()
        store.reserve.assert_not_called()
        store.recover_raw.assert_not_called()
        gate.verify_on_start_or_resume()
        gate(self.pool_item(pool), plan["schedule"][0], "FAKE-ATTEMPT")
        raw.assert_called_once()

    @staticmethod
    def pool_item(pool):
        return pool[0]

    def test_sampling_call_rng_context_and_seeds_are_identical_to_original_core(self):
        old, new = (ast.parse(inspect.getsource(function)) for function in (generate_record, generate_raw_record))

        def extract(tree):
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            return [ast.dump(node, include_attributes=False) for node in calls
                    if isinstance(node.func, ast.Attribute) and node.func.attr in (
                        "generate", "fork_rng", "inference_mode", "manual_seed", "manual_seed_all")]

        self.assertEqual(extract(old), extract(new))

    def test_actual_eleven_record_bytes_and_all_old_outcomes_remain_identical(self):
        plan, pool, declaration, digest = load_amended(sealed=False)
        decode = make_decoder(plan)
        store = AmendedRecordStore(RUN, plan, pool, declaration, digest, decode)
        rows = [row for row in store.records() if identity(row) in declaration["legacy_record_sha256"]]
        self.assertEqual(len(rows), 11)
        for row in rows:
            self.assertEqual(hashlib.sha256(json_bytes(row)).hexdigest(),
                             declaration["legacy_record_sha256"][identity(row)])
        result = recompute_amended(plan, pool, rows, original_records(plan), decode, declaration, digest)
        snapshot = read_json(DIRECTORY / "legacy_analysis.json")
        for key, value in snapshot.items():
            self.assertEqual(json_bytes({"value": result[key]}), json_bytes({"value": value}))
        verify_legacy_outcomes(result, snapshot)

    def test_raw_ids_and_text_exist_before_a_structural_validation_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            store = self.make_store(Path(temporary))
            job = store.plan["schedule"][0]
            attempt = store.reserve(job)
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = "FAKE-PROMPT"
            inputs = SimpleNamespace(to=lambda device: {"input_ids": torch.tensor([[0]])})
            tokenizer.side_effect = None
            tokenizer.return_value = inputs
            tokenizer.decode.side_effect = lambda ids, **kwargs: fake_decode(ids)
            model = SimpleNamespace(device="cpu", generate=Mock(return_value=torch.tensor([[0, 248044, 9]])))
            with (
                patch("p50c_amended_generate.build_probe_content", return_value="FAKE-CONTENT"),
                patch("p50c_amended_generate.torch.cuda.synchronize"),
                patch("p50c_amended_generate.torch.cuda.reset_peak_memory_stats"),
                patch("p50c_amended_generate.torch.cuda.max_memory_allocated", return_value=1),
                patch("p50c_amended_generate.torch.cuda.max_memory_reserved", return_value=1),
                patch("p50c_amended_generate.torch.random.fork_rng", return_value=nullcontext()),
                patch("p50c_amended_generate.torch.manual_seed"),
                patch("p50c_amended_generate.torch.cuda.manual_seed_all"),
            ):
                with self.assertRaises(ValueError):
                    generate_raw_record(model, tokenizer, store.plan, store.pool[job["problem_id"]], job,
                                        attempt, store.plan_hash, store, FAKE_HASHES)
            directory = store.raw_directory / attempt
            self.assertEqual(read_json(directory / "tokens.json")["token_ids"], [248044, 9])
            self.assertEqual((directory / "text.txt").read_text(encoding="utf-8"), fake_decode([248044, 9]))
            self.assertEqual(store.records(), [])
            self.assertEqual(store.budget()["accounted_tokens"], 8192)

    def lifecycle(self, *, interrupted=False, starts=1):
        plan, pool = fixture()
        model = FakeModel(FAKE_ORIGINALS)
        store = SimpleNamespace(plan_hash="FAKE-PLAN", recover_journal=Mock())
        events = []

        def load(_):
            model.values = dict(FAKE_ORIGINALS)
            events.append("load_original")
            return model, None

        def verify(actual_model, _):
            self.assertEqual(actual_model.values, FAKE_ORIGINALS)
            events.append("verify_original")
            return dict(FAKE_ORIGINALS)

        @contextmanager
        def edit(actual_model, **kwargs):
            self.assertEqual(kwargs, {"layers": (15, 19), "strength": 1.0})
            events.append("apply_edit")
            actual_model.values = dict(FAKE_HASHES)
            try:
                yield []
            finally:
                actual_model.values = dict(FAKE_ORIGINALS)
                events.append("restore")

        def gate_factory(actual_model, tokenizer, p, digest, actual_store):
            events.append("new_gate")
            return RawFirstEditedGenerator(actual_model, tokenizer, p, digest, actual_store,
                                           generator=Mock(), hasher=lambda value: value)

        def schedule(actual_store, gate, runtime, checks):
            gate.assert_ready()
            self.assertEqual(checks["edited_after_apply"], FAKE_HASHES)
            events.append("verified_work")
            if interrupted:
                raise KeyboardInterrupt()
            return "complete"

        with (
            patch("p50c_amended_worker.load_amended", return_value=(plan, pool, {}, "FAKE-AMENDMENT")),
            patch("p50c_amended_worker.RunLock", side_effect=lambda path: nullcontext()),
            patch("p50c_amended_worker.AmendedRecordStore", return_value=store),
            patch("p50c_amended_worker.make_decoder", return_value=fake_decode),
            patch("p50c_amended_worker.configure_determinism", return_value={}),
            patch("p50c_amended_worker.load_model", side_effect=load),
            patch("p50c_amended_worker.verify_targets", side_effect=verify),
            patch("p50c_amended_worker.edited_weights", side_effect=edit),
            patch("p50c_amended_worker.RawFirstEditedGenerator", side_effect=gate_factory),
            patch("p50c_amended_worker.run_schedule", side_effect=schedule),
            patch("p50c_amended_worker.save_summary", side_effect=lambda _, status, *a, **k: {
                "status": status, "edited_main_responses": 0}),
            patch("p50c_amended_worker.atomic_json") as written,
            patch("builtins.print"),
        ):
            codes = [execute() for _ in range(starts)]
        return events, codes, written

    def test_amended_every_start_and_resume_reapplies_and_verifies(self):
        events, codes, _ = self.lifecycle(starts=2)
        cycle = ["load_original", "verify_original", "apply_edit", "new_gate", "verified_work", "restore", "verify_original"]
        self.assertEqual(events, cycle * 2)
        self.assertEqual(codes, [0, 0])

    def test_amended_interrupt_restores_and_independently_verifies_originals(self):
        events, codes, written = self.lifecycle(interrupted=True)
        self.assertEqual(codes, [1])
        self.assertEqual(events[-2:], ["restore", "verify_original"])
        self.assertTrue(written.call_args.args[1]["target_hash_checks"]["verified_after_restore"])


if __name__ == "__main__":
    unittest.main()
