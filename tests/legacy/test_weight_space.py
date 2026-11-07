"""CPU regression tests; real tensor fixtures use the already-downloaded model."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open

import weight_edit
from download_model import MODEL_DIR, WEIGHT_FILE
from math_probe import GeneratedPresencePenalty, boxed_integer, repeated_ngram_fraction, run_probes


class ArtifactTests(unittest.TestCase):
    def test_invalid_artifact_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            weight_edit._verify_artifact("edit.npz", b"FAKE-INVALID-ARTIFACT")

    def test_corrupt_existing_data_is_not_overwritten_or_downloaded(self):
        with tempfile.TemporaryDirectory(dir=weight_edit.PROJECT_ROOT / ".cache") as temp:
            directory = Path(temp)
            path = directory / "manifest.json"
            path.write_bytes(b"FAKE-INVALID-MANIFEST")
            with patch.object(weight_edit, "EDIT_DIR", directory), patch.object(
                weight_edit, "urlopen"
            ) as network:
                with self.assertRaises(ValueError):
                    weight_edit.download_edit()
                network.assert_not_called()
            self.assertEqual(path.read_bytes(), b"FAKE-INVALID-MANIFEST")


@unittest.skipUnless(
    (MODEL_DIR / WEIGHT_FILE).is_file() and (weight_edit.EDIT_DIR / "edit.npz").is_file(),
    "Download the model and published edit to run tensor integration tests.",
)
class TensorEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.edits = weight_edit.load_edit()
        with safe_open(MODEL_DIR / WEIGHT_FILE, framework="pt", device="cpu") as source:
            cls.originals = {
                edit["name"]: source.get_tensor(edit["name"]).clone() for edit in cls.edits
            }

    def setUp(self):
        self.parameters = {
            name: torch.nn.Parameter(tensor.clone(), requires_grad=False)
            for name, tensor in self.originals.items()
        }
        self.model = SimpleNamespace(get_parameter=self.parameters.__getitem__)

    def assert_restored(self):
        for edit in self.edits:
            self.assertEqual(
                weight_edit.tensor_sha256(self.parameters[edit["name"]]), edit["base_sha256"]
            )

    def test_exact_edit_and_bit_exact_restoration(self):
        with weight_edit.edited_weights(self.model) as changes:
            self.assertEqual(sum(item["changed_entries"] for item in changes), 130188)
            for edit in self.edits:
                self.assertEqual(
                    weight_edit.tensor_sha256(self.parameters[edit["name"]]),
                    edit["edited_sha256"],
                )
        self.assert_restored()

    def test_restoration_after_exception_and_keyboard_interrupt(self):
        for exception in (RuntimeError, KeyboardInterrupt):
            with self.subTest(exception=exception):
                with self.assertRaises(exception):
                    with weight_edit.edited_weights(self.model):
                        raise exception("FAKE-TEST-INTERRUPTION")
                self.assert_restored()

    def test_single_layer_control_leaves_other_layer_unchanged(self):
        for layer in (15, 19):
            with self.subTest(layer=layer):
                with weight_edit.edited_weights(self.model, layers=(layer,)) as changes:
                    self.assertEqual(len(changes), 1)
                    for edit in self.edits:
                        key = "edited_sha256" if f".{layer}." in edit["name"] else "base_sha256"
                        self.assertEqual(
                            weight_edit.tensor_sha256(self.parameters[edit["name"]]), edit[key]
                        )
                self.assert_restored()

    def test_late_base_mismatch_does_not_partially_apply_first_tensor(self):
        second = self.parameters[self.edits[1]["name"]]
        with torch.no_grad():
            second[0, 0].add_(1)
        before = {name: weight_edit.tensor_sha256(value) for name, value in self.parameters.items()}
        with self.assertRaisesRegex(ValueError, "Base tensor hash"):
            with weight_edit.edited_weights(self.model):
                self.fail("A mismatched base model must not enter the context.")
        self.assertEqual(
            before, {name: weight_edit.tensor_sha256(value) for name, value in self.parameters.items()}
        )

    def test_float32_model_is_rejected_without_silent_conversion(self):
        name = self.edits[0]["name"]
        self.parameters[name] = torch.nn.Parameter(self.originals[name].float(), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "BF16"):
            with weight_edit.edited_weights(self.model):
                self.fail("Only the exact original BF16 model is supported.")
        self.assertEqual(self.parameters[name].dtype, torch.float32)

    def test_negative_sparse_index_is_rejected(self):
        edit = {**self.edits[0], "indices": self.edits[0]["indices"].copy()}
        edit["indices"][0] = -1
        with self.assertRaisesRegex(ValueError, "sparse edit"):
            weight_edit._patched_tensor(self.originals[edit["name"]], edit)
        self.assert_restored()

    def test_wrong_reconstructed_hash_is_rejected(self):
        edit = {**self.edits[0], "values": self.edits[0]["values"].copy()}
        edit["values"][0] ^= np.uint16(1)
        with self.assertRaisesRegex(ValueError, "Edited tensor hash"):
            weight_edit._patched_tensor(self.originals[edit["name"]], edit)
        self.assert_restored()

    def test_invalid_layer_selection_is_rejected(self):
        for layers in ((), (14,), (15, 15), (True,), "15", (15, 19, 15)):
            with self.subTest(layers=layers):
                with self.assertRaises(ValueError):
                    with weight_edit.edited_weights(self.model, layers=layers):
                        self.fail("Invalid layer selection was accepted.")
        self.assert_restored()

    def test_zero_strength_is_bit_exact_identity(self):
        with weight_edit.edited_weights(self.model, strength=0) as changes:
            self.assertTrue(all(item["changed_entries"] == 0 for item in changes))
            self.assert_restored()
        self.assert_restored()

    def test_half_strength_only_changes_published_indices_and_restores(self):
        with weight_edit.edited_weights(self.model, strength=0.5) as changes:
            for edit, change in zip(self.edits, changes):
                before = self.originals[edit["name"]].view(torch.uint16).numpy().reshape(-1)
                after = self.parameters[edit["name"]].detach().view(torch.uint16).numpy().reshape(-1)
                allowed = np.zeros(before.size, dtype=bool)
                allowed[edit["indices"]] = True
                self.assertTrue(np.array_equal(before[~allowed], after[~allowed]))
                self.assertGreater(change["changed_entries"], 0)
                self.assertNotEqual(change["applied_sha256"], edit["edited_sha256"])
                self.assertEqual(change["applied_sha256"], weight_edit.tensor_sha256(self.parameters[edit["name"]]))
        self.assert_restored()

    def test_invalid_strength_is_rejected_before_changes(self):
        for strength in (-0.1, 1.1, float("nan"), True, "0.5"):
            with self.subTest(strength=strength):
                with self.assertRaises(ValueError):
                    with weight_edit.edited_weights(self.model, strength=strength):
                        self.fail("Invalid strength was accepted.")
        self.assert_restored()


class PresencePenaltyTests(unittest.TestCase):
    def test_prompt_tokens_are_not_penalized(self):
        scores = torch.zeros(1, 5)
        result = GeneratedPresencePenalty(2, 1.5)(torch.tensor([[1, 2]]), scores)
        self.assertTrue(torch.equal(result, scores))

    def test_seen_generated_token_is_penalized_once_not_per_occurrence(self):
        scores = torch.zeros(1, 5)
        result = GeneratedPresencePenalty(2, 1.5)(torch.tensor([[1, 2, 3, 3]]), scores)
        self.assertTrue(torch.equal(result, torch.tensor([[0.0, 0.0, 0.0, -1.5, 0.0]])))
        self.assertTrue(torch.equal(scores, torch.zeros(1, 5)))

    def test_batch_rows_have_independent_seen_tokens(self):
        tokens = torch.tensor([[1, 2, 3, 3], [2, 3, 1, 1]])
        result = GeneratedPresencePenalty(2, 1.5)(tokens, torch.zeros(2, 5))
        expected = torch.tensor([[0.0, 0.0, 0.0, -1.5, 0.0], [0.0, -1.5, 0.0, 0.0, 0.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_invalid_parameters_fail_fast(self):
        for prompt, penalty in ((-1, 1), (True, 1), (2, float("nan")), (2, 3)):
            with self.subTest(prompt=prompt, penalty=penalty):
                with self.assertRaises(ValueError):
                    GeneratedPresencePenalty(prompt, penalty)


class ProbeMetricTests(unittest.TestCase):
    def test_grade_last_boxed_integer_only(self):
        self.assertEqual(boxed_integer(r"\boxed{3} then \boxed{+0013}"), "13")
        self.assertEqual(boxed_integer(r"Final: \boxed{-8}"), "-8")
        self.assertIsNone(boxed_integer(r"\boxed{3} then \boxed{3.0}"))
        self.assertIsNone(boxed_integer(r"\boxed{3} then \boxed{\frac{1}{2}}"))
        self.assertIsNone(boxed_integer(r"\boxed{3} then \boxed{"))
        self.assertIsNone(boxed_integer("The last number is 1048."))

    def test_repetition_is_a_fraction_and_handles_short_text(self):
        self.assertEqual(repeated_ngram_fraction([1, 2]), 0.0)
        self.assertEqual(repeated_ngram_fraction([1, 2, 3, 4, 5]), 0.0)
        self.assertAlmostEqual(repeated_ngram_fraction([1, 2, 1, 2, 1, 2]), 1 / 3)
        with self.assertRaises(ValueError):
            repeated_ngram_fraction([1, 2], n=0)

    def test_probe_budget_and_seed_are_validated_before_inference(self):
        for budget, seed in ((0, 0), (True, 0), (5000, 0), (32, -1)):
            with self.subTest(budget=budget, seed=seed):
                with self.assertRaises(ValueError):
                    run_probes(None, None, [], "test", max_new_tokens=budget, seed=seed)


if __name__ == "__main__":
    unittest.main()
