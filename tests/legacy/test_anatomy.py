"""Check static geometry, numeric plots, and hook cleanup without a GPU."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from edit_anatomy import analyze_edit, capture_teacher_projections, save_anatomy
from svg_plots import heatmap, line_chart
from weight_edit import PROJECT_ROOT


class AnatomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anatomy = analyze_edit()

    def test_geometry_has_expected_shapes_and_normalized_directions(self):
        self.assertEqual(len(self.anatomy["layers"]), 2)
        self.assertTrue(0 <= self.anatomy["direction_absolute_cosine"] <= 1.000001)
        for row in self.anatomy["layers"]:
            self.assertEqual(row["changed_columns"], 64)
            self.assertEqual(len(row["column_norms"]), 3584)
            self.assertEqual(len(row["direction"]), 1024)
            self.assertAlmostEqual(torch.tensor(row["direction"]).norm().item(), 1.0, places=5)
            self.assertGreater(row["top_direction_energy"], 0.99)
            self.assertAlmostEqual(row["cumulative_energy"][-1], 1.0, places=5)
            self.assertTrue(all(a <= b for a, b in zip(row["cumulative_energy"], row["cumulative_energy"][1:])))

    def test_five_figures_are_valid_svg_with_no_executable_elements(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".cache") as temp:
            paths = save_anatomy(self.anatomy, Path(temp))
            self.assertEqual(len(paths), 5)
            allowed = {"svg", "rect", "polyline", "text", "line", "title"}
            for path in paths:
                root = ET.parse(path).getroot()
                self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
                self.assertTrue(all(node.tag.split("}")[-1] in allowed for node in root.iter()))

    def test_teacher_text_names_are_validated_before_model_access(self):
        with self.assertRaises(ValueError):
            capture_teacher_projections(None, None, self.anatomy, {"../outside": "text"})

    def test_forward_failure_removes_our_hooks_but_keeps_existing_hooks(self):
        modules = {row["name"].removesuffix(".weight"): torch.nn.Linear(1, 1) for row in self.anatomy["layers"]}
        existing = [module.register_forward_hook(lambda *args: None) for module in modules.values()]

        class Batch(dict):
            def to(self, device):
                return self

        class Model:
            device = torch.device("cpu")

            def get_submodule(self, name):
                return modules[name]

            def __call__(self, **kwargs):
                raise RuntimeError("FAKE-FORWARD-FAILURE")

        def tokenizer(*args, **kwargs):
            return Batch(input_ids=torch.tensor([[1, 2]]))

        with patch("edit_anatomy.edited_weights", side_effect=lambda *args, **kwargs: nullcontext([])):
            with self.assertRaisesRegex(RuntimeError, "FAKE-FORWARD"):
                capture_teacher_projections(Model(), tokenizer, self.anatomy, {"fixture": "text"})
        self.assertTrue(all(len(module._forward_hooks) == 1 for module in modules.values()))
        for handle in existing:
            handle.remove()


class SvgTests(unittest.TestCase):
    def test_labels_are_escaped_and_existing_files_are_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".cache") as temp:
            path = Path(temp) / "plot.svg"
            label = "<script>FAKE-UNTRUSTED-LABEL</script>"
            line_chart(path, label, {"fixture": [0.0, 1.0]})
            original = path.read_bytes()
            root = ET.fromstring(original)
            self.assertEqual(root.find("{http://www.w3.org/2000/svg}title").text, label)
            self.assertEqual(root.findall(".//{http://www.w3.org/2000/svg}script"), [])
            with self.assertRaises(FileExistsError):
                line_chart(path, label, {"fixture": [0.0, 1.0]})
            self.assertEqual(path.read_bytes(), original)

    def test_malformed_or_nonfinite_plot_values_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".cache") as temp:
            path = Path(temp) / "invalid.svg"
            for series in ({}, {"bad": []}, {"bad": [float("nan")]}, {"bad": [float("inf")]}):
                with self.subTest(series=series):
                    with self.assertRaises(ValueError):
                        line_chart(path, "fixture", series)
            with self.assertRaises(ValueError):
                heatmap(path, "fixture", [[1.0], [2.0, 3.0]])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
