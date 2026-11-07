"""Inspect sparse edit geometry and teacher-forced MLP projections locally."""

import json
import re

import torch
from safetensors import safe_open

from download_model import MODEL_DIR, WEIGHT_FILE
from svg_plots import heatmap, line_chart
from weight_edit import UPSTREAM_COMMIT, _patched_tensor, edited_weights, load_edit

TEACHER_TEXTS = {
    "short_reasoning": "What is 2^10 + 24?\n2^10 is 1024. Adding 24 gives 1048. The answer is 1048.",
    "rechecking": (
        "What is 2^10 + 24?\n"
        + "2^10 is 1024. Adding 24 gives 1048. Let me check the calculation again. " * 4
    ),
}


def analyze_edit():
    layers = []
    with safe_open(MODEL_DIR / WEIGHT_FILE, framework="pt", device="cpu") as source:
        for edit in load_edit():
            base = source.get_tensor(edit["name"])
            patched = _patched_tensor(base, edit)
            delta = patched.float() - base.float()
            columns = torch.nonzero(delta.ne(0).any(dim=0), as_tuple=True)[0]
            active = delta[:, columns]
            left, singular, _ = torch.linalg.svd(active, full_matrices=False)
            energy = singular.square() / singular.square().sum()
            cumulative = energy.cumsum(0)
            direction = left[:, 0]
            if direction[direction.abs().argmax()] < 0:
                direction = -direction
            layers.append({
                "name": edit["name"], "layer": int(edit["name"].split(".")[3]),
                "changed_entries": edit["changed_entries"], "changed_columns": len(columns),
                "relative_frobenius_change": float(delta.norm() / base.float().norm()),
                "top_direction_energy": float(energy[0]),
                "rank_for_99_percent_energy": int(torch.searchsorted(cumulative, 0.99)) + 1,
                "singular_values": singular.tolist(), "cumulative_energy": cumulative.tolist(),
                "direction": direction.tolist(), "column_indices": columns.tolist(),
                "column_norms": delta.norm(dim=0).tolist(),
                "binned_heatmap": active.reshape(64, 16, len(columns)).mean(1).tolist(),
            })
    first = torch.tensor(layers[0]["direction"])
    second = torch.tensor(layers[1]["direction"])
    return {
        "edit_commit": UPSTREAM_COMMIT,
        "method": "FP32 SVD of the BF16 edit delta restricted to changed columns.",
        "direction_absolute_cosine": float(torch.dot(first, second).abs()),
        "layers": layers,
        "caution": "Static low-rank geometry does not establish a looping mechanism.",
    }


def save_anatomy(report, directory):
    with (directory / "anatomy.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    plots = [line_chart(
        directory / "spectrum.svg", "How many directions describe the edit?",
        {f"layer {row['layer']}": row["cumulative_energy"][:8] for row in report["layers"]},
        x_label="Number of singular components", y_label="Cumulative fraction of squared delta norm", x_start=1,
    )]
    for row in report["layers"]:
        layer = row["layer"]
        plots.append(line_chart(
            directory / f"columns_{layer}.svg", f"Layer {layer}: where weights changed",
            {"column delta norm": row["column_norms"]},
            x_label="Original MLP column index", y_label="L2 norm of each changed output column",
        ))
        plots.append(heatmap(
            directory / f"delta_{layer}.svg", f"Layer {layer}: binned weight changes",
            row["binned_heatmap"],
        ))
    return plots


def capture_teacher_projections(model, tokenizer, anatomy, texts=None):
    """Compare identical fixed token inputs, not diverging generated histories."""
    texts = TEACHER_TEXTS if texts is None else texts
    if not isinstance(texts, dict) or not 1 <= len(texts) <= 5:
        raise ValueError("Supply one to five named fixed texts.")
    for name, text in texts.items():
        if (
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name)
            or not isinstance(text, str) or not 1 <= len(text) <= 3000
        ):
            raise ValueError("Invalid teacher-forced text name or length.")
    directions = {
        row["layer"]: torch.tensor(row["direction"], device=model.device, dtype=torch.float32)
        for row in anatomy["layers"]
    }
    captured, handles, records = {}, [], []

    def hook_for(layer):
        def hook(module, arguments, output):
            values = output.detach().float()[0]
            if values.ndim != 2 or values.shape[-1] != 1024 or not torch.isfinite(values).all():
                raise ValueError("Unexpected MLP activation shape or nonfinite values.")
            projection = values @ directions[layer]
            captured[str(layer)] = {
                "projection": projection.cpu().tolist(),
                "absolute_cosine": (projection.abs() / values.norm(dim=-1).clamp_min(1e-8)).cpu().tolist(),
            }
        return hook

    try:
        for row in anatomy["layers"]:
            module = model.get_submodule(row["name"].removesuffix(".weight"))
            handles.append(module.register_forward_hook(hook_for(row["layer"])))
        for condition in ("base", "published_edit"):
            context = edited_weights(model, strength=0 if condition == "base" else 1)
            with context:
                for name, text in texts.items():
                    captured.clear()
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        model(**inputs, use_cache=False, logits_to_keep=1)
                    if len(captured) != 2:
                        raise RuntimeError("Both expected MLP hooks must run.")
                    records.append({
                        "condition": condition, "text_id": name,
                        "token_count": inputs["input_ids"].shape[1], "layers": dict(captured),
                    })
    finally:
        for handle in handles:
            handle.remove()
    return records


def save_projections(records, directory):
    with (directory / "teacher_projections.json").open("x", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, allow_nan=False)
    paths = []
    for name in sorted({row["text_id"] for row in records}):
        for layer in ("15", "19"):
            series = {
                row["condition"]: row["layers"][layer]["absolute_cosine"]
                for row in records if row["text_id"] == name
            }
            paths.append(line_chart(
                directory / f"projection_{name}_{layer}.svg",
                f"Layer {layer}: fixed input / {name}", series,
                x_label="Token position (identical input in both conditions)",
                y_label="Absolute cosine: MLP output vs leading edit direction",
            ))
    return paths
