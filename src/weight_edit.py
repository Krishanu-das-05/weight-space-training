"""Download data only, verify the published edit, and apply it reversibly."""

import hashlib
import io
import json
import math
import ssl
import zipfile
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import torch
import truststore

PROJECT_ROOT = Path(__file__).resolve().parent
EDIT_DIR = PROJECT_ROOT / ".cache" / "published-edit"
UPSTREAM_COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
SOURCE_URL = (
    "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/"
    f"{UPSTREAM_COMMIT}/exp8_math/runs/p50c/edit/"
)
ARTIFACTS = {
    "manifest.json": (
        1326,
        "83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225",
    ),
    "edit.npz": (
        387619,
        "e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7",
    ),
}
TENSOR_NAMES = tuple(
    f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    for layer in (15, 19)
)


def _verify_artifact(name, data):
    size, digest = ARTIFACTS[name]
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Published artifact failed size/SHA-256 verification: {name}")
    return data


def _read_artifact(directory, name):
    path = Path(directory) / name
    if not path.is_file():
        raise FileNotFoundError("Edit data missing. Run: .venv\\Scripts\\python.exe weight_edit.py")
    with path.open("rb") as handle:
        return _verify_artifact(name, handle.read(ARTIFACTS[name][0] + 1))


def download_edit():
    """Explicit network step; existing data is checked, never overwritten."""
    EDIT_DIR.mkdir(parents=True, exist_ok=True)
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    for name, (size, _) in ARTIFACTS.items():
        target = EDIT_DIR / name
        if target.exists():
            _read_artifact(EDIT_DIR, name)
        else:
            request = Request(SOURCE_URL + name, headers={"User-Agent": "weight-space-local/1.0"})
            with urlopen(request, context=context, timeout=60) as response:
                data = _verify_artifact(name, response.read(size + 1))
            with target.open("xb") as handle:
                handle.write(data)
        print(f"{name}: size and SHA-256 verified")


def load_edit(directory=EDIT_DIR):
    """Read the pinned JSON/NPZ without pickle or downloaded Python code."""
    manifest = json.loads(_read_artifact(directory, "manifest.json"))
    specs = manifest["tensors"]
    if (
        [spec["name"] for spec in specs] != list(TENSOR_NAMES)
        or manifest["changed_tensors"] != 2
        or manifest["changed_entries_total"] != 130188
    ):
        raise ValueError("Unexpected published edit manifest.")

    data = _read_artifact(directory, "edit.npz")
    expected_keys = {f"t{i}_{suffix}" for i in range(2) for suffix in ("idx", "val")}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if (
            {entry.filename for entry in members} != {f"{key}.npy" for key in expected_keys}
            or len(members) != 4
            or sum(entry.file_size for entry in members) > 1_000_000
        ):
            raise ValueError("Unexpected NPZ members or decompressed size.")
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        if set(arrays.files) != expected_keys:
            raise ValueError("Unexpected sparse edit arrays.")
        return [
            {**spec, "indices": arrays[f"t{i}_idx"].copy(), "values": arrays[f"t{i}_val"].copy()}
            for i, spec in enumerate(specs)
        ]


def tensor_sha256(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _patched_tensor(base, edit):
    if (
        base.dtype != torch.bfloat16
        or tuple(base.shape) != (1024, 3584)
        or edit["shape"] != [1024, 3584]
        or edit["dtype"] != "bfloat16"
        or edit["form"] != "sparse"
    ):
        raise ValueError("This edit requires the original BF16 (1024, 3584) MLP matrices.")
    if tensor_sha256(base) != edit["base_sha256"]:
        raise ValueError(f"Base tensor hash mismatch: {edit['name']}")

    indices, values = edit["indices"], edit["values"]
    if (
        indices.dtype != np.dtype("int32")
        or values.dtype != np.dtype("uint16")
        or indices.ndim != 1
        or values.shape != indices.shape
        or len(indices) != edit["changed_entries"]
        or len(indices) == 0
        or int(indices.min()) < 0
        or int(indices.max()) >= base.numel()
        or np.unique(indices).size != len(indices)
    ):
        raise ValueError("Invalid sparse edit indices or BF16 values.")

    bits = base.detach().cpu().contiguous().view(torch.uint16).numpy().copy()
    bits.reshape(-1)[indices] = values
    replacement = torch.from_numpy(bits).view(torch.bfloat16)
    if tensor_sha256(replacement) != edit["edited_sha256"]:
        raise ValueError(f"Edited tensor hash mismatch: {edit['name']}")
    return replacement


@contextmanager
def edited_weights(model, layers=(15, 19), directory=EDIT_DIR, *, strength=1.0):
    """Apply selected published tensors in memory, restoring even on errors."""
    if (
        not isinstance(layers, (tuple, list))
        or not 1 <= len(layers) <= 2
        or any(type(layer) is not int or layer not in (15, 19) for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("layers must select 15, 19, or both without duplicates.")
    if (
        type(strength) not in (int, float) or not math.isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise ValueError("strength must be finite and between 0 and 1.")
    chosen = {f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in layers}
    prepared = []
    for edit in load_edit(directory):
        if edit["name"] in chosen:
            parameter = model.get_parameter(edit["name"])
            original = parameter.detach().cpu().clone()
            replacement = _patched_tensor(original, edit)
            if strength == 0:
                replacement = original.clone()
            elif strength != 1:
                indices = torch.from_numpy(edit["indices"].astype(np.int64))
                mixed = original.clone()
                mixed.view(-1)[indices] = torch.lerp(
                    original.view(-1)[indices].float(),
                    replacement.view(-1)[indices].float(),
                    float(strength),
                ).to(torch.bfloat16)
                replacement = mixed
            prepared.append((parameter, original, replacement, edit, tensor_sha256(replacement)))

    # Validate every replacement before changing any live model parameter.
    try:
        with torch.no_grad():
            for parameter, _, replacement, _, _ in prepared:
                parameter.copy_(replacement)
        for parameter, _, _, _, applied_hash in prepared:
            if tensor_sha256(parameter) != applied_hash:
                raise RuntimeError("Applying an edited tensor failed hash verification.")
        yield [
            {
                **{key: edit[key] for key in ("name", "changed_entries", "changed_columns", "edited_sha256")},
                **_change_counts(original, replacement),
                "published_changed_entries": edit["changed_entries"],
                "applied_sha256": applied_hash,
                "strength": float(strength),
            }
            for _, original, replacement, edit, applied_hash in prepared
        ]
    finally:
        with torch.no_grad():
            for parameter, original, _, _, _ in prepared:
                parameter.copy_(original)
        for parameter, _, _, edit, _ in prepared:
            if tensor_sha256(parameter) != edit["base_sha256"]:
                raise RuntimeError("Restoring an original model tensor failed hash verification.")


def _change_counts(original, replacement):
    changed = (
        original.contiguous().view(torch.uint16).numpy()
        != replacement.contiguous().view(torch.uint16).numpy()
    )
    return {
        "changed_entries": int(np.count_nonzero(changed)),
        "changed_columns": int(np.count_nonzero(changed.any(axis=0))),
    }


if __name__ == "__main__":
    download_edit()
