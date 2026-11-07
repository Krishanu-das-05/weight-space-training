"""Download and verify the exact base checkpoint used by the published edit."""

import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3.5-0.8B"
WEIGHT_FILE = "model.safetensors-00001-of-00001.safetensors"
WEIGHT_SIZE = 1_746_942_600
WEIGHT_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"


def main():
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=MODEL_DIR,
        cache_dir=PROJECT_ROOT / ".cache" / "huggingface" / "hub",
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.jinja",
            "*.txt",
            "README.md",
            "LICENSE*",
        ],
        token=False,
        max_workers=4,
    )

    # Verify the entire file against the pinned revision's published LFS hash.
    weights = MODEL_DIR / WEIGHT_FILE
    if weights.stat().st_size != WEIGHT_SIZE:
        raise RuntimeError("The model weight file has an unexpected size.")
    with weights.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != WEIGHT_SHA256:
        raise RuntimeError("The model weight file failed SHA-256 verification.")
    print(f"Verified {MODEL_ID} at revision {REVISION}.")
    print(f"Local model: {MODEL_DIR}")


if __name__ == "__main__":
    main()
