"""Offline conversion using the already-installed system PyArrow, not the lab venv."""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "gsm8k-main-train"


def main():
    receipt = json.loads((CACHE / "receipt.json").read_text(encoding="utf-8"))
    source = CACHE / "train.parquet"
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if not isinstance(receipt, dict) or digest != receipt.get("sha256"):
        raise ValueError("The cached TRAIN artifact failed verification.")
    pa.PyExtensionType.set_auto_load(False)
    parquet = pq.ParquetFile(source)
    if not 30 <= parquet.metadata.num_rows <= 10000:
        raise ValueError("Unexpected TRAIN size.")
    schema = parquet.schema_arrow
    for name in ("question", "answer"):
        field = schema.field(name)
        if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            raise ValueError("Expected plain UTF-8 TRAIN columns.")
    rows = parquet.read(columns=["question", "answer"], use_threads=False).to_pylist()
    target = CACHE / "train.jsonl"
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if any(not isinstance(row[name], str) for name in ("question", "answer")):
                raise ValueError("Null/non-string TRAIN content.")
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    tool = {
        "tool": "preinstalled system Python / PyArrow; offline conversion only",
        "python": sys.version.split()[0], "pyarrow": pa.__version__,
        "rows": len(rows), "source_sha256": digest,
        "decoded_jsonl_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "package_installations": 0,
    }
    with (CACHE / "decoder.json").open("x", encoding="utf-8") as handle:
        json.dump(tool, handle, indent=2)
    print(json.dumps(tool, indent=2))


if __name__ == "__main__":
    main()
