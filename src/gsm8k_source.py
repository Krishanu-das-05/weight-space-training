"""One explicit GSM8K TRAIN download and deterministic, integer-only selection."""

import argparse
import hashlib
import json
import random
import re
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

import truststore

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "gsm8k-main-train"
REPO = "openai/gsm8k"
TRAIN_FILE = "main/train-00000-of-00001.parquet"
SAMPLING_SEED = 20261106


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def fetch_train_once():
    """Metadata plus one TRAIN payload; never retry or fetch TEST/other assets."""
    CACHE.mkdir(parents=True, exist_ok=True)
    receipt_path = CACHE / "receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or file_sha256(CACHE / "train.parquet") != receipt.get("sha256"):
            raise ValueError("Cached TRAIN data failed verification; no redownload is permitted.")
        return receipt
    incomplete_files = [
        name for name in (
            "download_attempt.json", "train.parquet", "train_payload_attempt.json",
            "pinned_source.json", "api_dataset_info.json", "train.jsonl",
            "decoder.json", "selection.json",
        )
        if (CACHE / name).exists() or (CACHE / name).is_symlink()
    ]
    if incomplete_files:
        raise ValueError(
            "Incomplete GSM8K cache: no valid receipt.json is available. "
            "No files were deleted and no download was retried. "
            "Before a separately authorized fresh attempt, manually delete these "
            "files from the GSM8K cache: " + ", ".join(incomplete_files) + "."
        )
    write_json(CACHE / "download_attempt.json", {
        "repo": REPO, "config": "main", "split": "train", "automatic_retry": False,
    })
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    headers = {"User-Agent": "weight-space-budget-plan/1.0"}
    metadata_url = f"https://huggingface.co/api/datasets/{REPO}?blobs=true"
    metadata_path = CACHE / "api_dataset_info.json"
    with urlopen(Request(metadata_url, headers=headers), context=context, timeout=60) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Dataset metadata exceeds the allowed size.")
    with metadata_path.open("xb") as handle:
        handle.write(raw)
    info = json.loads(raw)
    if not isinstance(info, dict) or info.get("id") != REPO:
        raise ValueError("Unexpected dataset identity.")
    revision = info.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full dataset commit hash is required.")
    card = info.get("cardData")
    license_value = card.get("license") if isinstance(card, dict) else None
    license_ok = license_value == "mit" or (
        isinstance(license_value, list) and all(isinstance(value, str) for value in license_value)
        and set(license_value) == {"mit"}
    )
    if not license_ok:
        raise ValueError("The approved MIT dataset license was not confirmed.")
    siblings = info.get("siblings")
    if not isinstance(siblings, list) or any(not isinstance(item, dict) for item in siblings):
        raise ValueError("Unexpected dataset file listing.")
    candidates = [item for item in siblings if item.get("rfilename") == TRAIN_FILE]
    if len(candidates) != 1:
        raise ValueError("Expected exactly the approved main TRAIN shard.")
    lfs = candidates[0].get("lfs")
    if not isinstance(lfs, dict):
        raise ValueError("Expected a dataset LFS size and SHA-256.")
    expected_hash, expected_size = lfs.get("sha256"), lfs.get("size")
    if (
        not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        or type(expected_size) is not int or not 1 <= expected_size <= 25_000_000
    ):
        raise ValueError("Invalid TRAIN artifact metadata.")
    source_url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{TRAIN_FILE}"
    write_json(CACHE / "pinned_source.json", {
        "repo": REPO, "config": "main", "split": "train", "revision": revision,
        "path": TRAIN_FILE, "license": "MIT", "url": source_url,
        "expected_sha256": expected_hash, "expected_size": expected_size,
    })
    write_json(CACHE / "train_payload_attempt.json", {"source_url": source_url, "automatic_retry": False})
    digest, size = hashlib.sha256(), 0
    with urlopen(Request(source_url, headers=headers), context=context, timeout=120) as response:
        with (CACHE / "train.parquet").open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > expected_size:
                    raise ValueError("TRAIN payload exceeds its pinned size.")
                digest.update(chunk)
                handle.write(chunk)
    if size != expected_size or digest.hexdigest() != expected_hash:
        raise ValueError("TRAIN payload failed pinned size/SHA-256 verification.")
    receipt = {
        "repo": REPO, "config": "main", "split": "train", "revision": revision,
        "license": "MIT", "remote_path": TRAIN_FILE, "source_url": source_url,
        "sha256": digest.hexdigest(), "bytes": size,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_payload_downloads": 1,
        "other_dataset_splits_downloaded": [],
    }
    write_json(receipt_path, receipt)
    return receipt


def integer_final_answer(annotation):
    if not isinstance(annotation, str) or "####" not in annotation:
        return None
    answer = annotation.rsplit("####", 1)[1].strip().replace(",", "")
    if not re.fullmatch(r"[+-]?[0-9]{1,30}", answer):
        return None
    return str(int(answer))


def select_development(rows, *, seed=SAMPLING_SEED, count=30):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("The sampling seed must be a nonnegative 32-bit integer.")
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("Choose 1-100 items.")
    if not isinstance(rows, list) or not count <= len(rows) <= 10000:
        raise ValueError("Invalid TRAIN row count.")
    for row in rows:
        if (
            not isinstance(row, dict) or set(row) != {"question", "answer"}
            or not isinstance(row["question"], str) or not 1 <= len(row["question"]) <= 20000
            or not isinstance(row["answer"], str) or not 1 <= len(row["answer"]) <= 50000
        ):
            raise ValueError("Invalid TRAIN row schema.")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected, selection, skipped = [], [], []
    for rank, index in enumerate(order):
        answer = integer_final_answer(rows[index]["answer"])
        identifier = f"gsm8k_train_{index:05d}"
        if answer is None:
            skipped.append({"id": identifier, "row_index": index, "rank": rank, "reason": "non_integer_final"})
            continue
        selected.append({"id": identifier, "question": rows[index]["question"], "answer": answer})
        selection.append({"id": identifier, "row_index": index, "rank": rank})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Not enough integer-answer TRAIN items.")
    if len({item["question"] for item in selected}) != count:
        raise ValueError("Duplicate selected questions; stop rather than resample silently.")
    return selected, {
        "sampling_seed": seed, "algorithm": "Python random.Random(seed).shuffle(range(train_rows))",
        "train_rows": len(rows), "selection": selection, "skipped": skipped,
        "answer_parser": "After the last ####: strip outer whitespace, remove commas, require signed ASCII integer.",
    }


def overlap_questions(development, heldout_questions):
    if not isinstance(development, list) or not isinstance(heldout_questions, list):
        raise ValueError("Expected question lists.")
    if any(not isinstance(text, str) for text in heldout_questions):
        raise ValueError("Invalid held-out question text.")
    exact, normalized = set(heldout_questions), {" ".join(text.split()) for text in heldout_questions}
    collisions = []
    for item in development:
        if not isinstance(item, dict) or not isinstance(item.get("question"), str) or not isinstance(item.get("id"), str):
            raise ValueError("Invalid development question.")
        if item["question"] in exact or " ".join(item["question"].split()) in normalized:
            collisions.append(item["id"])
    return collisions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-train-once", action="store_true")
    arguments = parser.parse_args()
    if not arguments.fetch_train_once:
        parser.error("The network step requires --fetch-train-once.")
    print(json.dumps(fetch_train_once(), indent=2))
