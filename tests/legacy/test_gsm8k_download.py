"""Download lifecycle tests with synthetic data and a fully mocked network."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gsm8k_source as source


class DownloadLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gsm8k_download_test_", dir=source.ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name) / "missing_parent" / "cache"
        for name, value in (("CACHE", self.cache), ("urlopen", None)):
            patcher = patch.object(source, name, value) if value is not None else patch.object(source, name)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if name == "urlopen":
                self.network = mocked
        context = patch.object(source.truststore, "SSLContext")
        self.ssl_context = context.start()
        self.addCleanup(context.stop)

    def write_fixture(self, name, content):
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / name).write_bytes(content)

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.cache.iterdir() if path.is_file()}

    def test_fresh_download_creates_missing_parents_with_mocked_io_only(self):
        payload = b"FAKE-TRAIN-DATA-FOR-TESTING"
        digest = hashlib.sha256(payload).hexdigest()
        metadata = {
            "id": source.REPO, "sha": "a" * 40, "cardData": {"license": ["mit"]},
            "siblings": [{"rfilename": source.TRAIN_FILE, "lfs": {"size": len(payload), "sha256": digest}}],
        }
        self.network.side_effect = [
            io.BytesIO(json.dumps(metadata).encode("utf-8")), io.BytesIO(payload),
        ]
        self.assertFalse(self.cache.parent.exists())
        receipt = source.fetch_train_once()
        self.assertTrue(self.cache.is_dir())
        self.assertEqual(receipt["sha256"], digest)
        self.assertEqual(self.network.call_count, 2)
        self.assertTrue((self.cache / "receipt.json").is_file())

    def test_marker_or_partial_payload_without_receipt_stops_without_io_or_mutation(self):
        for name in ("download_attempt.json", "train.parquet", "api_dataset_info.json"):
            with self.subTest(name=name):
                directory = self.cache / name.replace(".", "_")
                directory.mkdir(parents=True)
                (directory / name).write_bytes(b"FAKE-INCOMPLETE-ARTIFACT")
                with patch.object(source, "CACHE", directory):
                    before = {path.name: path.read_bytes() for path in directory.iterdir()}
                    with self.assertRaisesRegex(ValueError, r"manually delete.*" + name.replace(".", r"\.")):
                        source.fetch_train_once()
                    self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})
        self.network.assert_not_called()
        self.ssl_context.assert_not_called()

    def test_incomplete_cache_names_every_relevant_file_for_manual_cleanup(self):
        names = (
            "download_attempt.json", "train.parquet", "train_payload_attempt.json",
            "pinned_source.json", "api_dataset_info.json", "train.jsonl", "decoder.json", "selection.json",
        )
        for name in names:
            self.write_fixture(name, b"FAKE-INCOMPLETE-ARTIFACT")
        before = self.snapshot()
        with self.assertRaises(ValueError) as error:
            source.fetch_train_once()
        for name in names:
            self.assertIn(name, str(error.exception))
        self.assertIn("receipt.json", str(error.exception))
        self.assertIn("No files were deleted", str(error.exception))
        self.assertEqual(self.snapshot(), before)
        self.network.assert_not_called()
        self.ssl_context.assert_not_called()

    def test_valid_receipt_reuses_verified_cache_without_network(self):
        payload = b"FAKE-COMPLETED-TRAIN-CACHE"
        receipt = {"sha256": hashlib.sha256(payload).hexdigest()}
        self.write_fixture("train.parquet", payload)
        self.write_fixture("receipt.json", json.dumps(receipt).encode("utf-8"))
        self.write_fixture("download_attempt.json", b"FAKE-COMPLETED-MARKER")
        before = self.snapshot()
        self.assertEqual(source.fetch_train_once(), receipt)
        self.assertEqual(self.snapshot(), before)
        self.network.assert_not_called()
        self.ssl_context.assert_not_called()

    def test_corrupt_completed_cache_is_not_redownloaded(self):
        self.write_fixture("train.parquet", b"FAKE-CORRUPT-CACHE")
        self.write_fixture("receipt.json", json.dumps({"sha256": "0" * 64}).encode("utf-8"))
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "no redownload"):
            source.fetch_train_once()
        self.assertEqual(self.snapshot(), before)
        self.network.assert_not_called()

    def test_failed_fresh_attempt_cannot_be_retried_automatically(self):
        self.network.side_effect = RuntimeError("FAKE-NETWORK-FAILURE")
        with self.assertRaisesRegex(RuntimeError, "FAKE-NETWORK-FAILURE"):
            source.fetch_train_once()
        before = self.snapshot()
        self.network.reset_mock()
        with self.assertRaisesRegex(ValueError, "download_attempt.json"):
            source.fetch_train_once()
        self.assertEqual(self.snapshot(), before)
        self.network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
