"""Passive raw capture precedes response validation; old records stay immutable."""

import hashlib
import math
import re
from budget_sweep_io import atomic_bytes, atomic_json, identity, json_bytes, read_json, sha256
from budget_sweep_scoring import _tokens
from p50c_amended_scoring import amended_grade, closure_from_tokens
from p50c_completion_io import EditedRecordStore, require_edited_provenance


class RawCapture:
    def __init__(self, directory):
        self.directory = directory

    def tokens(self, metadata, token_ids):
        atomic_json(self.directory / "tokens.json", {**metadata, "token_ids": token_ids})

    def text(self, text):
        # This is persistence, not acceptance: the character bound is checked afterwards.
        atomic_bytes(self.directory / "text.txt", text.encode("utf-8"))
        atomic_json(self.directory / "capture.json", {
            "raw_tokens_sha256": sha256(self.directory / "tokens.json"),
            "raw_text_sha256": sha256(self.directory / "text.txt"),
            "decoded_character_length": len(text), "decodable": True,
        })

    def decode_error(self, exception):
        atomic_json(self.directory / "decode_error.json", {"type": type(exception).__name__,
                                                         "raw_tokens_retained": True})


class AmendedRecordStore(EditedRecordStore):
    def __init__(self, directory, plan, pool, declaration, amendment_hash, decode):
        super().__init__(directory, plan, pool)
        self.declaration, self.amendment_hash, self.decode = declaration, amendment_hash, decode
        self.raw_directory = self.directory / "raw_responses"
        self.raw_directory.mkdir(exist_ok=True)

    def capture(self, attempt):
        if not isinstance(attempt, str) or not re.fullmatch(r"[0-9a-f]{32}", attempt):
            raise ValueError("Invalid reserved attempt identity.")
        path = self.raw_directory / attempt
        if path.is_symlink():
            raise ValueError("Symlinked raw captures are forbidden.")
        path.mkdir(exist_ok=True)
        return RawCapture(path)

    def _header(self, raw, attempt):
        require_edited_provenance(raw, self.plan)
        key = identity(raw)
        job = self.jobs.get(key)
        if (
            job is None or any(raw.get(name) != value for name, value in job.items())
            or key in self.declaration["legacy_record_sha256"]
            or raw.get("problem_id") not in self.declaration["remaining_ids"]
            or raw.get("attempt_id") != attempt or raw.get("plan_sha256") != self.plan_hash
            or attempt == self.declaration.get("retained_unreconciled_attempt_id")
            or raw.get("amendment_sha256") != self.amendment_hash
            or raw.get("raw_captured_before_response_validation") is not True
            or type(raw.get("seed")) is not int or type(raw.get("max_new_tokens")) is not int
            or raw.get("thinking") is not True or raw.get("expected") != self.pool[job["problem_id"]]["answer"]
        ):
            raise ValueError("Raw capture identity/provenance differs from its approved job.")
        seconds = raw.get("seconds")
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid raw generation duration.")
        for name in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
            if type(raw.get(name)) is not int or raw[name] < 0:
                raise ValueError("Invalid raw GPU memory metric.")
        return key

    def build_record(self, attempt):
        if not isinstance(attempt, str) or not re.fullmatch(r"[0-9a-f]{32}", attempt):
            raise ValueError("Invalid raw attempt.")
        directory = self.raw_directory / attempt
        paths = [directory / name for name in ("tokens.json", "text.txt", "capture.json")]
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise ValueError("Raw-first capture is missing or symlinked.")
        raw, capture = read_json(paths[0]), read_json(paths[2])
        self._header(raw, attempt)
        tokens = _tokens(raw.get("token_ids"))
        text = self.decode(tokens)
        if not isinstance(text, str):
            raise ValueError("Pinned decoder did not produce text.")
        encoded = text.encode("utf-8")
        if (
            capture.get("decodable") is not True
            or type(capture.get("decoded_character_length")) is not int
            or capture["decoded_character_length"] != len(text)
            or paths[1].stat().st_size != len(encoded)
            or capture.get("raw_tokens_sha256") != sha256(paths[0])
            or capture.get("raw_text_sha256") != sha256(paths[1])
            or hashlib.sha256(encoded).hexdigest() != capture["raw_text_sha256"]
        ):
            raise ValueError("Raw bytes disagree with their captured hashes or pinned token decoder.")
        graded, violation, characters = amended_grade(
            text, tokens, raw["expected"], self.plan["stopping"]["terminal_tokens_for_grading"],
        )
        if graded["natural_eos_index"] is None and len(tokens) != raw["max_new_tokens"]:
            raise ValueError("Unfinished raw generation has an invalid stopping point.")
        return {
            **raw, "text": None if violation else text, "graded": graded,
            "text_bound_violation": violation, "text_character_length": characters,
            "raw_response": {
                path.name: {"file": path.relative_to(self.directory).as_posix(), "sha256": sha256(path)}
                for path in paths
            },
            "token_248046_count": tokens.count(248046), "contains_token_248046": 248046 in tokens,
            "thinking_closed_at_cap": closure_from_tokens(tokens, self.decode),
        }

    def check_record(self, row):
        require_edited_provenance(row, self.plan)
        key = identity(row)
        legacy = self.declaration["legacy_record_sha256"]
        if key in legacy:
            if hashlib.sha256(json_bytes(row)).hexdigest() != legacy[key]:
                raise ValueError("A saved pre-amendment response changed.")
            return super().check_record(row)
        if row != self.build_record(row.get("attempt_id")):
            raise ValueError("Amended response differs from raw-first reconstruction.")
        return key

    def recover_raw(self):
        rows = self.recover_journal()
        completed = {identity(row): row for row in rows}
        directories = list(self.raw_directory.iterdir())
        if len(directories) > 500:
            raise ValueError("Unexpected raw capture count.")
        for directory in sorted(directories):
            if directory.is_symlink() or not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                raise ValueError("Invalid raw capture directory.")
            tokens_path = directory / "tokens.json"
            if not tokens_path.exists():
                continue
            raw = read_json(tokens_path)
            key = self._header(raw, directory.name)
            if key in completed:
                if completed[key]["attempt_id"] != directory.name:
                    raise ValueError("Multiple completed raw attempts for one job.")
                continue
            if (directory / "decode_error.json").exists():
                raise ValueError("A raw decoding error needs investigation; IDs are preserved.")
            if not (directory / "capture.json").exists():
                tokens = _tokens(raw.get("token_ids"))
                text = self.decode(tokens)
                capture = RawCapture(directory)
                text_path = directory / "text.txt"
                if not text_path.exists():
                    capture.text(text)
                else:
                    encoded = text.encode("utf-8")
                    if text_path.stat().st_size != len(encoded) or sha256(text_path) != hashlib.sha256(encoded).hexdigest():
                        raise ValueError("Interrupted raw text differs from token decoding.")
                    atomic_json(directory / "capture.json", {
                        "raw_tokens_sha256": sha256(tokens_path), "raw_text_sha256": sha256(text_path),
                        "decoded_character_length": len(text), "decodable": True,
                    })
            row = self.build_record(directory.name)
            self.commit(row)
            completed[key] = row
        return self.records()
