"""Append-safe run records and conservative crash-aware token accounting."""

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

from budget_sweep_scoring import grade_generation

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def project_path(relative):
    if not isinstance(relative, str) or not 1 <= len(relative) <= 300:
        raise ValueError("Invalid project-relative path.")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path escapes the project.")
    return path


def read_json(path, limit=2_000_000):
    with Path(path).open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("JSON artifact exceeds its size limit.")
    def invalid_constant(value):
        raise ValueError("Non-finite JSON number.")
    value = json.loads(data, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def atomic_bytes(path, data, *, replace=False):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def atomic_json(path, value, *, replace=False):
    atomic_bytes(path, json_bytes(value), replace=replace)


def identity(job):
    return job["problem_id"] + "__" + job["condition"] + "__" + job["kind"]


class RunLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.handle = self.path.open("a+b")
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *exception):
        self.handle.close()


class RecordStore:
    def __init__(self, directory, plan, pool):
        self.directory = Path(directory).resolve()
        self.plan, self.pool = plan, {item["id"]: item for item in pool}
        self.plan_hash = sha256(self.directory / "plan.json")
        self.jobs = {identity(job): job for job in plan["schedule"]}
        if len(self.jobs) != len(plan["schedule"]):
            raise ValueError("Duplicate scheduled identity.")
        for name in ("durable_records", "budget_ledger", "recovery"):
            (self.directory / name).mkdir(exist_ok=True)
        self.journal = self.directory / "records.jsonl"

    def check_record(self, row):
        if not isinstance(row, dict):
            raise ValueError("Invalid record.")
        for name in ("problem_id", "condition", "kind"):
            if not isinstance(row.get(name), str):
                raise ValueError("Invalid record identity.")
        key = identity(row)
        job = self.jobs.get(key)
        if job is None or any(row.get(name) != value for name, value in job.items()):
            raise ValueError("Record differs from its scheduled job.")
        if type(row.get("seed")) is not int or type(row.get("max_new_tokens")) is not int:
            raise ValueError("Record seed/cap must be integers.")
        if row.get("plan_sha256") != self.plan_hash or not re.fullmatch(r"[0-9a-f]{32}", row.get("attempt_id", "")):
            raise ValueError("Record plan/attempt mismatch.")
        expected = self.pool[job["problem_id"]]["answer"]
        thinking = job["condition"] == "thinking_on"
        if row.get("expected") != expected or row.get("thinking") is not thinking:
            raise ValueError("Record grading metadata mismatch.")
        graded = grade_generation(
            row.get("text"), row.get("token_ids"), expected, thinking=thinking,
            eos_ids={248044}, terminal_tokens=self.plan["stopping"]["terminal_tokens_for_grading"],
        )
        length = graded["generated_tokens"]
        if length > job["max_new_tokens"] or (graded["natural_eos_index"] is None and length != job["max_new_tokens"]):
            raise ValueError("Invalid generation stopping point.")
        if row.get("graded") != graded or row.get("token_248046_count") != row["token_ids"].count(248046):
            raise ValueError("Derived record fields do not match raw tokens/text.")
        seconds = row.get("seconds")
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid generation duration.")
        if type(row.get("peak_gpu_allocated_bytes")) is not int or row["peak_gpu_allocated_bytes"] < 0:
            raise ValueError("Invalid peak GPU memory.")
        return key

    def records(self):
        found = {}
        files = list((self.directory / "durable_records").glob("*.json"))
        if len(files) > len(self.jobs):
            raise ValueError("Too many completed records.")
        for path in files:
            if path.is_symlink():
                raise ValueError("Symlinked records are forbidden.")
            row = read_json(path)
            key = self.check_record(row)
            if path.stem != key or key in found:
                raise ValueError("Duplicate/misnamed completed record.")
            found[key] = row
        return [found[key] for key in self.jobs if key in found]

    def budget(self, records=None):
        records = self.records() if records is None else records
        completed = {row["attempt_id"]: row for row in records}
        if len(completed) != len(records):
            raise ValueError("Duplicate completed attempt ID.")
        known, outstanding, seen = 0, 0, set()
        files = list((self.directory / "budget_ledger").glob("*.json"))
        if len(files) > 500:
            raise ValueError("Unexpected reservation count.")
        for path in files:
            if path.is_symlink():
                raise ValueError("Symlinked reservations are forbidden.")
            reservation = read_json(path, 10000)
            attempt = reservation.get("attempt_id")
            key = reservation.get("identity")
            job = self.jobs.get(key) if isinstance(key, str) else None
            if (
                not isinstance(attempt, str) or not re.fullmatch(r"[0-9a-f]{32}", attempt)
                or attempt in seen or path.stem != attempt or job is None
                or reservation.get("plan_sha256") != self.plan_hash
                or reservation.get("reserved_tokens") != job["max_new_tokens"]
            ):
                raise ValueError("Invalid token reservation.")
            seen.add(attempt)
            row = completed.get(attempt)
            if row is None:
                outstanding += reservation["reserved_tokens"]
            elif identity(row) != key:
                raise ValueError("Reservation identity mismatch.")
            else:
                known += len(row["token_ids"])
        if set(completed) - seen:
            raise ValueError("Completed record has no durable token reservation.")
        accounted = known + outstanding
        if accounted > self.plan["token_budget"]["limit"]:
            raise ValueError("Token ledger exceeds the approved budget.")
        return {"completed_actual_tokens": known, "unreconciled_reserved_tokens": outstanding,
                "accounted_tokens": accounted, "limit": self.plan["token_budget"]["limit"]}

    def reserve(self, job):
        if self.jobs.get(identity(job)) != job:
            raise ValueError("Unknown job.")
        if any(identity(row) == identity(job) for row in self.records()):
            raise ValueError("A completed generation must not be repeated.")
        if self.budget()["accounted_tokens"] + job["max_new_tokens"] > self.plan["token_budget"]["limit"]:
            return None
        attempt = uuid.uuid4().hex
        atomic_json(self.directory / "budget_ledger" / (attempt + ".json"), {
            "attempt_id": attempt, "identity": identity(job),
            "reserved_tokens": job["max_new_tokens"], "plan_sha256": self.plan_hash,
        })
        return attempt

    def commit(self, row):
        key = self.check_record(row)
        reservation = read_json(self.directory / "budget_ledger" / (row["attempt_id"] + ".json"))
        if (
            reservation["identity"] != key or reservation["reserved_tokens"] != row["max_new_tokens"]
            or reservation.get("plan_sha256") != self.plan_hash
        ):
            raise ValueError("Unfunded generation record.")
        atomic_json(self.directory / "durable_records" / (key + ".json"), row)
        with self.journal.open("ab") as handle:
            handle.write(json_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())

    def recover_journal(self):
        records = self.records()
        self.budget(records)
        if not self.journal.exists():
            atomic_bytes(self.journal, b"".join(json_bytes(row) for row in records))
            return records
        if self.journal.stat().st_size > 50_000_000:
            raise ValueError("Journal exceeds its bounded size.")
        raw = self.journal.read_bytes()
        parsed, damaged_tail = [], False
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ValueError("Unexpected journal corruption; recovery stopped.") from None
                damaged_tail = True
                break
            if len(parsed) >= len(records) or row != records[len(parsed)]:
                raise ValueError("Journal disagrees with committed records.")
            parsed.append(row)
        if damaged_tail:
            backup = self.directory / "recovery" / (uuid.uuid4().hex + ".jsonl")
            atomic_bytes(backup, raw)
            atomic_json(backup.with_suffix(".json"), {"original_sha256": hashlib.sha256(raw).hexdigest(),
                                                    "valid_lines": len(parsed), "committed_records": len(records)})
            atomic_bytes(self.journal, b"".join(json_bytes(row) for row in records), replace=True)
        elif len(parsed) < len(records):
            with self.journal.open("ab") as handle:
                if raw and not raw.endswith(b"\n"):
                    handle.write(b"\n")
                for row in records[len(parsed):]:
                    handle.write(json_bytes(row))
                handle.flush()
                os.fsync(handle.fileno())
        elif raw and not raw.endswith(b"\n"):
            with self.journal.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        return records
