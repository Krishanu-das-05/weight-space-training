"""Reconstruct the frozen flat lab locally without changing research sources."""

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKBENCH = ROOT / ".local" / "workbench"
ARCHIVE = ROOT / ".local" / "original-layout"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_object(path, limit=2_000_000):
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError("Expected a project-local regular JSON file.")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("JSON metadata exceeds its size limit.")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def checked_source(relative):
    if not isinstance(relative, str) or not 1 <= len(relative) <= 300:
        raise ValueError("Invalid relative source path.")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or (ROOT / relative).is_symlink():
        raise ValueError("Source is missing, symlinked or outside the project.")
    return path


def layout_sources():
    layout = read_object(ROOT / "repository-layout.json")
    if layout.get("schema_version") != 1 or not isinstance(layout.get("files"), dict):
        raise ValueError("Invalid source-layout manifest.")
    result = {}
    for name, entry in layout["files"].items():
        if (
            not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*\.py", name)
            or not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise ValueError("Invalid frozen source identity.")
        path = checked_source(entry.get("file"))
        expected_prefix = "tests/legacy/" if name.startswith("test_") else "src/"
        if entry["file"] != expected_prefix + name or sha256(path) != entry["sha256"]:
            raise ValueError("A byte-frozen source differs from its declared layout.")
        result[name] = path
    return result


def _copy_exact(source, target, *, hardlink=False):
    if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
        raise ValueError("Do not copy symlinked or external inputs.")
    if target.is_symlink() or not target.resolve().is_relative_to(WORKBENCH.resolve()):
        raise ValueError("Refusing a symlinked workbench target.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or source.stat().st_size != target.stat().st_size or sha256(source) != sha256(target):
            raise FileExistsError("Existing workbench input differs; nothing was overwritten: " + target.name)
        return
    if hardlink:
        os.link(source, target)
    else:
        shutil.copy2(source, target)
    if sha256(source) != sha256(target):
        raise RuntimeError("Copy failed its byte-integrity check.")


def _mirror(source, destination):
    if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
        raise ValueError("Only project-local data directories may be mirrored.")
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinked input trees are not supported.")
        if path.is_file():
            _copy_exact(path, destination / path.relative_to(source))
            count += 1
    return count


def prepare(*, with_local_data=False):
    sources = layout_sources()
    if WORKBENCH.is_symlink() or not WORKBENCH.resolve().is_relative_to(ROOT / ".local"):
        raise ValueError("Invalid workbench location.")
    WORKBENCH.mkdir(parents=True, exist_ok=True)
    (WORKBENCH / ".cache").mkdir(exist_ok=True)
    for name, source in sources.items():
        _copy_exact(source, WORKBENCH / name)
    historical_files = 0
    if (ARCHIVE / "snapshot.json").is_file():
        snapshot = read_object(ARCHIVE / "snapshot.json")
        if not isinstance(snapshot.get("files"), dict):
            raise ValueError("Invalid original-layout file map.")
        for name, metadata in snapshot["files"].items():
            if Path(name).name != name or not isinstance(metadata, dict):
                raise ValueError("Invalid original-layout snapshot.")
            original = checked_source(".local/original-layout/" + name)
            if sha256(original) != metadata.get("sha256"):
                raise ValueError("A preserved original file changed.")
            if name not in sources:
                _copy_exact(original, WORKBENCH / name)
                historical_files += 1
    mirrored = 0
    if with_local_data:
        for relative in (".cache/published-edit", ".cache/gsm8k-main-train", "results"):
            source = ROOT / relative
            if source.is_dir():
                mirrored += _mirror(source, WORKBENCH / relative)
        model = ROOT / "models" / "Qwen3.5-0.8B"
        if model.is_dir():
            for path in sorted(model.iterdir()):
                if path.is_file():
                    _copy_exact(path, WORKBENCH / "models" / model.name / path.name,
                                hardlink=path.suffix == ".safetensors")
                    mirrored += 1
    environment = ROOT / ".venv"
    link = WORKBENCH / ".venv"
    if environment.is_dir() and not link.exists():
        if not environment.resolve().is_relative_to(ROOT):
            raise ValueError("The existing environment must be project-local.")
        if os.name == "nt":
            subprocess.run([
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "New-Item -ItemType Junction -Path $env:WS_ENV_LINK -Target $env:WS_ENV_TARGET -ErrorAction Stop | Out-Null",
            ], check=True, capture_output=True, text=True, env={
                **os.environ, "WS_ENV_LINK": str(link), "WS_ENV_TARGET": str(environment),
            })
        else:
            link.symlink_to(environment, target_is_directory=True)
    if link.exists() and link.resolve() != environment.resolve():
        raise ValueError("The workbench environment link points somewhere unexpected.")
    return {
        "workspace": WORKBENCH.relative_to(ROOT).as_posix(), "source_and_test_files": len(sources),
        "historical_files": historical_files, "local_data_files": mirrored,
        "model_downloaded": False, "inference_performed": False,
        "large_checkpoint_storage": "Local hardlink only; do not edit either checkpoint path.",
    }
