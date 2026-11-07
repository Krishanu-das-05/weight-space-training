"""Offline publication gates; reports locations/rules, never matching secrets."""

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.workbench import layout_sources

ROOT_FILES = {
    ".gitignore", ".gitattributes", "README.md", "THIRD_PARTY_NOTICES.md",
    "requirements.in", "requirements.lock", "repository-layout.json",
}
PUBLIC_FOLDERS = {"src", "tests", "scripts", "tools", "notebooks", "docs"}
SECRET_RULES = {
    "github_credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "api_credential": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "credential_in_url": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    "jupyter_login_url": re.compile(r"https?://[^\s\"']+[?&]token=[A-Za-z0-9_-]{16,}"),
}
ASSIGNMENT = re.compile(
    r"""(?i)\b(?:password|api[_-]?key|access[_-]?token|client[_-]?secret)\b\s*[:=]\s*["']([^"'\n]{16,})["']"""
)
# Split home prefixes so the regex source is not mistaken for an actual home path.
HOME_PATH = re.compile(
    r"""(?i)(?:[a-z]:[/\\]Users[/\\][^/\\\s"']+|/"""
    r"""home/[^/\s"']+|/"""
    r"""Users/[^/\s"']+)"""
)
RAW_FIELDS = {"token_ids", "rendered_prompt", "user_message", "question", "text", "records"}


def secret_findings(text):
    result = []
    for name, pattern in SECRET_RULES.items():
        for match in pattern.finditer(text):
            result.append({"rule": name, "line": text.count("\n", 0, match.start()) + 1})
    for match in ASSIGNMENT.finditer(text):
        if not match.group(1).startswith(("FAKE-", "fake-", "${", "<")):
            result.append({"rule": "literal_credential_assignment", "line": text.count("\n", 0, match.start()) + 1})
    for match in HOME_PATH.finditer(text):
        result.append({"rule": "machine_specific_home_path", "line": text.count("\n", 0, match.start()) + 1})
    return result


def public_path(relative):
    if not isinstance(relative, str) or not 1 <= len(relative) <= 400:
        return False
    path = Path(relative)
    return (
        not path.is_absolute() and ".." not in path.parts
        and (relative in ROOT_FILES or path.parts[0] in PUBLIC_FOLDERS)
        and not any(part in (".env", ".local", ".cache", ".venv", "__pycache__", ".ipynb_checkpoints")
                    for part in path.parts)
        and path.suffix.lower() not in (".pem", ".key", ".pfx", ".p12", ".safetensors", ".parquet", ".pyc")
    )


def raw_fields(value):
    if isinstance(value, dict):
        return any(key in RAW_FIELDS or raw_fields(item) for key, item in value.items())
    if isinstance(value, list):
        return any(raw_fields(item) for item in value)
    return False


def notebook_issues(value):
    if not isinstance(value, dict) or value.get("nbformat") != 4 or not isinstance(value.get("cells"), list):
        return ["invalid_notebook_structure"]
    result = []
    if set(value.get("metadata", {})) - {"kernelspec", "language_info"}:
        result.append("unreviewed_notebook_metadata")
    for cell in value["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") not in ("code", "markdown", "raw"):
            result.append("invalid_notebook_cell")
            continue
        if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
            result.append("saved_notebook_execution")
        if cell.get("attachments"):
            result.append("embedded_notebook_attachments")
    return result


def svg_is_safe(text):
    return not (
        "<script" in text.lower() or "javascript:" in text.lower()
        or re.search(r"""(?:href|src)=["'](?!#)""", text)
    )


def publication_files(staged):
    command = ["git", "ls-files", "--cached"]
    if not staged:
        command.extend(["--others", "--exclude-standard"])
    command.append("-z")
    response = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return sorted(set(item.decode("utf-8") for item in response.stdout.split(b"\0") if item))


def check(staged=False):
    files = publication_files(staged)
    if not files:
        raise ValueError("No publication files were selected.")
    frozen = layout_sources()
    findings = []
    notebooks = 0
    for relative in files:
        path = ROOT / relative
        if not public_path(relative) or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            findings.append({"file": relative, "rule": "outside_publication_scope"})
            continue
        if not path.is_file() or path.stat().st_size > 2_000_000:
            findings.append({"file": relative, "rule": "missing_or_oversized_publication_file"})
            continue
        data = path.read_bytes()
        if path.suffix.lower() == ".png":
            if path.parent != ROOT / "docs" / "figures" or not data.startswith(b"\x89PNG\r\n\x1a\n"):
                findings.append({"file": relative, "rule": "unapproved_binary"})
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeError:
            findings.append({"file": relative, "rule": "unexpected_binary"})
            continue
        findings.extend({"file": relative, **item} for item in secret_findings(text))
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=relative)
            except SyntaxError:
                findings.append({"file": relative, "rule": "python_syntax"})
        if path.suffix == ".ipynb":
            notebooks += 1
            value = json.loads(text)
            findings.extend({"file": relative, "rule": rule} for rule in notebook_issues(value))
        if path.suffix == ".svg" and not svg_is_safe(text):
            findings.append({"file": relative, "rule": "executable_or_external_svg"})
        if path.parent == ROOT / "docs" / "results" and raw_fields(json.loads(text)):
            findings.append({"file": relative, "rule": "raw_data_in_curated_export"})
    report = {
        "publication_files": len(files), "byte_frozen_sources_and_tests": len(frozen),
        "clean_notebooks": notebooks, "staged_only": staged, "findings": findings,
        "inference_performed": False, "network_access": False,
        "scope": "Publication hygiene and obvious-secret scan; not a comprehensive vulnerability audit.",
    }
    print(json.dumps(report, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    arguments = parser.parse_args()
    try:
        raise SystemExit(check(arguments.staged))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(type(error).__name__ + ": repository checks could not complete.", file=sys.stderr)
        raise SystemExit(1) from None
