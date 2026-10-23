#!/usr/bin/env python3
"""Run public-safe checks that do not require private audio or model files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".aac",
    ".aiff",
    ".alac",
    ".flac",
    ".ggml",
    ".gguf",
    ".lrc",
    ".m4a",
    ".mp3",
    ".ogg",
    ".onnx",
    ".opus",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
}
# Test files that pass on a clean checkout with only requirements.txt installed.
# The full suite (test_core_logic, test_align_lrc_wrapper, ...) needs the Windows
# runtime (PowerShell wrapper, whisper.cpp, model caches) and runs on the release PC.
PUBLIC_SAFE_TESTS = [
    "scripts/test_installation.py",
    "scripts/test_runtime_paths.py",
    "scripts/test_file_digest.py",
    "scripts/test_model_cache.py",
    "scripts/test_r2_pipeline.py",
    "scripts/test_review_presentation.py",
    "scripts/test_lrc14_review_server.py",
    "scripts/test_multimodel_extra_models.py",
    "scripts/test_compare_report_runs.py",
    "scripts/test_archive_stale_reports.py",
    "scripts/test_weak_review_policy.py",
    "scripts/reviewers/test_common.py",
]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def assert_no_forbidden_tracked_files() -> None:
    violations: list[str] = []
    for name in tracked_files():
        path = Path(name)
        if name.endswith(".align-report.json") or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(name)
    if violations:
        joined = "\n".join(f"  - {item}" for item in violations)
        raise SystemExit(f"Forbidden generated/media/model files are tracked:\n{joined}")


def python_sources() -> list[str]:
    return [name for name in tracked_files() if name.endswith(".py")]


def main() -> int:
    assert_no_forbidden_tracked_files()
    run([sys.executable, "-m", "py_compile", *python_sources()])
    for test in PUBLIC_SAFE_TESTS:
        run([sys.executable, test])
    print("Public-safe checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
