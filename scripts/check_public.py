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
PYTHON_SOURCES = [
    "scripts/auto_lrc.py",
    "scripts/ctc_align.py",
    "scripts/evaluate_lrc.py",
    "scripts/export_alignment_audit.py",
    "scripts/run_benchmarks.py",
    "scripts/test_core_logic.py",
    "scripts/whisperx_refine.py",
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


def main() -> int:
    run([sys.executable, "scripts/test_core_logic.py"])
    run([sys.executable, "-m", "py_compile", *PYTHON_SOURCES])
    assert_no_forbidden_tracked_files()
    print("Public-safe checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
