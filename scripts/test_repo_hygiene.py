"""Repository hygiene checks (LRC 1.5 W1).

Ensures that tracked Python sources under ``scripts/`` are not shadowed by an
ignore rule, and that a newly created ``scripts/*.py`` file would be visible
to ``git status`` instead of being silently ignored.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _git_available() -> bool:
    try:
        result = _git("rev-parse", "--is-inside-work-tree")
    except (OSError, FileNotFoundError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def _check_ignore_no_index(paths: list[str]) -> list[str]:
    """Return the subset of ``paths`` that an ignore rule matches."""
    if not paths:
        return []
    result = _git("check-ignore", "--no-index", "--", *paths)
    # exit 0: some matched, 1: none matched, 128: error
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        raise AssertionError(f"git check-ignore failed: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


@unittest.skipUnless(_git_available(), "git work tree not available")
class RepoHygieneTests(unittest.TestCase):
    def test_tracked_scripts_are_not_ignored(self) -> None:
        listed = _git("ls-files", "--", "scripts")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        tracked = [
            line.strip()
            for line in listed.stdout.splitlines()
            if line.strip().endswith(".py")
        ]
        self.assertTrue(tracked, "expected tracked scripts/*.py files")
        ignored = _check_ignore_no_index(tracked)
        self.assertEqual(ignored, [], f"tracked scripts matched by .gitignore: {ignored}")

    def test_new_script_is_not_ignored(self) -> None:
        probe = SCRIPTS_DIR / "_hygiene_probe.py"
        self.assertFalse(probe.exists(), f"stale probe file present: {probe}")
        probe.write_text("# hygiene probe\n", encoding="utf-8")
        try:
            ignored = _check_ignore_no_index(["scripts/_hygiene_probe.py"])
            self.assertEqual(ignored, [], "a new scripts/*.py file must not be ignored")
            status = _git("status", "--porcelain", "--", "scripts/_hygiene_probe.py")
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertIn("?? scripts/_hygiene_probe.py", status.stdout)
        finally:
            try:
                os.remove(probe)
            except FileNotFoundError:
                pass

    def test_review_scratch_dir_stays_ignored(self) -> None:
        ignored = _check_ignore_no_index(["_review/anything.md"])
        self.assertEqual(ignored, ["_review/anything.md"])


if __name__ == "__main__":
    unittest.main()
