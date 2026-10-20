"""Tests for scripts/report_diff.py (LRC 1.5 W4)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_diff  # noqa: E402


class ReportDiffTests(unittest.TestCase):
    def test_added_only_is_clean(self) -> None:
        diff = report_diff.diff_reports({"a": 1}, {"a": 1, "b": 2})
        self.assertEqual(diff, {"added": ["b"], "removed": [], "changed": []})
        self.assertEqual(report_diff.unexpected_changes(diff, []), [])

    def test_changed_and_removed_are_unexpected(self) -> None:
        diff = report_diff.diff_reports({"a": 1, "c": [1, 2]}, {"a": 2, "c": [1, 3]})
        self.assertEqual(diff["changed"], ["a", "c.[1]"])
        self.assertEqual(report_diff.unexpected_changes(diff, []), ["a", "c.[1]"])
        self.assertEqual(report_diff.unexpected_changes(diff, ["a", "c"]), [])
        removed = report_diff.diff_reports({"a": 1}, {})
        self.assertEqual(report_diff.unexpected_changes(removed, ["a"]), ["a"])

    def test_path_keys_are_not_compared(self) -> None:
        before = {
            "audio_path": "D:/x.flac",
            "review_tsv_path": "D:/a.tsv",
            "multimodel_consensus": {"cache_path": "D:/c", "n": 1},
            "generation": {"paths": {"k": "D:/old"}, "cache_root": "D:/r"},
            "assignments": [{"temp_dir": "D:/t", "time": 1.0}],
        }
        after = json.loads(json.dumps(before))
        after["audio_path"] = "F:/x.flac"
        after["review_tsv_path"] = "F:/a.tsv"
        after["multimodel_consensus"]["cache_path"] = "F:/c"
        after["generation"]["paths"] = {"k": "F:/new", "extra": 1}
        after["generation"]["cache_root"] = "F:/r"
        after["assignments"][0]["temp_dir"] = "F:/t"
        diff = report_diff.diff_reports(before, after)
        self.assertEqual(diff, {"added": [], "removed": [], "changed": []})
        after["multimodel_consensus"]["n"] = 2
        self.assertEqual(report_diff.diff_reports(before, after)["changed"], ["multimodel_consensus.n"])

    def test_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text('{"x": 1, "y": {"z": 1}}', encoding="utf-8")
            (root / "b.json").write_text('{"x": 1, "y": {"z": 2}, "w": 0}', encoding="utf-8")
            self.assertEqual(report_diff.main([str(root / "a.json"), str(root / "b.json")]), 1)
            self.assertEqual(report_diff.main([str(root / "a.json"), str(root / "b.json"), "--allow-changed", "y.z"]), 0)
            self.assertEqual(report_diff.main([str(root / "a.json"), str(root / "a.json")]), 0)
            self.assertEqual(report_diff.main([str(root / "missing.json"), str(root / "a.json")]), 2)


if __name__ == "__main__":
    unittest.main()
