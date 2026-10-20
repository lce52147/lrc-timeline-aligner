"""Reviewer status classification, --strict-reviewers exit code, fallback marker (LRC 1.5 W5)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import r2_pipeline  # noqa: E402


class ReviewerStatusTests(unittest.TestCase):
    def test_seven_status_kinds(self) -> None:
        unsupported = {"songs": [{"HUB": {"status": "UNSUPPORTED_TEXT"}}]}
        self.assertEqual(r2_pipeline.reviewer_step_status("WX", 0), "OK")
        self.assertEqual(r2_pipeline.reviewer_step_status("WX", 1), "FAILED")
        self.assertEqual(r2_pipeline.reviewer_step_status("HUB", 2, unsupported), "MISSING_UNSUPPORTED")
        self.assertEqual(r2_pipeline.reviewer_step_status("WX", r2_pipeline.EXIT_TIMEOUT), "TIMEOUT")
        self.assertEqual(r2_pipeline.reviewer_step_status("WX", r2_pipeline.EXIT_LAUNCH_ERROR), "LAUNCH_ERROR")
        self.assertEqual(r2_pipeline.reviewer_step_status("HUB", 2), "FAILED")
        rows = r2_pipeline.reviewer_status_rows(
            {"HUBP": {"status": "SKIPPED_DEPENDENCY"}, "WX": {"status": "error:OSError", "exit_code": None, "seconds": 0.5}},
            Path("log.txt"),
        )
        self.assertEqual([row["status"] for row in rows], ["SKIPPED_DEPENDENCY", "error:OSError"])
        self.assertEqual(rows[1], {"name": "WX", "exit_code": None, "status": "error:OSError", "log_path": "log.txt", "elapsed_s": 0.5})

    def test_strict_exit_code(self) -> None:
        ok_rows = [{"status": "OK"}, {"status": "MISSING_UNSUPPORTED"}, {"status": "SKIPPED_DEPENDENCY"}]
        self.assertEqual(r2_pipeline.strict_reviewers_exit_code(ok_rows), 0)
        for bad in ("FAILED", "TIMEOUT", "LAUNCH_ERROR", "error:OSError"):
            self.assertEqual(r2_pipeline.strict_reviewers_exit_code(ok_rows + [{"status": bad}]), 4)

    def test_status_source(self) -> None:
        self.assertEqual(r2_pipeline.reviewer_status_source({"WX": {"status": "OK", "source": "cache"}}), "cache")
        self.assertEqual(r2_pipeline.reviewer_status_source({"WX": {"status": "OK"}}), "r2")

    def test_attach_and_fallback_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.align-report.json"
            path.write_text(json.dumps({"audio_path": "x", "assignments": []}), encoding="utf-8")
            rows = [{"name": "WX", "exit_code": 0, "status": "OK", "log_path": "l", "elapsed_s": 1.0}]
            report = r2_pipeline.attach_reviewer_status(path, rows, "r2")
            self.assertEqual(report["reviewer_status"], {"source": "r2", "steps": rows})
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["reviewer_status"]["source"], "r2")
            self.assertEqual(on_disk["audio_path"], "x")
            r2_pipeline._mark_r2_skipped_report(path, "no-candidate", rows)
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(on_disk["r2"]["skipped"])
            self.assertEqual(on_disk["reviewer_status"]["source"], "r2-fallback")
            self.assertEqual(on_disk["reviewer_status"]["steps"], rows)

    def test_batch_summary_reviewer_counts(self) -> None:
        import lrc_batch_summary as batch
        report = {"reviewer_status": {"source": "r2", "steps": [
            {"name": "WX", "status": "OK"}, {"name": "XLSR", "status": "TIMEOUT"}, {"name": "HUB", "status": "MISSING_UNSUPPORTED"},
        ]}}
        self.assertEqual(batch._reviewer_counts(report), (2, 3, "r2"))
        self.assertEqual(batch._reviewer_counts({}), (None, None, None))
        summary = {
            "n_songs": 1, "counts": {"ok": 1, "stale": 0, "lrc_missing": 0, "no_report": 0, "unreadable": 0},
            "total_known_seconds": 1.0, "total_review_rows": 0, "total_untrusted_rows": 0,
            "total_reviewers_ok": 2, "total_reviewers": 3, "attention": [],
            "songs": [{"name": "a.flac", "status": "ok", "seconds": 1.0, "reviewers_ok": 2, "reviewers_total": 3, "reviewers_source": "r2"}],
        }
        lines = batch.format_console(summary)
        self.assertIn("reviewers OK=2/3", lines[0])
        self.assertIn("  REVIEWERS a.flac: 2/3 OK (r2)", lines)

    def test_timeout_env(self) -> None:
        self.assertEqual(r2_pipeline.subprocess_timeout_seconds({}), 7200.0)
        self.assertEqual(r2_pipeline.subprocess_timeout_seconds({"LRC_SUBPROCESS_TIMEOUT_SECONDS": "90"}), 90.0)
        self.assertIsNone(r2_pipeline.subprocess_timeout_seconds({"LRC_SUBPROCESS_TIMEOUT_SECONDS": "0"}))
        self.assertEqual(r2_pipeline.subprocess_timeout_seconds({"LRC_SUBPROCESS_TIMEOUT_SECONDS": "x"}), 7200.0)


if __name__ == "__main__":
    unittest.main()
