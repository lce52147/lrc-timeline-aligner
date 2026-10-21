import json
import tempfile
import unittest
from pathlib import Path

import archive_stale_reports as asr


class ArchiveStaleReportsTest(unittest.TestCase):
    def _report(self, folder: Path, name: str, output: Path | None, sidecars=()):
        report = folder / f"{name}.align-report.json"
        report.write_text(json.dumps({"output_path": str(output) if output else ""}), encoding="utf-8")
        for suffix in sidecars:
            (folder / f"{name}{suffix}").write_text("x", encoding="utf-8")
        return report

    def test_plan_and_apply_move_only_reports_without_an_lrc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live_lrc = root / "song.lrc"
            live_lrc.write_text("[00:01.00]a", encoding="utf-8")
            live = self._report(root, "live--1", live_lrc, (".review.tsv",))
            stale = self._report(root, "old--2", root / "gone.lrc", (".review.tsv", ".auto.lrc", ".review-edits.json"))
            nopath = self._report(root, "odd--3", None)
            (root / "broken--4.align-report.json").write_text("{", encoding="utf-8")
            planned, skipped = asr.plan(root)
            self.assertEqual(planned, [stale])
            self.assertEqual({path.name for path, _ in skipped}, {"odd--3.align-report.json", "broken--4.align-report.json"})
            archive_dir = root / "_archive" / "x"
            moved = asr.archive(planned, archive_dir)
            self.assertEqual(len(moved), 4)
            self.assertFalse(stale.exists())
            self.assertTrue((archive_dir / "old--2.review.tsv").is_file())
            self.assertTrue(live.is_file())
            self.assertTrue((root / "live--1.review.tsv").is_file())
            self.assertTrue(nopath.is_file())

    def test_dry_run_moves_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = self._report(root, "old--2", root / "gone.lrc")
            self.assertEqual(asr.main(["--report-dir", str(root)]), 0)
            self.assertTrue(stale.is_file())
            self.assertEqual(asr.main(["--report-dir", str(root), "--apply", "--archive-dir", str(root / "arc")]), 0)
            self.assertFalse(stale.is_file())
            self.assertTrue((root / "arc" / "old--2.align-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
