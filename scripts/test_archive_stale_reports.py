import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import archive_stale_reports as asr

ALL_SUFFIXES = (".review.tsv", ".auto.lrc", ".review-edits.json", ".review-decisions.jsonl", ".romaji-overrides.json",
                ".orphaned-20261101T010203Z.auto.lrc", ".orphaned-20261101T010203Z.review-edits.json",
                ".discarded-20261025T144411Z.lrc", ".discarded-20261025T144411Z.auto.lrc")


class ArchiveStaleReportsTest(unittest.TestCase):
    def _report(self, folder: Path, name: str, audio: Path | None, sidecars=(), output: Path | None = None):
        report = folder / f"{name}.align-report.json"
        report.write_text(json.dumps({"audio_path": str(audio) if audio else "", "output_path": str(output or (folder / "gone.lrc"))}), encoding="utf-8")
        for suffix in sidecars:
            (folder / f"{name}{suffix}").write_text("x", encoding="utf-8")
        return report

    def _audio(self, folder: Path, name="song.flac"):
        path = folder / name
        path.write_bytes(b"a")
        return path

    def test_only_a_missing_audio_makes_a_report_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = self._audio(root)
            live = self._report(root, "live--1", audio, (".review.tsv",), output=root / "song.lrc")
            exported = self._report(root, "exported--5", audio, (".review.tsv",))  # LRC archived to checked/, audio still there
            stale = self._report(root, "old--2", root / "gone.flac", (".review.tsv", ".auto.lrc", ".review-edits.json"))
            noaudio = self._report(root, "odd--3", None)
            (root / "broken--4.align-report.json").write_text("{", encoding="utf-8")
            planned, skipped = asr.plan(root)
            self.assertEqual(planned, [stale])
            self.assertEqual({path.name for path, _ in skipped}, {"odd--3.align-report.json", "broken--4.align-report.json"})
            archive_dir = root / "_archive" / "x"
            moved = asr.archive(planned, archive_dir)
            self.assertEqual(len(moved), 4)
            self.assertFalse(stale.exists())
            self.assertTrue((archive_dir / "old--2.review.tsv").is_file())
            for kept in (live, exported, noaudio, root / "exported--5.review.tsv", root / "live--1.review.tsv"):
                self.assertTrue(kept.is_file(), kept.name)

    def test_every_file_of_the_stem_moves_and_nothing_else(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = self._report(root, "old--aaaaaaaaaa", root / "Song Name.flac", ALL_SUFFIXES)
            (root / "Song Name.harmony_cands.json").write_text("{}", encoding="utf-8")
            self._report(root, "old--aaaaaaaaaab", root / "other.flac", (".review.tsv",))  # same text prefix, different stem
            self._report(root, "keep--bbbbbbbbbb", self._audio(root), (".review.tsv", ".review-decisions.jsonl"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(asr.main(["--report-dir", str(root), "--apply", "--archive-dir", str(root / "arc")]), 0)
            arc = {path.name for path in (root / "arc").iterdir()}
            expected = {"old--aaaaaaaaaa.align-report.json", "Song Name.harmony_cands.json", "old--aaaaaaaaaab.align-report.json", "old--aaaaaaaaaab.review.tsv"}
            expected |= {"old--aaaaaaaaaa" + suffix for suffix in ALL_SUFFIXES}
            self.assertEqual(arc, expected)
            self.assertFalse(stale.exists())
            self.assertTrue((root / "keep--bbbbbbbbbb.review-decisions.jsonl").is_file())
            self.assertIn("by kind:", out.getvalue())
            self.assertIn(".review-decisions.jsonl x1", out.getvalue())
            self.assertIn(".orphaned-<stamp>.auto.lrc x1", out.getvalue())

    def test_the_harmony_sidecar_of_an_audio_name_still_in_use_stays(self) -> None:
        """The sidecar is named after the audio file: a song moved to a new folder and aligned again has a stale report and a live one
        with the same audio name, and the live one needs the sidecar."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "music_new").mkdir()
            live_audio = self._audio(root / "music_new", "01 Song.flac")
            stale = self._report(root, "01 Song--aaaaaaaaaa", root / "music_old" / "01 Song.flac", (".review.tsv",))
            live = self._report(root, "01 Song--bbbbbbbbbb", live_audio)
            sidecar = root / "01 Song.harmony_cands.json"
            sidecar.write_text("{}", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(asr.main(["--report-dir", str(root), "--apply", "--archive-dir", str(root / "arc")]), 0)
            self.assertEqual({path.name for path in (root / "arc").iterdir()}, {stale.name, "01 Song--aaaaaaaaaa.review.tsv"})
            self.assertTrue(sidecar.is_file())
            self.assertTrue(live.is_file())

    def test_two_stale_reports_with_one_audio_name_move_the_sidecar_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._report(root, "01 Song--aaaaaaaaaa", root / "old1" / "01 Song.flac")
            second = self._report(root, "01 Song--bbbbbbbbbb", root / "old2" / "01 Song.flac")
            (root / "01 Song.harmony_cands.json").write_text("{}", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(asr.main(["--report-dir", str(root), "--apply", "--archive-dir", str(root / "arc")]), 0)
            self.assertEqual({path.name for path in (root / "arc").iterdir()}, {first.name, second.name, "01 Song.harmony_cands.json"})
            self.assertFalse((root / "01 Song.harmony_cands.json").exists())

    def test_an_existing_target_gets_a_new_name_and_the_run_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arc = root / "arc"
            arc.mkdir()
            (arc / "old--2.align-report.json").write_text("previous", encoding="utf-8")
            (arc / "old--2.review.tsv").write_text("previous", encoding="utf-8")
            self._report(root, "old--2", root / "gone.flac", (".review.tsv", ".auto.lrc"))
            self._report(root, "older--3", root / "gone2.flac", (".review.tsv",))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(asr.main(["--report-dir", str(root), "--apply", "--archive-dir", str(arc)]), 0)
            self.assertEqual((arc / "old--2.align-report.json").read_text(encoding="utf-8"), "previous")
            names = {path.name for path in arc.iterdir()}
            self.assertEqual(names, {"old--2.align-report.json", "old--2.review.tsv", "old--2.align-report.dup2.json", "old--2.review.dup2.tsv",
                                     "old--2.auto.dup2.lrc", "older--3.align-report.json", "older--3.review.tsv"})
            self.assertIn("1 report(s) renamed", out.getvalue())
            self.assertEqual(sorted(path.name for path in root.glob("*.json")), [])

    def test_a_failed_move_is_reported_and_the_rest_still_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._report(root, "a--1", root / "gone1.flac", (".review.tsv",))
            self._report(root, "b--2", root / "gone2.flac", (".review.tsv",))
            real = asr.shutil.move

            def flaky(src, dst):
                if Path(src).name == "a--1.review.tsv":
                    raise PermissionError("locked")
                return real(src, dst)

            out = io.StringIO()
            with mock.patch.object(asr.shutil, "move", side_effect=flaky), contextlib.redirect_stdout(out):
                self.assertEqual(asr.main(["--report-dir", str(root), "--apply", "--archive-dir", str(root / "arc")]), 1)
            self.assertIn("FAILED a--1.review.tsv", out.getvalue())
            self.assertTrue((root / "a--1.review.tsv").is_file())
            self.assertTrue((root / "arc" / "b--2.review.tsv").is_file())
            self.assertTrue((root / "arc" / "a--1.align-report.json").is_file())

    def test_dry_run_moves_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = self._report(root, "old--2", root / "gone.flac")
            self.assertEqual(asr.main(["--report-dir", str(root)]), 0)
            self.assertTrue(stale.is_file())
            self.assertEqual(asr.main(["--report-dir", str(root), "--apply", "--archive-dir", str(root / "arc")]), 0)
            self.assertFalse(stale.is_file())
            self.assertTrue((root / "arc" / "old--2.align-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
