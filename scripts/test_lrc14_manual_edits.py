from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc
import lrc_manual_edits
import lrc_publication
import r2_pipeline


class ManualEditTests(unittest.TestCase):
    def test_publish_callback_runs_after_lrc_and_before_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lrc = root / "song.lrc"
            report = root / "song.align-report.json"
            events: list[object] = []

            def write_lrc(path: Path) -> None:
                path.write_bytes(b"\xef\xbb\xbf[00:01.00]line\r\n")
                events.append(("lrc", path.read_bytes()))

            def on_lrc(digest: str) -> None:
                events.append(("hash", digest))

            def write_report(path: Path) -> None:
                events.append("report")
                path.write_text(json.dumps({"ok": True}), encoding="utf-8")

            lrc_publication.publish_pair(
                lrc, report, write_lrc, write_report, on_lrc_staged=on_lrc
            )
            self.assertEqual(events[0][0], "lrc")
            self.assertEqual(events[1][0], "hash")
            self.assertEqual(events[1][1], hashlib.sha256(lrc.read_bytes()).hexdigest())
            self.assertEqual(events[2], "report")

    def _run_lyrics(self, root: Path, *, overwrite: bool = False, discard: bool = False) -> tuple[Path, Path, object]:
        audio = root / "song.flac"
        lyrics = root / "song.lyrics.txt"
        output = root / "song.lrc"
        reports = root / "reports"
        audio.write_bytes(b"fake flac")
        lyrics.write_text("[00:01.00]line\n", encoding="utf-8")
        argv = [
            str(audio), "--lyrics", str(lyrics), "--output", str(output),
            "--report-dir", str(reports), "--timing-source", "lyrics",
        ]
        if overwrite:
            argv.append("--overwrite")
        if discard:
            argv.append("--discard-manual-edits")
        args = auto_lrc.build_parser().parse_args(argv)
        with (
            mock.patch.object(auto_lrc, "probe_duration", return_value=2.0),
            mock.patch.object(auto_lrc, "audio_track_labels", return_value=("", "Song")),
        ):
            result = auto_lrc.process_audio(audio, args)
        report, _audit, _template = auto_lrc.report_artifact_paths(audio, output, reports)
        return output, report, result

    def test_process_publishes_hash_and_refuses_before_probe(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, report, _ = self._run_lyrics(root)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["published_lrc_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            original = output.read_bytes()
            output.write_bytes(original.replace(b"line", b"EDIT", 1))
            args = auto_lrc.build_parser().parse_args([
                str(root / "song.flac"), "--lyrics", str(root / "song.lyrics.txt"),
                "--output", str(output), "--report-dir", str(root / "reports"),
                "--timing-source", "lyrics", "--overwrite",
            ])
            with mock.patch.object(auto_lrc, "probe_duration", side_effect=AssertionError("probe called")):
                with self.assertRaisesRegex(auto_lrc.LrcError, "Manual review edits detected"):
                    auto_lrc.process_audio(root / "song.flac", args)

    def test_auto_artifact_alone_refuses_and_discard_archives_then_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, report, _ = self._run_lyrics(root)
            original = output.read_bytes()
            auto_path, edits_path = lrc_manual_edits.review_artifact_paths(report)
            auto_path.write_bytes(original)
            output.write_bytes(original.replace(b"line", b"EDIT", 1))
            args = auto_lrc.build_parser().parse_args([
                str(root / "song.flac"), "--lyrics", str(root / "song.lyrics.txt"),
                "--output", str(output), "--report-dir", str(root / "reports"),
                "--timing-source", "lyrics", "--overwrite",
            ])
            with self.assertRaisesRegex(auto_lrc.LrcError, "Manual review edits detected"):
                auto_lrc.process_audio(root / "song.flac", args)
            edits_path.write_text(json.dumps({"schema": 1, "auto_lrc_sha256": hashlib.sha256(original).hexdigest(), "edits": [{"entry": 1}]}), encoding="utf-8")
            edited = output.read_bytes()
            self._run_lyrics(root, overwrite=True, discard=True)
            stem = report.name[:-len(".align-report.json")]
            archives = [path for path in report.parent.glob(stem + ".discarded-*.lrc") if not path.name.endswith(".auto.lrc")]
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), edited)
            self.assertTrue(list(report.parent.glob(stem + ".discarded-*.auto.lrc")))
            self.assertTrue(list(report.parent.glob(stem + ".discarded-*.review-edits.json")))
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["published_lrc_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_empty_skeleton_and_legacy_report_without_hash_are_not_manual_edits(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "song.lrc"
            report = root / "song.align-report.json"
            output.write_bytes(b"[00:01.00]line\n")
            report.write_text(json.dumps({"assignments": [{"lrc_tag": "00:01.00"}]}), encoding="utf-8")
            self.assertIsNone(lrc_manual_edits.manual_edit_reason(output, report))
            auto_path, edits_path = lrc_manual_edits.review_artifact_paths(report)
            auto_path.write_bytes(output.read_bytes())
            edits_path.write_text(json.dumps({"auto_lrc_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "edits": []}), encoding="utf-8")
            self.assertIsNone(lrc_manual_edits.manual_edit_reason(output, report))

    def test_r2_strips_discard_for_baseline_and_guards_before_backend(self) -> None:
        raw = ["song.flac", "--overwrite", "--discard-manual-edits"]
        baseline = r2_pipeline.prepare_baseline_args(raw, output=Path("b.lrc"), report_dir=Path("br"))
        self.assertNotIn("--discard-manual-edits", baseline)
        self.assertIn("--discard-manual-edits", r2_pipeline.prepare_final_args(raw, Path("evidence.json")))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "song.flac"
            output = root / "song.lrc"
            reports = root / "reports"
            audio.write_bytes(b"audio")
            output.write_bytes(b"old")
            report, _a, _t = auto_lrc.report_artifact_paths(audio, output, reports)
            report.write_text(json.dumps({"published_lrc_sha256": "0" * 64}), encoding="utf-8")
            error = io.StringIO()
            with mock.patch.object(r2_pipeline, "_run_backend", side_effect=AssertionError("backend called")), redirect_stderr(error):
                result = r2_pipeline.main(["--work-dir", str(root / "work"), "--", str(audio), "--output", str(output), "--report-dir", str(reports), "--overwrite", "--timing-source", "lyrics"])
            self.assertEqual(result, 2)
            self.assertIn("Manual review edits detected", error.getvalue())

    def test_r2_journal_allows_recovery_path_to_proceed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "song.flac"
            output = root / "song.lrc"
            reports = root / "reports"
            audio.write_bytes(b"audio")
            output.write_bytes(b"old")
            report, _a, _t = auto_lrc.report_artifact_paths(audio, output, reports)
            report.write_text(json.dumps({"published_lrc_sha256": "0" * 64}), encoding="utf-8")
            lrc_publication.journal_path(report).write_text("{}", encoding="utf-8")
            with mock.patch.object(r2_pipeline, "_run_backend", return_value=1) as backend:
                result = r2_pipeline.main(["--work-dir", str(root / "work"), "--", str(audio), "--output", str(output), "--report-dir", str(reports), "--overwrite", "--timing-source", "lyrics"])
            self.assertEqual(result, 1)
            backend.assert_called_once()


if __name__ == "__main__":
    unittest.main()
