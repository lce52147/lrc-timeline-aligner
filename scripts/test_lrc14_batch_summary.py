from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc
import lrc_review_tsv
import r2_pipeline
import review_server

try:
    import lrc_batch_summary
except ModuleNotFoundError:
    lrc_batch_summary = None


def _module():
    if lrc_batch_summary is None:
        raise AssertionError("scripts/lrc_batch_summary.py is not implemented")
    return lrc_batch_summary


class BatchSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.report_dir = self.root / "reports"
        self.work_dir = self.root / "reviewer-work"
        self.report_dir.mkdir()
        self.work_dir.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _audio(self, relative: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"flac-stub")
        path.with_suffix(".lrc").write_text("[00:01.00]line\n", encoding="utf-8")
        return path

    def _report(
        self,
        audio: Path,
        *,
        audio_path: Path | None = None,
        seconds: float | None = 12.3456,
        review_tsv: Path | None = None,
    ) -> Path:
        batch = _module()
        path = batch.report_path(audio, audio.with_suffix(".lrc"), self.report_dir)
        payload: dict[str, object] = {"audio_path": str(audio_path or audio)}
        if seconds is not None:
            payload["stage_timings"] = {"total_seconds": seconds}
        if review_tsv is not None:
            payload["review_tsv_path"] = str(review_tsv)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _real_tsv(self, audio: Path) -> tuple[Path, dict[str, object]]:
        report: dict[str, object] = {
            "audio_path": str(audio),
            "lyric_preflight": {"status": "OK"},
            "multimodel_consensus": {
                "review_only_model_disagreements": [{"entry": 1}, {"entry": 2}],
            },
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 1.0,
                    "review_required": True,
                    "verification_status": "review-required",
                    "timestamp_status": "provisional_unresolved",
                    "final_trust_reason_codes": ["model-disagreement"],
                    "candidates": [{"time": 1.0}, {"time": 1.8}],
                },
                {
                    "entry": 2,
                    "timestamp": 2.0,
                    "review_required": False,
                    "verification_status": "unverified",
                    "final_trust_reason_codes": ["unverified"],
                    "candidates": [{"time": 2.0}],
                },
            ],
        }
        path = self.report_dir / f"{audio.stem}.review.tsv"
        entries = [auto_lrc.LyricEntry(['"one']), auto_lrc.LyricEntry(["two"])]
        lrc_review_tsv.write_review_tsv(path, report, entries)
        return path, report

    def _collect(self, audios: list[Path], *, since: datetime | None = None):
        return _module().collect(
            audios,
            report_dir=self.report_dir,
            work_dir=self.work_dir,
            since=since,
        )

    def _main(self, argv: list[str], *, env: dict[str, str] | None = None):
        output, errors = io.StringIO(), io.StringIO()
        with (
            mock.patch.dict(os.environ, env or {}, clear=False),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            code = _module().main(argv)
        return code, output.getvalue(), errors.getvalue()

    @staticmethod
    def _snapshot(root: Path) -> dict[str, tuple[str, int]]:
        result = {}
        for path in root.rglob("*"):
            if path.is_file() and not path.name.endswith(".batch-summary.json"):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                result[str(path.relative_to(root))] = (digest, path.stat().st_mtime_ns)
        return result

    def test_01_path_formulas_match_production_owners(self) -> None:
        batch = _module()
        audios = [
            self._audio("a/same.flac"),
            self._audio("b/same.flac"),
            self._audio("tail./song..flac"),
            self._audio("日本語/歌.flac"),
        ]
        for audio in audios:
            with self.subTest(audio=audio):
                expected_report = auto_lrc.report_artifact_paths(
                    audio, audio.with_suffix(".lrc"), self.report_dir
                )[0]
                self.assertEqual(
                    batch.report_path(audio, audio.with_suffix(".lrc"), self.report_dir),
                    expected_report,
                )
                self.assertEqual(
                    batch.r2_run_dir(self.work_dir, audio),
                    r2_pipeline.reviewer_work_directory(self.work_dir, audio),
                )

    def test_02_statuses_and_exit_codes(self) -> None:
        ok = self._audio("ok.flac")
        stale = self._audio("stale.flac")
        missing = self._audio("missing.flac")
        bad = self._audio("bad.flac")
        mismatch = self._audio("mismatch.flac")
        self._report(ok)
        stale_report = self._report(stale)
        old = (datetime.now().astimezone() - timedelta(hours=2)).timestamp()
        os.utime(stale_report, (old, old))
        batch = _module()
        batch.report_path(bad, bad.with_suffix(".lrc"), self.report_dir).write_text(
            "{not-json", encoding="utf-8"
        )
        self._report(mismatch, audio_path=self.root / "other.flac")

        records = self._collect(
            [ok, stale, missing, bad, mismatch],
            since=datetime.now().astimezone() - timedelta(hours=1),
        )
        self.assertEqual(
            [row["status"] for row in records],
            ["ok", "stale", "no-report", "unreadable", "unreadable"],
        )
        code, _out, _err = self._main([
            *(str(path) for path in [ok, stale, missing, bad, mismatch]),
            "--report-dir", str(self.report_dir), "--work-dir", str(self.work_dir),
            "--since", "1h", "--no-json",
        ])
        self.assertEqual(code, 1)
        code, _out, _err = self._main([
            str(ok), "--report-dir", str(self.report_dir),
            "--work-dir", str(self.work_dir), "--no-json",
        ])
        self.assertEqual(code, 0)

    def test_03_real_tsv_metrics_and_nonfatal_missing_or_garbled_tsv(self) -> None:
        audio = self._audio("metrics.flac")
        tsv, _fixture = self._real_tsv(audio)
        self._report(audio, review_tsv=tsv)
        row = self._collect([audio])[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["review_rows"], 1)
        self.assertEqual(row["untrusted_rows"], 1)
        self.assertEqual(row["review_rows"] + row["untrusted_rows"], 2)
        self.assertEqual(row["model_disagreement_rows"], 2)
        self.assertEqual(row["max_severity_score"], 7)

        for name, contents in (("missing.tsv", None), ("garbled.tsv", "bad\tdata\n")):
            with self.subTest(name=name):
                other = self._audio(f"{name}.flac")
                path = self.report_dir / name
                if contents is not None:
                    path.write_text(contents, encoding="utf-8")
                self._report(other, review_tsv=path)
                record = self._collect([other])[0]
                self.assertEqual(record["status"], "ok")
                self.assertIsNone(record["review_rows"])
                self.assertIsNone(record["untrusted_rows"])
                self.assertIsNone(record["model_disagreement_rows"])
                self.assertIsNone(record["max_severity_score"])
                self.assertTrue(record["notes"])

    def test_04_r2_timing_match_mismatch_staleness_and_fallback(self) -> None:
        audios = [self._audio(f"timing-{index}.flac") for index in range(4)]
        reports = [self._report(audio, seconds=10.1236 + index) for index, audio in enumerate(audios)]
        batch = _module()
        for index, audio in enumerate(audios[:3]):
            run_dir = batch.r2_run_dir(self.work_dir, audio)
            run_dir.mkdir(parents=True)
            final_path = reports[index] if index != 1 else self.root / "wrong-report.json"
            timing = {
                "total_seconds": 20.9876 + index,
                "final_report": {"path": str(final_path)},
            }
            timing_path = run_dir / "r2-timings.json"
            timing_path.write_text(json.dumps(timing), encoding="utf-8")
            if index == 2:
                old = (datetime.now().astimezone() - timedelta(hours=2)).timestamp()
                os.utime(timing_path, (old, old))

        records = self._collect(
            audios, since=datetime.now().astimezone() - timedelta(hours=1)
        )
        self.assertEqual(records[0]["seconds_source"], "r2-timings")
        self.assertEqual(records[0]["seconds"], 20.988)
        for record in records[1:3]:
            self.assertEqual(record["seconds_source"], "stage-timings")
            self.assertIsNone(record["r2_total_seconds"])
        self.assertEqual(records[3]["seconds_source"], "stage-timings")
        no_timing = self._audio("no-timing.flac")
        self._report(no_timing, seconds=None)
        record = self._collect([no_timing])[0]
        self.assertIsNone(record["seconds"])
        self.assertIsNone(record["seconds_source"])

    def test_05_attention_order_ties_and_top_five(self) -> None:
        batch = _module()
        specs = [
            ("Zulu", 4, 1, 7),
            ("alpha", 4, 2, 6),
            ("Beta", 4, 2, 6),
            ("gamma", 3, 9, 9),
            ("delta", 2, 0, 1),
            ("epsilon", 1, 0, 1),
            ("zero", 0, 99, 99),
        ]
        records = []
        targets = []
        for index, (name, review, disagreements, score) in enumerate(specs, 1):
            audio = self._audio(f"{name}.flac")
            targets.append(audio)
            records.append({
                "index": index,
                "audio": str(audio),
                "name": audio.name,
                "status": "ok",
                "seconds": 1.0,
                "review_rows": review,
                "untrusted_rows": 0,
                "model_disagreement_rows": disagreements,
                "max_severity_score": score,
            })
        summary = batch.summarize(
            records,
            report_dir=self.report_dir,
            review_server_path=SCRIPTS / "review_server.py",
            python_exe=sys.executable,
            targets=targets,
        )
        self.assertEqual(
            [row["name"] for row in summary["attention"]],
            ["alpha.flac", "Beta.flac", "Zulu.flac", "gamma.flac", "delta.flac"],
        )
        self.assertEqual(len(summary["attention"]), 5)
        self.assertNotIn("zero.flac", [row["name"] for row in summary["attention"]])
        command = summary["review_command"]
        self.assertEqual(command[:3], [
            sys.executable, str(SCRIPTS / "review_server.py"), "--report-dir"
        ])
        self.assertTrue(Path(command[3]).samefile(self.report_dir))
        expected_targets = [targets[1], targets[2], targets[0], targets[3], targets[4]]
        self.assertTrue(all(
            Path(actual).samefile(expected)
            for actual, expected in zip(command[4:], expected_targets, strict=True)
        ))

    def test_06_console_golden_lines(self) -> None:
        batch = _module()
        alpha = self._audio("alpha.flac")
        missing = self._audio("missing.flac")
        records = [
            {
                "index": 1, "audio": str(alpha), "name": alpha.name, "status": "ok",
                "seconds": 3720.0, "review_rows": 3, "untrusted_rows": 4,
                "model_disagreement_rows": 2, "max_severity_score": 7,
            },
            {
                "index": 2, "audio": str(missing), "name": missing.name,
                "status": "no-report", "seconds": None, "review_rows": None,
                "untrusted_rows": None, "model_disagreement_rows": None,
                "max_severity_score": None,
            },
        ]
        summary = batch.summarize(
            records,
            report_dir=self.report_dir,
            review_server_path=SCRIPTS / "review_server.py",
            python_exe=sys.executable,
            targets=[alpha, missing],
        )
        command = summary["review_command"]
        review_line = (
            f'REVIEW PAGE: "{command[0]}" "{command[1]}" '
            f'--report-dir "{command[3]}" "{command[4]}"'
        )
        self.assertEqual(batch.format_console(summary), [
            "BATCH SUMMARY: 2 songs | ok=1 stale=0 no-report=1 unreadable=0 | "
            "time 1h02m (known for 1/2) | review rows=3 (untrusted=4)",
            "  MISSING missing.flac: no-report",
            "  ATTENTION #1 alpha.flac: review=3 model-disagreement=2 max-score=7",
            review_line,
            "BATCH JSON: disabled",
        ])

        summary = batch.summarize(
            [{**records[0], "review_rows": 0}],
            report_dir=self.report_dir,
            review_server_path=SCRIPTS / "review_server.py",
            python_exe=sys.executable,
            targets=[alpha],
        )
        self.assertIn("REVIEW PAGE: none (no songs need review)", batch.format_console(summary))

    def test_07_json_contract_controls_cleanup_and_review_server_exclusion(self) -> None:
        batch = _module()
        audio = self._audio("json.flac")
        report = self._report(audio)
        records = self._collect([audio])
        fixed_now = datetime(2026, 10, 17, 9, 20, 30).astimezone()
        summary = batch.summarize(
            records,
            report_dir=self.report_dir,
            review_server_path=SCRIPTS / "review_server.py",
            python_exe=sys.executable,
            targets=[audio],
            now=fixed_now,
        )
        output = batch.write_json(summary, self.report_dir, now=fixed_now, pid=4321)
        self.assertEqual(output.name, "batch-20261017-092030-4321.batch-summary.json")
        self.assertFalse(output.name.endswith(".align-report.json"))
        payload = json.loads(
            output.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        self.assertEqual((payload["schema"], payload["kind"]), (1, "lrc-batch-summary"))
        self.assertEqual(payload["generated_at"], fixed_now.isoformat())
        self.assertFalse(any(path.suffix == ".tmp" for path in self.report_dir.iterdir()))
        found, _excluded = review_server._report_files([], self.report_dir)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].samefile(report))

        output.unlink()
        code, stdout, _stderr = self._main([
            str(audio), "--report-dir", str(self.report_dir),
            "--work-dir", str(self.work_dir), "--no-json",
        ])
        self.assertEqual(code, 0)
        self.assertIn("BATCH JSON: disabled", stdout)
        self.assertFalse(list(self.report_dir.glob("*.batch-summary.json")))
        code, stdout, _stderr = self._main([
            str(audio), "--report-dir", str(self.report_dir),
            "--work-dir", str(self.work_dir),
        ], env={"LRC_BATCH_SUMMARY": "0"})
        self.assertEqual(code, 0)
        self.assertIn("BATCH JSON: disabled", stdout)
        self.assertFalse(list(self.report_dir.glob("*.batch-summary.json")))

        with mock.patch.object(batch, "write_json", side_effect=OSError("full")):
            code, _stdout, stderr = self._main([
                str(audio), "--report-dir", str(self.report_dir),
                "--work-dir", str(self.work_dir),
            ])
        self.assertEqual(code, 1)
        self.assertEqual(stderr, "WARN: batch summary failed (OSError)\n")

        absent = self.root / "absent-reports"
        other = self._audio("absent.flac")
        code, _stdout, stderr = self._main([
            str(other), "--report-dir", str(absent),
            "--work-dir", str(self.work_dir),
        ])
        self.assertEqual(code, 1)
        self.assertFalse(absent.exists())
        self.assertEqual(
            stderr,
            f"WARN: report directory does not exist; batch JSON skipped: {absent.resolve()}\n",
        )

        ok_record = {**records[0], "status": "ok"}
        with mock.patch.object(batch, "collect", return_value=[ok_record]):
            code, stdout, stderr = self._main([
                str(other), "--report-dir", str(absent),
                "--work-dir", str(self.work_dir),
            ])
        self.assertEqual(code, 0)
        self.assertIn("BATCH JSON: disabled", stdout)
        self.assertIn("batch JSON skipped", stderr)

        explicit_json = self.root / "elsewhere" / "summary.json"
        explicit_json.parent.mkdir()
        with mock.patch.object(batch, "collect", return_value=[ok_record]):
            code, stdout, stderr = self._main([
                str(other), "--report-dir", str(absent),
                "--work-dir", str(self.work_dir),
                "--json-out", str(explicit_json),
            ])
        self.assertEqual(code, 0)
        self.assertTrue(explicit_json.is_file())
        self.assertIn(f"BATCH JSON: {explicit_json.resolve()}", stdout)
        self.assertIn("report directory does not exist", stderr)

    def test_08_cp950_pipes_replace_unencodable_names(self) -> None:
        audio = self._audio("🎵.flac")
        self._report(audio)
        missing_text = self.root / "🎤.txt"
        result = subprocess.run(
            [
                sys.executable, str(SCRIPTS / "lrc_batch_summary.py"),
                str(audio), str(missing_text), "--report-dir", str(self.report_dir),
                "--work-dir", str(self.work_dir), "--no-json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "cp950:strict"},
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("cp950", errors="replace"))
        self.assertIn("BATCH JSON: disabled", result.stdout.decode("cp950"))
        self.assertIn("WARN: skipped non-FLAC", result.stderr.decode("cp950"))
        self.assertNotIn(b"UnicodeEncodeError", result.stdout + result.stderr)

    def test_09_main_is_read_only_except_summary_json(self) -> None:
        audio = self._audio("readonly.flac")
        tsv, _report_fixture = self._real_tsv(audio)
        report = self._report(audio, review_tsv=tsv)
        run_dir = _module().r2_run_dir(self.work_dir, audio)
        run_dir.mkdir(parents=True)
        (run_dir / "r2-timings.json").write_text(json.dumps({
            "total_seconds": 5.0,
            "final_report": {"path": str(report)},
        }), encoding="utf-8")
        before = self._snapshot(self.root)
        code, _stdout, _stderr = self._main([
            str(audio), "--report-dir", str(self.report_dir),
            "--work-dir", str(self.work_dir),
        ])
        self.assertEqual(code, 0)
        self.assertEqual(self._snapshot(self.root), before)
        self.assertEqual(len(list(self.report_dir.glob("*.batch-summary.json"))), 1)

    def test_10_input_expansion_sort_dedupe_warnings_and_usage_error(self) -> None:
        album = self.root / "album"
        album.mkdir()
        upper = self._audio("album/B.flac")
        lower = self._audio("album/a.flac")
        nested = self._audio("album/nested/hidden.flac")
        text = album / "notes.txt"
        text.write_text("x", encoding="utf-8")
        for audio in (upper, lower):
            self._report(audio)
        json_out = self.root / "inputs.batch-summary.json"
        code, _stdout, stderr = self._main([
            str(album), str(lower), str(text),
            "--report-dir", str(self.report_dir), "--work-dir", str(self.work_dir),
            "--json-out", str(json_out),
        ])
        self.assertEqual(code, 0)
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual([row["audio"] for row in payload["songs"]], [str(lower.resolve()), str(upper.resolve())])
        self.assertNotIn(str(nested.resolve()), [row["audio"] for row in payload["songs"]])
        self.assertEqual(stderr.count("WARN: skipped non-FLAC"), 1)

        empty = self.root / "empty"
        empty.mkdir()
        code, _stdout, _stderr = self._main([str(empty), "--no-json"])
        self.assertEqual(code, 2)

    def test_11_eta_empty_and_median(self) -> None:
        batch = _module()
        self.assertEqual(batch.format_eta([], 9), "n/a")
        self.assertEqual(
            batch.format_eta([240.0, 271.0, 600.0], 33),
            "~2h29m (median 4m31s x 33 left, n=3)",
        )

    def test_12_import_isolation_and_forbidden_source_terms(self) -> None:
        module_path = SCRIPTS / "lrc_batch_summary.py"
        self.assertTrue(module_path.is_file())
        code = (
            "import sys;"
            f"sys.path.insert(0, {str(SCRIPTS)!r});"
            "import lrc_batch_summary;"
            "print('auto_lrc' in sys.modules, 'numpy' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False False")
        source = module_path.read_text(encoding="utf-8").casefold()
        for forbidden in (
            "checked_lrc_candidates", "checked-lrc", "lrc bk", "checked references"
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
