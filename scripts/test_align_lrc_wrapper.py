from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "align-lrc.ps1"


class AlignLrcWrapperSafetyTests(unittest.TestCase):
    def test_wrapper_progress_summary_and_encoding_contract(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8-sig")
        self.assertIn("$songCount = $Paths.Count", source)
        self.assertIn("songs queued", source)
        self.assertIn("Format-Elapsed", source)
        self.assertIn('$summaryScript = Join-Path $scriptDir "scripts\\lrc_batch_summary.py"', source)
        self.assertIn('"--paths-file", $summaryPathsFile', source)
        self.assertNotIn("$inputRoot", source)
        self.assertIn('"--since", $runStartIso', source)
        self.assertIn('"--report-dir", $ReportDirectory', source)
        self.assertIn('"--work-dir", $scriptDir', source)
        self.assertIn('"--json-out", $batchSummaryPath', source)
        self.assertIn('$batchTargetsPath = Join-Path $ReportDirectory "batch-targets.txt"', source)
        self.assertIn('total_review_rows', source)
        self.assertIn('$song.report_path', source)
        self.assertIn('Start-Process -FilePath $reviewBatchPath', source)
        self.assertNotIn("cmd /c", source)
        self.assertIn('$env:LRC_NO_REVIEW -eq "1"', source)
        self.assertIn('summaryGeneratedAt -lt $runStartedAt', source)
        self.assertIn("$PSNativeCommandUseErrorActionPreference = $false", source)
        self.assertIn("$previousConsoleOutputEncoding", source)
        self.assertIn("$previousPythonEncoding", source)
        self.assertIn('$reviewBatchPath = [System.IO.Path]::GetFullPath', source)

    def test_review_batch_is_ascii_crlf_and_uses_local_server(self) -> None:
        raw = (REPO_ROOT / "Review LRC.bat").read_bytes()
        self.assertNotIn(b"\x00", raw)
        self.assertTrue(raw.count(b"\r\n") >= 1)
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))
        text = raw.decode("ascii")
        self.assertIn("setlocal", text)
        self.assertIn("scripts\\review_server.py", text)
        self.assertIn('"%~dp0scripts\\review_server.py" %*', text)
        self.assertIn("LRC_NO_PAUSE", text)
        self.assertIn('--report-dir "%~dp0outputs\\reports"', text)
        self.assertNotIn("--targets-file", text)
        self.assertNotIn(" cd ", text.lower())

    def test_alignment_pause_uses_shared_no_pause_switch(self) -> None:
        text = (REPO_ROOT / "Align LRC.bat").read_text(encoding="ascii")
        self.assertIn("LRC_NO_PAUSE", text)
        self.assertNotIn("LRC_TOOLS_NO_PAUSE", text)

    def test_default_routes_through_r2_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flac = root / "Song.flac"
            lyrics = root / "Song.lyrics.txt"
            flac.write_bytes(b"not-a-real-flac")
            lyrics.write_text("test line\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(WRAPPER),
                    str(flac),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Arbiter: reviewer-validity", combined)
        self.assertIn("R2 reviewer acquisition", combined)

    def test_reviewer_validity_routes_through_r2_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flac = root / "Song.flac"
            lyrics = root / "Song.lyrics.txt"
            flac.write_bytes(b"not-a-real-flac")
            lyrics.write_text("test line\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(WRAPPER),
                    "-Arbiter",
                    "reviewer-validity",
                    str(flac),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("R2 reviewer acquisition", combined)

    def test_hub_reviewer_opt_in_is_accepted_without_changing_default(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8-sig")
        self.assertIn("[switch] $IncludeHubReviewer", source)
        self.assertIn('"--include-hub-reviewer"', source)

    def test_reviewer_profile_is_exposed_and_forwarded(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8-sig")
        self.assertIn('[ValidateSet("none", "strict", "balanced", "loose")]', source)
        self.assertIn('[string] $ReviewerProfile = "none"', source)
        self.assertIn('"--reviewer-profile", $ReviewerProfile', source)

    def test_refuses_to_overwrite_dropped_lyric_source_even_with_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flac = root / "Song.flac"
            lyric = root / "Song.lrc"
            flac.write_bytes(b"not-a-real-flac")
            lyric.write_text("[00:01.00]line\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(WRAPPER),
                    "-Output",
                    str(lyric),
                    "-Overwrite",
                    str(lyric),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite lyric source", combined.lower())


if __name__ == "__main__":
    unittest.main()
