from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "align-lrc.ps1"


class AlignLrcWrapperSafetyTests(unittest.TestCase):
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
