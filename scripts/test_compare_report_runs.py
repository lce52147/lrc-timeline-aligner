import json
import tempfile
import unittest
from pathlib import Path

import compare_report_runs as crr


def tag(cs: int) -> str:
    return f"[{cs // 6000:02d}:{cs % 6000 // 100:02d}.{cs % 100:02d}]"


class CompareReportRunsTest(unittest.TestCase):
    def _report(self, folder: Path, name: str, times, statuses, output: Path, audio: Path) -> None:
        rows = [{"entry": i + 1, "lrc_tag": tag(t), "verification_status": s} for i, (t, s) in enumerate(zip(times, statuses))]
        (folder / f"{name}.align-report.json").write_text(
            json.dumps({"assignments": rows, "output_path": str(output), "audio_path": str(audio)}), encoding="utf-8")

    def test_scores_against_bk_and_reports_moved_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, run, bk = root / "ref", root / "run", root / "bk"
            for folder in (ref, run, bk / "Album A", bk / "Album B"):
                folder.mkdir(parents=True)
            (bk / "Album A" / "01. song.lrc").write_text("\n".join(tag(t) + " x" for t in (1000, 2000, 3000)), encoding="utf-8")
            (bk / "Album B" / "01. song.lrc").write_text("\n".join(tag(t) + " y" for t in (1000, 2000, 3000)), encoding="utf-8")
            audio = root / "music" / "Album A" / "01. song.flac"
            out = audio.with_suffix(".lrc")
            self._report(ref, "01. song--abc123", (1000, 2150, 3100), ("trusted", "review-required", "unverified"), out, audio)
            self._report(run, "01. song--abc123", (1000, 2010, 3100), ("trusted", "unverified", "unverified"), out, audio)
            self._report(run, "02. other--def456", (500,), ("trusted",), root / "02. other.lrc", root / "02. other.flac")
            result = crr.compare(ref, run, bk)
            self.assertEqual(result["totals"]["run"]["songs"], 1)
            self.assertEqual(result["totals"]["ref"]["within_20cs"], 1)
            self.assertEqual(result["totals"]["run"]["within_20cs"], 2)
            self.assertEqual(result["totals"]["ref"]["over_1s"], 1)
            self.assertEqual(result["totals"]["run"]["over_1s"], 0)
            self.assertEqual(result["totals"]["ref"]["status"], {"REVIEW": 1, "UNTRUSTED": 1, "TRUSTED": 1})
            self.assertEqual(len(result["moved_lines"]), 1)
            self.assertEqual(result["moved_lines"][0]["shift_cs"], -140)
            self.assertEqual(result["moved_better"], 1)
            self.assertTrue(result["skipped"][0]["reason"].startswith("no-bk-match:02. other.lrc"))
            text = crr.markdown(result)
            self.assertIn("| lines within 0.20 s of BK | 1 | 2 |", text)

    def test_generic_output_name_matches_bk_by_audio_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, bk = root / "run", root / "bk" / "TUYU"
            run.mkdir(); bk.mkdir(parents=True)
            (bk / "01. ふり.lrc").write_text("\n".join(tag(t) + " x" for t in (100, 200)), encoding="utf-8")
            self._report(run, "01. ふり--c4063ef9f2", (100, 230), ("trusted", "trusted"),
                         root / "B2_outputs" / "song1" / "song1.lrc", root / "input" / "01. ふり.flac")
            result = crr.compare(run, run, root / "bk")
            self.assertEqual(result["totals"]["run"]["songs"], 1)
            self.assertEqual(result["totals"]["run"]["within_20cs"], 1)

    def test_ambiguous_bk_without_folder_hint_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, bk = root / "run", root / "bk"
            for folder in (run, bk / "X", bk / "Y"):
                folder.mkdir(parents=True)
            for sub in ("X", "Y"):
                (bk / sub / "s.lrc").write_text(tag(100) + " a", encoding="utf-8")
            self._report(run, "s--111111", (100,), ("trusted",), root / "s.lrc", root / "s.flac")
            result = crr.compare(run, run, bk)
            self.assertTrue(result["skipped"][0]["reason"].startswith("ambiguous-bk-match"))


if __name__ == "__main__":
    unittest.main()
