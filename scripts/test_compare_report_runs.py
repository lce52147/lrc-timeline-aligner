import json
import tempfile
import unittest
from pathlib import Path

import compare_report_runs as crr


def tag(cs: int) -> str:
    return f"[{cs // 6000:02d}:{cs % 6000 // 100:02d}.{cs % 100:02d}]"


def bare(cs: int) -> str:
    return tag(cs)[1:-1]


class CompareReportRunsTest(unittest.TestCase):
    def _report(self, folder: Path, name: str, times, statuses, output: Path, audio: Path) -> None:
        rows = [{"entry": i + 1, "lrc_tag": bare(t), "verification_status": s,
                 "final_trust_reason_codes": (["order-problem"] if s == "review-required" else [])}
                for i, (t, s) in enumerate(zip(times, statuses))]
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
            ref_cells = result["totals"]["ref"]["status_error"]
            self.assertEqual(ref_cells["REVIEW"], {"n": 1, "<=0.20s": 0, "0.20-0.50s": 0, "0.50-1.00s": 0, ">1.00s": 1})
            self.assertEqual(ref_cells["TRUSTED"]["<=0.20s"], 1)
            self.assertEqual(result["totals"]["ref"]["reason_error"]["order-problem"]["n"], 1)
            self.assertIn("| REVIEW | 1 | 0 | 0 | 0 | 1 | 0% |", text)
            self.assertIn("REVIEW reason codes vs BK (ref)", text)
            self.assertIn("| order-problem | 1 | 0 | 0 | 0 | 1 | 0% |", text)
            self.assertNotIn("REVIEW reason codes vs BK (run)", text)

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

    def test_bilingual_bk_counts_each_timestamp_once_and_matches_loosely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, bk = root / "run", root / "bk"
            run.mkdir(); bk.mkdir()
            (bk / "Ever Be My Love (TV size).lrc").write_text(
                "[00:00.00]title\n" + "\n".join(f"{tag(t)} orig\n{tag(t)} 翻譯" for t in (100, 200, 300)), encoding="utf-8")
            self._report(run, "07. Ever be my love--8140201fbb", (0, 100, 200, 330), ("trusted",) * 4,
                         root / "o.lrc", root / "07. Ever be my love.flac")
            result = crr.compare(run, run, bk)
            self.assertEqual(result["skipped"], [])
            self.assertEqual(result["totals"]["run"]["lines"], 4)
            self.assertEqual(result["totals"]["run"]["within_20cs"], 3)

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




class DemotionLadderTest(unittest.TestCase):
    def test_ladder_demotes_only_lines_whose_codes_are_all_weak(self) -> None:
        lines = [
            {"codes": ["weak"], "delta_cs": 5},
            {"codes": ["weak"], "delta_cs": 8},
            {"codes": ["weak", "strong"], "delta_cs": 300},
            {"codes": ["strong"], "delta_cs": 150},
        ]
        reason_error = {}
        for item in lines:
            for code in item["codes"]:
                crr.add_error(reason_error.setdefault(code, crr.error_cells()), item["delta_cs"])
        ladder = crr.demotion_ladder(lines, reason_error)
        self.assertEqual([step["code"] for step in ladder], ["weak", "strong"])
        first, second = ladder
        self.assertEqual((first["demoted"], first["demoted_right"], first["demoted_over_1s"]), (2, 2, 0))
        self.assertEqual((first["kept"], first["kept_over_1s"]), (2, 2))
        self.assertEqual((second["demoted"], second["demoted_over_1s"], second["kept"]), (4, 2, 0))
        sole = crr.sole_trigger_table(lines)
        self.assertEqual(sole["weak"]["n"], 2)
        self.assertEqual(sole["strong"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
