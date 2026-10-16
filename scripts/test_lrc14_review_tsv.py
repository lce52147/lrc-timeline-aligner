from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import auto_lrc
import lrc_review_tsv
import r2_pipeline


HEADER = [
    "entry", "time", "lyric", "status", "reason_codes", "severity_rank",
    "severity_score", "severity_reasons", "timestamp_status", "lrc_time",
    "prev_time", "next_time", "candidate_times", "candidate_spread_s",
]


def report_fixture() -> dict[str, object]:
    return {
        "lyric_preflight": {"status": "OK"},
        "multimodel_consensus": {
            "review_only_model_disagreements": [{"entry": 3}]
        },
        "central_timing_state": [
            {"entry": 3, "timestamp_status": "provisional_unresolved", "canonical_written_centiseconds": 330},
            {"entry": 1, "timestamp_status": "provisional_unresolved", "canonical_written_centiseconds": 110},
            {"entry": 4, "timestamp_status": "provisional_unresolved", "canonical_written_centiseconds": 440},
            {"entry": 2, "timestamp_status": "provisional_unresolved", "canonical_written_centiseconds": 220},
        ],
        "assignments": [
            {
                "entry": 1, "timestamp": 1.1, "canonical_written_centiseconds": 110,
                "review_required": True, "verification_status": "review-required",
                "final_trust_reason_codes": [{"code": "model-disagreement"}],
                "candidates": [{"time": 1.1}, {"time": 1.8}],
                "alignment_hypotheses": [{"raw_candidate_time": 1.1}],
            },
            {
                "entry": 2, "timestamp": 2.2, "canonical_written_centiseconds": 220,
                "review_required": True, "verification_status": "review-required",
                "timestamp_status": "selected_valid",
                "final_trust_reason_codes": ["candidate-disagreement"],
                "candidates": [{"time": 2.2}],
            },
            {
                "entry": 3, "timestamp": 3.3, "canonical_written_centiseconds": 330,
                "review_required": True, "verification_status": "review-required",
                "final_trust_reason_codes": ["candidate-disagreement"],
                "candidates": [{"time": 3.3}, {"time": 3.9}],
            },
            {
                "entry": 4, "timestamp": 4.4, "canonical_written_centiseconds": 440,
                "review_required": False, "verification_status": "unverified",
                "timestamp_status": "provisional_unresolved",
                "final_trust_reason_codes": ["unverified"],
                "candidates": [{"time": 4.4}],
            },
        ],
    }


class ReviewSeverityTests(unittest.TestCase):
    def test_header_bom_and_sanitization(self) -> None:
        report = report_fixture()
        entries = [
            auto_lrc.LyricEntry(["one\tline"]),
            auto_lrc.LyricEntry(["two\nline"]),
            auto_lrc.LyricEntry(["three"]),
            auto_lrc.LyricEntry(["four"]),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "song.review.tsv"
            lrc_review_tsv.write_review_tsv(path, report, entries)
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            self.assertNotIn(b"\r", raw)
            lines = raw.decode("utf-8-sig").splitlines()
            self.assertTrue(lines[0].startswith("# fast_singing="))
            self.assertIn("severity_rule=lrc14-pmd-v1", lines[0])
            header = lines[1].split("\t")
            self.assertEqual(header, HEADER)
            self.assertEqual(header[:5], ["entry", "time", "lyric", "status", "reason_codes"])
            self.assertTrue(all(len(line.split("\t")) == 14 for line in lines[2:]))
            self.assertIn("one line", "\n".join(lines))
            self.assertIn("two line", "\n".join(lines))

    def test_ranking_and_features(self) -> None:
        report = report_fixture()
        ranked = lrc_review_tsv.rank_rows(report, report["assignments"])
        self.assertEqual([row["entry"] for row in ranked], [1, 3, 2, 4])
        features = lrc_review_tsv.severity_features(report)
        self.assertEqual([features[i]["severity_score"] for i in (1, 3, 2, 4)], [7, 7, 6, 2])
        self.assertEqual(features[1]["candidate_times"], [1.1, 1.8])
        self.assertEqual(features[1]["candidate_spread_s"], 0.7)
        self.assertEqual(features[2]["timestamp_status"], "provisional_unresolved")

    def test_central_entry_mapping_and_edges(self) -> None:
        report = report_fixture()
        features = lrc_review_tsv.severity_features(report)
        self.assertEqual(features[1]["lrc_time"], 1.1)
        self.assertEqual(features[1]["prev_time"], None)
        self.assertEqual(features[4]["next_time"], None)
        self.assertEqual(features[2]["prev_time"], 1.1)
        self.assertEqual(features[2]["next_time"], 3.3)

    def test_writer_does_not_mutate_report(self) -> None:
        report = report_fixture()
        before = copy.deepcopy(report)
        entries = [auto_lrc.LyricEntry([str(i)]) for i in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            lrc_review_tsv.write_review_tsv(Path(folder) / "x.tsv", report, entries)
        self.assertEqual(report, before)

    def test_top_review_summary(self) -> None:
        report = report_fixture()
        entries = [auto_lrc.LyricEntry([str(i)]) for i in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.tsv"
            lrc_review_tsv.write_review_tsv(path, report, entries)
            summary = lrc_review_tsv.format_top_review(path)
            self.assertLessEqual(len(summary), 6)
            self.assertTrue(summary[0].startswith("REVIEW TOP 5 (of 4 listed, 3 review):"))
            empty = Path(folder) / "empty.tsv"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(lrc_review_tsv.format_top_review(empty), ["REVIEW TOP 5: none"])

    def test_missing_review_tsv_summary_is_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "song.flac"
            lyrics = root / "song.txt"
            audio.write_bytes(b"audio")
            lyrics.write_text("[00:01.00]line\n", encoding="utf-8")
            args = auto_lrc.build_parser().parse_args([
                str(audio), "--lyrics", str(lyrics), "--output", str(root / "song.lrc"),
                "--timing-source", "lyrics", "--report-dir", str(root / "reports"), "--overwrite",
            ])
            output = io.StringIO()
            with (
                mock.patch.object(auto_lrc, "probe_duration", return_value=2.0),
                mock.patch.object(auto_lrc, "audio_track_labels", return_value=("", "Song")),
                mock.patch.object(auto_lrc, "write_review_tsv"),
                redirect_stdout(output),
            ):
                auto_lrc.process_audio(audio, args)
        self.assertIn("REVIEW TOP 5: unavailable (FileNotFoundError)", output.getvalue())

    def test_cli_and_r2_summary_flag(self) -> None:
        args = auto_lrc.build_parser().parse_args(["song.flac", "--no-review-summary"])
        self.assertTrue(args.no_review_summary)
        raw = ["--timing-source", "auto", "song.flac"]
        baseline = r2_pipeline.prepare_baseline_args(
            raw, output=Path("scratch/baseline.lrc"), report_dir=Path("scratch/reports")
        )
        self.assertIn("--no-review-summary", baseline)
        self.assertEqual(baseline[-7:], [
            "--arbiter", "off", "--output", str(Path("scratch/baseline.lrc")),
            "--report-dir", str(Path("scratch/reports")), "--overwrite",
        ])
        final = r2_pipeline.prepare_final_args(raw, Path("evidence.json"))
        self.assertNotIn("--no-review-summary", final)


if __name__ == "__main__":
    unittest.main()
