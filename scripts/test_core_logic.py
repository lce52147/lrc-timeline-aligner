#!/usr/bin/env python3
"""Public-safe unit tests for alignment trust and anchor matching logic."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from auto_lrc import (
    AnchorHint,
    LrcError,
    LyricEntry,
    apply_ctc_local_fusion_to_whisperx,
    match_anchor_entries,
    update_report_confidence_metrics,
)
from export_alignment_audit import build_rows, write_markdown


class AnchorHintTests(unittest.TestCase):
    def test_entry_number_disambiguates_repeated_lyrics(self) -> None:
        entries = [
            LyricEntry(["repeat"]),
            LyricEntry(["middle"]),
            LyricEntry(["repeat"]),
        ]

        matches = match_anchor_entries(entries, [AnchorHint(3, LyricEntry(["repeat"], 1234))])

        self.assertEqual([(index + 1, anchor.source_time_cs) for index, anchor in matches], [(3, 1234)])

    def test_entry_number_rejects_text_mismatch(self) -> None:
        entries = [
            LyricEntry(["repeat"]),
            LyricEntry(["middle"]),
            LyricEntry(["repeat"]),
        ]

        with self.assertRaisesRegex(LrcError, "text mismatch"):
            match_anchor_entries(entries, [AnchorHint(2, LyricEntry(["repeat"], 1234))])


class TimingTrustTests(unittest.TestCase):
    def test_timing_trusted_assignment_counts_as_trusted_without_changing_score(self) -> None:
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                {
                    "entry": 2,
                    "segment": 2,
                    "score": 0.48,
                    "timestamp": 20.0,
                    "timing_trusted": True,
                },
            ],
            "low_confidence_entries": [{"entry": 2, "score": 0.48, "lyric": "low text score"}],
            "review_required_count": 0,
        }

        update_report_confidence_metrics(report)

        self.assertEqual(report["trusted_entries"], 2)
        self.assertEqual(report["trusted_percent"], 100.0)
        self.assertEqual(report["low_confidence_count"], 0)
        self.assertEqual(report["timing_trusted_entries"], 1)
        self.assertEqual(report["assignments"][1]["score"], 0.48)  # type: ignore[index]

    def test_ctc_raw_consensus_resolves_local_fusion_review(self) -> None:
        whisperx_report: dict[str, object] = {
            "backend": "whisperx",
            "strategy": "whisperx-hybrid-experimental",
            "timing_entries": 2,
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                {"entry": 2, "segment": 2, "score": 0.55, "timestamp": 20.0},
            ],
            "low_confidence_entries": [{"entry": 2, "score": 0.55, "lyric": "second"}],
            "suspicious_alignments": [],
            "review_required_count": 0,
        }
        ctc_report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "ctc_score": 0.1},
                {"entry": 2, "timestamp": 22.5, "ctc_score": 0.1},
            ],
        }
        raw_report: dict[str, object] = {
            "backend": "whispercpp_raw",
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.9, "timestamp": 10.0},
                {"entry": 2, "segment": 2, "score": 0.76, "timestamp": 22.3},
            ],
        }

        timestamps, report, changes = apply_ctc_local_fusion_to_whisperx(
            [10.0, 20.0],
            whisperx_report,
            ctc_report,
            duration=30.0,
            raw_report=raw_report,
        )

        self.assertEqual(timestamps, [10.0, 22.5])
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["reason"], "ctc-raw-consensus-over-whisperx")
        self.assertTrue(changes[0]["fusion_trusted"])
        self.assertEqual(report["review_required_count"], 0)
        self.assertEqual(report["trusted_percent"], 100.0)
        self.assertEqual(report["low_confidence_count"], 0)
        assignments = report["assignments"]  # type: ignore[assignment]
        self.assertTrue(assignments[1]["timing_trusted"])  # type: ignore[index]
        self.assertEqual(assignments[1]["score"], 0.55)  # type: ignore[index]


class AuditExportTests(unittest.TestCase):
    def test_audit_rows_expose_timing_trust_reason_and_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lrc_path = tmp_path / "song.lrc"
            lrc_path.write_text("[00:10.00]first\n[00:22.50]second\n", encoding="utf-8")
            report: dict[str, object] = {
                "backend": "hybrid",
                "strategy": "unit-test",
                "timing_entries": 2,
                "trusted_percent": 100.0,
                "timing_trusted_entries": 1,
                "review_required_count": 0,
                "assignments": [
                    {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                    {
                        "entry": 2,
                        "segment": 2,
                        "score": 0.55,
                        "timestamp": 22.5,
                        "timing_trusted": True,
                        "timing_trusted_reason": "multi-backend-time-consensus",
                        "timing_trusted_sources": ["ctc", "raw"],
                        "timing_trusted_candidate_times": {
                            "selected": 22.5,
                            "whisperx": 20.0,
                            "ctc": 22.5,
                            "raw": 22.3,
                        },
                    },
                ],
                "suspicious_alignments": [],
            }

            rows = build_rows(lrc_path, report)
            output = tmp_path / "audit.md"
            write_markdown(rows, report, output)
            content = output.read_text(encoding="utf-8")

        self.assertEqual(rows[1]["timing_trusted"], "yes")
        self.assertEqual(rows[1]["timing_trust_sources"], "ctc;raw")
        self.assertIn("multi-backend-time-consensus", content)
        self.assertIn("ctc;raw", content)
        self.assertIn("timing_trusted", content)


if __name__ == "__main__":
    unittest.main()
