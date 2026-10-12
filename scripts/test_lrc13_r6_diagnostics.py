from __future__ import annotations

import unittest

import auto_lrc


class AlignmentDiagnosticSeverityTests(unittest.TestCase):
    def test_high_severity_flag_is_not_downgraded_by_medium_flag(self) -> None:
        entries = [
            auto_lrc.LyricEntry(["before"]),
            auto_lrc.LyricEntry(["this is a deliberately long target lyric line"]),
            auto_lrc.LyricEntry(["after"]),
        ]
        timestamps = [10.0, 20.0, 22.5]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.95},
                {"entry": 2, "timestamp": 20.0, "score": 0.95},
                {"entry": 3, "timestamp": 22.5, "score": 0.95},
            ]
        }
        forced = [
            auto_lrc.AsrSegment(10.0, 10.5, "before"),
            auto_lrc.AsrSegment(16.5, 18.0, "target"),
            auto_lrc.AsrSegment(22.5, 23.0, "after"),
        ]

        auto_lrc.add_alignment_diagnostics(entries, timestamps, report, forced, 30.0)

        row = next(
            item for item in report["suspicious_alignments"]  # type: ignore[index]
            if item["entry"] == 2
        )
        self.assertIn("possible_midline_anchor", row["flags"])
        self.assertIn("long_line_close_to_next_entry", row["flags"])
        self.assertEqual(row["severity"], "high")


if __name__ == "__main__":
    unittest.main()
