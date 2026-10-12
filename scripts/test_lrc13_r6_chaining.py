from __future__ import annotations

import unittest

import auto_lrc


class AlignmentDiagnosticChainingTests(unittest.TestCase):
    def test_followup_boundary_warning_does_not_chain(self) -> None:
        entries = [
            auto_lrc.LyricEntry(["before"]),
            auto_lrc.LyricEntry(["this is another deliberately long target lyric line"]),
            auto_lrc.LyricEntry(["third"]),
            auto_lrc.LyricEntry(["fourth"]),
        ]
        timestamps = [10.0, 20.0, 22.0, 24.0]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.95},
                {"entry": 2, "timestamp": 20.0, "score": 0.95},
                {"entry": 3, "timestamp": 22.0, "score": 0.95},
                {"entry": 4, "timestamp": 24.0, "score": 0.95},
            ]
        }
        forced = [
            auto_lrc.AsrSegment(10.0, 10.5, "before"),
            auto_lrc.AsrSegment(16.5, 18.0, "target"),
            auto_lrc.AsrSegment(22.0, 22.5, "third"),
            auto_lrc.AsrSegment(24.0, 24.5, "fourth"),
        ]

        auto_lrc.add_alignment_diagnostics(entries, timestamps, report, forced, 30.0)

        rows = {
            item["entry"]: item
            for item in report["suspicious_alignments"]  # type: ignore[index]
        }
        self.assertIn("follows_suspicious_long_line", rows[3]["flags"])
        self.assertNotIn(4, rows)


if __name__ == "__main__":
    unittest.main()
