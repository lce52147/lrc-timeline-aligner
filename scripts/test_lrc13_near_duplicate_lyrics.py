"""A verse variant that differs only in its opening words is raw-ASR-ambiguous:
its raw timestamp neither triggers nor bounds a local CTC window."""
from __future__ import annotations

import json
import pathlib
import unittest
from unittest.mock import patch

import auto_lrc
from auto_lrc import LyricEntry, has_ambiguous_lyric_prefix
from test_lrc13_collapsed_local_window import ctc_report

TEXTS = [
    "the first line of the song",
    "im dreaming without you",
    "a lonely middle verse line",
    "im living without you",
    "and the very last line here",
]


class RecordingHelper:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float, list[str]]] = []

    def __call__(self, command, **_kwargs):  # type: ignore[no-untyped-def]
        start = float(command[command.index("--start") + 1])
        end = float(command[command.index("--end") + 1])
        transcript = json.loads(
            pathlib.Path(command[command.index("--transcript") + 1]).read_text(encoding="utf-8")
        )
        self.calls.append((round(start, 3), round(end, 3), transcript))
        rows = [
            {"start": start + 0.5 + i, "ctc_score": 0.6, "token_spans": [], "first_token_candidates": []}
            for i in range(len(transcript))
        ]
        pathlib.Path(command[command.index("--output") + 1]).write_text(
            json.dumps({"entries": rows}), encoding="utf-8"
        )
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()


class NearDuplicateLyricTests(unittest.TestCase):
    def test_shared_suffix_is_ambiguous_like_a_shared_prefix(self) -> None:
        entries = [LyricEntry([text]) for text in TEXTS]
        self.assertTrue(has_ambiguous_lyric_prefix(entries, 1))
        self.assertTrue(has_ambiguous_lyric_prefix(entries, 3))
        self.assertFalse(has_ambiguous_lyric_prefix(entries, 0))
        self.assertFalse(has_ambiguous_lyric_prefix(entries, 2))

    def test_variant_raw_time_does_not_bound_the_local_window(self) -> None:
        # Row 3 has a low path score.  Raw ASR is confident that row 4 starts
        # at 30 s, but row 4 is a variant of row 2 and actually sits at 45 s.
        times = [10.0, 20.0, 28.0, 45.0, 55.0]
        report = ctc_report(times, [2.0] * 5, runs=[[3]])
        raw = {
            "assignments": [
                {"entry": i + 1, "timestamp": t, "score": 0.2} for i, t in enumerate(times)
            ]
        }
        raw["assignments"][3] = {"entry": 4, "timestamp": 30.0, "score": 0.95}
        entries = [LyricEntry([text]) for text in TEXTS]
        helper = RecordingHelper()
        with patch.object(auto_lrc.subprocess, "run", helper), patch.object(
            auto_lrc, "default_ctc_python", lambda: pathlib.Path("python")
        ):
            auto_lrc.apply_ctc_local_window_realign(
                pathlib.Path("dummy.flac"), entries, list(times), report, raw, 60.0,
                type("Args", (), {"whisperx_device": "cpu"})(),
            )
        # One window for row 3, ending at row 4's own CTC start, not at 30 s.
        self.assertEqual([(start, end) for start, end, _ in helper.calls], [(19.95, 44.96)])


if __name__ == "__main__":
    unittest.main()
