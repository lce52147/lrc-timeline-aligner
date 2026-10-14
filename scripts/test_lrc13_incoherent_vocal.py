"""Incoherent whole-song rows take the collapsed fallback; local-window rows that
start in vocal silence never become candidates (same F1 judgment as the final
self-check)."""
from __future__ import annotations

import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import auto_lrc
from auto_lrc import LyricEntry
from test_lrc13_collapsed_local_window import FakeHelper, ctc_report, local_window_times, silent_raw


def spans(points: list[tuple[float, float]]) -> tuple[auto_lrc.TimingTokenSpan, ...]:
    return auto_lrc._timing_spans_from_value(
        [
            {"char": chr(97 + i), "start": start, "end": round(start + 0.02, 3), "score": score}
            for i, (start, score) in enumerate(points)
        ]
    )


def candidate(index: int, source: str, time: float, points, *, current: bool = False):
    return auto_lrc.make_timing_candidate(
        entry_index=index,
        entry_text=f"line {index + 1}",
        source=source,
        raw_time=time,
        spans=spans(points),
        confidence=0.3,
        identity_support="supported",
        sequence_support="supported",
        current=current,
    )


def steady(start: float, score: float = 0.4) -> list[tuple[float, float]]:
    return [(round(start + 0.3 * i, 3), score) for i in range(6)]


class IncoherentFallbackTests(unittest.TestCase):
    def row_sets(self, middle_current):
        # Row 2's local-window alternative carries no opening proof (weak
        # tokens), so neither candidate is valid and Central falls back.
        return (
            (candidate(0, "ctc-current", 10.0, steady(10.0), current=True),),
            (
                middle_current,
                candidate(1, "ctc-local-window", 20.0, steady(20.0, 0.01)),
            ),
            (candidate(2, "ctc-current", 40.0, steady(40.0), current=True),),
        )

    def test_fractured_opening_takes_the_local_window(self) -> None:
        # One token, then a 6 s gap: extent 7.5 s (not collapsed) but the
        # opening is fractured, so Central rejects it as temporal-incoherence.
        broken = candidate(1, "ctc-current", 30.0, [(30.0, 0.4)] + steady(36.0)[:5], current=True)
        sets = self.row_sets(broken)
        self.assertFalse(auto_lrc.current_ctc_path_collapsed(sets[1]))
        self.assertTrue(auto_lrc.current_ctc_path_incoherent(sets[1]))
        state = auto_lrc.build_final_timing_state(sets)
        decision = state.decisions[1]
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.candidate.source, "ctc-local-window")
        self.assertAlmostEqual(decision.written_time.seconds, 20.0, places=2)

    def test_merely_unproven_current_is_kept(self) -> None:
        weak = candidate(1, "ctc-current", 30.0, steady(30.0, 0.01), current=True)
        sets = self.row_sets(weak)
        self.assertFalse(auto_lrc.current_ctc_path_incoherent(sets[1]))
        state = auto_lrc.build_final_timing_state(sets)
        self.assertEqual(state.decisions[1].candidate.source, "ctc-current")


class LocalWindowVocalSilenceTests(unittest.TestCase):
    TIMES = [10.0, 20.0, 22.0, 45.0, 60.0]
    EXTENTS = [2.0, 2.0, 20.0, 2.0, 2.0]

    def test_onset_in_vocal_silence_is_rejected(self) -> None:
        # Vocals 0-30 s and 40-70 s; row 3's window onset at 33 s is silent,
        # rows 2 and 4 start inside singing.
        x = np.zeros(70 * auto_lrc.SAMPLE_RATE, dtype=np.float32)
        x[: 30 * auto_lrc.SAMPLE_RATE] = 0.25
        x[40 * auto_lrc.SAMPLE_RATE:] = 0.25
        features = types.SimpleNamespace(segments=[(0, 30), (40, 70)])
        report = ctc_report(self.TIMES, self.EXTENTS)
        entries = [LyricEntry([f"line {i + 1}"]) for i in range(len(self.TIMES))]
        helper = FakeHelper({1: 10.0, 2: 26.0, 3: 33.0, 4: 47.0})
        with tempfile.TemporaryDirectory() as tmp:
            stem = pathlib.Path(tmp) / "vocals.mp3"
            stem.write_bytes(b"mock")
            report["ctc_audio_source"] = "vocal-stem"
            report["ctc_audio_path"] = str(stem)
            with patch.object(auto_lrc, "decode_audio", return_value=x), patch.object(
                auto_lrc, "analyze_audio", return_value=features
            ), patch.object(auto_lrc.subprocess, "run", helper), patch.object(
                auto_lrc, "default_ctc_python", lambda: pathlib.Path("python")
            ):
                levels = auto_lrc.vocal_stem_levels(report)
                auto_lrc.apply_ctc_local_window_realign(
                    pathlib.Path("dummy.flac"), entries, list(self.TIMES), report,
                    silent_raw(self.TIMES), 70.0, type("Args", (), {"whisperx_device": "cpu"})(),
                )
        self.assertEqual(levels.status, "checked")
        self.assertTrue(levels.absent(33.0))
        self.assertFalse(levels.absent(26.0))
        self.assertEqual(local_window_times(report), {2: 26.0, 4: 47.0})
        rejected = report["ctc_local_window_realign"]["rejected_rows"]  # type: ignore[index]
        self.assertEqual(
            [(row["entry"], row["reason"]) for row in rejected], [(3, "local-window-vocal-absent")]
        )

    def test_without_a_vocal_stem_nothing_is_rejected(self) -> None:
        report = ctc_report(self.TIMES, self.EXTENTS)
        entries = [LyricEntry([f"line {i + 1}"]) for i in range(len(self.TIMES))]
        helper = FakeHelper({1: 10.0, 2: 26.0, 3: 33.0, 4: 47.0})
        with patch.object(auto_lrc.subprocess, "run", helper), patch.object(
            auto_lrc, "default_ctc_python", lambda: pathlib.Path("python")
        ):
            auto_lrc.apply_ctc_local_window_realign(
                pathlib.Path("dummy.flac"), entries, list(self.TIMES), report,
                silent_raw(self.TIMES), 70.0, type("Args", (), {"whisperx_device": "cpu"})(),
            )
        self.assertEqual(local_window_times(report), {2: 26.0, 3: 33.0, 4: 47.0})
        self.assertEqual(report["ctc_local_window_realign"]["rejected_rows"], [])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
