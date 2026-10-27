"""V2 segment realign stage: structural runs become bounded CTC passes and ledger hypotheses only."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
# The helper-call counts below assume no on-disk model cache.
os.environ.setdefault("LRC_MODEL_CACHE", "0")

import numpy as np  # noqa: E402

import auto_lrc  # noqa: E402
from auto_lrc import LyricEntry  # noqa: E402

VERSE1 = ["asa no hikari ga", "mado wo tataku", "kimi no koe ga", "kikoeta ki ga shita"]
CHORUS = ["kono te wo nobashite", "tooku made tonde yuku", "futari no yume"]
VERSE2 = ["yoru ga kite mo", "hoshi wa kienai", "zutto soba ni", "ite hoshii dake"]
BRIDGE = ["kaze ga fuku"]
LYRICS = VERSE1 + CHORUS + VERSE2 + CHORUS + BRIDGE + CHORUS
# verse 1, chorus (gaps 3 s and 8 s), verse 2, chorus, bridge, chorus
TIMES = [10.0, 14.0, 18.0, 22.0, 30.0, 33.0, 41.0, 55.0, 59.0, 63.0, 67.0, 80.0, 83.0, 91.0, 100.0, 110.0, 113.0, 121.0]
SEGMENTS = [(10.0, 26.0), (30.0, 47.0), (55.0, 71.0), (80.0, 97.0), (100.0, 108.0), (110.0, 127.0)]
DURATION = 130.0


def spans(start: float, count: int = 4) -> list[dict[str, object]]:
    return [
        {"char": chr(97 + i), "start": round(start + 0.3 * i, 3), "end": round(start + 0.3 * i + 0.02, 3), "score": 0.4}
        for i in range(count)
    ]


def ctc_report(times: list[float], weak: set[int] = frozenset()) -> dict[str, object]:
    return {
        "backend": "ctc",
        "assignments": [
            {
                "entry": index + 1, "timestamp": time, "score": 0.9,
                "ctc_score": 0.02 if index in weak else 0.5,
                "timing_repair": "ctc-forced-align", "ctc_token_spans": spans(time),
            }
            for index, time in enumerate(times)
        ],
        "ctc_low_score_runs": [], "ctc_low_score_entries": [], "ctc_missing_entries": [], "suspicious_alignments": [],
    }


def raw_report(times: list[float], weak: set[int] = frozenset()) -> dict[str, object]:
    """Raw ASR agrees with every row it heard; the weak rows were not heard (borrowed)."""
    return {
        "assignments": [
            {"entry": index + 1, "timestamp": time, "score": 0.2 if index in weak else 0.9, "borrowed": index in weak}
            for index, time in enumerate(times)
        ]
    }


class FakeHelper:
    """Stands in for ctc_align.py: answers each window with the onsets it was given per lyric."""

    def __init__(self, onsets: dict[str, float], stray: dict[str, float] | None = None) -> None:
        self.onsets = onsets
        self.stray = stray or {}          # lyric -> time of a lone first token before the real onset
        self.calls: list[tuple[float, float, list[str]]] = []

    def __call__(self, command, **_kwargs):  # type: ignore[no-untyped-def]
        start = float(command[command.index("--start") + 1])
        end = float(command[command.index("--end") + 1])
        transcript = json.loads(Path(command[command.index("--transcript") + 1]).read_text(encoding="utf-8"))
        self.calls.append((round(start, 3), round(end, 3), list(transcript)))
        rows = []
        for offset, text in enumerate(transcript):
            onset = self.onsets.get(text, start + 0.5 + offset)
            row = {"start": onset, "ctc_score": 0.6, "token_spans": spans(onset), "first_token_candidates": []}
            if text in self.stray:
                row["start"] = self.stray[text]
                row["token_spans"] = spans(self.stray[text], 1) + spans(onset)
            rows.append(row)
        Path(command[command.index("--output") + 1]).write_text(json.dumps({"entries": rows}), encoding="utf-8")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()


class Patched:
    """Swap the model helpers for fakes for the duration of one test."""

    def __init__(self, helper: FakeHelper, vocal_absent: bool = False) -> None:
        self.helper = helper
        self.vocal_absent = vocal_absent

    def __enter__(self) -> "Patched":
        self.saved = (
            auto_lrc._run_model_helper_command, auto_lrc.default_ctc_python, auto_lrc.analyze_audio,
            auto_lrc.vocal_stem_levels, auto_lrc._ctc_model_blob_sha256,
        )
        helper = self.helper
        absent = self.vocal_absent

        class Levels:
            status = "checked"
            threshold_db = -40.0

            def absent(self, seconds: float) -> bool:
                return absent

        auto_lrc._run_model_helper_command = lambda command, **kwargs: helper(command, **kwargs)  # type: ignore[assignment]
        auto_lrc.default_ctc_python = lambda: Path("python")  # type: ignore[assignment]
        auto_lrc.analyze_audio = lambda path, duration: auto_lrc.AudioFeatures(  # type: ignore[assignment]
            duration, np.zeros(0), np.zeros(0), np.zeros(0), list(SEGMENTS)
        )
        auto_lrc.vocal_stem_levels = lambda report: Levels()  # type: ignore[assignment]
        auto_lrc._ctc_model_blob_sha256 = lambda: "test-model-blob"  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> bool:
        (
            auto_lrc._run_model_helper_command, auto_lrc.default_ctc_python, auto_lrc.analyze_audio,
            auto_lrc.vocal_stem_levels, auto_lrc._ctc_model_blob_sha256,
        ) = self.saved
        return False


def entries() -> list[LyricEntry]:
    return [LyricEntry([text]) for text in LYRICS]


def args(**extra: object):  # type: ignore[no-untyped-def]
    base: dict[str, object] = {"whisperx_device": "cpu", "segment_realign": "auto"}
    base.update(extra)
    return type("Args", (), base)()


def hypotheses(report: dict[str, object], source: str = auto_lrc.SEGMENT_REALIGN_SOURCE) -> dict[int, list[dict]]:
    found: dict[int, list[dict]] = {}
    for index, assignment in enumerate(report["assignments"]):  # type: ignore[index]
        for item in assignment.get("alignment_hypotheses") or []:
            if item["source"] == source:
                found.setdefault(index + 1, []).append(item)
    return found


class SegmentRealignStageTests(unittest.TestCase):
    def test_displaced_chorus_line_gets_a_bounded_pass_and_a_hypothesis(self) -> None:
        times = list(TIMES)
        times[13] = 125.0            # last chorus-2 line placed after the bridge
        report = ctc_report(times, weak={13})
        helper = FakeHelper({LYRICS[13]: 91.0})
        with Patched(helper):
            result = auto_lrc.apply_segment_realign(
                Path("dummy.flac"), entries(), list(times), report, raw_report(times, weak={13}), DURATION, args(),
                window_cache={},
            )
        self.assertEqual(result["status"], "candidates", result)
        self.assertEqual(result["flagged_entries"], [14])
        self.assertEqual(len(helper.calls), 1)
        start, end, transcript = helper.calls[0]
        self.assertEqual(transcript, [LYRICS[12], LYRICS[13]], "the preceding row is CTC context")
        self.assertAlmostEqual(start, 83.0 - 0.05, delta=0.11)
        self.assertAlmostEqual(end, 100.0 - 0.04, delta=0.11)
        found = hypotheses(report)
        self.assertEqual(sorted(found), [14], "the context row gets no hypothesis")
        (item,) = found[14]
        self.assertEqual(item["raw_candidate_time"], 91.0)
        self.assertEqual(item["stage"], auto_lrc.SEGMENT_REALIGN_STAGE)
        self.assertEqual(item["identity_support"], "supported")
        self.assertEqual(item["sequence_support"], "supported")
        self.assertEqual(item["mode"], "run-14-14")
        self.assertEqual(item["parent"]["source"], "ctc-global-mms")
        self.assertTrue(report["assignments"][13]["segment_realign_candidate_generated"])
        self.assertEqual(report["assignments"][13]["timestamp"], 125.0, "nothing is overwritten")
        self.assertEqual(times[13], 125.0)
        self.assertIs(report["segment_realign"], result)
        self.assertEqual(result["runs"][0]["entries"], [14])
        self.assertEqual(result["runs"][0]["helper"], "helper")

    def test_stray_first_token_does_not_set_the_onset(self) -> None:
        times = list(TIMES)
        times[13] = 125.0
        report = ctc_report(times, weak={13})
        helper = FakeHelper({LYRICS[13]: 91.0}, stray={LYRICS[13]: 87.2})   # breath 3.8 s before the line
        with Patched(helper):
            result = auto_lrc.apply_segment_realign(
                Path("dummy.flac"), entries(), list(times), report, raw_report(times, weak={13}), DURATION, args()
            )
        self.assertEqual(result["status"], "candidates", result)
        (item,) = hypotheses(report)[14]
        self.assertEqual(item["raw_candidate_time"], 91.0)
        self.assertEqual(item["mode"], "run-14-14-onset-shifted")
        self.assertEqual(item["spans"][0]["start"], 91.0, "the stray span is dropped")
        self.assertEqual(result["runs"][0]["onset_shifted"], 1)
        # a row whose tokens are dense from the first one keeps its first token
        self.assertEqual(auto_lrc.segment_realign_onset({"start": 5.0, "token_spans": spans(5.0)}), (5.0, spans(5.0), False))
        # a sustained first vowel (one 1.2 s gap, then dense) is not a stray token
        held = [{"char": "a", "start": 5.0, "end": 5.02, "score": 0.4}] + spans(6.2)
        self.assertEqual(auto_lrc.segment_realign_onset({"start": 5.0, "token_spans": held})[0], 5.0)

    def test_consistent_song_makes_no_helper_call(self) -> None:
        report = ctc_report(TIMES)
        helper = FakeHelper({})
        with Patched(helper):
            result = auto_lrc.apply_segment_realign(
                Path("dummy.flac"), entries(), list(TIMES), report, raw_report(TIMES), DURATION, args()
            )
        self.assertEqual(result["status"], "no-runs", result)
        self.assertEqual(result["flagged_entries"], [])
        self.assertEqual(helper.calls, [])
        self.assertEqual(hypotheses(report), {})

    def test_vocal_absent_candidate_is_rejected(self) -> None:
        times = list(TIMES)
        times[13] = 125.0
        report = ctc_report(times, weak={13})
        helper = FakeHelper({LYRICS[13]: 91.0})
        with Patched(helper, vocal_absent=True):
            result = auto_lrc.apply_segment_realign(
                Path("dummy.flac"), entries(), list(times), report, raw_report(times, weak={13}), DURATION, args()
            )
        self.assertEqual(result["status"], "no-candidates")
        self.assertEqual(result["runs"][0]["rejected"][0]["reason"], "vocal-absent")
        self.assertEqual(hypotheses(report), {})

    def test_confirming_onset_is_not_a_candidate(self) -> None:
        # the sibling chorus gap wants this line 1.5 s earlier, but the bounded pass
        # agrees with where the row already is: a false alarm, no hypothesis
        times = list(TIMES)
        times[13] = 92.5
        report = ctc_report(times, weak={13})
        helper = FakeHelper({LYRICS[13]: 92.5})
        with Patched(helper):
            result = auto_lrc.apply_segment_realign(
                Path("dummy.flac"), entries(), list(times), report, raw_report(times, weak={13}), DURATION, args()
            )
        self.assertEqual(result["flagged_entries"], [14], result)
        self.assertEqual(result["status"], "no-candidates")
        self.assertEqual(len(helper.calls), 1)
        (rejected,) = result["runs"][0]["rejected"]
        self.assertEqual(rejected["reason"], "agrees-with-current")
        self.assertEqual(rejected["candidate"], 92.5)
        self.assertEqual(rejected["current"], 92.5)
        self.assertEqual(hypotheses(report), {})
        self.assertNotIn("segment_realign_candidate_generated", report["assignments"][13])
        # the same flag with a pass that disagrees by 1.5 s is a candidate
        report = ctc_report(times, weak={13})
        helper = FakeHelper({LYRICS[13]: 91.0})
        with Patched(helper):
            result = auto_lrc.apply_segment_realign(
                Path("dummy.flac"), entries(), list(times), report, raw_report(times, weak={13}), DURATION, args()
            )
        self.assertEqual(result["status"], "candidates", result)
        self.assertEqual(hypotheses(report)[14][0]["raw_candidate_time"], 91.0)
        self.assertEqual(result["runs"][0]["rejected"], [])

    def test_weak_anchors_leave_the_song_alone(self) -> None:
        weak = set(range(5, len(TIMES)))
        times = list(TIMES)
        times[13] = 125.0
        report = ctc_report(times, weak=weak)
        helper = FakeHelper({LYRICS[13]: 91.0})
        with Patched(helper):
            result = auto_lrc.apply_segment_realign(
                Path("dummy.flac"), entries(), list(times), report, raw_report(times, weak=weak), DURATION, args()
            )
        self.assertEqual(result["status"], "weak-anchors", result)
        self.assertEqual(helper.calls, [])
        self.assertEqual(hypotheses(report), {})

    def test_stage_wrapper_honours_the_off_switch(self) -> None:
        report = ctc_report(TIMES)
        helper = FakeHelper({})
        with Patched(helper):
            auto_lrc.run_segment_realign_stage(
                Path("dummy.flac"), entries(), list(TIMES), report, raw_report(TIMES), DURATION,
                args(segment_realign="off"), {},
            )
        self.assertEqual(report["segment_realign"]["status"], "off")
        self.assertEqual(helper.calls, [])
        os.environ["LRC_SEGMENT_REALIGN"] = "off"
        try:
            self.assertEqual(auto_lrc.segment_realign_mode(args()), "off")
        finally:
            del os.environ["LRC_SEGMENT_REALIGN"]
        self.assertEqual(auto_lrc.segment_realign_mode(args()), "auto")
        self.assertEqual(auto_lrc.segment_realign_mode(type("Args", (), {})()), "off", "off by default until the 2.0 replay is accepted")

    def test_row_status_proxy(self) -> None:
        rows = entries()
        strong = {"ctc_score": 0.5, "ctc_token_spans": spans(10.0)}
        weak = {"ctc_score": 0.02, "ctc_token_spans": spans(10.0)}
        self.assertEqual(auto_lrc.segment_realign_row_status(rows, 0, 10.0, strong, {"timestamp": 10.2, "score": 0.9}), "TRUSTED")
        # a trusted raw onset that disagrees no longer demotes a confident CTC row
        self.assertEqual(auto_lrc.segment_realign_row_status(rows, 0, 10.0, strong, {"timestamp": 12.0, "score": 0.9}), "UNTRUSTED")
        self.assertEqual(
            auto_lrc.segment_realign_row_status(rows, 0, 10.0, strong, {"timestamp": 12.0, "score": 0.9, "borrowed": True}),
            "UNTRUSTED",
        )
        self.assertEqual(auto_lrc.segment_realign_row_status(rows, 0, 10.0, strong, None), "UNTRUSTED")
        quiet = {"ctc_score": 0.12, "ctc_token_spans": spans(10.0)}
        self.assertEqual(auto_lrc.segment_realign_row_status(rows, 0, 10.0, quiet, None), "UNTRUSTED")
        self.assertEqual(auto_lrc.segment_realign_row_status(rows, 0, 10.0, {"ctc_score": 0.10, "ctc_token_spans": spans(10.0)}, None), "REVIEW")
        self.assertEqual(auto_lrc.segment_realign_row_status(rows, 0, 10.0, weak, None), "REVIEW")
        self.assertEqual(auto_lrc.segment_realign_row_status(rows, 0, 10.0, None, None), "REVIEW")


if __name__ == "__main__":
    unittest.main()
