from __future__ import annotations

import json
import unittest
from pathlib import Path

import auto_lrc
from auto_lrc import LyricEntry


def span_dicts(start: float, extent: float, count: int = 4) -> list[dict[str, object]]:
    step = extent / count
    return [
        {
            "char": chr(97 + i),
            "start": round(start + step * i, 3),
            "end": round(start + step * i + min(step, 0.02), 3),
            "score": 0.4,
        }
        for i in range(count)
    ]


def ctc_report(times: list[float], extents: list[float], runs: list[list[int]] | None = None) -> dict[str, object]:
    runs = runs or []
    low = sorted({entry for run in runs for entry in run})
    return {
        "backend": "ctc",
        "assignments": [
            {
                "entry": index + 1,
                "timestamp": time,
                "score": 0.9,
                "ctc_score": 0.02 if index + 1 in low else 0.5,
                "timing_repair": "ctc-forced-align",
                "ctc_token_spans": span_dicts(time, extent),
            }
            for index, (time, extent) in enumerate(zip(times, extents))
        ],
        "ctc_low_score_runs": runs,
        "ctc_low_score_entries": [
            {"entry": entry, "text": f"line {entry}", "ctc_score": 0.02} for entry in low
        ],
        "ctc_missing_entries": [],
        "suspicious_alignments": [],
    }


def silent_raw(times: list[float]) -> dict[str, object]:
    # Raw ASR agrees with CTC and is never trusted enough to anchor a window:
    # the shape of the Latin/English passages where nothing else triggers.
    return {
        "assignments": [
            {"entry": index + 1, "timestamp": time, "score": 0.2}
            for index, time in enumerate(times)
        ]
    }


class FakeHelper:
    """Stands in for ctc_align.py; answers each window with fixed onsets."""

    def __init__(self, onsets: dict[int, float] | None = None) -> None:
        self.onsets = onsets or {}
        self.calls: list[tuple[float, float, list[str]]] = []

    def __call__(self, command, **_kwargs):  # type: ignore[no-untyped-def]
        start = float(command[command.index("--start") + 1])
        end = float(command[command.index("--end") + 1])
        transcript = json.loads(
            Path(command[command.index("--transcript") + 1]).read_text(encoding="utf-8")
        )
        self.calls.append((round(start, 3), round(end, 3), transcript))
        rows = []
        for offset, text in enumerate(transcript):
            entry = int(str(text).split()[-1])
            onset = self.onsets.get(entry, start + 0.5 + offset * 0.5)
            rows.append(
                {
                    "start": onset,
                    "ctc_score": 0.6,
                    "token_spans": span_dicts(onset, 1.0),
                    "first_token_candidates": [],
                }
            )
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps({"entries": rows}), encoding="utf-8"
        )

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()


def run_local_window(
    times: list[float],
    report: dict[str, object],
    duration: float,
    helper: FakeHelper,
) -> list[dict[str, object]]:
    entries = [LyricEntry([f"line {index + 1}"]) for index in range(len(times))]
    original_run = auto_lrc.subprocess.run
    original_python = auto_lrc.default_ctc_python
    auto_lrc.subprocess.run = helper  # type: ignore[assignment]
    auto_lrc.default_ctc_python = lambda: Path("python")  # type: ignore[assignment]
    try:
        _, _, changes = auto_lrc.apply_ctc_local_window_realign(
            Path("dummy.flac"),
            entries,
            list(times),
            report,
            silent_raw(times),
            duration,
            type("Args", (), {"whisperx_device": "cpu"})(),
        )
    finally:
        auto_lrc.subprocess.run = original_run  # type: ignore[assignment]
        auto_lrc.default_ctc_python = original_python  # type: ignore[assignment]
    return changes


def local_window_times(report: dict[str, object]) -> dict[int, float]:
    found: dict[int, float] = {}
    for index, assignment in enumerate(report["assignments"]):  # type: ignore[arg-type]
        for item in assignment.get("alignment_hypotheses") or []:
            if item["source"] == "ctc-local-window":
                found[index + 1] = float(item["raw_candidate_time"])
    return found


class SharedCollapseJudgmentTests(unittest.TestCase):
    def test_region_flags_cover_the_collapsed_row_and_its_neighbours(self) -> None:
        self.assertEqual(
            auto_lrc.ctc_collapsed_region_flags([False, False, True, False, False]),
            (False, True, True, True, False),
        )
        self.assertEqual(auto_lrc.ctc_collapsed_region_flags([]), ())

    def test_assignment_and_candidate_views_agree(self) -> None:
        for extent in (0.3, 3.0, 20.0):
            assignment = {"ctc_token_spans": span_dicts(30.0, extent)}
            candidate = auto_lrc.make_timing_candidate(
                entry_index=0,
                entry_text="line 1",
                source="ctc-current",
                raw_time=30.0,
                spans=auto_lrc._timing_spans_from_value(assignment["ctc_token_spans"]),
                confidence=0.3,
                current=True,
            )
            self.assertEqual(
                auto_lrc.assignment_ctc_path_collapsed(assignment),
                auto_lrc.current_ctc_path_collapsed((candidate,)),
                extent,
            )


class CollapsedRegionLocalWindowTests(unittest.TestCase):
    # Row 3 owns a 20 s span (collapsed); rows 2-4 form the region.  Raw ASR is
    # silent and no path score is low, so only the collapse can trigger.
    TIMES = [10.0, 20.0, 22.0, 45.0, 60.0]
    EXTENTS = [2.0, 2.0, 20.0, 2.0, 2.0]
    ONSETS = {1: 10.0, 2: 26.0, 3: 35.0, 4: 47.0}

    def test_collapsed_region_triggers_one_window_bounded_by_sane_rows(self) -> None:
        report = ctc_report(self.TIMES, self.EXTENTS)
        helper = FakeHelper(self.ONSETS)
        changes = run_local_window(self.TIMES, report, 70.0, helper)

        # 50 s > the 45 s split limit, but no row inside the region can bound
        # a half, so the region is aligned whole between rows 1 and 5.
        self.assertEqual(len(helper.calls), 1)
        start, end, transcript = helper.calls[0]
        self.assertAlmostEqual(start, 9.95)
        self.assertAlmostEqual(end, 59.96)
        self.assertEqual(transcript, ["line 1", "line 2", "line 3", "line 4"])
        self.assertEqual(changes[0]["entries"], [2, 3, 4])
        self.assertEqual(local_window_times(report), {2: 26.0, 3: 35.0, 4: 47.0})
        # Candidates only: the backend baseline is untouched.
        self.assertEqual(
            [item["timestamp"] for item in report["assignments"]],  # type: ignore[union-attr]
            self.TIMES,
        )

    def test_sane_path_without_alarms_runs_no_window(self) -> None:
        extents = [2.0] * 5
        report = ctc_report(self.TIMES, extents)
        helper = FakeHelper()
        changes = run_local_window(self.TIMES, report, 70.0, helper)
        self.assertEqual(helper.calls, [])
        self.assertEqual(changes, [])

    def test_runtime_chain_writes_local_window_for_unresolved_region_rows(self) -> None:
        """local window -> Central candidate collection -> evaluated decision."""
        report = ctc_report(self.TIMES, self.EXTENTS)
        run_local_window(self.TIMES, report, 70.0, FakeHelper(self.ONSETS))
        entries = [LyricEntry([f"line {index + 1}"]) for index in range(len(self.TIMES))]
        candidate_sets, diagnostics = auto_lrc.collect_central_timing_candidates(
            Path("dummy.flac"),
            entries,
            list(self.TIMES),
            report,
            70.0,
            type("Args", (), {})(),
        )
        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc",
            candidate_sets,
            report,
            diagnostics,
            arbiter_mode="reviewer-validity",
            timing_source_policy="ctc-first",
        )
        # Every region row reaches Central with a local-window alternative...
        for row, expected in ((1, 26.0), (2, 35.0), (3, 47.0)):
            local = [c for c in candidate_sets[row] if c.source == "ctc-local-window"]
            self.assertEqual(len(local), 1, row)
            self.assertAlmostEqual(local[0].written_time.seconds, expected, places=2)
        self.assertFalse(
            any(c.source == "ctc-local-window" for c in candidate_sets[0] + candidate_sets[4])
        )
        # ...and the collapsed row no longer writes its broken whole-song time.
        decision = evaluated.state.decisions[2]
        self.assertEqual(decision.candidate.source, "ctc-local-window")
        self.assertAlmostEqual(decision.written_time.seconds, 35.0, places=2)


class CollapsedRegionSplitTests(unittest.TestCase):
    def test_wide_run_splits_only_between_sane_rows(self) -> None:
        # Low-score run 2-7; row 6 collapsed so rows 5-7 are the region.  The
        # old midpoint split bounded the first half by row 5's broken time.
        # 8 s spacing stays under the 10 s detached-section rule.
        times = [10.0, 18.0, 26.0, 34.0, 42.0, 44.0, 46.0, 58.0]
        extents = [2.0, 2.0, 2.0, 2.0, 2.0, 20.0, 2.0, 2.0]
        report = ctc_report(times, extents, runs=[[2, 3, 4, 5, 6, 7]])
        helper = FakeHelper()
        run_local_window(times, report, 80.0, helper)
        bounds = [(start, end) for start, end, _ in helper.calls]
        self.assertEqual(bounds, [(9.95, 33.96), (25.95, 57.96)])
        region_bounds = {round(times[i] - 0.04, 3) for i in (4, 5, 6)} | {
            round(times[i] - 0.05, 3) for i in (4, 5, 6)
        }
        for start, end in bounds:
            self.assertNotIn(start, region_bounds)
            self.assertNotIn(end, region_bounds)

    def test_sane_wide_run_keeps_the_midpoint_split(self) -> None:
        times = [10.0, 18.0, 26.0, 34.0, 42.0, 58.0]
        extents = [2.0] * 6
        report = ctc_report(times, extents, runs=[[2, 3, 4, 5]])
        helper = FakeHelper()
        run_local_window(times, report, 80.0, helper)
        splits = report["ctc_local_window_realign"]["split_windows"]  # type: ignore[index]
        self.assertEqual(splits[0]["split_into"], [[2, 3], [4, 5]])

    def test_region_wider_than_the_collapsed_cap_is_skipped(self) -> None:
        times = [10.0, 20.0, 22.0, 45.0, 120.0]
        extents = [2.0, 2.0, 20.0, 2.0, 2.0]
        report = ctc_report(times, extents)
        helper = FakeHelper()
        run_local_window(times, report, 130.0, helper)
        self.assertEqual(helper.calls, [])
        errors = report["ctc_local_window_realign"]["errors"]  # type: ignore[index]
        self.assertTrue(any("over-wide" in str(item) for item in errors))


if __name__ == "__main__":
    unittest.main()
