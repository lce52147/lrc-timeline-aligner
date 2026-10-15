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


def run_with_shadows(
    times: list[float],
    report: dict[str, object],
    duration: float,
    helper: FakeHelper,
) -> None:
    entries = [LyricEntry([f"line {index + 1}"]) for index in range(len(times))]
    cache: dict[tuple[int, int, str, str], list[object]] = {}
    args = type("Args", (), {"whisperx_device": "cpu"})()
    original_run = auto_lrc.subprocess.run
    original_python = auto_lrc.default_ctc_python
    auto_lrc.subprocess.run = helper  # type: ignore[assignment]
    auto_lrc.default_ctc_python = lambda: Path("python")  # type: ignore[assignment]
    try:
        auto_lrc.apply_ctc_local_window_realign(
            Path("dummy.flac"), entries, list(times), report, silent_raw(times), duration, args,
            window_cache=cache,
        )
        auto_lrc.run_ctc_local_window_shadows(
            Path("dummy.flac"), entries, list(times), report, silent_raw(times), duration, args, cache,
        )
    finally:
        auto_lrc.subprocess.run = original_run  # type: ignore[assignment]
        auto_lrc.default_ctc_python = original_python  # type: ignore[assignment]


class LocalWindowShadowTests(unittest.TestCase):
    # Row 4 collapsed.  Radius 1 covers rows 3-5, radius 2 rows 2-6, radius 3
    # every row; each radius plans a different window.
    TIMES = [10.0, 18.0, 26.0, 34.0, 50.0, 58.0, 66.0]
    EXTENTS = [2.0, 2.0, 2.0, 20.0, 2.0, 2.0, 2.0]

    def test_radius_widens_the_region(self) -> None:
        flags = [False, False, False, True, False, False, False]
        self.assertEqual(
            auto_lrc.ctc_collapsed_region_flags(flags, radius=2),
            (False, True, True, True, True, True, False),
        )

    def test_shadow_rows_are_recorded_but_never_reach_central(self) -> None:
        report = ctc_report(self.TIMES, self.EXTENTS)
        helper = FakeHelper()
        run_with_shadows(self.TIMES, report, 80.0, helper)

        self.assertEqual(len(helper.calls), 3)
        self.assertEqual(sorted(local_window_times(report)), [3, 4, 5])
        shadow = report["ctc_local_window_shadow"]  # type: ignore[index]
        self.assertEqual(sorted(shadow), ["region-radius-2", "region-radius-3"])
        self.assertEqual(sorted(shadow["region-radius-2"]["rows"]), ["2", "3", "4", "5", "6"])
        self.assertEqual(
            sorted(shadow["region-radius-3"]["rows"], key=int),
            ["1", "2", "3", "4", "5", "6", "7"],
        )
        self.assertFalse(shadow["region-radius-2"]["central_candidate"])
        # The primary record is the radius-1 pass only.
        primary = report["ctc_local_window_realign"]  # type: ignore[index]
        self.assertEqual([item["entries"] for item in primary["windows"]], [[3, 4, 5]])

        entries = [LyricEntry([f"line {index + 1}"]) for index in range(len(self.TIMES))]
        candidate_sets, _ = auto_lrc.collect_central_timing_candidates(
            Path("dummy.flac"), entries, list(self.TIMES), report, 80.0, type("Args", (), {})(),
        )
        local_rows = {
            row + 1
            for row, items in enumerate(candidate_sets)
            if any(item.source == "ctc-local-window" for item in items)
        }
        self.assertEqual(local_rows, {3, 4, 5})

    def test_shadow_reuses_windows_the_primary_pass_aligned(self) -> None:
        report = ctc_report(self.TIMES, self.EXTENTS)
        helper = FakeHelper()
        entries = [LyricEntry([f"line {index + 1}"]) for index in range(len(self.TIMES))]
        cache: dict[tuple[int, int, str, str], list[object]] = {}
        args = type("Args", (), {"whisperx_device": "cpu"})()
        original_run = auto_lrc.subprocess.run
        original_python = auto_lrc.default_ctc_python
        auto_lrc.subprocess.run = helper  # type: ignore[assignment]
        auto_lrc.default_ctc_python = lambda: Path("python")  # type: ignore[assignment]
        try:
            for shadow in (False, True):
                auto_lrc.apply_ctc_local_window_realign(
                    Path("dummy.flac"), entries, list(self.TIMES), report,
                    silent_raw(self.TIMES), 80.0, args,
                    region_radius=1, shadow=shadow, window_cache=cache,
                )
        finally:
            auto_lrc.subprocess.run = original_run  # type: ignore[assignment]
            auto_lrc.default_ctc_python = original_python  # type: ignore[assignment]
        self.assertEqual(len(helper.calls), 1)
        self.assertEqual(
            sorted(report["ctc_local_window_shadow"]["region-radius-1"]["rows"]),  # type: ignore[index]
            ["3", "4", "5"],
        )

    def test_song_without_collapse_runs_no_shadow(self) -> None:
        report = ctc_report(self.TIMES, [2.0] * len(self.TIMES))
        helper = FakeHelper()
        run_with_shadows(self.TIMES, report, 80.0, helper)
        self.assertEqual(helper.calls, [])
        self.assertNotIn("ctc_local_window_shadow", report)

    def test_shadow_failure_is_recorded_not_raised(self) -> None:
        report = ctc_report(self.TIMES, self.EXTENTS)
        good = FakeHelper()
        calls = {"count": 0}

        def flaky(command, **kwargs):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            if calls["count"] > 1:
                raise RuntimeError("helper died")
            return good(command, **kwargs)

        run_with_shadows(self.TIMES, report, 80.0, flaky)  # type: ignore[arg-type]
        shadow = report["ctc_local_window_shadow"]  # type: ignore[index]
        self.assertIn("helper died", shadow["region-radius-2"]["error"])
        self.assertEqual(sorted(local_window_times(report)), [3, 4, 5])


def timing_candidate(
    index: int, source: str, time: float, *, extent: float = 1.0,
    confidence: float = 0.5, current: bool = False,
) -> auto_lrc.TimingCandidate:
    return auto_lrc.make_timing_candidate(
        entry_index=index,
        entry_text=f"line {index + 1}",
        source=source,
        raw_time=time,
        spans=auto_lrc._timing_spans_from_value(span_dicts(time, extent)),
        confidence=confidence,
        identity_support="supported",
        sequence_support="supported",
        current=current,
    )


def wide_candidate(index: int, time: float) -> auto_lrc.TimingCandidate:
    return auto_lrc.make_timing_candidate(
        entry_index=index,
        entry_text=f"line {index + 1}",
        source=auto_lrc.CTC_WIDE_REGION_FALLBACK_SOURCE,
        raw_time=time,
        spans=auto_lrc._timing_spans_from_value(span_dicts(time, 1.0)),
        confidence=0.3,
        sequence_support="supported",
    )


class WideRegionPromotionTests(unittest.TestCase):
    # Row 2's whole-song path spans 20 s (collapsed), so Central leaves it
    # unresolved; rows 1-3 are all inside the wide region.
    def sets(self, wide_time: float, next_time: float = 44.0):
        return (
            (timing_candidate(0, "ctc-current", 10.0, current=True),),
            (
                timing_candidate(1, "ctc-current", 20.0, extent=20.0, confidence=0.02, current=True),
                wide_candidate(1, wide_time),
            ),
            (timing_candidate(2, "ctc-current", next_time, current=True),),
        )

    def test_unresolved_row_takes_the_wide_time_and_stays_unresolved(self) -> None:
        state = auto_lrc.build_final_timing_state(self.sets(15.0))
        decision = state.decisions[1]
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.candidate.source, auto_lrc.CTC_WIDE_REGION_FALLBACK_SOURCE)
        self.assertAlmostEqual(decision.written_time.seconds, 15.0, places=2)
        self.assertFalse(state.audits[1].timing_trusted)

    def test_wide_time_past_the_next_row_is_not_promoted(self) -> None:
        state = auto_lrc.build_final_timing_state(self.sets(50.0, next_time=44.0))
        self.assertEqual(state.decisions[1].candidate.source, "ctc-current")
        self.assertAlmostEqual(state.decisions[1].written_time.seconds, 20.0, places=2)

    def test_a_small_shift_keeps_the_provisional_time(self) -> None:
        state = auto_lrc.build_final_timing_state(self.sets(20.3))
        self.assertEqual(state.decisions[1].candidate.source, "ctc-current")

    def test_wide_candidate_never_beats_a_valid_row(self) -> None:
        sets = (
            (timing_candidate(0, "ctc-current", 10.0, current=True), wide_candidate(0, 8.0)),
            *self.sets(15.0)[1:],
        )
        state = auto_lrc.build_final_timing_state(sets)
        self.assertEqual(state.decisions[0].status, "selected_valid")
        self.assertEqual(state.decisions[0].candidate.source, "ctc-current")

    def test_row_outside_the_wide_region_is_not_promoted(self) -> None:
        sets = self.sets(15.0)
        # Decisions as Central reaches them before any promotion.
        without_wide = tuple(
            tuple(c for c in items if c.source != auto_lrc.CTC_WIDE_REGION_FALLBACK_SOURCE)
            for items in sets
        )
        decisions = list(auto_lrc.build_final_timing_state(without_wide).decisions)
        self.assertEqual(
            auto_lrc._ctc_wide_region_promotions(sets, decisions, (False, False, False)), {}
        )
        forced = auto_lrc._ctc_wide_region_promotions(sets, decisions, (False, True, False))
        self.assertEqual(list(forced), [1])


class WideRegionAdmissionTests(unittest.TestCase):
    def report_with(self, hypotheses: list[dict[str, object]], **extra: object) -> dict[str, object]:
        assignment: dict[str, object] = {"entry": 1, "timestamp": 20.0}
        for item in hypotheses:
            auto_lrc.record_alignment_hypothesis(
                assignment,
                source=auto_lrc.CTC_WIDE_REGION_FALLBACK_SOURCE,
                stage="ctc-local-window-shadow",
                raw_time=float(item["time"]),
                confidence=0.3,
                sequence_support="supported",
                mode=f"region-radius-{item['radius']}",
            )
        return {"assignments": [assignment], **extra}

    def admit(self, report: dict[str, object], candidates=None):
        candidates = candidates or (timing_candidate(0, "ctc-current", 20.0, current=True),)
        entries = [LyricEntry(["line 1"])]
        sets, rows = auto_lrc._attach_ctc_wide_region_candidates((tuple(candidates),), report, entries, ())
        wide = [c for c in sets[0] if c.source == auto_lrc.CTC_WIDE_REGION_FALLBACK_SOURCE]
        return wide, rows

    def test_radius_two_agreement_admits_the_radius_three_time(self) -> None:
        wide, rows = self.admit(self.report_with([{"radius": 3, "time": 15.0}, {"radius": 2, "time": 15.2}]))
        self.assertEqual(len(wide), 1)
        self.assertAlmostEqual(wide[0].written_time.seconds, 15.0, places=2)
        self.assertTrue(rows[0]["admitted"])

    def test_non_posterior_support_admits_without_a_peer(self) -> None:
        candidates = (
            timing_candidate(0, "ctc-current", 20.0, current=True),
            timing_candidate(0, "ctc-opening-local-retry", 15.05),
        )
        wide, _ = self.admit(self.report_with([{"radius": 3, "time": 15.0}]), candidates)
        self.assertEqual(len(wide), 1)

    def test_posterior_support_alone_is_not_agreement(self) -> None:
        candidates = (
            timing_candidate(0, "ctc-current", 20.0, current=True),
            timing_candidate(0, "ctc-first-token-posterior", 15.05),
        )
        wide, rows = self.admit(
            self.report_with([{"radius": 3, "time": 15.0}, {"radius": 2, "time": 17.0}]), candidates
        )
        self.assertEqual(wide, [])
        self.assertFalse(rows[0]["admitted"])

    def test_contradicting_recurrence_onset_vetoes(self) -> None:
        report = self.report_with(
            [{"radius": 3, "time": 15.0}, {"radius": 2, "time": 15.0}],
            mutual_acoustic_recurrence_hypotheses=[{"entry": 1, "candidate": 20.2}],
        )
        wide, rows = self.admit(report)
        self.assertEqual(wide, [])
        self.assertTrue(rows[0]["vetoed"])

    def test_agreeing_recurrence_onset_does_not_veto(self) -> None:
        report = self.report_with(
            [{"radius": 3, "time": 15.0}, {"radius": 2, "time": 15.0}],
            mutual_acoustic_recurrence_hypotheses=[{"entry": 1, "candidate": 15.3}],
        )
        wide, _ = self.admit(report)
        self.assertEqual(len(wide), 1)

    def test_shadow_records_wide_hypotheses_that_only_reach_central_gated(self) -> None:
        times = [10.0, 18.0, 26.0, 34.0, 50.0, 58.0, 66.0]
        extents = [2.0, 2.0, 2.0, 20.0, 2.0, 2.0, 2.0]
        report = ctc_report(times, extents)
        run_with_shadows(times, report, 80.0, FakeHelper({4: 40.0, 6: 60.0}))
        modes = {
            index + 1: sorted(
                item["mode"] for item in assignment.get("alignment_hypotheses") or []
                if item["source"] == auto_lrc.CTC_WIDE_REGION_FALLBACK_SOURCE
            )
            for index, assignment in enumerate(report["assignments"])  # type: ignore[arg-type]
        }
        self.assertEqual(modes[1], ["region-radius-3"])
        self.assertEqual(modes[4], ["region-radius-2", "region-radius-3"])
        entries = [LyricEntry([f"line {index + 1}"]) for index in range(len(times))]
        candidate_sets, diagnostics = auto_lrc.collect_central_timing_candidates(
            Path("dummy.flac"), entries, list(times), report, 80.0, type("Args", (), {})(),
        )
        for items in candidate_sets:
            self.assertLessEqual(
                sum(item.source == auto_lrc.CTC_WIDE_REGION_FALLBACK_SOURCE for item in items), 1
            )
        rows = diagnostics["ctc_wide_region_fallback"]["rows"]
        self.assertEqual([row["entry"] for row in rows], list(range(1, 8)))
        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", candidate_sets, report, diagnostics,
            arbiter_mode="reviewer-validity", timing_source_policy="ctc-first",
        )
        # Valid rows keep their own clocks.
        for decision in evaluated.state.decisions:
            if decision.status == "selected_valid":
                self.assertNotEqual(decision.candidate.source, auto_lrc.CTC_WIDE_REGION_FALLBACK_SOURCE)


if __name__ == "__main__":
    unittest.main()
