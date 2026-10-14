"""Cross-GPU robustness of the segment-skeleton gates and the vocal-onset trust
leak (sister c06/c09, 2026-10-14).  No BK is read anywhere."""
from __future__ import annotations

import copy
import functools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import auto_lrc
import segment_group_acoustic
import segment_skeleton_production
import test_option_a_skeleton as option_a
from auto_lrc import AudioFeatures, LyricEntry
from segment_asr_corroboration import check_asr_corroboration
from segment_group_acoustic import check_group_placement, placement_fallback_near_edge
from segment_structure import VocalPassage


class PlacementNearEdgeTests(unittest.TestCase):
    def invoke(self, neighbour_score: float) -> dict:
        def run(cmd, *, timeout):
            import json
            out = Path(cmd[cmd.index("--output") + 1])
            transcript = json.loads(Path(cmd[cmd.index("--transcript") + 1]).read_text(encoding="utf8"))
            out.write_text(json.dumps({"entries": [{"ctc_score": neighbour_score} for _ in transcript]}),
                           encoding="utf8")
            return SimpleNamespace(returncode=0, stderr="")
        return check_group_placement(
            rows=[{"ctc_score": .40}] * 3, passage_ids=[0, 1, 2],
            passages=[VocalPassage(0, 1), VocalPassage(2, 3), VocalPassage(4, 5)],
            lyric_texts=["aa", "bb", "cc"], stem=Path("unused-stem"), duration=6.,
            python=Path("unused-python"), helper=Path("unused-helper"),
            device="auto", runner=run)

    def test_verdict_decided_by_gpu_rounding_is_near_edge(self) -> None:
        # .40 vs .38+.02: the c06 situation (0.577823 vs 0.577825).
        for neighbour in (.38, .380001, .379999):
            with self.subTest(neighbour=neighbour):
                result = self.invoke(neighbour)
                self.assertTrue(result["near_edge"])

    def test_clear_verdicts_are_not_near_edge(self) -> None:
        self.assertFalse(self.invoke(.10)["near_edge"])
        self.assertFalse(self.invoke(.60)["near_edge"])

    def test_marginal_group_that_cannot_change_the_rollback_is_not_near_edge(self) -> None:
        groups = [
            {"rows": [1, 10], "supported": False, "decision_margin": -.3},
            {"rows": [11, 11], "supported": True, "decision_margin": .001},
            {"rows": [12, 40], "supported": True, "decision_margin": .3},
        ]
        # 10/40 rows clearly unsupported already exceeds 15%.
        self.assertFalse(placement_fallback_near_edge(groups, 40))
        groups[0]["rows"] = [1, 5]
        groups[1]["rows"] = [6, 6]
        # 5/40 vs 6/40: still both under 15%.
        self.assertFalse(placement_fallback_near_edge(groups, 40))
        groups[1]["rows"] = [6, 7]
        # 5/40 (12.5%) vs 7/40 (17.5%): the marginal group decides.
        self.assertTrue(placement_fallback_near_edge(groups, 40))


class AsrCorroborationNearEdgeTests(unittest.TestCase):
    def test_row_moving_near_the_material_cutoff_marks_near_edge(self) -> None:
        result = check_asr_corroboration(
            lyric_texts=["alpha", "beta"],
            global_times=[1.0, 20.0], skeleton_times=[11.0, 20.22],
            raw_asr_events=[(0., 2., "zzz"), (10., 12., "alpha"), (15., 19.1, "beta")])
        self.assertTrue(result["fallback"])
        self.assertTrue(result["near_edge"])

    def test_clear_corroboration_is_not_near_edge(self) -> None:
        result = check_asr_corroboration(
            lyric_texts=["alpha", "beta", "gamma"],
            global_times=[1., 3., 5.], skeleton_times=[11., 13., 15.],
            raw_asr_events=[(0., 7., "noise"), (10., 12., "alpha"),
                            (12., 14., "beta"), (14., 16., "gamma")])
        self.assertFalse(result["fallback"])
        self.assertFalse(result["near_edge"])


class SkeletonGateOverrideTests(unittest.TestCase):
    def run_case(self, placement: dict, override: str | None):
        real = segment_skeleton_production.run_segment_skeleton
        with tempfile.TemporaryDirectory() as name, \
                mock.patch.object(segment_group_acoustic, "check_group_placement",
                                  return_value=placement), \
                mock.patch.object(option_a, "run_segment_skeleton",
                                  functools.partial(real, gate_override=override)):
            segment_skeleton_production._RUNTIME_CACHE.clear()
            return option_a.OptionAProductionTests.case(None, Path(name))

    @staticmethod
    def placement(*, fallback: bool, near_edge: bool) -> dict:
        return {"status": "OK", "groups": [], "unsupported_rows": [1, 2] if fallback else [],
                "unsupported_fraction": .5 if fallback else 0., "fallback": fallback,
                "near_edge": near_edge}

    def test_near_edge_rollback_reports_the_gate(self) -> None:
        times, _report, status = self.run_case(self.placement(fallback=True, near_edge=True), None)
        self.assertEqual(status["status"], "FALLBACK_CTC")
        self.assertEqual(status["reason"], "GROUP_PLACEMENT_UNSUPPORTED")
        self.assertEqual(status["near_edge_gates"], ["GROUP_PLACEMENT"])
        self.assertEqual(times, [1.2, 2.2, 4.2, 5.2])

    def test_segment_override_keeps_the_skeleton_but_rows_stay_unsupported(self) -> None:
        _times, report, status = self.run_case(
            self.placement(fallback=True, near_edge=True), "segment")
        self.assertEqual(status["status"], "SEGMENT_CURRENT")
        self.assertEqual(status["near_edge_gates"], ["GROUP_PLACEMENT"])
        self.assertTrue(report["assignments"][0]["group_placement_unsupported"])
        self.assertTrue(report["assignments"][1]["group_placement_unsupported"])

    def test_override_never_relaxes_a_clear_rollback(self) -> None:
        _times, _report, status = self.run_case(
            self.placement(fallback=True, near_edge=False), "segment")
        self.assertEqual(status["status"], "FALLBACK_CTC")
        self.assertEqual(status["near_edge_gates"], [])

    def test_fallback_override_rolls_back_a_near_edge_pass(self) -> None:
        times, _report, status = self.run_case(
            self.placement(fallback=False, near_edge=True), "fallback")
        self.assertEqual(status["status"], "FALLBACK_CTC")
        self.assertEqual(status["reason"], "NEAR_EDGE_ALTERNATIVE")
        self.assertEqual(times, [1.2, 2.2, 4.2, 5.2])

    def test_fallback_override_ignored_when_no_gate_is_near_edge(self) -> None:
        _times, _report, status = self.run_case(
            self.placement(fallback=False, near_edge=False), "fallback")
        self.assertEqual(status["status"], "SEGMENT_CURRENT")


class DualPathCompetitionTests(unittest.TestCase):
    def competition(self, outcomes: list[tuple[str, float]], seen: list[object]):
        def run(_audio, _entries, _duration, args, _anchor):
            seen.append(getattr(args, auto_lrc.SEGMENT_GATE_OVERRIDE_ATTR))
            status_name, quality = outcomes[len(seen) - 1]
            status = {"status": status_name, "reason": status_name,
                      "near_edge_gates": ["GROUP_PLACEMENT"]}
            setattr(args, auto_lrc.SEGMENT_GATE_STATUS_ATTR, status)
            report = {"segment_skeleton_status": status, "run": len(seen)}
            selection = SimpleNamespace(selected=SimpleNamespace(quality=quality))
            return [1.0], report, "ctc", selection
        return run

    def call(self, outcomes):
        seen: list[object] = []
        args = SimpleNamespace()
        with mock.patch.object(auto_lrc, "run_auto_backend_competition",
                               side_effect=self.competition(outcomes, seen)):
            result = auto_lrc.run_backend_competition_over_skeleton_gate(
                Path("a.flac"), [], 10.0, args, None)
        return result, seen, args

    def test_better_central_quality_wins(self) -> None:
        result, seen, args = self.call([("SEGMENT_CURRENT", 50.0), ("FALLBACK_CTC", 60.0)])
        self.assertEqual(seen, [None, "fallback"])
        self.assertEqual(result[1]["run"], 2)
        self.assertEqual(result[1]["segment_gate_dual_evaluation"]["chosen_path"], "fallback")
        self.assertIsNone(getattr(args, auto_lrc.SEGMENT_GATE_OVERRIDE_ATTR))

    def test_tie_is_broken_by_path_not_by_run_order(self) -> None:
        first, _seen, _args = self.call([("SEGMENT_CURRENT", 50.0), ("FALLBACK_CTC", 50.0)])
        second, seen, _args = self.call([("FALLBACK_CTC", 50.0), ("SEGMENT_CURRENT", 50.0)])
        self.assertEqual(seen, [None, "segment"])
        self.assertEqual(first[1]["segment_gate_dual_evaluation"]["chosen_path"], "segment")
        self.assertEqual(second[1]["segment_gate_dual_evaluation"]["chosen_path"], "segment")

    def test_clear_gate_runs_once(self) -> None:
        seen: list[object] = []

        def run(_audio, _entries, _duration, args, _anchor):
            seen.append(1)
            setattr(args, auto_lrc.SEGMENT_GATE_STATUS_ATTR,
                    {"status": "SEGMENT_CURRENT", "near_edge_gates": []})
            return [1.0], {}, "ctc", SimpleNamespace(selected=SimpleNamespace(quality=1.0))
        with mock.patch.object(auto_lrc, "run_auto_backend_competition", side_effect=run):
            result = auto_lrc.run_backend_competition_over_skeleton_gate(
                Path("a.flac"), [], 10.0, SimpleNamespace(), None)
        self.assertEqual(len(seen), 1)
        self.assertNotIn("segment_gate_dual_evaluation", result[1])


def _token_spans(start: float, extent: float, count: int = 10) -> list[dict]:
    step = extent / count
    return [{"char": chr(97 + i), "start": round(start + i * step, 3),
             "end": round(start + i * step + .02, 3), "score": .25} for i in range(count)]


class VocalOnsetConfirmationScopeTests(unittest.TestCase):
    def generate(self, report: dict) -> list:
        frame_times = np.arange(0.0, 40.0, 0.02, dtype=np.float32)
        onset = np.zeros_like(frame_times)
        onset[int(round(10.0 / 0.02))] = 1.0
        features = AudioFeatures(duration=40.0, frame_times=frame_times,
                                 rms_db=np.full_like(frame_times, -20.0),
                                 onset_strength=onset, segments=[])
        entries = [LyricEntry(["target line"]), LyricEntry(["second words"]),
                   LyricEntry(["third phrase"])]
        return auto_lrc.apply_vocal_onset_confirmation_hypotheses(
            Path(__file__), entries, [10.0, 20.0, 30.0], report, 40.0, features=features)

    @staticmethod
    def report(second_extent: float = 4.0) -> dict:
        return {"backend": "ctc", "assignments": [
            {"entry": 1, "timestamp": 10.0, "score": .9, "ctc_token_spans": _token_spans(10.0, 4.0)},
            {"entry": 2, "timestamp": 20.0, "score": .9,
             "ctc_token_spans": _token_spans(20.0, second_extent)},
            {"entry": 3, "timestamp": 30.0, "score": .9, "ctc_token_spans": _token_spans(30.0, 4.0)},
        ]}

    def test_control_confirms_the_coherent_row(self) -> None:
        generated = self.generate(self.report())
        self.assertEqual([item["entry"] for item in generated], [1])

    def test_row_next_to_a_collapsed_path_is_not_confirmed(self) -> None:
        report = self.report(second_extent=0.3)
        self.assertEqual(self.generate(report), [])
        # Radius 1, the same region the local-window generator and Central's
        # collapsed fallback use: the collapsed row and both neighbours.
        self.assertEqual(report["vocal_onset_confirmation_skipped"],
                         [{"entry": entry, "reason": "collapsed-ctc-region"} for entry in (1, 2, 3)])

    @staticmethod
    def raw_row(entry: int, raw: float, score: float, *, borrowed: bool = False) -> dict:
        return {"entry": entry, "flags": ["unresolved_raw_ctc_disagreement"],
                "candidate_timestamps": {"raw_asr": raw}, "raw_asr_score": score,
                "raw_asr_borrowed": borrowed}

    def test_clock_after_later_rows_raw_onsets_is_not_confirmed(self) -> None:
        # sister c06 #15: CTC 89.61 while #16-#22 raw sit at 64-88 s.
        report = self.report()
        report["suspicious_alignments"] = [self.raw_row(2, 5.0, .9), self.raw_row(3, 6.0, .6)]
        self.assertEqual(self.generate(report), [])
        self.assertEqual(report["vocal_onset_confirmation_skipped"][0]["contradicting_entries"], [2, 3])

    def test_one_raw_witness_or_weak_or_borrowed_raw_is_not_enough(self) -> None:
        for rows in (
            [self.raw_row(2, 5.0, .9)],
            [self.raw_row(2, 5.0, .9), self.raw_row(3, 6.0, .3)],
            [self.raw_row(2, 5.0, .9), self.raw_row(3, 6.0, .9, borrowed=True)],
        ):
            with self.subTest(rows=rows):
                report = self.report()
                report["suspicious_alignments"] = copy.deepcopy(rows)
                self.assertEqual([item["entry"] for item in self.generate(report)], [1])


class ValidDistantAlternativeTests(unittest.TestCase):
    @staticmethod
    def candidate(time: float, *, source: str, current: bool, confidence: float):
        spans = [auto_lrc.TimingTokenSpan(round(time + i * .2, 3), round(time + i * .2 + .02, 3),
                                          .4, chr(97 + i)) for i in range(12)]
        return auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="target", source=source, raw_time=time,
            spans=tuple(spans), confidence=confidence, current=current,
            identity_support="supported", sequence_support="supported")

    def decision(self, alternative_time: float, alternative_source: str = "ctc-local-window"):
        current = self.candidate(89.61, source="ctc-current", current=True, confidence=.9)
        alternative = self.candidate(alternative_time, source=alternative_source,
                                     current=False, confidence=.3)
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(current, alternative),
            previous_candidate=None, selection_revision=1)
        return decision

    def test_valid_known_lyric_alternative_far_away_is_material(self) -> None:
        decision = self.decision(61.59)
        self.assertEqual(decision.candidate.source, "ctc-current")
        rejected = {item.candidate.source: item for item in decision.rejected_candidates}
        self.assertTrue(rejected["ctc-local-window"].valid)
        self.assertFalse(auto_lrc.candidate_has_bound_direct_onset_evidence(
            rejected["ctc-local-window"].candidate))
        self.assertTrue(auto_lrc._decision_has_material_distant_timing_hypothesis(
            decision, decision.written_time.seconds))

    def test_valid_alternative_within_a_second_is_not_material(self) -> None:
        decision = self.decision(90.30)
        self.assertFalse(auto_lrc._decision_has_material_distant_timing_hypothesis(
            decision, decision.written_time.seconds))


if __name__ == "__main__":
    unittest.main()
