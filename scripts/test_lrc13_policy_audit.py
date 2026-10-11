"""Regression contracts for reachable N2 timing policy paths."""
from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path
import unittest

import auto_lrc


class TimingPolicyRegressionTests(unittest.TestCase):
    def test_ctc_first_is_default_but_legacy_still_explicit(self) -> None:
        parser = auto_lrc.build_parser()
        self.assertEqual(
            parser.parse_args(["song.flac"]).timing_source_policy, "ctc-first"
        )
        self.assertEqual(
            parser.parse_args(["--timing-source-policy", "default", "song.flac"]).timing_source_policy,
            "default",
        )

    def test_evaluation_retains_actual_policy(self) -> None:
        self.assertIn(
            "timing_source_policy",
            {field.name for field in fields(auto_lrc.EvaluatedBackendTiming)},
        )

    def test_all_reachable_private_replays_inherit_selected_policy(self) -> None:
        tree = ast.parse(Path(auto_lrc.__file__).read_text(encoding="utf-8"))
        replay_calls = [
            call for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "evaluate_backend_timing"
            and (17000 <= call.lineno <= 18000 or 32000 <= call.lineno <= 34000)
        ]
        self.assertEqual(len(replay_calls), 5, "New replay calls must be reviewed")
        for call in replay_calls:
            with self.subTest(line=call.lineno):
                value = next(
                    (kw.value for kw in call.keywords if kw.arg == "timing_source_policy"),
                    None,
                )
                self.assertIsNotNone(value, "Private replay may silently use legacy policy")
                self.assertIsInstance(value, ast.Attribute)
                self.assertEqual(value.attr, "timing_source_policy")
                self.assertIsInstance(value.value, ast.Name)
                self.assertEqual(value.value.id, "selected_eval")


    def test_ctc_evidence_binding_runs_on_actual_generation_path(self) -> None:
        tree = ast.parse(Path(auto_lrc.__file__).read_text(encoding="utf-8"))
        generation = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_auto_backend_competition"
        )
        calls = [
            node for node in ast.walk(generation)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        binders = [
            node for node in calls
            if node.func.id == "bind_ctc_generation_candidate_evidence"
        ]
        self.assertEqual(len(binders), 1)
        self.assertEqual(
            [arg.id for arg in binders[0].args if isinstance(arg, ast.Name)],
            ["generation_candidates", "entries", "whisperx_evidence_candidates"],
        )
        previews = [
            node.lineno for node in calls
            if node.func.id == "prepare_backend_timing_preview"
        ]
        self.assertEqual(len(previews), 1)
        self.assertLess(binders[0].lineno, previews[0])

    def test_ctc_candidate_receives_row_and_tail_evidence_when_seed_is_whisper(self) -> None:
        from unittest import mock
        report = {"backend": "ctc", "assignments": [{"entry": 1, "timestamp": 3.0}]}
        candidate = {"backend": "ctc", "timestamps": [3.0], "report": report}
        entries = [auto_lrc.LyricEntry(["hello"])]
        evidence = [([2.9], {"backend": "whisperx"}, "complete-independent-whisperx")]
        with (
            mock.patch.object(auto_lrc, "bind_runtime_whisperx_row_evidence") as row,
            mock.patch.object(auto_lrc, "bind_runtime_whisperx_ctc_consensus") as consensus,
            mock.patch.object(auto_lrc, "bind_runtime_ctc_assignment_tail_fallback") as tail,
        ):
            auto_lrc.bind_ctc_generation_candidate_evidence([candidate], entries, evidence)
            row.assert_called_once_with(report, entries, evidence)
            consensus.assert_called_once_with(report, entries, [3.0], evidence)
            tail.assert_called_once_with(report, entries)

    def test_ctc_first_retains_whisper_candidates_for_disagreement_evidence(self) -> None:
        ctc = auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="hello", source="ctc-current",
            raw_time=3.0, current=True,
        )
        whisper = auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="hello", source="whisperx-forced",
            raw_time=1.0,
        )
        candidate_sets = ((ctc, whisper),)
        self.assertIs(
            auto_lrc.apply_central_timing_source_policy(candidate_sets, "ctc-first"),
            candidate_sets,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=candidate_sets[0],
            previous_candidate=None,
            selection_revision=1,
            timing_source_policy="ctc-first",
        )
        self.assertEqual(decision.written_time.centiseconds, 300)
        self.assertIn(
            whisper.candidate_id,
            {item.candidate.candidate_id for item in decision.rejected_candidates},
        )

    def test_evaluator_does_not_drop_whisper_evidence_before_central(self) -> None:
        tree = ast.parse(Path(auto_lrc.__file__).read_text(encoding="utf-8"))
        evaluator = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "evaluate_backend_timing"
        )
        self.assertFalse(any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_central_timing_source_policy"
            for node in ast.walk(evaluator)
        ))

    def test_ctc_first_evaluator_retains_all_candidates_but_commits_ctc(self) -> None:
        from test_core_logic import CentralTimingStateTests
        fixture = CentralTimingStateTests()
        ctc = fixture.candidate(
            3.0, fixture.coherent_spans(3.0), source="ctc-full", current=True
        )
        whisper = fixture.candidate(
            1.0, fixture.coherent_spans(1.0), source="whisperx-forced"
        )
        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", ((ctc, whisper),), fixture.report(1, backend="ctc"), {},
            timing_source_policy="ctc-first",
        )
        self.assertEqual(evaluated.candidate_sets[0], (ctc, whisper))
        self.assertEqual(evaluated.state.decisions[0].written_time.centiseconds, 300)
        self.assertIn(
            whisper.candidate_id,
            {item.candidate.candidate_id
             for item in evaluated.state.decisions[0].rejected_candidates},
        )


if __name__ == "__main__":
    unittest.main()