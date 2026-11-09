"""Central timing state test methods, part b (mixed into CentralTimingStateTests)."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from unittest import mock
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import auto_lrc
import split_patch
import audio_features
import central_commit
import ctc_refinement
import local_whisper_retry
import lyrics_io
import runtime_base
from auto_lrc import (
    AnchorHint,
    AudioFeatures,
    LrcError,
    LyricEntry,
    apply_ctc_acoustic_backtrack,
    apply_ctc_crossline_initial_recovery,
    apply_ctc_forward_supported_prefix_recovery,
    apply_final_ctc_timing_guard,
    apply_ctc_weak_prefix_recovery,
    apply_ctc_local_window_realign,
    apply_ctc_local_fusion_to_whisperx,
    apply_ctc_micro_refinement_to_whisperx,
    apply_vocal_onset_tiebreak,
    apply_whisperx_acoustic_boundary_refinement,
    annotate_ctc_boundary_evidence,
    build_parser,
    choose_alignment_candidate,
    choose_onset_consensus_time,
    ctc_forward_supported_prefix_candidate,
    ctc_prefix_boundary_evidence,
    ctc_right_edge_pileup_evidence,
    ctc_path_fracture_evidence,
    duplicate_block_offset_evidence,
    lyric_prefix_probe_text,
    flag_unresolved_ctc_boundary_issues,
    flag_unresolved_raw_ctc_disagreements,
    has_ambiguous_lyric_prefix,
    match_anchor_entries,
    load_lyrics,
    refresh_ctc_confidence_diagnostics,
    raw_asr_is_fallback_eligible,
    remove_generated_title_cards,
    score_line_timing_candidates,
    should_accept_zero_gap_boundary_realign,
    should_accept_ctc_boundary_realign,
    has_ctc_compressed_prefix,
    should_prefer_ctc_over_review_exploded_hybrid,
    split_group_text,
    update_report_confidence_metrics,
    assignment_timing_is_trusted,
)
from evaluate_lrc import summarize
from export_alignment_audit import build_rows, write_markdown
from multimodel_consensus import model_disagreement_review


class _CentralTimingStatePartB:
    def test_sol_r2_fractured_prefix_requires_independent_opening_segment_candidate(self) -> None:
        fractured = self.candidate(
            12.0,
            (*self.coherent_spans(12.0), self.span(16.2)),
            source="ctc-prefix-probe",
            prefix_scope=True,
            confidence=0.95,
        )
        opening = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="ctc-opening-segment",
            confidence=0.8,
            acoustic="supported",
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(fractured, opening),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.selected_candidate_id, opening.candidate_id)
        self.assertIn(
            "temporal-incoherence",
            next(
                item for item in decision.rejected_candidates
                if item.candidate.candidate_id == fractured.candidate_id
            ).rejection_reasons,
        )

    def test_sol_r4_acoustic_peak_change_updates_candidate_and_evidence_revisions(self) -> None:
        first = self.candidate(
            12.0, self.coherent_spans(12.0), acoustic="supported", acoustic_time=12.02
        )
        second = self.candidate(
            12.0, self.coherent_spans(12.0), acoustic="supported", acoustic_time=12.08
        )
        self.assertNotEqual(first.candidate_id, second.candidate_id)
        self.assertNotEqual(first.generation_revision, second.generation_revision)
        self.assertNotEqual(
            auto_lrc.normalize_timing_candidate(first).candidate_revision,
            auto_lrc.normalize_timing_candidate(second).candidate_revision,
        )

    def test_sol_r4_candidate_a_acoustic_peak_cannot_support_candidate_b(self) -> None:
        candidate_a = self.candidate(
            12.0, self.coherent_spans(12.0), acoustic="supported", acoustic_time=12.0
        )
        candidate_b = self.candidate(
            13.0, self.coherent_spans(13.0), acoustic="supported", acoustic_time=12.0
        )
        self.assertTrue(auto_lrc.candidate_has_bound_acoustic_evidence(candidate_a))
        self.assertFalse(auto_lrc.candidate_has_bound_acoustic_evidence(candidate_b))

    def test_sol_r4_missing_peak_provenance_cannot_create_independent_ownership(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="generic",
            source="ctc-prefix-probe",
            raw_time=12.0,
            spans=self.coherent_spans(12.0),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.ownership_independent)

    def test_sol_r6_prefix_source_name_does_not_grant_sequence_support(self) -> None:
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="ctc-prefix-probe",
            prefix_scope=True,
            sequence="unavailable",
            acoustic="supported",
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate)
        self.assertFalse(evaluation.valid)
        self.assertEqual(candidate.sequence_support, "unavailable")

    def test_sol_r6_unavailable_neighbor_and_no_sequence_proof_stay_unresolved(self) -> None:
        previous = self.previous(10.0, pileup=True)
        prefix = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="ctc-prefix-probe",
            prefix_scope=True,
            sequence="unavailable",
            acoustic="supported",
            entry_index=1,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(prefix,),
            previous_candidate=previous,
            selection_revision=2,
        )
        self.assertEqual(decision.status, "provisional_unresolved")

    def test_sol_r7_invalid_prefix_cannot_clear_historical_severe_finding(self) -> None:
        prefix = self.candidate(
            12.0,
            (*self.coherent_spans(12.0), self.span(16.2)),
            source="ctc-prefix-probe",
            prefix_scope=True,
            current=True,
        )
        state = auto_lrc.build_final_timing_state(((prefix,),))
        report: dict[str, object] = {
            "assignments": [{
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary"],
            }]
        }
        auto_lrc.commit_final_timing_state(state, report, {})
        assignment = report["assignments"][0]
        self.assertEqual(assignment["resolved_timing_findings"], [])
        self.assertEqual(
            assignment["historical_legacy_timing_findings"][0]["code"],
            "ctc_unresolved_boundary",
        )
        self.assertFalse(assignment["timing_trusted"])

    def test_sol_r7_active_and_historical_evidence_scopes_do_not_mix(self) -> None:
        candidate = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(3)),
            current=True,
        )
        state = auto_lrc.build_final_timing_state(((candidate,),))
        report: dict[str, object] = {
            "assignments": [{
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary"],
            }]
        }
        auto_lrc.commit_final_timing_state(state, report, {})
        assignment = report["assignments"][0]
        active_historical = [
            item for item in assignment["active_timing_findings"]
            if item["evidence_scope"] == "active-historical-boundary"
        ]
        active_central = [
            item for item in assignment["active_timing_findings"]
            if item["evidence_scope"] == "active-central-evaluator"
        ]
        self.assertEqual(
            [item["code"] for item in active_historical],
            ["ctc_unresolved_boundary"],
        )
        self.assertTrue(active_central)
        self.assertEqual(
            active_historical[0]["finding_id"],
            assignment["historical_legacy_timing_findings"][0]["finding_id"],
        )
        self.assertTrue(all(
            item["evidence_scope"] == "historical-legacy-boundary"
            for item in assignment["historical_legacy_timing_findings"]
        ))

    def test_sol_r7_permutation_preserves_multi_island_rejection_winner_and_trust(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(3)),
            current=True,
        )
        multi_island = self.candidate(
            12.0,
            tuple(self.span(time) for time in (12.0, 15.2, 18.5)),
            source="ctc-prefix-probe",
            prefix_scope=True,
            confidence=0.99,
        )
        valid = self.candidate(
            12.5,
            self.coherent_spans(12.5),
            source="ctc-opening-segment",
            confidence=0.8,
            acoustic="supported",
        )
        outcomes = set()
        for items in permutations((current, multi_island, valid)):
            decision = auto_lrc.select_timing_decision(
                entry_index=0,
                candidates=tuple(items),
                previous_candidate=None,
                selection_revision=1,
            )
            audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
            rejected_multi = next(
                item for item in decision.rejected_candidates
                if item.candidate.candidate_id == multi_island.candidate_id
            )
            outcomes.add((
                decision.selected_candidate_id,
                audit.timing_trusted,
                rejected_multi.evidence.island_count,
                rejected_multi.evidence.temporal_coherence,
            ))
        self.assertEqual(outcomes, {(valid.candidate_id, True, 3, "incoherent")})

    def test_revision_safe_clear_binds_both_candidates_and_tail_revision(self) -> None:
        previous = self.previous(10.0)
        current = self.candidate(12.0, self.coherent_spans(12.0), entry_index=1)
        evidence = auto_lrc.normalize_timing_candidate(current, previous)
        ownership = evidence.ownership
        self.assertEqual(ownership.status, "clear")
        self.assertEqual(ownership.current_candidate_id, current.candidate_id)
        self.assertEqual(ownership.current_generation_revision, current.generation_revision)
        self.assertEqual(ownership.previous_candidate_id, previous.candidate_id)
        self.assertEqual(
            ownership.previous_tail_evidence_revision,
            auto_lrc.derive_reliable_tail(previous).tail_evidence_revision,
        )
        self.assertEqual(
            evidence.referenced_tail_evidence_revision,
            ownership.previous_tail_evidence_revision,
        )

    def test_relation_revision_mismatch_fails_closed(self) -> None:
        previous = self.previous(10.0)
        current = self.candidate(12.0, self.coherent_spans(12.0), entry_index=1)
        regions = auto_lrc.analyze_candidate_regions(current)
        tail = auto_lrc.derive_reliable_tail(previous)
        relation = auto_lrc.classify_candidate_ownership(current, regions, tail)
        stale = replace(relation, previous_tail_evidence_revision="stale")
        evidence = auto_lrc.normalize_timing_candidate(
            current,
            previous,
            current_regions=regions,
            previous_tail=tail,
            ownership_relation=stale,
        )
        self.assertFalse(evidence.minimum_proof_satisfied)
        self.assertIn("ownership-revision-mismatch", evidence.structural_failures)

    def test_tail_pileup_and_truncation_make_ownership_unknown(self) -> None:
        for previous in (
            self.previous(10.0, pileup=True),
            self.candidate(
                9.8,
                (self.span(9.8), self.span(9.9)),
                current=True,
                truncated=True,
            ),
        ):
            with self.subTest(reason=auto_lrc.derive_reliable_tail(previous).reason):
                tail = auto_lrc.derive_reliable_tail(previous)
                self.assertEqual(tail.validity, "invalid")
                evaluation = auto_lrc.evaluate_timing_candidate(
                    self.candidate(12.0, self.coherent_spans(12.0), entry_index=1),
                    previous,
                )
                self.assertEqual(evaluation.evidence.ownership.status, "unknown")
                self.assertFalse(evaluation.valid)

    def test_tail_fracture_preserves_onset_but_invalidates_tail_evidence(self) -> None:
        previous = self.candidate(
            12.0,
            (*self.coherent_spans(12.0), self.span(16.2), self.span(16.3)),
            current=True,
        )
        self.assertTrue(auto_lrc.evaluate_timing_candidate(previous).valid)
        tail = auto_lrc.derive_reliable_tail(previous)
        self.assertEqual(tail.validity, "invalid")
        self.assertEqual(tail.reason, "tail-fracture-invalid")

    def test_e14_like_unknown_fracture_cannot_create_clear_successor(self) -> None:
        previous = self.candidate(
            12.0,
            (self.span(12.0, 0.01), self.span(15.2), self.span(15.3)),
            current=True,
        )
        regions = auto_lrc.analyze_candidate_regions(previous)
        self.assertGreater(len(regions.severe_gap_indexes), 0)
        self.assertEqual(auto_lrc.derive_reliable_tail(previous).validity, "invalid")
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(18.0, self.coherent_spans(18.0), entry_index=1),
            previous,
        )
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertFalse(evaluation.valid)

    def test_historical_onset_failure_does_not_invalidate_valid_tail_relation(self) -> None:
        previous = self.candidate(10.0, self.coherent_spans(10.0), current=True)
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, entry_index=1)
        state = auto_lrc.build_final_timing_state(((previous,), (current,)))
        report = self.report(2)
        report["assignments"][0]["flags"] = ["ctc_unresolved_boundary"]
        effective, _findings = auto_lrc.evaluate_final_timing_state_for_report(state, report)
        self.assertEqual(
            [audit.timing_trusted for audit in effective.audits],
            [False, True],
        )
        self.assertEqual(state.decisions[1].evidence.ownership.status, "clear")

    def test_unknown_without_candidate_bound_onset_remains_unresolved(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="whisperx-word",
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertFalse(evaluation.evidence.ownership_independent)
        self.assertFalse(evaluation.valid)

    def test_sequence_or_confidence_cannot_bypass_unknown_ownership(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            confidence=1.0,
            sequence="supported",
            entry_index=1,
        )
        self.assertFalse(auto_lrc.evaluate_timing_candidate(candidate, previous).valid)

    def test_revision_bound_direct_onset_can_bypass_unknown_ownership(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            direct=True,
            direct_independent=True,
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertTrue(evaluation.evidence.ownership_independent)
        self.assertTrue(evaluation.valid)

    def test_self_bound_direct_onset_cannot_bypass_unknown_ownership(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="whisperx-forced-first",
            direct=True,
            direct_independent=False,
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertFalse(evaluation.evidence.ownership_independent)
        self.assertFalse(evaluation.valid)

    def test_unapproved_direct_onset_producer_cannot_bypass_unknown(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="generic line 1",
            source="whisperx-forced-first",
            raw_time=12.0,
            spans=self.coherent_spans(12.0),
            confidence=1.0,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported",
            direct_onset_time=12.0,
            direct_onset_source_artifact={"claimed": "independent"},
            direct_onset_evidence_producer="whisperx-forced-first",
            direct_onset_evidence_kind="phonetic-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"source": "whisperx-forced-first"},
        )
        self.assertFalse(auto_lrc.candidate_has_bound_direct_onset_evidence(candidate))
        self.assertFalse(auto_lrc.evaluate_timing_candidate(candidate, previous).valid)

    def test_bare_direct_support_without_provenance_cannot_bypass_unknown(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            direct=True,
            direct_bound=False,
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertFalse(evaluation.evidence.ownership_independent)
        self.assertFalse(evaluation.valid)

    def test_unresolved_unknown_line_does_not_poison_clear_successor(self) -> None:
        first = self.previous(10.0, pileup=True)
        second = self.candidate(12.0, self.coherent_spans(12.0), current=True, entry_index=1)
        third = self.candidate(14.0, self.coherent_spans(14.0), current=True, entry_index=2)
        state = auto_lrc.build_final_timing_state(((first,), (second,), (third,)))
        self.assertEqual(
            [audit.timing_trusted for audit in state.audits],
            [False, False, True],
        )
        self.assertEqual(state.decisions[2].evidence.ownership.status, "clear")

    def test_egakumirai_forced_first_5957_cannot_self_certify_over_unknown_tail(self) -> None:
        previous = self.previous(59.20, pileup=True)
        forced = self.candidate(
            59.57,
            (),
            source="whisperx-forced-first",
            confidence=0.58,
            direct=True,
            direct_independent=False,
            entry_index=1,
        )
        selected = self.candidate(
            62.28,
            (),
            source="whisperx-word",
            confidence=0.835,
            direct=False,
            entry_index=1,
        )
        forced_eval = auto_lrc.evaluate_timing_candidate(forced, previous)
        selected_eval = auto_lrc.evaluate_timing_candidate(selected, previous)
        self.assertFalse(forced_eval.valid)
        self.assertFalse(forced_eval.evidence.ownership_independent)
        self.assertFalse(selected_eval.valid)
        state = auto_lrc.build_final_timing_state(((previous,), (forced, selected)))
        self.assertFalse(state.audits[1].timing_trusted)
        self.assertEqual(state.decisions[1].status, "provisional_unresolved")

    def test_generic_acoustic_peak_cannot_replace_phonetic_opening_proof(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            (),
            source="audio-validated-current",
            acoustic="supported",
            acoustic_time=12.0,
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.ownership_independent)

    def test_whisperx_advisory_refinement_never_mutates_baseline(self) -> None:
        entries = [LyricEntry(["first"]), LyricEntry(["second"])]
        baseline = [10.0, 20.0]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {"entry": 2, "timestamp": 20.0, "score": 0.9},
            ]
        }
        asr = [
            auto_lrc.AsrSegment(10.0, 12.0, "first", 0.9),
            auto_lrc.AsrSegment(20.0, 22.0, "second", 0.9),
        ]
        forced = [
            auto_lrc.AsrSegment(9.5, 12.0, "first", 0.9, [("f", 9.5)]),
            auto_lrc.AsrSegment(17.0, 22.0, "second", 0.9, [("s", 17.0)]),
        ]
        refined, changes = auto_lrc.apply_whisperx_lyric_refinement(
            entries, baseline, report, asr, forced, 30.0
        )
        self.assertEqual(refined, baseline)
        self.assertEqual([row["timestamp"] for row in report["assignments"]], baseline)
        self.assertTrue(changes)
        self.assertTrue(report["whisperx_refinement_candidate_only"])
        self.assertFalse(report["whisperx_refinement_mutates_baseline"])

    def test_lyric_identity_contract_rejects_corrupted_anchor_text(self) -> None:
        entries = [LyricEntry(["合成試験の文字列です"])]
        report: dict[str, object] = {
            "suspicious_alignments": [
                {"entry": 1, "anchor_profile": {"entry_text": "合成試験の文宇列です"}}
            ]
        }
        with self.assertRaisesRegex(auto_lrc.LrcError, "Phonetic anchor text diverged"):
            auto_lrc.assert_lyric_identity_contract(entries, report, initialize=True)

    def test_review_state_is_not_an_implicit_output_gate(self) -> None:
        args = auto_lrc.build_parser().parse_args(["song.flac"])
        self.assertFalse(args.strict_review)
        self.assertFalse(args.fail_on_review_required)
        source = auto_lrc.algorithm_source_text()
        self.assertNotIn("Refusing to write untrusted LRC", source)
        self.assertNotIn("blocking untrusted LRC", source)

    def test_production_source_contains_no_song_specific_alignment_literals(self) -> None:
        source = auto_lrc.algorithm_source_text()
        for forbidden in ("Georgette", "エガクミライ", "\\u79c1"):
            self.assertNotIn(forbidden, source)

    def test_forward_and_reverse_worklists_preserve_adjacency_and_result(self) -> None:
        candidates = tuple(
            (
                self.candidate(
                    10.0 + index * 2.0,
                    self.coherent_spans(10.0 + index * 2.0),
                    current=True,
                    entry_index=index,
                ),
            )
            for index in range(4)
        )
        forward = auto_lrc.build_final_timing_state(candidates, worklist_order="forward")
        reverse = auto_lrc.build_final_timing_state(candidates, worklist_order="reverse")
        self.assertEqual(
            [decision.selected_candidate_id for decision in forward.decisions],
            [decision.selected_candidate_id for decision in reverse.decisions],
        )
        self.assertEqual(forward.audits, reverse.audits)
        self.assertEqual(
            [decision.evidence.ownership_relation_revision for decision in forward.decisions],
            [decision.evidence.ownership_relation_revision for decision in reverse.decisions],
        )

    def test_backend_quality_prefers_timing_coverage_to_review_sparsity(self):
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1, backend="synthetic")
        evaluated = auto_lrc.evaluate_backend_timing("synthetic", ((candidate,),), report, {})
        rich = replace(evaluated.summary, timing_trusted_percent=80.0, overall_trusted_percent=40.0)
        sparse = replace(evaluated.summary, timing_trusted_percent=60.0, overall_trusted_percent=60.0)
        self.assertGreater(auto_lrc.central_evaluated_quality({}, rich),
                           auto_lrc.central_evaluated_quality({}, sparse))
        self.assertLess(auto_lrc.central_evaluated_quality({"collapse_detected": True}, rich),
                        auto_lrc.central_evaluated_quality({}, sparse))

    def test_central_backend_rank_overrides_opposite_generation_metrics(self) -> None:
        invalid = self.candidate(
            12.0,
            tuple(self.span(12.0 + index * 0.06, 0.01) for index in range(3)),
            current=True,
        )
        valid = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report_a = self.report(1, backend="backend-a")
        report_a.update({"trusted_percent": 100.0, "review_required_count": 0})
        report_b = self.report(1, backend="backend-b")
        report_b.update({"trusted_percent": 0.0, "review_required_count": 1})
        evaluated_a = auto_lrc.evaluate_backend_timing(
            "backend-a", ((invalid,),), report_a, {}
        )
        evaluated_b = auto_lrc.evaluate_backend_timing(
            "backend-b", ((valid,),), report_b, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated_a, evaluated_b))
        self.assertIs(selection.selected, evaluated_b)
        self.assertGreater(evaluated_b.quality, evaluated_a.quality)

    def test_central_quality_ignores_candidate_selection_record(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        first = self.report(1, backend="synthetic")
        second = self.report(1, backend="synthetic")
        first["candidate_selection"] = {"selected_quality": 100.0}
        second["candidate_selection"] = {"selected_quality": -100.0}
        left = auto_lrc.evaluate_backend_timing("synthetic", ((candidate,),), first, {})
        right = auto_lrc.evaluate_backend_timing("synthetic", ((candidate,),), second, {})
        self.assertEqual(left.quality, right.quality)
        self.assertEqual(left.summary, right.summary)

    def test_preview_commit_reuses_identity_without_rerunning_evaluator(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        original = auto_lrc.evaluate_final_timing_state_for_report
        original_collect = auto_lrc.collect_central_timing_candidates
        original_probe = auto_lrc.apply_final_ctc_timing_guard
        split_patch.assign("evaluate_final_timing_state_for_report", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("re-evaluated")))  # type: ignore[assignment]
        central_commit.collect_central_timing_candidates = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("regenerated"))  # type: ignore[assignment]
        split_patch.assign("apply_final_ctc_timing_guard", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("re-probed")))  # type: ignore[assignment]
        try:
            committed = auto_lrc.commit_evaluated_backend(
                auto_lrc.select_evaluated_backend((evaluated,)), report
            )
        finally:
            split_patch.assign("evaluate_final_timing_state_for_report", original)  # type: ignore[assignment]
            central_commit.collect_central_timing_candidates = original_collect  # type: ignore[assignment]
            split_patch.assign("apply_final_ctc_timing_guard", original_probe)  # type: ignore[assignment]
        self.assertIs(committed, evaluated.state)
        self.assertEqual(report["trusted_percent"], evaluated.summary.overall_trusted_percent)

    def test_commit_rejects_candidate_set_changed_after_preview(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        changed = self.candidate(13.0, self.coherent_spans(13.0), current=True)
        stale = replace(evaluated, candidate_sets=((changed,),))
        with self.assertRaises(LrcError):
            auto_lrc.commit_evaluated_backend(
                auto_lrc.select_evaluated_backend((stale,)), report
            )

    def test_commit_rejects_every_changed_report_evaluation_input_without_writes(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        baseline = self.report(1, backend="ctc")
        baseline.update({
            "ctc_missing_count": 0,
            "ctc_low_score_count": 0,
            "ctc_very_low_score_count": 0,
        })
        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", ((candidate,),), baseline, {"nested": {"probe": [1]}}
        )

        mutations = {
            "severe-flag": lambda report: report["assignments"][0].update(
                {"flags": ["ctc_unresolved_boundary"]}
            ),
            "content-score": lambda report: report["assignments"][0].update(
                {"score": 0.1}
            ),
            "segment": lambda report: report["assignments"][0].update(
                {"segment": {"start": 12.0, "end": 13.0}}
            ),
            "backend": lambda report: report.update({"backend": "whisperx"}),
            "quality-input": lambda report: report.update({"ctc_low_score_count": 1}),
            "assignment-identity": lambda report: report["assignments"][0].update(
                {"entry": 99}
            ),
            "timing-revision": lambda report: report["assignments"][0].update(
                {"timing_evidence_revision": 99}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                report = copy.deepcopy(baseline)
                mutate(report)
                before = copy.deepcopy(report)
                with self.assertRaises(LrcError):
                    auto_lrc.commit_evaluated_backend(
                        auto_lrc.select_evaluated_backend((evaluated,)), report
                    )
                self.assertEqual(report, before)

    def test_post_preview_severe_flag_counterexample_is_rejected(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        report["assignments"][0]["flags"] = ["ctc_unresolved_boundary"]
        before = copy.deepcopy(report)
        with self.assertRaises(LrcError):
            auto_lrc.commit_evaluated_backend(
                auto_lrc.select_evaluated_backend((evaluated,)), report
            )
        self.assertEqual(report, before)
        self.assertNotIn("timing_trusted", report["assignments"][0])

    def test_generation_diagnostics_are_deep_frozen_and_digest_guarded(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        diagnostics: dict[str, object] = {"nested": {"probe": [1, 2]}}
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, diagnostics
        )
        diagnostics["nested"]["probe"].append(3)
        self.assertEqual(
            auto_lrc.thaw_generation_diagnostics(evaluated),
            {"nested": {"probe": [1, 2]}},
        )
        stale = replace(
            evaluated,
            generation_diagnostics_json='{"nested":{"probe":[9]}}',
        )
        before = copy.deepcopy(report)
        with self.assertRaises(LrcError):
            auto_lrc.commit_evaluated_backend(
                auto_lrc.select_evaluated_backend((stale,)), report
            )
        self.assertEqual(report, before)

    def test_nonfinite_projection_and_diagnostics_fail_closed(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        report["assignments"][0]["score"] = float("nan")
        with self.assertRaises(LrcError):
            auto_lrc.evaluate_backend_timing(
                "synthetic", ((candidate,),), report, {}
            )

        clean_report = self.report(1)
        with self.assertRaises(LrcError):
            auto_lrc.evaluate_backend_timing(
                "synthetic",
                ((candidate,),),
                clean_report,
                {"nested": {"probe": float("inf")}},
            )

    def test_commit_writes_only_truthful_central_candidate_selection(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        expected = auto_lrc.central_candidate_selection_payload(selection)
        committed = auto_lrc.commit_evaluated_backend(selection, report)
        self.assertIs(committed, evaluated.state)
        self.assertEqual(report["candidate_selection"], expected)
        self.assertEqual(report["backend"], expected["selected_backend"])

    def test_commit_rejects_incomplete_or_mutated_candidate_selection_without_writes(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        expected = auto_lrc.central_candidate_selection_payload(selection)
        mutations = {
            "incomplete": lambda payload: payload.clear(),
            "backend": lambda payload: payload.update({"selected_backend": "whisperx"}),
            "quality": lambda payload: payload.update({
                "selected_quality": float(payload["selected_quality"]) - 1.0
            }),
            "input-revision": lambda payload: payload.update({"selected_input_revision": "fake"}),
            "evaluation-revision": lambda payload: payload.update({"selected_evaluation_revision": "fake"}),
            "selected-candidate": lambda payload: payload["selected_candidate"].update(
                {"backend": "whisperx"}
            ),
            "alternatives": lambda payload: payload.update({"alternatives": [{"backend": "fake"}]}),
            "reason": lambda payload: payload.update({"selected_reason": "fake selection"}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                attempt = copy.deepcopy(report)
                attempt["candidate_selection"] = copy.deepcopy(expected)
                mutate(attempt["candidate_selection"])
                before = copy.deepcopy(attempt)
                with self.assertRaises(LrcError):
                    auto_lrc.commit_evaluated_backend(selection, attempt)
                self.assertEqual(attempt, before)

    def test_commit_accepts_exact_preexisting_central_selection_payload(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        report["candidate_selection"] = auto_lrc.central_candidate_selection_payload(selection)
        committed = auto_lrc.commit_evaluated_backend(selection, report)
        self.assertIs(committed, evaluated.state)

    def test_central_selected_reason_satisfies_active_powershell_consumer(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1, backend="synthetic")
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        payload = auto_lrc.central_candidate_selection_payload(selection)
        self.assertIn("immutable central evaluation", payload["selected_reason"])
        self.assertIn(evaluated.evaluation_revision[:12], payload["selected_reason"])
        consumer = (
            Path(auto_lrc.__file__).resolve().parents[1] / "align-lrc.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("$report.candidate_selection.selected_reason", consumer)
        self.assertTrue(payload["selected_reason"])

    def test_mocked_auto_and_explicit_previews_run_each_producer_once(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        counts = {
            "anchor": 0,
            "score": 0,
            "duplicate": 0,
            "validation": 0,
            "collect": 0,
        }

        def anchors(_entries, timestamps, _report, _path, _duration):
            counts["anchor"] += 1
            return list(timestamps), []

        def score(_entries, timestamps, _report, _duration):
            counts["score"] += 1
            return list(timestamps)

        def duplicate(_audio, _entries, timestamps, report, _duration):
            counts["duplicate"] += 1
            return list(timestamps), report, []

        def validation(_audio, _entries, _timestamps, _report, _duration):
            counts["validation"] += 1
            return []

        def collect(*_args, **_kwargs):
            counts["collect"] += 1
            return ((candidate,),), {"probe_calls": 1}

        restores = []
        restores.append(split_patch.assign("apply_anchor_hints", anchors))  # type: ignore[assignment]
        restores.append(split_patch.assign("score_line_timing_candidates", score))  # type: ignore[assignment]
        restores.append(split_patch.assign("apply_short_duplicate_acoustic_recovery", duplicate))  # type: ignore[assignment]
        restores.append(split_patch.assign("apply_post_score_audio_validation", validation))  # type: ignore[assignment]
        restores.append(split_patch.assign("collect_central_timing_candidates", collect))  # type: ignore[assignment]
        restores.append(split_patch.assign("ctc_alignment_audio_path", lambda _report, audio: audio))  # type: ignore[assignment]
        try:
            previews = []
            for backend in ("auto-candidate", "explicit-candidate"):
                report = self.report(1, backend=backend)
                _times, preview_report, evaluated, _changes = auto_lrc.prepare_backend_timing_preview(
                    Path("dummy.flac"),
                    [LyricEntry(["generic"])],
                    [12.0],
                    report,
                    20.0,
                    object(),  # type: ignore[arg-type]
                    None,
                )
                previews.append((preview_report, evaluated))
        finally:
            for restore in reversed(restores):
                restore()
        self.assertEqual(counts, {key: 2 for key in counts})
        selection = auto_lrc.select_evaluated_backend(
            tuple(evaluated for _report, evaluated in previews)
        )
        selected_report = next(
            report for report, evaluated in previews
            if evaluated is selection.selected
        )
        committed = auto_lrc.commit_evaluated_backend(
            selection, selected_report
        )
        self.assertIs(committed, selection.selected.state)
        self.assertEqual(counts, {key: 2 for key in counts})

    def test_auto_competition_previews_each_usable_candidate_once(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        counts = {"inference": 0, "preview": 0, "commit": 0}
        preview_backends: list[str] = []
        cache_ids: list[int] = []

        def infer(*_args, **_kwargs):
            counts["inference"] += 1
            return [12.0], self.report(1, backend="ctc"), None

        def generation_candidates(*_args, **_kwargs):
            return [
                {
                    "backend": "ctc",
                    "timestamps": [12.0],
                    "report": self.report(1, backend="ctc"),
                    "error": None,
                },
                {
                    "backend": "whisperx",
                    "timestamps": [12.0],
                    "report": self.report(1, backend="whisperx"),
                    "error": None,
                },
            ]

        def preview(_audio, _entries, timestamps, report, _duration, _args, _anchor, *, retry_helper_cache):
            counts["preview"] += 1
            preview_backends.append(str(report["backend"]))
            self.assertIsNotNone(retry_helper_cache)
            cache_ids.append(id(retry_helper_cache))
            if retry_helper_cache:
                self.assertEqual(retry_helper_cache.get("sentinel"), "first-preview")
            else:
                retry_helper_cache["sentinel"] = "first-preview"
            evaluated = auto_lrc.evaluate_backend_timing(
                str(report["backend"]), ((candidate,),), report, {"anchor_changes": []}
            )
            return list(timestamps), report, evaluated, []

        restores = []
        restores.append(split_patch.assign("default_ctc_ready", lambda: True))  # type: ignore[assignment]
        restores.append(split_patch.assign("default_whisperx_ready", lambda: False))  # type: ignore[assignment]
        restores.append(split_patch.assign("try_ctc_candidate", infer))  # type: ignore[assignment]
        restores.append(split_patch.assign("_central_preview_generation_candidates", generation_candidates))  # type: ignore[assignment]
        restores.append(split_patch.assign("prepare_backend_timing_preview", preview))  # type: ignore[assignment]
        restores.append(split_patch.assign("commit_evaluated_backend", lambda *_args, **_kwargs: counts.update(  # type: ignore[assignment]
            {"commit": counts["commit"] + 1}
        )))
        try:
            # The bare ``object()`` args cannot drive a real WhisperX pass, so the
            # python-availability probe is pinned off regardless of whether the
            # machine running the tests has a .venv-asr.
            with mock.patch.object(auto_lrc, "_whisperx_python_available", return_value=False):
                timestamps, report, backend, selection = auto_lrc.run_auto_backend_competition(
                    Path("dummy.flac"),
                    [LyricEntry(["generic"])],
                    20.0,
                    object(),  # type: ignore[arg-type]
                    None,
                )
        finally:
            for restore in reversed(restores):
                restore()
        self.assertEqual(timestamps, [12.0])
        self.assertEqual(report["backend"], backend)
        self.assertEqual(counts["inference"], 1)
        self.assertEqual(counts["preview"], len(selection.candidates))
        self.assertEqual(preview_backends, ["ctc", "whisperx"])
        self.assertEqual(len(set(cache_ids)), 1)
        self.assertEqual(counts["commit"], 0)
        self.assertIsInstance(selection, auto_lrc.BackendSelection)

    def test_process_auto_reuses_selection_and_explicit_previews_once(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        real_commit = auto_lrc.commit_evaluated_backend

        with TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            audio = temp / "generic.flac"
            lyrics = temp / "generic.txt"
            audio.write_bytes(b"synthetic")
            lyrics.write_text("ordinary lyric line\n", encoding="utf-8")

            for timing_source in ("auto", "ctc"):
                with self.subTest(timing_source=timing_source):
                    report = self.report(1, backend="ctc")
                    evaluated = auto_lrc.evaluate_backend_timing(
                        "ctc", ((candidate,),), report, {"anchor_changes": []}
                    )
                    selection = auto_lrc.select_evaluated_backend((evaluated,))
                    counts = {
                        "backend_inference": 0,
                        "competition": 0,
                        "preview": 0,
                        "commit": 0,
                    }

                    def explicit_inference(*_args, **_kwargs):
                        counts["backend_inference"] += 1
                        return [12.0], report

                    def competition(*_args, **_kwargs):
                        counts["competition"] += 1
                        return [12.0], report, "ctc", selection

                    def preview(*_args, **_kwargs):
                        counts["preview"] += 1
                        return [12.0], report, evaluated, []

                    def commit(selected, selected_report):
                        counts["commit"] += 1
                        return real_commit(selected, selected_report)

                    restores = []
                    restores.append(split_patch.assign("probe_duration", lambda _path: 20.0))  # type: ignore[assignment]
                    restores.append(split_patch.assign("find_anchor_hints", lambda *_args: None))  # type: ignore[assignment]
                    restores.append(split_patch.assign("default_ctc_ready", lambda: True))  # type: ignore[assignment]
                    restores.append(split_patch.assign("default_whisperx_ready", lambda: False))  # type: ignore[assignment]
                    restores.append(split_patch.assign("run_ctc_alignment", explicit_inference))  # type: ignore[assignment]
                    restores.append(split_patch.assign("run_auto_backend_competition", competition))  # type: ignore[assignment]
                    restores.append(split_patch.assign("prepare_backend_timing_preview", preview))  # type: ignore[assignment]
                    restores.append(split_patch.assign("commit_evaluated_backend", commit))  # type: ignore[assignment]
                    restores.append(split_patch.assign("audio_track_labels", lambda _path: ("", "generic")))  # type: ignore[assignment]
                    output = temp / f"{timing_source}.lrc"
                    report_dir = temp / f"{timing_source}-reports"
                    args = build_parser().parse_args([
                        str(audio),
                        "--lyrics", str(lyrics),
                        "--output", str(output),
                        "--report-dir", str(report_dir),
                        "--timing-source", timing_source,
                        "--whisper-language", "en",
                        "--overwrite",
                    ])
                    try:
                        result = auto_lrc.process_audio(audio, args)
                    finally:
                        for restore in reversed(restores):
                            restore()
                    self.assertEqual(result, output.resolve())
                    self.assertEqual(counts["commit"], 1)
                    if timing_source == "auto":
                        self.assertEqual(counts["competition"], 1)
                        self.assertEqual(counts["backend_inference"], 0)
                        self.assertEqual(counts["preview"], 0)
                    else:
                        self.assertEqual(counts["competition"], 0)
                        self.assertEqual(counts["backend_inference"], 1)
                        self.assertEqual(counts["preview"], 1)

    def test_review_boolean_count_percent_and_rows_are_one_contract(self) -> None:
        candidate = self.candidate(
            12.0,
            tuple(self.span(12.0 + index * 0.06, 0.01) for index in range(3)),
            current=True,
        )
        report = self.report(1)
        report["review_required"] = False
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        auto_lrc.commit_evaluated_backend(
            auto_lrc.select_evaluated_backend((evaluated,)), report
        )
        self.assertFalse(report["review_required"])
        self.assertEqual(report["review_required_count"], 0)
        self.assertEqual(report["review_required_percent"], 0.0)
        self.assertFalse(report["assignments"][0]["review_required"])

        actionable = self.report(1)
        actionable["assignments"][0]["flags"] = ["candidate_disagreement"]  # type: ignore[index]
        actionable["suspicious_alignments"] = [{
            "entry": 1,
            "flags": ["candidate_disagreement"],
            "review_required": False,
        }]
        positive_evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), actionable, {}
        )
        auto_lrc.commit_evaluated_backend(
            auto_lrc.select_evaluated_backend((positive_evaluated,)), actionable
        )
        self.assertTrue(actionable["review_required"])
        self.assertEqual(actionable["review_required_count"], 1)
        self.assertEqual(actionable["review_required_percent"], 100.0)
        self.assertTrue(actionable["assignments"][0]["review_required"])

    def test_metric_refresh_repairs_latent_stale_root_review_boolean(self) -> None:
        report: dict[str, object] = {
            "timing_entries": 1,
            "review_required": False,
            "assignments": [{
                "timing_repair": "synthetic",
                "score": 0.9,
                "timestamp_status": "provisional_unresolved",
                "timing_trusted": False,
                "review_required": True,
            }],
        }
        auto_lrc.update_report_confidence_metrics(report)
        self.assertFalse(report["review_required"])
        self.assertEqual(report["review_required_count"], 0)

        actionable = copy.deepcopy(report)
        actionable["assignments"][0]["flags"] = ["previous-tail-overlap"]  # type: ignore[index]
        auto_lrc.update_report_confidence_metrics(actionable)
        self.assertTrue(actionable["review_required"])
        self.assertEqual(actionable["review_required_count"], 1)
        self.assertTrue(actionable["assignments"][0]["review_required"])  # type: ignore[index]

    def test_auto_and_explicit_paths_share_preview_and_single_commit(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        process = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_audio"
        )
        competition = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_auto_backend_competition"
        )
        self.assertTrue(any(
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "prepare_backend_timing_preview"
            for node in ast.walk(process)
        ))
        self.assertTrue(any(
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "prepare_backend_timing_preview"
            for node in ast.walk(competition)
        ))
        commits = [
            node for node in ast.walk(process)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "commit_evaluated_backend"
        ]
        self.assertEqual(len(commits), 1)

    def test_production_source_has_no_fixture_or_phoneme_recovery_branch(self) -> None:
        source = auto_lrc.algorithm_source_text()
        self.assertNotIn("Georgette", source)
        self.assertNotIn("motsuretamama", source)
        self.assertNotIn("romaji.startswith", source)

    def test_no_timestamp_mutation_after_central_finalizer(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        process = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_audio"
        )
        finalizer_call = next(
            node for node in ast.walk(process)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "commit_evaluated_backend"
        )
        late_targets = []
        for node in ast.walk(process):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            if node.lineno <= finalizer_call.lineno:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            late_targets.extend(
                ast.unparse(target) for target in targets
                if "timestamp" in ast.unparse(target)
                or "assignments" in ast.unparse(target)
            )
        self.assertEqual(late_targets, [])

    def _projection_selection(
        self,
        root: Path,
        *,
        include_nonselected: bool = False,
    ) -> tuple[dict[str, object], auto_lrc.BackendSelection]:
        audio = root / "projection.flac"
        audio.write_bytes(b"projection-audio")
        args = build_parser().parse_args([str(audio)])
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1, backend="ctc")
        evaluated = auto_lrc.evaluate_backend_timing("ctc", ((candidate,),), report, {})
        evaluated_items = [evaluated]
        preview_records: list[dict[str, object]] = [
            {"timestamps": [12.0], "report": report, "evaluated": evaluated}
        ]
        generation_candidates: list[dict[str, object]] = [
            {"backend": "ctc", "report": report, "timestamps": [12.0], "error": None}
        ]
        if include_nonselected:
            other = self.report(1, backend="whispercpp")
            other_evaluated = auto_lrc.evaluate_backend_timing(
                "whispercpp", ((candidate,),), other, {}
            )
            evaluated_items.append(other_evaluated)
            preview_records.append({
                "timestamps": [12.0], "report": other, "evaluated": other_evaluated
            })
            generation_candidates.append({
                "backend": "whispercpp", "report": other, "timestamps": [12.0], "error": None
            })
        selection = auto_lrc.select_evaluated_backend(tuple(evaluated_items))
        seed = auto_lrc._build_backend_provenance_projection_seed(
            audio, args, generation_candidates, preview_records, selection
        )
        report[auto_lrc._RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = seed
        auto_lrc.commit_evaluated_backend(selection, report)
        return report, selection

    def test_backend_projection_contains_selected_and_nonselected_capabilities(self) -> None:
        with TemporaryDirectory() as temp_name:
            report, selection = self._projection_selection(Path(temp_name), include_nonselected=True)
        projection = report[auto_lrc._BACKEND_PROVENANCE_PROJECTION_KEY]
        self.assertEqual(projection["selected_backend"], selection.selected.backend)  # type: ignore[index]
        self.assertEqual(len(projection["candidates"]), 2)  # type: ignore[index]
        statuses = {item["selection_status"] for item in projection["candidates"]}  # type: ignore[index]
        self.assertEqual(statuses, {"selected", "nonselected"})
        self.assertEqual(projection["source_audio"]["sha256"]["status"], "RETRIEVED")  # type: ignore[index]
        serialized = json.dumps(projection, ensure_ascii=False)
        self.assertNotIn("projection.flac", serialized)
        self.assertNotIn("temp", serialized.lower())

    def test_backend_projection_digest_and_current_source_binding_fail_closed(self) -> None:
        with TemporaryDirectory() as temp_name:
            report, _selection = self._projection_selection(Path(temp_name))
        projection = report[auto_lrc._BACKEND_PROVENANCE_PROJECTION_KEY]
        projection["selection"]["selected_quality"] = -1.0  # type: ignore[index]
        with self.assertRaises(LrcError):
            auto_lrc._validate_backend_provenance_projection(report)

        with TemporaryDirectory() as temp_name:
            report, _selection = self._projection_selection(Path(temp_name))
        projection = report[auto_lrc._BACKEND_PROVENANCE_PROJECTION_KEY]
        projection["algorithm_source_sha256"] = "stale-source"  # type: ignore[index]
        payload = dict(projection)
        payload.pop("projection_revision", None)
        projection["projection_revision"] = auto_lrc._timing_content_digest(payload)  # type: ignore[index]
        with self.assertRaises(LrcError):
            auto_lrc._validate_backend_provenance_projection(report)

    def test_backend_projection_missing_capability_fields_are_explicit(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "projection.flac"
            audio.write_bytes(b"projection-audio")
            args = build_parser().parse_args([str(audio)])
            with split_patch.patch("_whispercpp_runtime_artifact_hashes",
                return_value=None,
            ):
                summary = auto_lrc._public_backend_capability_summary(
                    audio,
                    args,
                    "whisperx",
                    {"backend": "whisperx", "error": "No usable WhisperX candidate"},
                )
        capability = summary["capability"]
        self.assertEqual(capability["helper_sha256"]["status"], "ABSENT")
        self.assertEqual(capability["package_version"]["status"], "ABSENT")
        self.assertEqual(capability["source_audio_sha256"]["status"], "RETRIEVED")
        self.assertEqual(capability["capability_revision"]["status"], "UNAVAILABLE")
        self.assertIsNone(capability["capability_revision"]["value"])

    def test_writer_preserves_public_projection_and_strips_private_seed(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            report, _selection = self._projection_selection(root)
            report[auto_lrc._RUNTIME_WHISPERX_TEMP_PATHS_KEY] = (
                "C:/temp/private-input.wav",
            )
            path = root / "report.json"
            auto_lrc.write_alignment_report(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertIn(auto_lrc._BACKEND_PROVENANCE_PROJECTION_KEY, text)
        self.assertNotIn(auto_lrc._RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY, text)
        self.assertNotIn("private-input.wav", text)
        self.assertNotIn("_runtime_", text)

    def test_backend_projection_does_not_change_committed_timing_metrics(self) -> None:
        with TemporaryDirectory() as temp_name:
            report, selection = self._projection_selection(Path(temp_name))
        self.assertEqual(report["central_timing_state"][0]["timestamp"], 12.0)
        self.assertEqual(report["overall_trusted_entries"], selection.selected.summary.overall_trusted_entries)
        self.assertEqual(report["review_required_count"], selection.selected.summary.review_required_count)
