"""Central timing state test methods, part a (mixed into CentralTimingStateTests)."""

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


class _CentralTimingStatePartA:
    def span(self, start: float, score: float = 0.20, token: str = "x") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(
        self,
        time: float,
        spans: tuple[auto_lrc.TimingTokenSpan, ...],
        *,
        source: str = "ctc-full",
        current: bool = False,
        confidence: float = 0.80,
        acoustic: auto_lrc.EvidenceState = "unavailable",
        acoustic_time: float | None = None,
        sequence: auto_lrc.EvidenceState = "supported",
        prefix_scope: bool = False,
        pileup: bool = False,
        truncated: bool = False,
        direct: bool = False,
        direct_bound: bool = True,
        direct_independent: bool = False,
        entry_index: int = 0,
    ) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry_index,
            entry_text=f"generic line {entry_index}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=confidence,
            identity_support="supported",
            sequence_support=sequence,
            acoustic_support=acoustic,
            acoustic_strength=0.8 if acoustic == "supported" else None,
            acoustic_onset_time=(
                time if acoustic == "supported" and acoustic_time is None
                else acoustic_time
            ),
            acoustic_source_artifact=(
                {
                    "source": "synthetic-acoustic",
                    "time": time if acoustic_time is None else acoustic_time,
                    "strength": 0.8,
                }
                if acoustic == "supported"
                else None
            ),
            acoustic_evidence_producer=(
                "synthetic-independent-acoustic" if acoustic == "supported" else None
            ),
            acoustic_evidence_kind="generic-acoustic",
            acoustic_evidence_independent=bool(acoustic == "supported"),
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct and direct_bound else None,
            direct_onset_source_artifact=(
                {"source": "synthetic-direct", "time": time}
                if direct and direct_bound
                else None
            ),
            direct_onset_evidence_producer=(
                "phonetic-onset-fusion" if direct and direct_independent else None
            ),
            direct_onset_evidence_kind=(
                "phonetic-onset" if direct and direct_independent else "generic-onset"
            ),
            direct_onset_evidence_independent=bool(direct and direct_independent),
            current=current,
            prefix_scope=prefix_scope,
            right_edge_pileup=pileup,
            window_truncated=truncated,
            source_artifact={"source": source, "time": time},
        )

    def previous(self, end: float = 10.0, *, pileup: bool = False) -> auto_lrc.TimingCandidate:
        return self.candidate(
            end - 0.26,
            (
                self.span(end - 0.26),
                self.span(end - 0.18),
                self.span(end - 0.10),
            ),
            current=True,
            pileup=pileup,
            entry_index=0,
        )

    def coherent_spans(self, start: float) -> tuple[auto_lrc.TimingTokenSpan, ...]:
        return (
            self.span(start, 0.20, "a"),
            self.span(start + 0.08, 0.22, "b"),
            self.span(start + 0.16, 0.18, "c"),
        )

    def report(self, count: int, *, backend: str = "synthetic") -> dict[str, object]:
        return {
            "backend": backend,
            "assignments": [
                {"timing_repair": "synthetic", "score": 0.9}
                for _ in range(count)
            ],
            "timing_entries": count,
        }

    def test_canonical_written_time_recomputes_at_centisecond(self) -> None:
        candidate = self.candidate(59.996, self.coherent_spans(60.0))
        evidence = auto_lrc.normalize_timing_candidate(candidate)
        self.assertEqual(candidate.written_time.centiseconds, 6000)
        self.assertEqual(candidate.written_time.lrc_tag, "01:00.00")
        self.assertEqual(evidence.computed_for_centiseconds, 6000)

    def test_previous_tail_overlap_is_structurally_invalid(self) -> None:
        previous = self.previous(10.0)
        candidate = self.candidate(9.90, self.coherent_spans(9.90), entry_index=1)
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertFalse(evaluation.valid)
        self.assertIn("previous-tail-overlap", evaluation.rejection_reasons)

    def test_20ms_weak_prefix_is_not_timing_proof(self) -> None:
        previous = self.previous(10.0)
        spans = tuple(self.span(10.02 + offset * 0.06, 0.01) for offset in range(4))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(10.02, spans, entry_index=1), previous
        )
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.minimum_proof_satisfied)

    def test_60_to_150ms_weak_prefix_is_not_timing_proof(self) -> None:
        previous = self.previous(10.0)
        for gap in (0.06, 0.10, 0.15):
            spans = tuple(self.span(10.0 + gap + offset * 0.06, 0.01) for offset in range(5))
            evaluation = auto_lrc.evaluate_timing_candidate(
                self.candidate(10.0 + gap, spans, entry_index=1), previous
            )
            self.assertFalse(evaluation.valid, gap)

    def test_strong_onset_survives_tail_fracture(self) -> None:
        spans = (*self.coherent_spans(12.0), self.span(16.5), self.span(16.6))
        evaluation = auto_lrc.evaluate_timing_candidate(self.candidate(12.0, spans))
        self.assertTrue(evaluation.valid)
        self.assertIn(evaluation.evidence.fracture_region, {"middle", "tail"})

    def test_weak_token_zero_with_supported_following_run_can_prove_onset(self) -> None:
        spans = (
            self.span(12.0, 0.01),
            self.span(12.08, 0.20),
            self.span(12.16, 0.18),
        )
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, acoustic="supported")
        )
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.longest_supported_run, 2)

    def test_prefix_stretched_several_seconds_is_rejected(self) -> None:
        spans = (
            self.span(12.0), self.span(12.1), self.span(12.2),
            self.span(17.0), self.span(17.1),
        )
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, source="ctc-prefix-probe", prefix_scope=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("temporal-incoherence", evaluation.rejection_reasons)

    def test_multi_second_internal_gap_is_rejected_for_prefix_candidate(self) -> None:
        spans = (self.span(20.0), self.span(20.1), self.span(24.0), self.span(24.1))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(20.0, spans, source="ctc-prefix-probe", prefix_scope=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertGreater(float(evaluation.evidence.max_internal_gap or 0.0), 3.0)

    def test_right_edge_pileup_never_commits(self) -> None:
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, self.coherent_spans(12.0), pileup=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("right-edge-pileup", evaluation.rejection_reasons)

    def test_window_truncation_never_commits(self) -> None:
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, self.coherent_spans(12.0), truncated=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("window-truncated", evaluation.rejection_reasons)

    def test_timestamp_evidence_revision_mismatch_fails_final_audit(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(candidate,),
            previous_candidate=None,
            selection_revision=1,
        )
        stale = replace(decision, timing_evidence_revision=0)
        audit = auto_lrc.audit_final_timing_decisions((stale,))[0]
        self.assertFalse(audit.timing_trusted)
        self.assertIn("revision-mismatch", audit.failures)

    def test_valid_repair_candidate_commits_with_provenance(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        repair = self.candidate(
            12.8,
            self.coherent_spans(12.8),
            source="ctc-prefix-probe",
            confidence=0.90,
            acoustic="supported",
            prefix_scope=True,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, repair),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertEqual(decision.selected_candidate_id, repair.candidate_id)
        self.assertIsNotNone(decision.recovery)
        self.assertIn("candidate-dominates-current", decision.recovery.passed_invariants)  # type: ignore[union-attr]

    def test_no_valid_candidate_stays_unresolved(self) -> None:
        candidate = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
        )
        state = auto_lrc.build_final_timing_state(((candidate,),))
        self.assertEqual(state.decisions[0].status, "provisional_unresolved")
        self.assertFalse(state.audits[0].timing_trusted)

    def test_late_fracture_does_not_destroy_onset_proof(self) -> None:
        spans = (*self.coherent_spans(30.0), self.span(34.5), self.span(34.6))
        evidence = auto_lrc.normalize_timing_candidate(self.candidate(30.0, spans))
        self.assertEqual(evidence.temporal_coherence, "coherent")
        self.assertNotIn("temporal-incoherence", evidence.structural_failures)

    def test_weaker_new_candidate_cannot_override_valid_current(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.90)
        weaker = self.candidate(
            12.5, self.coherent_spans(12.5), source="ctc-local", confidence=0.60
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(weaker, current),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.selected_candidate_id, current.candidate_id)

    def test_current_replacement_diagnostics_preserve_decision_and_vectors(self):
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=.8)
        challenger = self.candidate(12.5, self.coherent_spans(12.5), source="ctc-local", confidence=.9)
        kwargs = dict(entry_index=0, candidates=(current, challenger),
                      previous_candidate=None, selection_revision=1)
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "events.jsonl"
            with mock.patch.dict(auto_lrc.os.environ, {}, clear=True):
                disabled = auto_lrc.select_timing_decision(**kwargs)
            with mock.patch.dict(auto_lrc.os.environ, {"LRC_CURRENT_REPLACEMENT_DIAGNOSTICS": str(path)}):
                enabled = auto_lrc.select_timing_decision(**kwargs)
            self.assertEqual(disabled, enabled)
            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(event["path"], "dominance-over-valid-current")
            self.assertEqual(len(event["current"]["vector"]), 5)
            self.assertEqual(len(event["challenger"]["vector"]), 5)
            self.assertEqual(event["challenger"]["candidate_id"], challenger.candidate_id)

    def test_gross_rescue_preserves_valid_current_for_subthreshold_replacement(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.80)
        challenger = self.candidate(
            12.5,
            self.coherent_spans(12.5),
            source="ctc-local",
            confidence=0.90,
            acoustic="supported",
        )
        baseline = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, challenger),
            previous_candidate=None,
            selection_revision=1,
        )
        rescued = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, challenger),
            previous_candidate=None,
            selection_revision=1,
            arbiter_mode="gross-rescue",
        )
        self.assertEqual(baseline.selected_candidate_id, challenger.candidate_id)
        self.assertEqual(rescued.selected_candidate_id, current.candidate_id)
        self.assertIsNotNone(rescued.arbiter)
        self.assertTrue(rescued.arbiter.use_current)  # type: ignore[union-attr]

    def test_gross_rescue_keeps_large_non_asr_central_replacement(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.80)
        challenger = self.candidate(
            12.6,
            self.coherent_spans(12.6),
            source="ctc-local",
            confidence=0.90,
            acoustic="supported",
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, challenger),
            previous_candidate=None,
            selection_revision=1,
            arbiter_mode="gross-rescue",
        )
        self.assertEqual(decision.selected_candidate_id, challenger.candidate_id)
        self.assertIsNotNone(decision.arbiter)
        self.assertFalse(decision.arbiter.use_current)  # type: ignore[union-attr]

    def test_gross_rescue_trace_is_committed_per_row(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.80)
        challenger = self.candidate(
            12.5,
            self.coherent_spans(12.5),
            source="ctc-local",
            confidence=0.90,
            acoustic="supported",
        )
        state = auto_lrc.build_final_timing_state(
            ((current, challenger),), arbiter_mode="gross-rescue"
        )
        report = self.report(1)
        committed = auto_lrc.commit_final_timing_state(state, report, {})
        trace = report["assignments"][0]["arbiter"]
        self.assertEqual(committed, state)
        self.assertEqual(trace["mode"], "gross-rescue")
        self.assertTrue(trace["reverted_to_current"])
        self.assertEqual(trace["selected_source"], "ctc-local")
        self.assertEqual(trace["shift_ms"], 500)

    def test_cli_arbiter_defaults_off_and_accepts_gross_rescue(self) -> None:
        parser = auto_lrc.build_parser()
        self.assertEqual(parser.parse_args(["song.flac"]).arbiter, "off")
        self.assertEqual(
            parser.parse_args(["song.flac", "--arbiter", "gross-rescue"]).arbiter,
            "gross-rescue",
        )

    def test_cli_accepts_reviewer_validity_and_evidence_path(self) -> None:
        parser = auto_lrc.build_parser()
        args = parser.parse_args(["song.flac", "--arbiter", "reviewer-validity", "--reviewer-evidence", "sidecar.json"])
        self.assertEqual(args.arbiter, "reviewer-validity")
        self.assertEqual(str(args.reviewer_evidence), "sidecar.json")

    def test_reviewer_sidecar_missing_falls_back_to_empty_rows(self) -> None:
        with TemporaryDirectory() as temp_name:
            rows = auto_lrc.load_reviewer_validity_evidence(Path(temp_name) / "missing.json", 2)
        self.assertEqual(rows, (None, None))

    def test_reviewer_sidecar_maps_entries_and_shared_policy(self) -> None:
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "sidecar.json"
            path.write_text(json.dumps({
                "schema": 1, "policy": "R2b",
                "reviewers": ["HUBP", "WX", "XLSR"], "required_agreements": 1,
                "offsets": {"HUBP": 0.02}, "offset_sample_counts": {"HUBP": 12},
                "rows": [{"entry": 2, "expected_current_seconds": 12.0, "reviewer_times": {"HUBP": 12.02}}],
            }), encoding="utf-8")
            rows = auto_lrc.load_reviewer_validity_evidence(path, 2)
        self.assertIsNone(rows[0])
        self.assertEqual(rows[1]["expected_current_seconds"], 12.0)
        self.assertEqual(rows[1]["offsets"], {"HUBP": 0.02})
        self.assertEqual(rows[1]["required_agreements"], 1)

    def test_reviewer_evidence_binds_into_generation_diagnostics_and_round_trips(self) -> None:
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "sidecar.json"
            path.write_text(json.dumps({
                "schema": 1, "policy": "R2b",
                "reviewers": ["HUBP", "WX", "XLSR"], "required_agreements": 1,
                "offsets": {"HUBP": 0.02}, "offset_sample_counts": {"HUBP": 8},
                "rows": [{"entry": 1, "expected_current_seconds": 12.0, "reviewer_times": {"HUBP": 12.02}}],
            }), encoding="utf-8")
            args = mock.Mock(arbiter="reviewer-validity", reviewer_evidence=path)
            diagnostics = auto_lrc.bind_reviewer_validity_diagnostics({}, args, 1)
        rows = auto_lrc.reviewer_validity_evidence_from_diagnostics(diagnostics, 1)
        self.assertEqual(rows[0]["expected_current_seconds"], 12.0)
        self.assertEqual(rows[0]["offsets"], {"HUBP": 0.02})

    def reviewer_validity_row(self, current_seconds: float) -> dict[str, object]:
        return {
            "expected_current_seconds": current_seconds,
            "reviewer_times": {"HUBP": current_seconds + 0.02, "WX": None, "XLSR": None},
            "offsets": {"HUBP": 0.02},
            "reviewers": ["HUBP", "WX", "XLSR"],
            "required_agreements": 1,
        }

    def test_reviewer_validity_endorsement_promotes_only_soft_invalid_current(self) -> None:
        current = self.candidate(12.0, tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)), current=True)
        baseline = auto_lrc.build_final_timing_state(((current,),))
        self.assertFalse(baseline.audits[0].timing_trusted)
        endorsed = auto_lrc.build_final_timing_state(((current,),), arbiter_mode="reviewer-validity", reviewer_evidence=(self.reviewer_validity_row(12.0),))
        self.assertEqual(endorsed.decisions[0].selected_candidate_id, current.candidate_id)
        self.assertEqual(endorsed.decisions[0].status, "selected_valid")
        self.assertTrue(endorsed.audits[0].timing_trusted)

    def test_reviewer_validity_endorsed_soft_invalid_current_survives_ordinary_challenger(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        challenger = self.candidate(
            12.6,
            self.coherent_spans(12.6),
            source="ctc-local",
            confidence=0.95,
            acoustic="supported",
        )
        baseline = auto_lrc.build_final_timing_state(((current, challenger),))
        self.assertEqual(baseline.decisions[0].selected_candidate_id, challenger.candidate_id)

        endorsed = auto_lrc.build_final_timing_state(
            ((current, challenger),),
            arbiter_mode="reviewer-validity",
            reviewer_evidence=(self.reviewer_validity_row(12.0),),
        )
        self.assertEqual(endorsed.decisions[0].selected_candidate_id, current.candidate_id)
        self.assertEqual(endorsed.decisions[0].status, "selected_valid")
        self.assertTrue(endorsed.audits[0].timing_trusted)

    def test_reviewer_validity_endorsed_soft_invalid_current_yields_to_independent_consensus(self) -> None:
        current = self.candidate(
            12.69,
            tuple(self.span(12.69 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        ctc_whisper_consensus = self.candidate(
            12.22,
            self.coherent_spans(12.22),
            source="ctc-local-whisper-vocal-independent-consensus",
            confidence=0.85,
            acoustic="supported",
        )
        direct_whisper = self.candidate(
            12.26,
            self.coherent_spans(12.26),
            source="local-whisper-vocal-independent-fusion",
            confidence=0.88,
            acoustic="supported",
            direct=True,
            direct_independent=True,
        )

        endorsed = auto_lrc.build_final_timing_state(
            ((current, ctc_whisper_consensus, direct_whisper),),
            arbiter_mode="reviewer-validity",
            reviewer_evidence=({
                "expected_current_seconds": 12.69,
                "reviewer_times": {"HUBP": 12.26, "WX": 12.27, "XLSR": 12.69},
                "offsets": {"HUBP": 0.0, "WX": 0.0, "XLSR": 0.0},
                "reviewers": ["HUBP", "WX", "XLSR"],
                "required_agreements": 1,
            },),
        )

        self.assertEqual(
            endorsed.decisions[0].selected_candidate_id,
            direct_whisper.candidate_id,
        )
        self.assertEqual(endorsed.decisions[0].status, "selected_valid")
        self.assertTrue(endorsed.audits[0].timing_trusted)

    def test_reviewer_validity_endorsed_current_blocks_consensus_with_less_reviewer_support(self) -> None:
        current = self.candidate(
            12.69,
            tuple(self.span(12.69 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        ctc_whisper_consensus = self.candidate(
            12.22,
            self.coherent_spans(12.22),
            source="ctc-local-whisper-vocal-independent-consensus",
            confidence=0.85,
            acoustic="supported",
        )
        direct_whisper = self.candidate(
            12.26,
            self.coherent_spans(12.26),
            source="local-whisper-vocal-independent-fusion",
            confidence=0.88,
            acoustic="supported",
            direct=True,
            direct_independent=True,
        )

        endorsed = auto_lrc.build_final_timing_state(
            ((current, ctc_whisper_consensus, direct_whisper),),
            arbiter_mode="reviewer-validity",
            reviewer_evidence=({
                "expected_current_seconds": 12.69,
                "reviewer_times": {"HUBP": 12.69, "WX": 12.68, "XLSR": 12.26},
                "offsets": {"HUBP": 0.0, "WX": 0.0, "XLSR": 0.0},
                "reviewers": ["HUBP", "WX", "XLSR"],
                "required_agreements": 1,
            },),
        )

        self.assertEqual(endorsed.decisions[0].selected_candidate_id, current.candidate_id)

    def test_reviewer_validity_endorsed_current_keeps_current_on_tied_reviewer_support(self) -> None:
        current = self.candidate(
            12.69,
            tuple(self.span(12.69 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        ctc_whisper_consensus = self.candidate(
            12.22,
            self.coherent_spans(12.22),
            source="ctc-local-whisper-vocal-independent-consensus",
            confidence=0.85,
            acoustic="supported",
        )
        direct_whisper = self.candidate(
            12.26,
            self.coherent_spans(12.26),
            source="local-whisper-vocal-independent-fusion",
            confidence=0.88,
            acoustic="supported",
            direct=True,
            direct_independent=True,
        )

        endorsed = auto_lrc.build_final_timing_state(
            ((current, ctc_whisper_consensus, direct_whisper),),
            arbiter_mode="reviewer-validity",
            reviewer_evidence=({
                "expected_current_seconds": 12.69,
                "reviewer_times": {"HUBP": 12.69, "WX": 12.26, "XLSR": None},
                "offsets": {"HUBP": 0.0, "WX": 0.0},
                "reviewers": ["HUBP", "WX", "XLSR"],
                "required_agreements": 1,
            },),
        )

        self.assertEqual(endorsed.decisions[0].selected_candidate_id, current.candidate_id)

    def test_reviewer_validity_does_not_relax_temporal_incoherence(self) -> None:
        current = self.candidate(12.0, (self.span(12.0), self.span(12.1), self.span(12.2), self.span(17.0), self.span(17.1)), source="ctc-prefix-probe", prefix_scope=True, current=True)
        state = auto_lrc.build_final_timing_state(((current,),), arbiter_mode="reviewer-validity", reviewer_evidence=(self.reviewer_validity_row(12.0),))
        self.assertFalse(state.audits[0].timing_trusted)
        self.assertIn("temporal-incoherence", state.audits[0].failures)

    def test_reviewer_validity_missing_evidence_falls_back_to_central(self) -> None:
        current = self.candidate(12.0, tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)), current=True)
        baseline = auto_lrc.build_final_timing_state(((current,),))
        fallback = auto_lrc.build_final_timing_state(((current,),), arbiter_mode="reviewer-validity", reviewer_evidence=(None,))
        self.assertEqual(fallback.decisions[0].status, baseline.decisions[0].status)
        self.assertEqual(fallback.written_times, baseline.written_times)
        self.assertEqual(fallback.audits[0].timing_trusted, baseline.audits[0].timing_trusted)

    def test_candidate_permutation_does_not_change_selection(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.80)
        stronger = self.candidate(
            12.5, self.coherent_spans(12.5), source="ctc-local", confidence=0.90,
            acoustic="supported",
        )
        weaker = self.candidate(
            12.8, self.coherent_spans(12.8), source="whisperx-word", confidence=0.60
        )
        selected = set()
        for items in permutations((current, stronger, weaker)):
            decision = auto_lrc.select_timing_decision(
                entry_index=0,
                candidates=tuple(items),
                previous_candidate=None,
                selection_revision=1,
            )
            selected.add(decision.selected_candidate_id)
        self.assertEqual(selected, {stronger.candidate_id})

    def test_cross_producer_confidence_alone_cannot_dominate_tight_independent_consensus(self) -> None:
        invalid_current = self.candidate(
            5.0,
            tuple(self.span(5.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=1.0,
        )
        near_ctc = self.candidate(
            10.0,
            self.coherent_spans(10.0),
            source="ctc-raw-independent-consensus",
            confidence=0.90,
            direct=True,
            direct_independent=True,
        )
        near_whisper = self.candidate(
            10.08,
            self.coherent_spans(10.08),
            source="raw-asr-vocal-fusion",
            confidence=0.90,
            direct=True,
            direct_independent=True,
        )
        distant_high_confidence = self.candidate(
            30.0,
            self.coherent_spans(30.0),
            source="whisperx-vocal-independent-fusion",
            confidence=0.99,
            direct=True,
            direct_independent=True,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(
                invalid_current,
                near_ctc,
                near_whisper,
                distant_high_confidence,
            ),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertIn(decision.written_time.seconds, {10.0, 10.08})

    def test_candidate_id_is_content_derived(self) -> None:
        first = self.candidate(12.0, self.coherent_spans(12.0), source="ctc-local")
        second = self.candidate(12.0, self.coherent_spans(12.0), source="ctc-local")
        self.assertEqual(first.candidate_id, second.candidate_id)
        self.assertEqual(first.generation_revision, second.generation_revision)

    def test_non_ctc_unavailable_tail_can_use_independent_minimum_proof(self) -> None:
        previous = self.previous(10.0, pileup=True)
        whisperx = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="whisperx-word",
            acoustic="supported",
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(whisperx, previous)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        # Generic acoustic onset may support an opening when ownership is clear,
        # but it is not phonetic proof and cannot bypass an unknown previous tail.
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.ownership_independent)

    def test_acoustic_only_candidate_lacks_identity_proof(self) -> None:
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="generic",
            source="acoustic-only",
            raw_time=12.0,
            spans=self.coherent_spans(12.0),
            confidence=0.9,
            identity_support="unavailable",
            sequence_support="supported",
            acoustic_support="supported",
        )
        self.assertFalse(auto_lrc.evaluate_timing_candidate(candidate).valid)

    def test_no_local_median_or_p90_is_not_an_absolute_hard_failure(self) -> None:
        spans = tuple(self.span(12.0 + index * 0.80, 0.20) for index in range(4))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, source="ctc-prefix-probe", prefix_scope=True)
        )
        self.assertNotIn("temporal-incoherence", evaluation.evidence.structural_failures)

    def test_final_state_is_frozen_and_uses_tuples(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        self.assertIsInstance(state.written_times, tuple)
        with self.assertRaises(FrozenInstanceError):
            state.written_times = ()  # type: ignore[misc]

    def test_previous_onset_downgrade_does_not_invalidate_revision_safe_clear_tail(self) -> None:
        first = self.candidate(10.0, self.coherent_spans(10.0), current=True, entry_index=0)
        first_decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(first,),
            previous_candidate=None,
            selection_revision=1,
        )
        second = self.candidate(12.0, self.coherent_spans(12.0), current=True, entry_index=1)
        second_decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(second,),
            previous_candidate=first,
            selection_revision=2,
        )
        stale_first = replace(first_decision, timing_evidence_revision=0)
        audits = auto_lrc.audit_final_timing_decisions((stale_first, second_decision))
        self.assertFalse(audits[0].timing_trusted)
        self.assertTrue(audits[1].timing_trusted)
        self.assertEqual(second_decision.evidence.ownership.status, "clear")
        self.assertEqual(
            second_decision.evidence.referenced_tail_evidence_revision,
            auto_lrc.derive_reliable_tail(first).tail_evidence_revision,
        )

    def test_independent_onset_proof_survives_previous_downgrade(self) -> None:
        first = self.candidate(
            10.0,
            tuple(self.span(10.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            entry_index=0,
        )
        second = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            current=True,
            direct=True,
            direct_independent=True,
            entry_index=1,
        )
        state = auto_lrc.build_final_timing_state(((first,), (second,)))
        self.assertFalse(state.audits[0].timing_trusted)
        self.assertTrue(state.audits[1].timing_trusted)

    def test_numeric_evidence_is_invariant_to_lyric_and_romaji_text(self) -> None:
        evaluations = []
        for text in ("alpha line", "beta line"):
            candidate = auto_lrc.make_timing_candidate(
                entry_index=0,
                entry_text=text,
                source="ctc-full",
                raw_time=12.0,
                spans=self.coherent_spans(12.0),
                confidence=0.8,
                identity_support="supported",
                sequence_support="supported",
                current=True,
                source_artifact={"numeric_fixture": 1},
            )
            evaluations.append(auto_lrc.evaluate_timing_candidate(candidate))
        self.assertEqual([item.valid for item in evaluations], [True, True])
        self.assertEqual(
            [item.evidence.structural_failures for item in evaluations],
            [(), ()],
        )
        self.assertEqual(
            [item.evidence.temporal_coherence for item in evaluations],
            ["coherent", "coherent"],
        )

    def test_single_raw_token_is_insufficient_timing_proof(self) -> None:
        candidate = self.candidate(
            12.0,
            (self.span(12.0, 0.9),),
            source="ctc-first-token-posterior",
            confidence=0.9,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate)
        self.assertFalse(evaluation.valid)
        self.assertIn("insufficient-minimum-proof", evaluation.rejection_reasons)

    def test_valid_earlier_repair_can_replace_invalid_current(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.2,
        )
        repair = self.candidate(
            11.5,
            self.coherent_spans(11.5),
            source="ctc-local-prefix",
            confidence=0.9,
            acoustic="supported",
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, repair),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.selected_candidate_id, repair.candidate_id)
        self.assertEqual(decision.written_time.seconds, 11.5)

    def test_previous_tail_distance_is_hard_gated_before_ranking(self) -> None:
        previous = self.previous(10.0)
        current = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            current=True,
            confidence=0.8,
            entry_index=1,
        )
        borrowing = self.candidate(
            9.96,
            self.coherent_spans(9.96),
            source="ctc-first-token-posterior",
            confidence=0.99,
            entry_index=1,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(borrowing, current),
            previous_candidate=previous,
            selection_revision=2,
        )
        self.assertEqual(decision.selected_candidate_id, current.candidate_id)
        rejected = next(
            item for item in decision.rejected_candidates
            if item.candidate.candidate_id == borrowing.candidate_id
        )
        self.assertFalse(rejected.valid)
        self.assertEqual(rejected.evidence.ownership.status, "unknown")
        self.assertAlmostEqual(float(rejected.evidence.tail_distance_seconds), 0.04, places=3)
        self.assertIn("previous-tail-ownership-unknown", rejected.rejection_reasons)
        state = auto_lrc.build_final_timing_state(
            ((previous,), (borrowing, current))
        )
        report: dict[str, object] = {
            "assignments": [
                {"timing_repair": "ctc-forced-align", "score": 0.9},
                {"timing_repair": "ctc-forced-align", "score": 0.9},
            ]
        }
        auto_lrc.commit_final_timing_state(state, report, {})
        report_rejections = report["assignments"][1]["candidate_rejections"]
        reported = next(
            item for item in report_rejections
            if item["candidate_id"] == borrowing.candidate_id
        )
        self.assertEqual(
            reported["evidence"]["ownership"]["reliable_tail_end"], 9.92
        )
        self.assertAlmostEqual(
            float(reported["evidence"]["tail_distance_seconds"]), 0.04, places=3
        )
        self.assertIn("previous-tail-ownership-unknown", reported["reasons"])

    def test_candidate_beyond_reliable_tail_uncertainty_is_valid(self) -> None:
        previous = self.previous(10.0)
        candidate = self.candidate(
            10.5,
            self.coherent_spans(10.5),
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.ownership.status, "clear")
        self.assertGreater(float(evaluation.evidence.tail_distance_seconds), 0.2)

    def test_successful_recovery_clears_severe_flag_with_provenance(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.2,
        )
        repair = self.candidate(
            12.8,
            self.coherent_spans(12.8),
            source="ctc-prefix-probe",
            confidence=0.9,
            acoustic="supported",
            prefix_scope=True,
        )
        state = auto_lrc.build_final_timing_state(((current, repair),))
        decision = state.decisions[0]
        self.assertIsNotNone(decision.recovery)
        resolved_id = auto_lrc._timing_content_digest({
            "entry": decision.entry_index,
            "flag": "ctc_unresolved_boundary",
            "candidate_revision": decision.evidence.candidate_revision,
        })
        decision = replace(
            decision,
            recovery=replace(
                decision.recovery,
                resolved_finding_ids=(resolved_id,),
            ),
        )
        state = replace(state, decisions=(decision,))
        report: dict[str, object] = {
            "assignments": [{
                "timestamp": 12.0,
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary"],
            }]
        }
        state = auto_lrc.commit_final_timing_state(state, report, {})
        assignment = report["assignments"][0]
        self.assertTrue(assignment["timing_trusted"])
        self.assertNotIn("ctc_unresolved_boundary", assignment["flags"])
        self.assertEqual(assignment["active_timing_findings"], [])
        resolved = assignment["resolved_timing_findings"]
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["resolved_by_candidate_id"], repair.candidate_id)
        self.assertEqual(resolved[0]["resolution_mode"], "challenger-recovery")
        auto_lrc.assert_report_timestamp_equality(report, state)

    def test_valid_current_with_historical_severe_finding_cannot_silently_clear(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        state = auto_lrc.build_final_timing_state(((current,),))
        self.assertIsNone(state.decisions[0].recovery)
        report: dict[str, object] = {
            "assignments": [{
                "timestamp": 12.0,
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary"],
            }]
        }

        final_state = auto_lrc.commit_final_timing_state(state, report, {})

        assignment = report["assignments"][0]
        self.assertIn("ctc_unresolved_boundary", assignment["flags"])
        self.assertEqual(assignment["resolved_timing_findings"], [])
        self.assertFalse(assignment["timing_trusted"])
        self.assertTrue(assignment["review_required"])
        self.assertFalse(final_state.audits[0].timing_trusted)
        auto_lrc.assert_report_timestamp_equality(report, final_state)

    def test_successful_recovery_only_clears_explicitly_resolved_finding(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.2,
        )
        repair = self.candidate(
            12.8,
            self.coherent_spans(12.8),
            source="ctc-opening-segment",
            confidence=0.9,
            acoustic="supported",
        )
        state = auto_lrc.build_final_timing_state(((current, repair),))
        decision = state.decisions[0]
        self.assertIsNotNone(decision.recovery)
        resolved_id = auto_lrc._timing_content_digest({
            "entry": decision.entry_index,
            "flag": "ctc_unresolved_boundary",
            "candidate_revision": decision.evidence.candidate_revision,
        })
        recovery = replace(
            decision.recovery,
            resolved_finding_ids=(resolved_id,),
        )
        decision = replace(decision, recovery=recovery)
        state = replace(state, decisions=(decision,))
        report: dict[str, object] = {
            "assignments": [{
                "timestamp": 12.0,
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary", "ctc_path_fracture"],
            }]
        }

        final_state = auto_lrc.commit_final_timing_state(state, report, {})

        assignment = report["assignments"][0]
        self.assertNotIn("ctc_unresolved_boundary", assignment["flags"])
        self.assertIn("ctc_path_fracture", assignment["flags"])
        self.assertEqual(
            [item["code"] for item in assignment["resolved_timing_findings"]],
            ["ctc_unresolved_boundary"],
        )
        self.assertEqual(
            [item["code"] for item in assignment["active_timing_findings"]],
            ["ctc_path_fracture"],
        )
        resolved = assignment["resolved_timing_findings"][0]
        self.assertEqual(resolved["resolved_by_candidate_id"], repair.candidate_id)
        self.assertEqual(
            resolved["resolved_candidate_revision"], decision.evidence.candidate_revision
        )
        self.assertEqual(
            resolved["resolved_evidence_revision"], decision.timing_evidence_revision
        )
        self.assertEqual(resolved["resolved_canonical_written_centiseconds"], 1280)
        self.assertEqual(resolved["resolved_selection_revision"], decision.selection_revision)
        self.assertEqual(resolved["resolution_mode"], "challenger-recovery")
        self.assertTrue(resolved["passed_invariants"])
        self.assertFalse(assignment["timing_trusted"])
        self.assertTrue(assignment["review_required"])
        self.assertFalse(final_state.audits[0].timing_trusted)
        auto_lrc.assert_report_timestamp_equality(report, final_state)

    def test_report_equality_detects_late_alias_mutation(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        report: dict[str, object] = {
            "assignments": [{
                "timestamp": 12.0,
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
            }]
        }
        auto_lrc.commit_final_timing_state(state, report, {})
        report["assignments"][0]["canonical_written_centiseconds"] = 1201
        with self.assertRaises(LrcError):
            auto_lrc.assert_report_timestamp_equality(report, state)

    def test_final_timing_state_rejects_internal_cardinality_mismatch(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        with self.assertRaises(LrcError):
            auto_lrc.FinalTimingState(
                decisions=state.decisions,
                written_times=state.written_times,
                audits=(),
            )

    def test_final_commit_relabels_precentral_scoring_provenance(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        report: dict[str, object] = {
            "assignments": [{
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "reasons": ["legacy-precentral-score"],
                "penalties": [{"kind": "legacy-penalty", "value": 0.4}],
            }]
        }
        auto_lrc.commit_final_timing_state(state, report, {})
        assignment = report["assignments"][0]
        self.assertNotIn("reasons", assignment)
        self.assertNotIn("penalties", assignment)
        self.assertEqual(assignment["precentral_reasons"], ["legacy-precentral-score"])
        self.assertEqual(
            assignment["precentral_penalties"],
            [{"kind": "legacy-penalty", "value": 0.4}],
        )

    def test_predecessor_onset_failure_does_not_cascade_across_clear_lines(self) -> None:
        candidates = tuple(
            self.candidate(
                10.0 + index * 2.0,
                self.coherent_spans(10.0 + index * 2.0),
                current=True,
                entry_index=index,
            )
            for index in range(3)
        )
        decisions = []
        previous = None
        for index, candidate in enumerate(candidates):
            decisions.append(auto_lrc.select_timing_decision(
                entry_index=index,
                candidates=(candidate,),
                previous_candidate=previous,
                selection_revision=index + 1,
            ))
            previous = candidate
        decisions[0] = replace(decisions[0], timing_evidence_revision=0)
        audits = auto_lrc.audit_final_timing_decisions(tuple(decisions))
        self.assertEqual([audit.timing_trusted for audit in audits], [False, True, True])
        self.assertTrue(all(
            decision.evidence.ownership.status == "clear"
            for decision in decisions[1:]
        ))

    def test_first_local_onset_edge_guard_is_an_explicit_caller_contract(self) -> None:
        frame_times = np.round(np.arange(0.0, 2.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 1.1))[0][0]] = 1.0
        features = AudioFeatures(
            duration=2.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        self.assertIsNone(auto_lrc.first_local_onset(features, 1.0, 1.5))
        self.assertAlmostEqual(
            auto_lrc.first_local_onset(
                features, 1.0, 1.5, edge_guard_seconds=0.0
            ) or 0.0,
            1.1,
            places=3,
        )

    def test_write_lrc_reads_immutable_canonical_times_without_mutation(self) -> None:
        candidate = self.candidate(12.006, self.coherent_spans(12.01), current=True)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        before = state.written_times
        original_labels = auto_lrc.audio_track_labels
        lyrics_io.audio_track_labels = lambda _path: ("", "generic")  # type: ignore[assignment]
        try:
            with TemporaryDirectory() as temp_name:
                root = Path(temp_name)
                output = root / "output.lrc"
                auto_lrc.write_lrc(
                    output,
                    root / "audio.flac",
                    root / "lyrics.txt",
                    [LyricEntry(["generic line"])],
                    state.written_times,
                    20.0,
                    False,
                )
                written = output.read_text(encoding="utf-8-sig")
        finally:
            lyrics_io.audio_track_labels = original_labels  # type: ignore[assignment]
        self.assertIs(state.written_times, before)
        self.assertIn(f"[{state.written_times[0].lrc_tag}]generic line", written)

    def test_audio_track_labels_preserve_artist_case(self) -> None:
        payload = json.dumps({"format": {"tags": {"ARTIST": "AC/DC feat. eVe", "TITLE": "Song"}}})
        completed = subprocess.CompletedProcess(["ffprobe"], 0, payload, "")
        with split_patch.patch("run_command", return_value=completed):
            artist, title = auto_lrc.audio_track_labels(Path("dummy.flac"))
        self.assertEqual(artist, "AC/DC feat. eVe")
        self.assertEqual(title, "Song")

    def test_process_audio_round_trips_metadata_and_offset_once(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "song.flac"
            lyrics = root / "song.txt"
            output = root / "song.lrc"
            reports = root / "reports"
            audio.write_bytes(b"audio")
            lyrics.write_text(
                "[ar:MiXeD ARTIST]\n[al:Album Name]\n[offset:+500]\n[00:01.00]line\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args([
                str(audio),
                "--lyrics", str(lyrics),
                "--output", str(output),
                "--timing-source", "lyrics",
                "--report-dir", str(reports),
                "--overwrite",
            ])
            with (
                split_patch.patch("probe_duration", return_value=2.0),
                mock.patch.object(lyrics_io, "audio_track_labels", return_value=("MiXeD ARTIST", "Song")),
            ):
                auto_lrc.process_audio(audio, args)
            written = output.read_text(encoding="utf-8-sig")
        self.assertEqual(written.count("[ar:MiXeD ARTIST]"), 1)
        self.assertEqual(written.count("[al:Album Name]"), 1)
        self.assertEqual(written.count("[offset:+500]"), 1)
        self.assertIn("[00:01.00]line", written)
        self.assertNotIn("[00:01.50]line", written)

    def test_segment_review_count_deduplicates_same_entry(self) -> None:
        import segment_skeleton_production

        suspicious = [
            {"entry": 2, "review_required": True, "flags": ["a"]},
            {"entry": 2, "review_required": True, "flags": ["b"]},
            {"entry": 3, "review_required": True, "flags": ["c"]},
            {"entry": 4, "review_required": False, "flags": ["d"]},
        ]
        self.assertEqual(
            segment_skeleton_production._review_required_row_count(suspicious),
            2,
        )

    def test_sol_r1_late_supported_pair_cannot_prove_opening(self) -> None:
        spans = (
            self.span(12.0, 0.01),
            self.span(12.08, 0.01),
            self.span(12.16, 0.20),
            self.span(12.24, 0.20),
        )
        evidence = auto_lrc.normalize_timing_candidate(self.candidate(12.0, spans))
        self.assertIsNone(evidence.onset_proof_end)
        self.assertEqual(evidence.opening_supported_run, 0)
        self.assertEqual(evidence.later_supported_run, 2)

    def test_sol_r1_weak_token_zero_contiguous_token_one_and_acoustic_prove_opening(self) -> None:
        spans = (self.span(12.0, 0.01), self.span(12.08, 0.20))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, acoustic="supported")
        )
        self.assertTrue(evaluation.valid)
        self.assertAlmostEqual(float(evaluation.evidence.onset_proof_end), 12.10, places=3)

    def test_sol_r1_weak_token_zero_distant_token_one_cannot_prove_opening(self) -> None:
        spans = (self.span(12.0, 0.01), self.span(12.5, 0.20))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, acoustic="supported")
        )
        self.assertFalse(evaluation.valid)
        self.assertIsNone(evaluation.evidence.onset_proof_end)

    def test_sol_r1_scores_separated_by_multi_second_gap_are_not_consecutive(self) -> None:
        spans = (self.span(12.0, 0.20), self.span(15.1, 0.20))
        evidence = auto_lrc.normalize_timing_candidate(self.candidate(12.0, spans))
        self.assertEqual(evidence.longest_supported_run, 1)
        self.assertEqual(evidence.opening_supported_run, 1)
        self.assertIsNone(evidence.onset_proof_end)

    def test_sol_r2_prefix_single_existing_severe_gap_is_incoherent(self) -> None:
        spans = (
            self.span(12.0), self.span(12.08), self.span(15.10), self.span(15.18)
        )
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, prefix_scope=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertEqual(evaluation.evidence.severe_gap_count, 1)
        self.assertEqual(evaluation.evidence.island_count, 2)
        self.assertEqual(evaluation.evidence.temporal_coherence, "incoherent")

    def test_sol_r2_multiple_severe_gaps_cannot_hide_in_inflated_median(self) -> None:
        spans = tuple(self.span(time) for time in (12.0, 15.2, 18.5, 21.9))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, prefix_scope=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertEqual(evaluation.evidence.severe_gap_count, 3)
        self.assertEqual(evaluation.evidence.island_count, 4)
        self.assertEqual(evaluation.evidence.temporal_coherence, "incoherent")

    def test_sol_r2_full_line_strong_opening_survives_late_tail_fracture(self) -> None:
        spans = (*self.coherent_spans(12.0), self.span(16.2), self.span(16.3))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, prefix_scope=False)
        )
        self.assertTrue(evaluation.valid)
        self.assertGreater(evaluation.evidence.severe_gap_count, 0)
        self.assertIn(evaluation.evidence.fracture_region, {"middle", "tail"})
