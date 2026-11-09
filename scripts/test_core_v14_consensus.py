#!/usr/bin/env python3
"""V14.2-V14.4 central consensus, evidence recovery and joint boundary graph tests."""

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


class V142CentralConsensusRegressionTests(unittest.TestCase):
    def candidate(
        self,
        source: str,
        time: float,
        *,
        current: bool = False,
        independent: bool = True,
        confidence: float = 0.9,
    ) -> auto_lrc.TimingCandidate:
        artifact = {"source": source, "time": time}
        if source.startswith("raw-"):
            onset_producer = "raw-asr-vocal-fusion"
            onset_kind = "cross-backend-onset"
        elif "whisperx" in source:
            onset_producer = "whisperx-vocal-fusion"
            onset_kind = "cross-backend-onset"
        elif "recurrence" in source:
            onset_producer = "global-recurrence-search"
            onset_kind = "recurrence-onset"
        else:
            onset_producer = "consonant-onset-detector"
            onset_kind = "vocal-onset"
        occurrence_evidence = None
        if onset_kind == "cross-backend-onset":
            occurrence_entries = [LyricEntry(["generic lyric"])]
            occurrence_assignments = [
                {"timestamp": 1.0, "score": .95, "segment": 1}
            ]
            occurrence_binding = auto_lrc._entry_occurrence_binding(
                occurrence_entries, occurrence_assignments, 0
            )
            occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
                occurrence_entries,
                0,
                producer=onset_producer,
                occurrence_binding=occurrence_binding,
                upstream_evidence_revision="v142-direct-upstream",
            )
        return auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="generic lyric",
            source=source,
            raw_time=time,
            confidence=confidence,
            identity_support="supported" if independent else "unsupported",
            sequence_support="supported",
            acoustic_support="supported" if independent else "unavailable",
            direct_onset_support="supported" if independent else "unavailable",
            acoustic_strength=0.9 if independent else None,
            acoustic_onset_time=time if independent else None,
            acoustic_source_artifact=artifact if independent else None,
            acoustic_evidence_producer=f"{source}-acoustic" if independent else None,
            acoustic_evidence_kind="vocal-onset",
            acoustic_evidence_independent=independent,
            direct_onset_time=time if independent else None,
            direct_onset_source_artifact=artifact if independent else None,
            direct_onset_evidence_producer=onset_producer if independent else None,
            direct_onset_evidence_kind=onset_kind,
            direct_onset_evidence_independent=independent,
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=independent,
            current=current,
            source_artifact=artifact,
        )

    def test_tight_independent_valid_cluster_resolves_invalid_current(self) -> None:
        current = self.candidate("ctc-current", 10.50, current=True, independent=False)
        raw = self.candidate("raw-vocal-independent-fusion", 10.00)
        recurrence = self.candidate("global-acoustic-recurrence", 10.06)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, raw, recurrence),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertIn(decision.candidate.source, {
            "raw-vocal-independent-fusion",
            "global-acoustic-recurrence",
        })
        self.assertLessEqual(abs(decision.written_time.seconds - 10.03), 0.04)

    def test_two_equally_supported_separated_clusters_remain_unresolved(self) -> None:
        current = self.candidate("ctc-current", 10.50, current=True, independent=False)
        first_a = self.candidate("raw-vocal-independent-fusion", 10.00)
        first_b = self.candidate("global-acoustic-recurrence", 10.06)
        second_a = self.candidate("raw-vocal-independent-fusion", 11.00)
        second_b = self.candidate("global-acoustic-recurrence", 11.06)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, first_a, first_b, second_a, second_b),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.candidate.candidate_id, current.candidate_id)

    def test_precentral_diagnostic_distant_candidate_blocks_review_supersede(self) -> None:
        row = {
            "entry": 1,
            "flags": ["candidate_disagreement"],
            "candidate_timestamps": {"output": 261.70},
            "precentral_diagnostic": {
                "candidates": [
                    {"source": "selected_path", "time": 261.70},
                    {"source": "ctc_repeated_leading_term_onset", "time": 259.42},
                ]
            },
        }

        self.assertTrue(
            auto_lrc._precentral_candidate_gap_exceeds_half_second(row, 261.70)
        )

    def test_postcentral_projection_replaces_stale_output_and_resolves_superseded_review(self) -> None:
        current = self.candidate("ctc-current", 11.00, current=True, independent=False)
        selected = self.candidate("raw-vocal-independent-fusion", 10.00)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, selected),
            previous_candidate=None,
            selection_revision=1,
        )
        state = auto_lrc.FinalTimingState(
            decisions=(decision,),
            written_times=(decision.written_time,),
            audits=auto_lrc.audit_final_timing_decisions((decision,)),
        )
        report = {
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "candidates": [{"source": decision.candidate.source, "time": 10.0}],
                "rejected_candidates": [],
            }],
            "suspicious_alignments": [{
                "entry": 1,
                "flags": ["candidate_disagreement"],
                "severity": "high",
                "review_required": True,
                "chosen_time": 11.0,
                "confidence": 0.2,
                "candidate_timestamps": {"output": 11.0, "raw_asr": 10.0},
                "candidate_disagreement_seconds": 1.0,
            }],
        }
        auto_lrc._project_suspicious_alignments_to_central_state(report, state)
        row = report["suspicious_alignments"][0]
        self.assertEqual(row["chosen_time"], 10.0)
        self.assertEqual(row["candidate_timestamps"]["output"], 10.0)
        self.assertEqual(row["precentral_diagnostic"]["chosen_time"], 11.0)
        self.assertEqual(row["severity"], "resolved")
        self.assertFalse(row["review_required"])
        self.assertEqual(row["confidence"], 0.9)
        self.assertEqual(row["candidate_disagreement_seconds"], 0.0)
        self.assertEqual(row["timestamp_revision"], decision.timestamp_revision)
        self.assertEqual(row["timing_evidence_revision"], decision.timing_evidence_revision)

    def test_distant_valid_alternative_keeps_review_open(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 10.00, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 11.00)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
        row = {
            "entry": 1,
            "flags": ["candidate_disagreement"],
            "severity": "high",
            "review_required": True,
        }
        self.assertTrue(audit.timing_trusted)
        self.assertFalse(auto_lrc._central_decision_supersedes_suspicious_row(decision, audit, row))

    def test_final_postcentral_material_conflict_propagates_without_suspicious_row(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 10.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 11.0)
        report = {
            "backend": "synthetic",
            "timing_entries": 1,
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "timing_repair": "synthetic",
                "score": 0.9,
            }],
            "suspicious_alignments": [],
        }
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((current, distant),), report, {}
        )
        self.assertTrue(evaluated.state.audits[0].timing_trusted)
        auto_lrc.commit_evaluated_backend(
            auto_lrc.select_evaluated_backend((evaluated,)), report
        )
        committed = auto_lrc.commit_final_timing_state(
            evaluated.state, report, {},
            finding_states=evaluated.finding_states, final_output=True,
        )
        self.assertIs(committed.decisions, evaluated.state.decisions)
        self.assertIs(committed.written_times, evaluated.state.written_times)
        self.assertTrue(evaluated.state.audits[0].timing_trusted)
        self.assertFalse(committed.audits[0].timing_trusted)
        auto_lrc.assert_report_timestamp_equality(report, committed)
        self.assertEqual(report["review_required_count"], 1)
        self.assertTrue(report["assignments"][0]["review_required"])
        self.assertFalse(report["assignments"][0]["timing_trusted"])
        self.assertEqual(report["assignments"][0]["timestamp_status"], "provisional_unresolved")
        self.assertEqual(report["assignments"][0]["verification_status"], "review-required")
        self.assertEqual(report["suspicious_alignments"], [])

    def test_material_unknown_ownership_is_not_falsification(self) -> None:
        unknown_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id="previous",
            previous_generation_revision="generation",
            previous_region_revision="region",
            tail_evidence_revision="tail",
            validity="unknown",
            reliable_tail_start=None,
            reliable_tail_end=None,
            uncertainty_seconds=None,
            confidence=None,
            reason="tail-unavailable",
        )
        current = self.candidate("raw-vocal-independent-fusion", 10.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 11.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            previous_tail=unknown_tail,
            selection_revision=1,
        )
        self.assertTrue(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_explicit_ownership_violation_can_falsify_material_alternative(self) -> None:
        valid_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id="previous",
            previous_generation_revision="generation",
            previous_region_revision="region",
            tail_evidence_revision="tail",
            validity="valid",
            reliable_tail_start=9.5,
            reliable_tail_end=10.0,
            uncertainty_seconds=0.01,
            confidence=0.9,
            reason="valid-tail",
        )
        current = self.candidate("raw-vocal-independent-fusion", 11.0, current=True)
        violating = self.candidate("whisperx-vocal-independent-fusion", 9.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, violating),
            previous_candidate=None,
            previous_tail=valid_tail,
            selection_revision=1,
        )
        self.assertFalse(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_material_candidate_without_minimum_proof_is_not_material(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 10.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 11.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        rejected = next(
            item for item in decision.rejected_candidates
            if item.candidate.candidate_id == distant.candidate_id
        )
        malformed = replace(
            rejected,
            evidence=replace(rejected.evidence, minimum_proof_satisfied=False),
        )
        decision = replace(decision, rejected_candidates=(malformed,))
        self.assertFalse(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_malformed_ownership_violation_remains_material(self) -> None:
        valid_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id="previous",
            previous_generation_revision="generation",
            previous_region_revision="region",
            tail_evidence_revision="tail",
            validity="valid",
            reliable_tail_start=9.5,
            reliable_tail_end=10.0,
            uncertainty_seconds=0.01,
            confidence=0.9,
            reason="valid-tail",
        )
        current = self.candidate("raw-vocal-independent-fusion", 11.0, current=True)
        violating = self.candidate("whisperx-vocal-independent-fusion", 9.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, violating),
            previous_candidate=None,
            previous_tail=valid_tail,
            selection_revision=1,
        )
        rejected = next(
            item for item in decision.rejected_candidates
            if item.candidate.candidate_id == violating.candidate_id
        )
        malformed_ownership = replace(
            rejected.evidence.ownership,
            previous_tail_validity="unknown",
            previous_tail_evidence_revision=None,
            relation_revision="",
        )
        malformed = replace(
            rejected,
            evidence=replace(rejected.evidence, ownership=malformed_ownership),
        )
        decision = replace(decision, rejected_candidates=(malformed,))
        self.assertTrue(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_postcentral_projection_recomputes_severity_boundary(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 12.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 14.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        state = auto_lrc.FinalTimingState(
            decisions=(decision,),
            written_times=(decision.written_time,),
            audits=auto_lrc.audit_final_timing_decisions((decision,)),
        )
        report = {
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "timing_repair": "synthetic",
                "score": 0.9,
                "candidates": [],
                "rejected_candidates": [],
            }],
            "suspicious_alignments": [{
                "entry": 1,
                "flags": ["unresolved_raw_ctc_disagreement"],
                "severity": "medium",
                "review_required": True,
                "chosen_time": 10.0,
                "candidate_timestamps": {"output": 10.0, "raw_asr": 9.8},
                "raw_asr_score": 0.78,
                "candidate_disagreement_seconds": 0.2,
            }],
        }
        auto_lrc._project_suspicious_alignments_to_central_state(report, state)
        row = report["suspicious_alignments"][0]
        self.assertEqual(row["candidate_disagreement_seconds"], 2.2)
        self.assertEqual(row["severity"], "high")

    def test_postcentral_projection_downshifts_sole_raw_severity(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 12.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 14.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        state = auto_lrc.FinalTimingState(
            decisions=(decision,),
            written_times=(decision.written_time,),
            audits=auto_lrc.audit_final_timing_decisions((decision,)),
        )
        report = {
            "assignments": [{
                "entry": 1,
                "timestamp": 11.0,
                "timing_repair": "synthetic",
                "score": 0.9,
                "candidates": [],
                "rejected_candidates": [],
            }],
            "suspicious_alignments": [{
                "entry": 1,
                "flags": ["unresolved_raw_ctc_disagreement"],
                "severity": "high",
                "review_required": True,
                "chosen_time": 11.0,
                "candidate_timestamps": {"output": 11.0, "raw_asr": 11.2},
                "raw_asr_score": 0.78,
                "candidate_disagreement_seconds": 0.2,
            }],
        }
        auto_lrc._project_suspicious_alignments_to_central_state(report, state)
        row = report["suspicious_alignments"][0]
        self.assertEqual(row["candidate_disagreement_seconds"], 0.8)
        self.assertEqual(row["severity"], "medium")

    def test_postcentral_projection_preserves_independent_high_severity(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 12.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 14.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        state = auto_lrc.FinalTimingState(
            decisions=(decision,),
            written_times=(decision.written_time,),
            audits=auto_lrc.audit_final_timing_decisions((decision,)),
        )
        report = {
            "assignments": [{
                "entry": 1,
                "timestamp": 11.0,
                "timing_repair": "synthetic",
                "score": 0.9,
                "candidates": [],
                "rejected_candidates": [],
            }],
            "suspicious_alignments": [{
                "entry": 1,
                "flags": ["unresolved_raw_ctc_disagreement", "phonetic_anchor_disagreement"],
                "severity": "high",
                "review_required": True,
                "chosen_time": 11.0,
                "candidate_timestamps": {"output": 11.0, "raw_asr": 10.8},
                "raw_asr_score": 0.78,
                "candidate_disagreement_seconds": 0.2,
            }],
        }
        auto_lrc._project_suspicious_alignments_to_central_state(report, state)
        row = report["suspicious_alignments"][0]
        self.assertEqual(row["candidate_disagreement_seconds"], 1.2)
        self.assertEqual(row["severity"], "high")


class V143EvidenceRecoveryRegressionTests(unittest.TestCase):
    def span(self, start: float, score: float = 0.5, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        current: bool = False,
        spans: tuple[auto_lrc.TimingTokenSpan, ...] | None = None,
        truncated: bool = False,
    ) -> auto_lrc.TimingCandidate:
        if spans is None:
            spans = (
                self.span(time, 0.3, "a"),
                self.span(time + 0.08, 0.4, "b"),
                self.span(time + 0.16, 0.5, "c"),
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            current=current,
            window_truncated=truncated,
            source_artifact={"entry": entry, "time": time, "source": source, "truncated": truncated},
        )

    def capability(self) -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": "ctc",
            "helper_path": "ctc_align.py",
            "helper_sha256": "helper-sha",
            "helper_protocol_revision": "protocol-revision",
            "alignment_audio_path": "audio.flac",
            "alignment_audio_sha256": "audio-sha",
            "alignment_audio_source": "vocal-stem",
            "sample_rate": auto_lrc.SAMPLE_RATE,
            "model_identity": "MMS_FA:known-lyric",
            "device": "cpu",
        }
        return auto_lrc.BoundedRetryCapability(
            **payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(payload)
            ),
        )

    def tail_evidence(
        self,
        *,
        producer: str = "whisperx-forced-row",
        tail_start: float = 9.8,
        tail_end: float = 10.4,
        uncertainty: float = 0.02,
        confidence: float = 0.95,
    ) -> auto_lrc.IndependentTailEvidence:
        provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=confidence,
            producer=producer,
            backend_lineage="whisper-family" if producer.startswith("whisper") else "ctc",
            model_identity="test-model",
            helper_revision="test-helper",
            capability_revision="test-capability",
            audio_revision="test-audio",
            transcript_revision="test-transcript",
            row_revision="test-row",
        )
        assert provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=confidence,
            producer=producer,
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )

    def test_ctc_assignment_tail_fallback_restores_missing_tail_geometry(self) -> None:
        entries = [LyricEntry(["unique predecessor"])]
        report: dict[str, object] = {
            "backend": "ctc",
            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: self.capability(),
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "score": 0.9,
                "timing_repair_source": "torchaudio-mms-fa",
                "ctc_token_spans": [
                    {"char": "a", "start": 10.0, "end": 10.02, "score": 0.5},
                    {"char": "b", "start": 10.10, "end": 10.12, "score": 0.6},
                    {"char": "c", "start": 10.20, "end": 10.22, "score": 0.7},
                ],
            }],
        }
        auto_lrc.bind_runtime_ctc_assignment_tail_fallback(report, entries)
        runtime = report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY]
        self.assertIn(0, runtime)
        tail = runtime[0][0]
        self.assertEqual(tail.producer, "ctc-assignment-terminal-row")
        self.assertGreaterEqual(tail.tail_end, 10.22)
        self.assertGreaterEqual(tail.uncertainty_seconds, 0.0)

    def test_ctc_assignment_tail_fallback_never_overwrites_distinct_backend_tail(self) -> None:
        entries = [LyricEntry(["unique predecessor"])]
        external = self.tail_evidence()
        report: dict[str, object] = {
            "backend": "ctc",
            auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY: {0: (external,)},
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "score": 0.9,
                "ctc_token_spans": [
                    {"char": "a", "start": 10.0, "end": 10.02, "score": 0.5},
                    {"char": "b", "start": 10.10, "end": 10.12, "score": 0.6},
                    {"char": "c", "start": 10.20, "end": 10.22, "score": 0.7},
                ],
            }],
        }
        auto_lrc.bind_runtime_ctc_assignment_tail_fallback(report, entries)
        runtime = report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY]
        self.assertIs(runtime[0][0], external)
        self.assertEqual(runtime[0][0].producer, "whisperx-forced-row")

    def test_truncated_retry_expands_next_bounded_window(self) -> None:
        entries = [LyricEntry(["previous"]), LyricEntry(["target"]), LyricEntry(["next"])]
        weak_prev = self.candidate(
            0, 9.0, current=True,
            spans=(self.span(9.0, 0.2, "a"), self.span(9.08, 0.01, "b")),
        )
        weak_current = self.candidate(
            1, 10.0, current=True,
            spans=(self.span(10.0, 0.01, "a"), self.span(10.08, 0.01, "b"), self.span(10.16, 0.01, "c")),
        )
        following = self.candidate(2, 12.0, current=True)
        base_sets = ((weak_prev,), (weak_current,), (following,))
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 3,
            "assignments": [
                {"entry": 1, "timestamp": 9.0, "timing_repair": "ctc", "score": 0.9},
                {"entry": 2, "timestamp": 10.0, "timing_repair": "ctc", "score": 0.9},
                {"entry": 3, "timestamp": 12.0, "timing_repair": "ctc", "score": 0.9},
            ],
        }
        cap = self.capability()
        epoch = auto_lrc.freeze_base_retry_epoch(
            "ctc", report, base_sets, auto_lrc.inactive_legacy_timing_diagnostics(), cap
        )
        first = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, base_sets, 30.0)[0]
        truncated = self.candidate(
            1, 10.1, source="ctc-opening-local-retry", current=False, truncated=True,
            spans=(self.span(10.1, 0.2, "a"), self.span(10.18, 0.3, "b"), self.span(10.26, 0.4, "c")),
        )
        expanded_sets = (base_sets[0], (weak_current, truncated), base_sets[2])
        epoch2 = auto_lrc.freeze_base_retry_epoch(
            "ctc", report, expanded_sets, auto_lrc.inactive_legacy_timing_diagnostics(), cap
        )
        second = auto_lrc.plan_crossline_retry_requests(epoch2, cap, entries, expanded_sets, 30.0)[0]
        self.assertAlmostEqual(
            second.window_end - first.window_end,
            auto_lrc.CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
            places=6,
        )
        self.assertNotEqual(first.window_definition_revision, second.window_definition_revision)

    def _install_unique_raw_identity(
        self,
        report: dict[str, object],
        entries: list[LyricEntry],
        raw_time: float,
        raw_score: float = 0.95,
    ) -> None:
        index = 0
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[index]))
        lyric_sha = __import__("hashlib").sha256(target.encode("utf-8")).hexdigest()
        binding_payload = {
            "mode": "global-unique",
            "entry_index": index,
            "lyric_sha256": lyric_sha,
        }
        binding = {
            **binding_payload,
            "binding_revision": auto_lrc._timing_content_digest(binding_payload),
        }
        payload = {
            "entry_index": index,
            "lyric_sha256": lyric_sha,
            "raw_time": round(raw_time, 6),
            "raw_score": round(raw_score, 6),
            "segment": 0,
            "occurrence_binding": binding,
            "producer": "raw-asr-content-identity",
        }
        report[auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY] = {
            index: {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}
        }

    def test_local_ctc_candidate_can_bind_existing_raw_identity_without_moving_timestamp(self) -> None:
        entries = [LyricEntry(["globally unique lyric"])]
        report: dict[str, object] = {}
        self._install_unique_raw_identity(report, entries, 10.10)
        parent = self.candidate(
            0, 10.0, source="ctc-opening-local-retry", current=False,
            spans=(self.span(10.0, 0.4, "a"), self.span(10.08, 0.5, "b"), self.span(10.16, 0.6, "c")),
        )
        augmented = auto_lrc._augment_local_ctc_candidates_with_runtime_backend_consensus(
            ((parent,),), report, entries
        )[0]
        composites = [item for item in augmented if item.source == "ctc-local-raw-independent-consensus"]
        self.assertEqual(len(composites), 1)
        composite = composites[0]
        self.assertEqual(composite.written_time, parent.written_time)
        self.assertTrue(composite.independent_content_identity)
        self.assertIsNotNone(composite.direct_onset_evidence)
        self.assertEqual(composite.direct_onset_evidence.kind, "cross-backend-onset")
        self.assertAlmostEqual(composite.direct_onset_evidence.onset_time, 10.10, places=6)

    def test_s1_segment_ctc_binds_real_independent_raw_at_same_clock(self) -> None:
        entries = [LyricEntry(["unique lyric instance"])]
        report: dict[str, object] = {}
        self._install_unique_raw_identity(report, entries, 32.10)
        parent = self.candidate(
            0, 32.0, source="segment-constrained-ctc", current=False,
            spans=(self.span(32.0, .32, "a"), self.span(32.08, .54, "b"),
                   self.span(32.16, .48, "c")),
        )
        items = auto_lrc._augment_local_ctc_candidates_with_runtime_backend_consensus(
            ((parent,),), report, entries)[0]
        found = [x for x in items if x.source == "ctc-local-raw-independent-consensus"]
        self.assertEqual(len(found), 1)
        attached = found[0]
        self.assertEqual(attached.written_time, parent.written_time)
        self.assertTrue(attached.independent_content_identity)
        self.assertTrue(auto_lrc.candidate_occurrence_evidence_is_valid(attached, entries))
        self.assertTrue(auto_lrc.candidate_has_bound_direct_onset_evidence(attached, entries))
        self.assertEqual(attached.spans, parent.spans)

    def test_s1_segment_ctc_cannot_bind_distant_asr_observation(self) -> None:
        entries = [LyricEntry(["unique lyric instance"])]
        report: dict[str, object] = {}
        self._install_unique_raw_identity(report, entries, 37.4)
        parent = self.candidate(
            0, 32.0, source="segment-constrained-ctc", current=False,
            spans=(self.span(32.0, .3, "a"), self.span(32.1, .3, "b"),
                   self.span(32.2, .3, "c")),
        )
        added = auto_lrc._augment_local_ctc_candidates_with_runtime_backend_consensus(
            ((parent,),), report, entries)[0]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].candidate_id, parent.candidate_id)

    def test_s1_segment_ctc_no_raw_identity_cannot_become_trusted(self) -> None:
        entries = [LyricEntry(["unique lyric instance"])]
        parent = self.candidate(
            0, 32.0, source="segment-constrained-ctc", current=False,
            spans=(self.span(32.0, .3, "a"), self.span(32.1, .3, "b"),
                   self.span(32.2, .3, "c")),
        )
        added = auto_lrc._augment_local_ctc_candidates_with_runtime_backend_consensus(
            ((parent,),), {}, entries)[0]
        self.assertEqual(len(added), 1)
        self.assertFalse(added[0].independent_content_identity)
        self.assertFalse(auto_lrc.candidate_has_bound_direct_onset_evidence(added[0], entries))

    def test_local_ctc_candidate_does_not_bind_distant_raw_identity(self) -> None:
        entries = [LyricEntry(["globally unique lyric"])]
        report: dict[str, object] = {}
        self._install_unique_raw_identity(report, entries, 10.50)
        parent = self.candidate(0, 10.0, source="ctc-opening-local-retry", current=False)
        augmented = auto_lrc._augment_local_ctc_candidates_with_runtime_backend_consensus(
            ((parent,),), report, entries
        )[0]
        self.assertFalse(any(
            item.source == "ctc-local-raw-independent-consensus" for item in augmented
        ))

    def test_candidate_seed_ranking_ignores_provenance_id_for_capped_strength_ties(self) -> None:
        def assignment(local_id: str, retry_id: str) -> dict[str, object]:
            return {
                "alignment_hypotheses": [
                    {
                        "source": "ctc-local-window",
                        "spans": [{"start": 215.013, "score": 0.716128}],
                        "hypothesis_id": local_id,
                    },
                    {
                        "source": "ctc-opening-local-retry-identity",
                        "spans": [{"start": 201.51, "score": 0.666215}],
                        "hypothesis_id": retry_id,
                        "window_truncated": False,
                        "right_edge_pileup": False,
                    },
                ]
            }

        first = auto_lrc._candidate_seed_rows(assignment("z-local", "a-retry"))
        second = auto_lrc._candidate_seed_rows(assignment("a-local", "z-retry"))

        self.assertEqual(first[0]["time"], 215.013)
        self.assertEqual(second[0]["time"], 215.013)
        self.assertEqual(first[0]["kind"], "ctc-local-first-token")
        self.assertEqual(second[0]["kind"], "ctc-local-first-token")

    def test_complete_retry_publishes_identity_only_occurrence_evidence(self) -> None:
        spans = tuple(
            self.span(10.0 + 0.08 * index, 0.4 + 0.05 * index, token)
            for index, token in enumerate("abcd")
        )
        candidate = self.candidate(
            0, 10.0, source="ctc-opening-local-retry", current=False, spans=spans
        )
        base = {
            "retry_epoch_revision": "epoch",
            "request_revision": "request",
            "entry_index": 0,
            "status": "candidate",
            "reason": "complete-known-lyric-target-row",
            "candidate": candidate,
            "output_revision": "output",
        }
        outcome = auto_lrc.CrossLineRetryOutcome(
            **base, outcome_revision="outcome-revision"
        )
        report: dict[str, object] = {"assignments": [{"entry": 1}]}
        hypothesis = auto_lrc.record_crossline_retry_identity_hypothesis(report, outcome)
        self.assertIsNotNone(hypothesis)
        assert hypothesis is not None
        self.assertEqual(hypothesis["source"], "ctc-opening-local-retry-identity")
        self.assertEqual(hypothesis["acoustic_support"], "unavailable")
        self.assertEqual(hypothesis["direct_onset_support"], "unavailable")
        assignment = report["assignments"][0]
        identity = auto_lrc._bound_local_occurrence_identity(assignment)
        self.assertIsNotNone(identity)
        self.assertEqual(identity["source"], "ctc-opening-local-retry-identity")
        seeds = auto_lrc._candidate_seed_rows(assignment)
        self.assertTrue(any(row["kind"] == "ctc-local-retry-first-token" for row in seeds))

    def test_truncated_retry_cannot_publish_occurrence_identity(self) -> None:
        spans = tuple(
            self.span(10.0 + 0.08 * index, 0.5, token)
            for index, token in enumerate("abcd")
        )
        candidate = self.candidate(
            0, 10.0, source="ctc-opening-local-retry", current=False,
            spans=spans, truncated=True,
        )
        base = {
            "retry_epoch_revision": "epoch",
            "request_revision": "request",
            "entry_index": 0,
            "status": "candidate",
            "reason": "complete-known-lyric-target-row",
            "candidate": candidate,
            "output_revision": "output",
        }
        outcome = auto_lrc.CrossLineRetryOutcome(
            **base, outcome_revision="outcome-revision"
        )
        report: dict[str, object] = {"assignments": [{"entry": 1}]}
        self.assertIsNone(auto_lrc.record_crossline_retry_identity_hypothesis(report, outcome))
        self.assertIsNone(auto_lrc._bound_local_occurrence_identity(report["assignments"][0]))
    def test_trusted_reference_can_use_unshifted_local_ctc_geometry(self) -> None:
        reference = self.candidate(
            0, 20.0, source="raw-vocal-independent-fusion", current=False, spans=()
        )
        assignment: dict[str, object] = {"entry": 1}
        spans = [
            {"char": token, "start": 20.10 + index * 0.08, "end": 20.12 + index * 0.08, "score": 0.5}
            for index, token in enumerate("abcd")
        ]
        hypothesis = auto_lrc.record_alignment_hypothesis(
            assignment,
            source="ctc-local-window",
            stage="ctc-local-window-realign",
            raw_time=20.10,
            confidence=0.7,
            spans=spans,
            identity_support="supported",
            sequence_support="supported",
            window={"kind": "test-local-window"},
            mode="local-known-lyric-row",
        )
        geometry = auto_lrc._trusted_recurrence_reference_geometry(assignment, reference)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        start, parsed_spans, provenance = geometry
        self.assertAlmostEqual(start, 20.10, places=6)
        self.assertAlmostEqual(parsed_spans[0].start, 20.10, places=6)
        self.assertEqual(provenance["geometry_hypothesis_id"], hypothesis["hypothesis_id"])
        self.assertEqual(provenance["mode"], "self-bound-local-ctc-geometry")
        self.assertAlmostEqual(provenance["selected_start"], 20.0, places=6)

    def test_trusted_reference_never_backfills_distant_local_geometry(self) -> None:
        reference = self.candidate(
            0, 20.0, source="raw-vocal-independent-fusion", current=False, spans=()
        )
        assignment: dict[str, object] = {"entry": 1}
        spans = [
            {"char": token, "start": 20.50 + index * 0.08, "end": 20.52 + index * 0.08, "score": 0.5}
            for index, token in enumerate("abcd")
        ]
        auto_lrc.record_alignment_hypothesis(
            assignment,
            source="ctc-local-window",
            stage="ctc-local-window-realign",
            raw_time=20.50,
            confidence=0.9,
            spans=spans,
            identity_support="supported",
            sequence_support="supported",
            window={"kind": "test-local-window"},
            mode="local-known-lyric-row",
        )
        self.assertIsNone(
            auto_lrc._trusted_recurrence_reference_geometry(assignment, reference)
        )


class V144JointBoundaryGraphRegressionTests(unittest.TestCase):
    def span(self, start: float, score: float = 0.4, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        current: bool = True,
        spans: tuple[auto_lrc.TimingTokenSpan, ...] | None = None,
    ) -> auto_lrc.TimingCandidate:
        if spans is None:
            spans = (
                self.span(time, 0.30, "a"),
                self.span(time + 0.08, 0.40, "b"),
                self.span(time + 0.16, 0.50, "c"),
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"generic line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            current=current,
            source_artifact={"entry": entry, "time": time, "source": source},
        )

    def capability(self) -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": "ctc",
            "helper_path": "ctc_align.py",
            "helper_sha256": "helper-sha-v144",
            "helper_protocol_revision": "protocol-v144",
            "alignment_audio_path": "audio.flac",
            "alignment_audio_sha256": "audio-sha-v144",
            "alignment_audio_source": "vocal-stem",
            "sample_rate": auto_lrc.SAMPLE_RATE,
            "model_identity": "MMS_FA:known-lyric",
            "device": "cpu",
        }
        return auto_lrc.BoundedRetryCapability(
            **payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(payload)
            ),
        )

    def tail_evidence(
        self,
        *,
        producer: str = "joint-ctc-boundary-previous-tail",
        tail_start: float = 9.0,
        tail_end: float = 10.0,
        uncertainty: float = 0.20,
        confidence: float = 0.90,
    ) -> auto_lrc.IndependentTailEvidence:
        cap = self.capability()
        provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=confidence,
            producer=producer,
            backend_lineage=cap.backend_lineage,
            model_identity=cap.model_identity,
            helper_revision=auto_lrc._tail_helper_revision(
                cap.helper_sha256,
                cap.helper_protocol_revision,
            ),
            capability_revision=cap.capability_revision,
            audio_revision=cap.alignment_audio_sha256,
            transcript_revision="test-transcript",
            row_revision="test-row",
        )
        assert provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=confidence,
            producer=producer,
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )

    def fixture(self):
        entries = [
            LyricEntry(["previous lyric"]),
            LyricEntry(["current lyric"]),
            LyricEntry(["next lyric"]),
        ]
        previous = self.candidate(0, 9.74)
        current = self.candidate(1, 10.00)
        following = self.candidate(2, 12.00)
        candidate_sets = ((previous,), (current,), (following,))
        report: dict[str, object] = {
            "backend": "ctc",
            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: self.capability(),
            "timing_entries": 3,
            "assignments": [
                {"entry": 1, "timestamp": 9.74, "timing_repair": "ctc", "score": 0.9},
                {"entry": 2, "timestamp": 10.00, "timing_repair": "ctc", "score": 0.9},
                {"entry": 3, "timestamp": 12.00, "timing_repair": "ctc", "score": 0.9},
            ],
        }
        cap = self.capability()
        helper_revision = auto_lrc._tail_helper_revision(
            cap.helper_sha256,
            cap.helper_protocol_revision,
        )
        report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY] = auto_lrc.TailEvidenceContext(
            backend_lineage=cap.backend_lineage,
            model_identity=cap.model_identity,
            helper_revision=helper_revision,
            capability_revision=cap.capability_revision,
            audio_revision=cap.alignment_audio_sha256,
            transcript_revisions=("test-transcript",),
            row_revisions=("test-row",),
            scopes=(auto_lrc.TailEvidenceScope(
                entry_index=0,
                producer="joint-ctc-boundary-previous-tail",
                backend_lineage=cap.backend_lineage,
                model_identity=cap.model_identity,
                helper_revision=helper_revision,
                capability_revision=cap.capability_revision,
                audio_revision=cap.alignment_audio_sha256,
                transcript_revisions=("test-transcript",),
                row_revisions=("test-row",),
            ),),
        )
        epoch = auto_lrc.freeze_base_retry_epoch(
            "ctc", report, candidate_sets,
            auto_lrc.inactive_legacy_timing_diagnostics(), cap,
        )
        return entries, candidate_sets, report, cap, epoch

    def joint_request(self):
        entries, candidate_sets, report, cap, epoch = self.fixture()
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        requests = auto_lrc.plan_joint_boundary_retry_requests(
            epoch, cap, entries, candidate_sets, 20.0, None, context
        )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        assert isinstance(context, auto_lrc.TailEvidenceContext)
        assert len(context.scopes) == 1
        scope = context.scopes[0]
        report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY] = replace(
            context,
            transcript_revisions=tuple(sorted({
                *context.transcript_revisions,
                request.transcript_digest,
                *request.transcript_digests,
            })),
            row_revisions=tuple(sorted({
                *context.row_revisions,
                request.row_mapping_revision,
            })),
            scopes=(replace(
                scope,
                transcript_revisions=tuple(sorted({
                    *scope.transcript_revisions,
                    request.transcript_digest,
                    *request.transcript_digests,
                })),
                row_revisions=tuple(sorted({
                    *scope.row_revisions,
                    request.row_mapping_revision,
                })),
            ),),
        )
        return entries, candidate_sets, report, cap, epoch, request

    def joint_payload(
        self,
        request: auto_lrc.CrossLineRetryRequest,
        entries: list[LyricEntry],
        *,
        current_start: float = 10.00,
        touch_left_edge: bool = False,
    ) -> dict[str, object]:
        previous_start = request.window_start + 0.05 if touch_left_edge else 9.20
        row_specs = (
            (request.entry_index - 1, tuple(previous_start + 0.08 * i for i in range(4))),
            (request.entry_index, tuple(current_start + 0.08 * i for i in range(4))),
        )
        rows: list[dict[str, object]] = []
        for local_index, (global_index, starts) in enumerate(row_specs):
            spans = [
                {
                    "char": chr(97 + token_index),
                    "start": start,
                    "end": start + 0.02,
                    "score": 0.30 + 0.05 * token_index,
                }
                for token_index, start in enumerate(starts)
            ]
            rows.append({
                "entry": local_index + 1,
                "text": auto_lrc.entry_sung_text(entries[global_index]),
                "romaji": "".join(str(span["char"]) for span in spans),
                "start": spans[0]["start"],
                "end": spans[-1]["end"],
                "ctc_score": 0.50,
                "tokens": len(spans),
                "token_spans": spans,
            })
        return {
            "window_start": round(request.window_start, 3),
            "window_end": round(request.window_end, 3),
            "entries": rows,
        }

    def test_joint_planner_uses_previous_and_current_rows_only(self) -> None:
        entries, candidate_sets, report, cap, epoch = self.fixture()
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        decision = auto_lrc.build_final_timing_state(candidate_sets).decisions[1]
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.evidence.ownership.status, "unknown")
        requests = auto_lrc.plan_joint_boundary_retry_requests(
            epoch, cap, entries, candidate_sets, 20.0, None, context
        )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.ordinal, 2)
        self.assertEqual(request.context_entry_indexes, (0, 1))
        self.assertEqual(request.row_mapping, ((0, 0), (1, 1)))
        self.assertEqual(len(request.transcript_digests), 2)
        self.assertTrue(
            auto_lrc.crossline_retry_request_is_current(
                request,
                epoch,
                cap,
                entries,
                candidate_sets,
                20.0,
                None,
                context,
            )
        )

    def test_retry_planners_respect_ctc_first_when_whisper_would_transiently_win(self) -> None:
        entries, base_candidate_sets, report, cap, _epoch = self.fixture()
        whisper = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="generic line 1",
            source="whisper-policy-probe",
            raw_time=10.70,
            confidence=0.95,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported",
            direct_onset_time=10.70,
            direct_onset_source_artifact={"kind": "policy-probe"},
            direct_onset_evidence_producer="phonetic-onset-fusion",
            direct_onset_evidence_kind="phonetic-onset",
            direct_onset_evidence_independent=True,
            current=False,
            source_artifact={"kind": "policy-probe"},
        )
        candidate_sets = (
            base_candidate_sets[0],
            (*base_candidate_sets[1], whisper),
            base_candidate_sets[2],
        )
        epoch = auto_lrc.freeze_base_retry_epoch(
            "ctc",
            report,
            candidate_sets,
            auto_lrc.inactive_legacy_timing_diagnostics(),
            cap,
        )
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]

        default_state = auto_lrc.build_final_timing_state(candidate_sets)
        self.assertEqual(default_state.decisions[1].status, "selected_valid")
        self.assertEqual(default_state.decisions[1].candidate.source, "whisper-policy-probe")
        ctc_first_state = auto_lrc.build_final_timing_state(
            candidate_sets,
            timing_source_policy="ctc-first",
        )
        self.assertEqual(ctc_first_state.decisions[1].status, "provisional_unresolved")

        joint_requests = auto_lrc.plan_joint_boundary_retry_requests(
            epoch,
            cap,
            entries,
            candidate_sets,
            20.0,
            None,
            context,
            timing_source_policy="ctc-first",
        )
        regular_requests = auto_lrc.plan_crossline_retry_requests(
            epoch,
            cap,
            entries,
            candidate_sets,
            20.0,
            None,
            context,
            timing_source_policy="ctc-first",
        )
        self.assertEqual([request.entry_index for request in joint_requests], [1])
        self.assertEqual([request.entry_index for request in regular_requests], [1])

    def test_retry_executor_passes_ctc_first_policy_to_production_planner(self) -> None:
        entries, candidate_sets, report, cap, epoch = self.fixture()
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        requests = auto_lrc.plan_crossline_retry_requests(
            epoch,
            cap,
            entries,
            candidate_sets,
            20.0,
            None,
            context,
            timing_source_policy="ctc-first",
        )
        self.assertTrue(requests)
        with split_patch.patch("plan_crossline_retry_requests", return_value=()
        ) as planner:
            result = auto_lrc.execute_crossline_retry_epoch(
                epoch,
                cap,
                entries,
                candidate_sets,
                20.0,
                (requests[0],),
                tail_evidence_context=context,
                timing_source_policy="ctc-first",
            )
        self.assertEqual(result, ())
        self.assertEqual(planner.call_args.args[-1], "ctc-first")

    def test_complete_joint_pair_is_observation_only(self) -> None:
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries)
        )
        self.assertEqual(outcome.status, "candidate")
        self.assertEqual(outcome.reason, "complete-joint-boundary-rows")
        assert outcome.candidate is not None
        self.assertEqual(outcome.candidate.source, "ctc-joint-boundary-observation")
        self.assertEqual(outcome.candidate.direct_onset_support, "unavailable")
        self.assertIsNone(outcome.candidate.direct_onset_evidence)
        # Even a geometrically strong pair observation cannot by itself cross
        # the old unknown-tail relation; only the separately bound consensus may.
        raw_eval = auto_lrc.evaluate_timing_candidate(
            outcome.candidate, candidate_sets[0][0]
        )
        self.assertFalse(raw_eval.valid)
        self.assertIn("previous-tail-ownership-unknown", raw_eval.rejection_reasons)
        self.assertIsNotNone(outcome.previous_tail_evidence)
        assert outcome.previous_tail_evidence is not None
        self.assertEqual(
            outcome.previous_tail_evidence.entry_index,
            request.entry_index - 1,
        )
        self.assertEqual(
            outcome.previous_tail_evidence.producer,
            "joint-ctc-boundary-previous-tail",
        )
        self.assertGreater(
            outcome.previous_tail_evidence.tail_end,
            outcome.previous_tail_evidence.tail_start,
        )
        self.assertIsNotNone(
            auto_lrc._retry_diagnostic_payload(outcome)["previous_tail_evidence"]
        )

    def test_strong_joint_previous_tail_rebinds_without_moving_timestamp(self) -> None:
        entries, _fixture_sets, report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries)
        )
        assert outcome.previous_tail_evidence is not None
        predecessor = self.candidate(
            0, 9.74, source="raw-vocal-independent-fusion", spans=()
        )
        # The joint CTC tail is authoritative only for a disjoint current
        # opening; same-lineage CTC ownership is covered by Contract-A tests.
        current = self.candidate(1, 10.00, source="raw-vocal-independent-fusion")
        following = self.candidate(2, 12.00)
        candidate_sets = ((predecessor,), (current,), (following,))
        changed = auto_lrc.merge_runtime_independent_tail_evidence(
            report, outcome.previous_tail_evidence
        )
        self.assertTrue(changed)
        tails = auto_lrc.runtime_independent_tail_evidence(
            report, len(candidate_sets), report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        )
        before = auto_lrc.build_final_timing_state(candidate_sets)
        after = auto_lrc.build_final_timing_state(
            candidate_sets,
            independent_tail_evidence=tails,
            tail_evidence_context=report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY],
        )
        self.assertEqual(before.written_times, after.written_times)
        self.assertEqual(after.decisions[1].evidence.ownership.status, "clear")
        self.assertTrue(after.audits[1].timing_trusted)

    def test_joint_previous_tail_merge_is_monotonic(self) -> None:
        report: dict[str, object] = {}
        cap = self.capability()
        report[auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY] = cap
        existing = self.tail_evidence()
        weaker = self.tail_evidence(
            producer="joint-ctc-boundary-previous-tail-weaker",
            tail_start=9.2,
            tail_end=9.8,
            uncertainty=0.05,
            confidence=0.99,
        )
        later = self.tail_evidence(
            producer="joint-ctc-boundary-previous-tail-later",
            tail_end=10.4,
            uncertainty=0.10,
            confidence=0.80,
        )
        self.assertTrue(auto_lrc.merge_runtime_independent_tail_evidence(report, existing))
        self.assertTrue(auto_lrc.merge_runtime_independent_tail_evidence(report, weaker))
        retained = auto_lrc.runtime_independent_tail_evidence(report, 1)[0]
        assert retained is not None
        self.assertEqual(len(retained), 2)
        self.assertTrue(auto_lrc.merge_runtime_independent_tail_evidence(report, later))
        retained = auto_lrc.runtime_independent_tail_evidence(report, 1)[0]
        assert retained is not None
        self.assertEqual(len(retained), 3)

    def test_joint_previous_tail_rejects_wrong_row_mapping_and_edge_pair(self) -> None:
        entries, _candidate_sets, _report, cap, _epoch, request = self.joint_request()
        wrong_text = self.joint_payload(request, entries)
        wrong_text["entries"][0]["text"] = "wrong immutable lyric"
        invalid = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, wrong_text
        )
        self.assertEqual(invalid.status, "invalid")
        self.assertIsNone(invalid.previous_tail_evidence)

        edge_pair = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries, touch_left_edge=True)
        )
        self.assertEqual(edge_pair.status, "candidate")
        self.assertIsNone(edge_pair.previous_tail_evidence)
        assert edge_pair.candidate is not None
        self.assertEqual(edge_pair.candidate.source, "ctc-joint-boundary-observation-weak")

    def test_joint_consensus_keeps_parent_timestamp_and_resolves_unknown_boundary(self) -> None:
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries)
        )
        augmented = auto_lrc._augment_candidates_with_joint_boundary_consensus(
            candidate_sets, (outcome,), entries
        )
        composites = [
            candidate for candidate in augmented[1]
            if candidate.source == "ctc-joint-boundary-consensus"
        ]
        self.assertEqual(len(composites), 1)
        composite = composites[0]
        self.assertEqual(composite.written_time, candidate_sets[1][0].written_time)
        self.assertTrue(auto_lrc.candidate_has_bound_direct_onset_evidence(composite))
        self.assertTrue(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(composite))
        assert composite.direct_onset_evidence is not None
        self.assertEqual(composite.direct_onset_evidence.kind, "joint-lyric-boundary")
        self.assertEqual(
            composite.direct_onset_evidence.producer,
            "joint-ctc-boundary-consensus",
        )
        evaluation = auto_lrc.evaluate_timing_candidate(
            composite, candidate_sets[0][0]
        )
        self.assertTrue(evaluation.valid)
        self.assertTrue(evaluation.evidence.minimum_proof_satisfied)
        self.assertEqual(evaluation.evidence.structural_failures, ())

    def test_joint_consensus_preserves_joint_row_tail_when_parent_has_no_spans(self) -> None:
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries)
        )
        assert outcome.candidate is not None
        geometryless = self.candidate(
            1, 10.00, source="ctc-current", current=False, spans=()
        )
        modified = (candidate_sets[0], (geometryless,), candidate_sets[2])
        augmented = auto_lrc._augment_candidates_with_joint_boundary_consensus(
            modified, (outcome,), entries
        )
        composite = next(
            candidate for candidate in augmented[1]
            if candidate.source == "ctc-joint-boundary-consensus"
        )
        self.assertEqual(composite.written_time, geometryless.written_time)
        self.assertEqual(composite.spans, outcome.candidate.spans)
        tail = auto_lrc.derive_reliable_tail(composite)
        self.assertEqual(tail.validity, "valid")
        decision = auto_lrc.select_timing_decision(
            entry_index=2,
            candidates=candidate_sets[2],
            previous_candidate=composite,
            previous_tail=tail,
            selection_revision=1,
        )
        self.assertEqual(decision.evidence.ownership.status, "clear")

    def test_joint_consensus_does_not_borrow_spans_that_start_elsewhere(self) -> None:
        # Regression (17fd62e5): a span-less parent borrowed observation spans
        # starting 0.06 s away, so the joint candidate failed its own
        # candidate-span-revision-mismatch check (43 joints lost, ふり #55).
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries,
            self.joint_payload(request, entries, current_start=10.06),
        )
        assert outcome.candidate is not None
        self.assertGreater(
            abs(outcome.candidate.spans[0].start - 10.00), auto_lrc.HOP_SECONDS
        )
        geometryless = self.candidate(
            1, 10.00, source="ctc-current", current=False, spans=()
        )
        modified = (candidate_sets[0], (geometryless,), candidate_sets[2])
        augmented = auto_lrc._augment_candidates_with_joint_boundary_consensus(
            modified, (outcome,), entries
        )
        composite = next(
            candidate for candidate in augmented[1]
            if candidate.source == "ctc-joint-boundary-consensus"
        )
        self.assertEqual(composite.written_time, geometryless.written_time)
        self.assertEqual(composite.spans, ())
        evaluation = auto_lrc.evaluate_timing_candidate(
            composite, candidate_sets[0][0]
        )
        self.assertNotIn("candidate-span-revision-mismatch", evaluation.rejection_reasons)

    def test_joint_observation_does_not_fuse_distant_parent(self) -> None:
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries)
        )
        distant = self.candidate(1, 10.50, source="ctc-local-window", current=False)
        modified = (candidate_sets[0], (distant,), candidate_sets[2])
        augmented = auto_lrc._augment_candidates_with_joint_boundary_consensus(
            modified, (outcome,), entries
        )
        self.assertFalse(any(
            candidate.source == "ctc-joint-boundary-consensus"
            for candidate in augmented[1]
        ))

    def test_left_edge_truncated_joint_pair_cannot_certify_boundary(self) -> None:
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries,
            self.joint_payload(request, entries, touch_left_edge=True),
        )
        self.assertEqual(outcome.status, "candidate")
        assert outcome.candidate is not None
        self.assertEqual(outcome.candidate.source, "ctc-joint-boundary-observation-weak")
        self.assertTrue(outcome.candidate.window_truncated)
        augmented = auto_lrc._augment_candidates_with_joint_boundary_consensus(
            candidate_sets, (outcome,), entries
        )
        self.assertFalse(any(
            candidate.source == "ctc-joint-boundary-consensus"
            for candidate in augmented[1]
        ))

    def test_joint_boundary_evidence_producer_cannot_be_spoofed(self) -> None:
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="generic",
            source="synthetic",
            raw_time=10.0,
            spans=(self.span(10.0), self.span(10.08), self.span(10.16)),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported",
            direct_onset_time=10.0,
            direct_onset_source_artifact={"source": "spoof"},
            direct_onset_evidence_producer="wrong-producer",
            direct_onset_evidence_kind="joint-lyric-boundary",
            direct_onset_evidence_independent=True,
        )
        self.assertFalse(auto_lrc.candidate_has_bound_direct_onset_evidence(candidate))
        self.assertFalse(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(candidate))

    def test_duplicate_offset_model_preserves_four_pair_evidence_floor(self) -> None:
        entries = [LyricEntry(["A"]), LyricEntry(["B"]), LyricEntry(["A"]), LyricEntry(["B"])]
        timestamps = [0.0, 1.0, 50.0, 51.0]
        consensus = auto_lrc.duplicate_lyric_offset_consensus(entries, timestamps)
        self.assertFalse(consensus["available"])
        self.assertEqual(consensus["reason"], "fewer-than-four-duplicate-pairs")
        self.assertEqual(consensus["pair_count"], 2)

    def test_duplicate_offsets_are_clustered_by_lyric_index_displacement(self) -> None:
        entries = [
            LyricEntry(["A"]), LyricEntry(["B"]),
            LyricEntry(["A"]), LyricEntry(["B"]),
            LyricEntry(["C"]), LyricEntry(["D"]),
            LyricEntry(["X"]), LyricEntry(["Y"]),
            LyricEntry(["C"]), LyricEntry(["D"]),
        ]
        timestamps = [
            0.00, 1.00, 75.68, 76.70,
            100.00, 101.00, 150.00, 180.00, 232.92, 233.94,
        ]
        consensus = auto_lrc.duplicate_lyric_offset_consensus(entries, timestamps)
        self.assertTrue(consensus["available"])
        self.assertEqual(consensus["mode"], "lyric-index-displacement-clusters")
        clusters = {
            int(cluster["entry_delta"]): cluster
            for cluster in consensus["clusters"]
        }
        self.assertEqual(set(clusters), {2, 4})
        self.assertTrue(clusters[2]["available"])
        self.assertTrue(clusters[4]["available"])
        self.assertAlmostEqual(float(clusters[2]["median_delta_seconds"]), 75.69, places=2)
        self.assertAlmostEqual(float(clusters[4]["median_delta_seconds"]), 132.93, places=2)
        self.assertLessEqual(float(clusters[2]["mad_seconds"]), 0.02)
        self.assertLessEqual(float(clusters[4]["mad_seconds"]), 0.02)


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
