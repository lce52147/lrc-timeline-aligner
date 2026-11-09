#!/usr/bin/env python3
"""Georgette continued, occurrence recurrence, fixed-point and boundary recovery tests."""

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


class GeorgetteBoundaryRegressionContinuedTests(unittest.TestCase):
    def test_final_guard_prefers_independent_prefix_probe_over_weak_zero_gap_path(self) -> None:
        frame_times = np.round(np.arange(10.0, 15.0, 0.01), 3).astype(np.float32)
        onset = np.zeros(len(frame_times), dtype=np.float32)
        onset[int(round((12.75 - 10.0) / 0.01))] = 1.0
        features = AudioFeatures(
            duration=20.0,
            frame_times=frame_times,
            rms_db=np.full(len(frame_times), -20.0, dtype=np.float32),
            onset_strength=onset,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        original_analyze_vocal = auto_lrc.analyze_vocal_onsets
        original_prefix_probe = auto_lrc.run_ctc_prefix_probe
        split_patch.assign("analyze_audio", lambda *_args, **_kwargs: features)  # type: ignore[assignment]
        split_patch.assign("analyze_vocal_onsets", lambda *_args, **_kwargs: features)  # type: ignore[assignment]
        ctc_refinement.run_ctc_prefix_probe = lambda *_args, **_kwargs: {  # type: ignore[assignment]
            "accepted": True,
            "candidate": 12.75,
            "ctc_score": 0.18,
            "prefix_mean_score": 0.14,
            "prefix_weak_ratio": 0.25,
            "prefix_span_seconds": 0.9,
            "prefix_text": "everlastin",
        }
        try:
            entries = [LyricEntry(["... again"]), LyricEntry(["everlastin ever"])]
            report: dict[str, object] = {
                "backend": "ctc",
                "timing_entries": 2,
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 10.0,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "ctc_token_spans": [
                            {"char": "n", "start": 11.9, "end": 12.0, "score": 0.01},
                        ],
                    },
                    {
                        "entry": 2,
                        "timestamp": 12.02,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "ctc_score": 0.1,
                        "ctc_token_spans": [
                            {"char": "e", "start": 12.02, "end": 12.04, "score": 0.01},
                            {"char": "v", "start": 12.08, "end": 12.10, "score": 0.001},
                            {"char": "e", "start": 12.14, "end": 12.16, "score": 0.001},
                            {"char": "r", "start": 12.20, "end": 12.22, "score": 0.001},
                            {"char": "l", "start": 12.28, "end": 12.30, "score": 0.001},
                            {"char": "a", "start": 12.80, "end": 12.82, "score": 0.2},
                            {"char": "s", "start": 12.9, "end": 12.92, "score": 0.001},
                            {"char": "t", "start": 13.0, "end": 13.02, "score": 0.001},
                        ],
                    },
                ],
                "suspicious_alignments": [],
            }
            args = auto_lrc.build_parser().parse_args(["dummy.txt"])
            timestamps, report, changes = apply_final_ctc_timing_guard(
                Path("dummy.flac"), entries, [10.0, 12.02], report, 20.0, args
            )
            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 12.75, places=3)
            self.assertEqual(changes[0]["reason"], "independent-lyric-prefix-ctc-probe")
            self.assertTrue(report["assignments"][1]["ctc_prefix_probe_recovery"])  # type: ignore[index]
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]
            split_patch.assign("analyze_vocal_onsets", original_analyze_vocal)  # type: ignore[assignment]
            ctc_refinement.run_ctc_prefix_probe = original_prefix_probe  # type: ignore[assignment]

    def test_untrusted_crossline_recovery_does_not_leave_stale_effective_index(self) -> None:
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 70.0,
                    "ctc_token_spans": [{"start": 77.943, "end": 77.963, "score": 0.01}],
                },
                {
                    "entry": 2,
                    "timestamp": 77.963,
                    "ctc_token_spans": [
                        {"start": 77.963, "end": 77.983, "score": 0.003},
                        {"start": 83.205, "end": 83.225, "score": 0.099},
                        {"start": 83.325, "end": 83.345, "score": 0.009},
                    ],
                },
                {"entry": 3, "timestamp": 92.33, "ctc_token_spans": [{"start": 92.33, "end": 92.35, "score": 0.1}]},
            ]
        }
        timestamps, report, changes = apply_ctc_crossline_initial_recovery(
            [70.0, 77.963, 92.33], report, 100.0
        )
        self.assertEqual(len(changes), 1)
        current = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(current["ctc_crossline_initial_recovery_trusted"])
        self.assertNotIn("ctc_effective_token_start_index", current)

    def test_0247_inherits_previous_right_edge_pileup_contamination(self) -> None:
        previous_tail = [
            {
                "char": "x",
                "start": 166.687 + index * 0.04,
                "end": 166.707 + index * 0.04,
                "score": 0.005,
            }
            for index in range(20)
        ]
        # Make the final token land exactly on the local-window edge.
        shift = 167.447 - previous_tail[-1]["end"]
        for span in previous_tail:
            span["start"] += shift
            span["end"] += shift

        assignments: list[object] = [
            {
                "entry": 1,
                "timestamp": 147.340,
                "ctc_score": 0.019063,
                "ctc_local_window": {"start": 136.896, "end": 167.447},
                "ctc_token_spans": previous_tail,
            },
            {
                "entry": 2,
                "timestamp": 167.468,
                "ctc_score": 0.500062,
                "ctc_token_spans": [
                    {"char": "k", "start": 167.468, "end": 167.488, "score": 0.003825},
                    {"char": "o", "start": 167.588, "end": 167.608, "score": 0.618277},
                    {"char": "n", "start": 171.272, "end": 171.292, "score": 0.028629},
                    {"char": "o", "start": 172.594, "end": 172.614, "score": 0.261168},
                    {"char": "m", "start": 172.954, "end": 172.974, "score": 0.924104},
                    {"char": "a", "start": 173.074, "end": 173.094, "score": 0.942513},
                    {"char": "m", "start": 173.575, "end": 173.595, "score": 0.035145},
                    {"char": "a", "start": 173.715, "end": 173.735, "score": 0.957866},
                ],
            },
        ]

        annotate_ctc_boundary_evidence(assignments)

        self.assertTrue(assignments[0]["ctc_right_edge_pileup"])  # type: ignore[index]
        evidence = assignments[1]["ctc_boundary_evidence"]  # type: ignore[index]
        self.assertTrue(evidence["prior_line_truncation_contamination"])
        self.assertTrue(evidence["realign_required"])

        report: dict[str, object] = {"assignments": assignments}
        timestamps = [147.340, 167.468]
        timestamps, report, changes = apply_ctc_crossline_initial_recovery(
            timestamps, report, 185.028
        )
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["mode"], "short-borrowed-tail-prefix")
        self.assertAlmostEqual(timestamps[1], 167.588, places=3)
        self.assertTrue(assignments[1]["ctc_crossline_initial_recovery_trusted"])  # type: ignore[index]


class OccurrenceRecurrenceZeroTrustTests(unittest.TestCase):
    def test_bound_vocal_onset_can_prove_single_reliable_opening_without_fracture(self) -> None:
        spans = [auto_lrc.TimingTokenSpan(10.0, 10.02, 0.30, "a")]
        spans.append(auto_lrc.TimingTokenSpan(10.30, 10.32, 0.01, "b"))
        spans.extend(
            auto_lrc.TimingTokenSpan(10.70 + index * 0.42, 10.72 + index * 0.42, 0.20, "c")
            for index in range(8)
        )
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="target",
            source="ctc-vocal-onset-confirmed",
            raw_time=10.0,
            spans=tuple(spans),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=0.95,
            acoustic_onset_time=10.04,
            acoustic_source_artifact={"audio": "independent"},
            acoustic_evidence_producer="consonant-onset-detector",
            acoustic_evidence_kind="vocal-onset",
            acoustic_evidence_independent=True,
            direct_onset_time=10.04,
            direct_onset_source_artifact={"audio": "independent"},
            direct_onset_evidence_producer="consonant-onset-detector",
            direct_onset_evidence_kind="vocal-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"producer": "vocal-onset"},
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate)
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.temporal_coherence, "coherent")
        self.assertTrue(evaluation.evidence.minimum_proof_satisfied)

    def test_bound_vocal_onset_does_not_rescue_short_collapsed_path(self) -> None:
        spans = (
            auto_lrc.TimingTokenSpan(10.0, 10.02, 0.30, "a"),
            auto_lrc.TimingTokenSpan(10.18, 10.20, 0.01, "b"),
            auto_lrc.TimingTokenSpan(10.40, 10.42, 0.20, "c"),
        )
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="target",
            source="ctc-vocal-onset-confirmed",
            raw_time=10.0,
            spans=spans,
            confidence=0.95,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=1.0,
            acoustic_onset_time=10.0,
            acoustic_source_artifact={"audio": "independent"},
            acoustic_evidence_producer="consonant-onset-detector",
            acoustic_evidence_kind="vocal-onset",
            acoustic_evidence_independent=True,
            direct_onset_time=10.0,
            direct_onset_source_artifact={"audio": "independent"},
            direct_onset_evidence_producer="consonant-onset-detector",
            direct_onset_evidence_kind="vocal-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"producer": "vocal-onset"},
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate)
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.minimum_proof_satisfied)

    def test_independent_content_identity_is_recurrence_source_scoped(self) -> None:
        common = dict(
            entry_index=0,
            entry_text="same lyric",
            raw_time=20.0,
            spans=(),
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=0.9,
            acoustic_onset_time=20.0,
            acoustic_source_artifact={"recurrence": 1},
            acoustic_evidence_producer="global-recurrence-search",
            acoustic_evidence_kind="recurrence-onset",
            acoustic_evidence_independent=True,
            direct_onset_time=20.0,
            direct_onset_source_artifact={"recurrence": 1},
            direct_onset_evidence_producer="global-recurrence-search",
            direct_onset_evidence_kind="recurrence-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"recurrence": 1},
        )
        recurrence_unbound = auto_lrc.make_timing_candidate(source="global-acoustic-recurrence", **common)
        recurrence = auto_lrc.make_timing_candidate(
            source="global-acoustic-recurrence", independent_content_identity=True, **common
        )
        ordinary = auto_lrc.make_timing_candidate(
            source="ctc-vocal-onset-confirmed", independent_content_identity=True, **common
        )
        self.assertFalse(auto_lrc.candidate_supplies_independent_content_identity(recurrence_unbound))
        self.assertTrue(auto_lrc.candidate_supplies_independent_content_identity(recurrence))
        self.assertFalse(auto_lrc.candidate_supplies_independent_content_identity(ordinary))

    def test_vocal_onset_confirmation_never_mutates_baseline(self) -> None:
        frame_times = np.arange(0.0, 20.0, 0.02, dtype=np.float32)
        onset = np.zeros_like(frame_times)
        onset[int(round(10.0 / 0.02))] = 1.0
        features = AudioFeatures(
            duration=20.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset,
            segments=[],
        )
        original = auto_lrc.analyze_vocal_onsets
        split_patch.assign("analyze_vocal_onsets", lambda _audio, _duration: features)  # type: ignore[assignment]
        try:
            spans = [
                {"char": chr(97 + (index % 20)), "start": 10.0 + index * 0.40, "end": 10.02 + index * 0.40, "score": 0.25}
                for index in range(10)
            ]
            report: dict[str, object] = {
                "backend": "ctc",
                "assignments": [{"entry": 1, "timestamp": 10.0, "score": 0.9, "ctc_token_spans": spans}],
            }
            timestamps = [10.0]
            before = copy.deepcopy(report)
            generated = auto_lrc.apply_vocal_onset_confirmation_hypotheses(
                Path(__file__), [LyricEntry(["target"])], timestamps, report, 20.0
            )
            self.assertEqual(timestamps, [10.0])
            self.assertEqual(report["assignments"][0]["timestamp"], before["assignments"][0]["timestamp"])  # type: ignore[index]
            self.assertEqual(len(generated), 1)
            self.assertFalse(report["vocal_onset_confirmation_authoritative"])
            self.assertFalse(report["vocal_onset_confirmation_mutates_baseline"])
        finally:
            split_patch.assign("analyze_vocal_onsets", original)  # type: ignore[assignment]

    def test_global_recurrence_search_requires_a_distinct_full_line_match(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(0.0, 0.15, size=(8, 1200)).astype(np.float32)
        template = rng.normal(0.0, 1.0, size=(8, 200)).astype(np.float32)
        features[:, 100:300] = template
        features[:, 700:900] = template + rng.normal(0.0, 0.01, size=template.shape).astype(np.float32)
        result = auto_lrc._search_recurrence_start(features, 2.0, 6.0, 10.0, 22.0)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(float(result["start"]), 14.0, delta=0.12)
        self.assertGreater(float(result["score"]), 0.95)
        self.assertGreater(float(result["margin"]), auto_lrc.RECURRENCE_MIN_DISTINCT_MARGIN)
    def test_recurrence_identity_provenance_fails_closed_on_margin_or_audio_mismatch(self) -> None:
        lyric = LyricEntry(["same lyric"])
        audio = Path(__file__)
        audio_sha = auto_lrc._path_sha256(audio)
        lyric_sha = auto_lrc.hashlib.sha256(
            auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(lyric)).encode("utf-8")
        ).hexdigest()
        base = {
            "source": "global-acoustic-recurrence",
            "stage": "occurrence-level-global-recurrence-search",
            "raw_candidate_time": 20.0,
            "authoritative": False,
            "mode": "full-line-acoustic-recurrence-with-trusted-reference",
            "parent": {"source": "central-trusted-repeat-reference", "raw_candidate_time": 10.0},
            "window": {
                "search_start": 15.0,
                "search_end": 25.0,
                "audio_sha256": audio_sha,
                "feature_revision": auto_lrc.RECURRENCE_FEATURE_REVISION,
                "reference_candidate_id": "trusted-ref",
                "similarity": auto_lrc.RECURRENCE_MIN_SIMILARITY + 0.1,
                "distinct_margin": auto_lrc.RECURRENCE_MIN_DISTINCT_MARGIN + 0.1,
                "normalized_lyric_sha256": lyric_sha,
            },
        }
        base["hypothesis_id"] = auto_lrc._timing_content_digest(base)
        self.assertTrue(auto_lrc.recurrence_hypothesis_provenance_is_valid(base, lyric, audio))
        weak = copy.deepcopy(base)
        weak["window"]["distinct_margin"] = auto_lrc.RECURRENCE_MIN_DISTINCT_MARGIN - 0.001  # type: ignore[index]
        self.assertFalse(auto_lrc.recurrence_hypothesis_provenance_is_valid(weak, lyric, audio))
        wrong_audio = copy.deepcopy(base)
        wrong_audio["window"]["audio_sha256"] = "0" * 64  # type: ignore[index]
        self.assertFalse(auto_lrc.recurrence_hypothesis_provenance_is_valid(wrong_audio, lyric, audio))

    def test_global_recurrence_ambiguous_distinct_matches_fail_margin_gate(self) -> None:
        rng = np.random.default_rng(11)
        features = rng.normal(0.0, 0.05, size=(8, 1400)).astype(np.float32)
        template = rng.normal(0.0, 1.0, size=(8, 200)).astype(np.float32)
        features[:, 100:300] = template
        features[:, 600:800] = template
        features[:, 1000:1200] = template
        result = auto_lrc._search_recurrence_start(features, 2.0, 6.0, 10.0, 26.0)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(float(result["score"]), 0.95)
        self.assertLess(float(result["margin"]), auto_lrc.RECURRENCE_MIN_DISTINCT_MARGIN)


class FixedPointPhaseCZeroTrustTests(unittest.TestCase):
    @staticmethod
    def _features_with_peaks(peaks: list[tuple[float, float]]) -> AudioFeatures:
        frame_times = np.arange(0.0, 6.0, 0.02, dtype=np.float32)
        onset = np.zeros_like(frame_times)
        for time, strength in peaks:
            onset[int(round(time / 0.02))] = strength
        return AudioFeatures(
            duration=6.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -30.0),
            onset_strength=onset,
            segments=[],
        )

    def _phonetic_runtime_fixture(
        self, audio: Path
    ) -> tuple[AudioFeatures, dict[str, object], dict[str, object]]:
        frame_times = np.arange(0.0, 20.0, 0.02, dtype=np.float32)
        onset = np.zeros_like(frame_times)
        onset[int(round(10.04 / 0.02))] = 0.95
        features = AudioFeatures(
            duration=20.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -30.0),
            onset_strength=onset,
            segments=[],
        )
        peak = {"time": 10.0, "score": auto_lrc.PHONETIC_ONSET_MIN_POSTERIOR + 0.10}
        spans = [
            {
                "char": chr(97 + index),
                "start": 10.003 + index * 0.12,
                "end": 10.053 + index * 0.12,
                "score": 0.8,
            }
            for index in range(8)
        ]
        assignment: dict[str, object] = {
            "entry": 1,
            "timestamp": 10.003,
            "score": 0.9,
            "timing_repair": "ctc",
            "ctc_token_spans": spans,
            "ctc_first_token_candidates": [peak],
        }
        current_candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="provenance-check",
            source="ctc-current",
            raw_time=10.003,
            spans=tuple(
                auto_lrc.TimingTokenSpan(
                    float(span["start"]), float(span["end"]), float(span["score"]), str(span["char"])
                )
                for span in spans
            ),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
        )
        window: dict[str, object] = {
            "feature_revision": auto_lrc.PHONETIC_ONSET_FEATURE_REVISION,
            "audio_sha256": auto_lrc._path_sha256(audio),
            "audio_duration_seconds": 20.0,
            "current_candidate_id": current_candidate.candidate_id,
            "current_raw_time": 10.003,
            "current_written_centiseconds": 1000,
            "current_time": 10.0,
            "ctc_peak_time": 10.0,
            "ctc_peak_score": peak["score"],
            "ctc_peak_digest": auto_lrc._timing_content_digest(peak),
            "onset_time": 10.04,
            "onset_strength": 0.95,
            "onset_prominence": 0.95,
        }
        holder: dict[str, object] = {}
        hypothesis = auto_lrc.record_alignment_hypothesis(
            holder,
            source="ctc-phonetic-onset-fusion",
            stage="independent-phonetic-onset-fusion",
            raw_time=10.0,
            confidence=0.9,
            spans=[],
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=float(window["onset_strength"]),
            acoustic_onset_time=10.04,
            direct_onset_time=10.04,
            parent_source="ctc-current",
            parent_time=10.003,
            window=window,
            mode="coherent-ctc-first-token-plus-independent-onset",
        )
        return features, assignment, hypothesis

    def test_tail_release_chooses_first_new_cluster_not_strongest_late_peak(self) -> None:
        features = self._features_with_peaks([
            (1.92, 0.90),
            (2.00, 0.94),
            (2.08, 0.92),
            (2.16, 0.91),
            (2.46, 0.86),
            (3.20, 1.00),
        ])
        result = auto_lrc._first_post_tail_cluster_onset(
            features,
            2.05,
            3.50,
            min_strength=0.82,
            min_prominence=0.50,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result["time"], 2.46, places=2)
        self.assertGreaterEqual(result["cluster_gap"], auto_lrc.TAIL_RELEASE_CLUSTER_BREAK_SECONDS)
        self.assertNotAlmostEqual(result["time"], 3.20, places=2)

    def test_tail_release_abstains_without_observable_cluster_break(self) -> None:
        features = self._features_with_peaks([
            (1.98, 0.90),
            (2.10, 0.93),
            (2.20, 0.95),
            (2.34, 0.96),
        ])
        result = auto_lrc._first_post_tail_cluster_onset(
            features,
            2.05,
            2.40,
            min_strength=0.82,
            min_prominence=0.50,
        )
        self.assertIsNone(result)

    def _mutual_pair_fixture(self) -> tuple[list[LyricEntry], list[object], dict[str, object], dict[str, object]]:
        audio = Path(__file__)
        entries = [LyricEntry(["same lyric"]), LyricEntry(["same lyric"])]
        assignments: list[object] = [
            {"entry": 1, "timestamp": 9.8, "timing_repair": "ctc", "score": 0.3},
            {"entry": 2, "timestamp": 29.8, "timing_repair": "ctc", "score": 0.3},
        ]
        for index, assignment in enumerate(assignments):
            assert isinstance(assignment, dict)
            start_time = 10.0 + index * 20.0
            auto_lrc.record_alignment_hypothesis(
                assignment,
                source="ctc-local-window",
                stage="ctc-local-window-realign",
                raw_time=start_time,
                confidence=0.6,
                spans=[
                    {"char": chr(97 + n), "start": start_time + n * 0.08, "end": start_time + n * 0.08 + 0.02, "score": 0.2}
                    for n in range(6)
                ],
                identity_support="supported",
                sequence_support="supported",
                parent_source="ctc-global-mms",
                parent_time=float(assignment["timestamp"]),
                mode="local-window-independent-candidate",
            )
        left_identity = auto_lrc._bound_local_occurrence_identity(assignments[0])  # type: ignore[arg-type]
        right_identity = auto_lrc._bound_local_occurrence_identity(assignments[1])  # type: ignore[arg-type]
        self.assertIsNotNone(left_identity)
        self.assertIsNotNone(right_identity)
        normalized = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[0]))
        lyric_sha = auto_lrc.hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        payload: dict[str, object] = {
            "audio_sha256": auto_lrc._path_sha256(audio),
            "feature_revision": auto_lrc.MUTUAL_RECURRENCE_FEATURE_REVISION,
            "normalized_lyric_sha256": lyric_sha,
            "left_entry": 1,
            "right_entry": 2,
            "left_seed": {"time": 10.0, "strength": 0.8, "kind": "ctc-local-first-token", "evidence_id": "L"},
            "right_seed": {"time": 30.0, "strength": 0.9, "kind": "ctc-local-first-token", "evidence_id": "R"},
            "left_identity_evidence": left_identity,
            "right_identity_evidence": right_identity,
            "left_bounds": [5.0, 20.0],
            "right_bounds": [25.0, 40.0],
            "left_candidate": 10.2,
            "right_candidate": 30.2,
            "template_duration": 5.0,
            "similarity": auto_lrc.MUTUAL_RECURRENCE_MIN_SIMILARITY + 0.10,
            "runner_similarity": 0.40,
            "distinct_margin": auto_lrc.MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN + 0.10,
        }
        pair_id = auto_lrc._timing_content_digest(payload)
        def row(target: int, peer: int, raw: float, peer_time: float) -> dict[str, object]:
            result: dict[str, object] = {
                "source": "mutual-acoustic-recurrence",
                "stage": "occurrence-level-mutual-recurrence-search",
                "raw_candidate_time": raw,
                "authoritative": False,
                "mode": "reciprocal-exact-duplicate-pitchclass-recurrence",
                "parent": {"source": "reciprocal-untrusted-duplicate-pair", "raw_candidate_time": raw},
                "window": {
                    **payload,
                    "pair_id": pair_id,
                    "target_entry": target,
                    "peer_entry": peer,
                    "peer_candidate": peer_time,
                    "onset_strength": 0.90,
                    "onset_prominence": 0.40,
                },
            }
            result["hypothesis_id"] = auto_lrc._timing_content_digest(result)
            return result
        left = row(1, 2, 10.2, 30.2)
        right = row(2, 1, 30.2, 10.2)
        assignments[0]["alignment_hypotheses"].append(left)  # type: ignore[index]
        assignments[1]["alignment_hypotheses"].append(right)  # type: ignore[index]
        return entries, assignments, left, right

    def test_mutual_recurrence_provenance_binds_actual_exact_duplicate_peer(self) -> None:
        entries, assignments, left, _right = self._mutual_pair_fixture()
        self.assertTrue(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            left, entries[0], Path(__file__), assignments, entries, 0
        ))
        wrong_entries = [entries[0], LyricEntry(["different lyric"])]
        self.assertFalse(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            left, wrong_entries[0], Path(__file__), assignments, wrong_entries, 0
        ))

    def test_mutual_recurrence_provenance_rejects_pair_tampering(self) -> None:
        entries, assignments, left, _right = self._mutual_pair_fixture()
        tampered = copy.deepcopy(left)
        tampered["window"]["distinct_margin"] = auto_lrc.MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN - 0.001  # type: ignore[index]
        self.assertFalse(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            tampered, entries[0], Path(__file__), assignments, entries, 0
        ))

    def test_mutual_recurrence_is_only_independent_identity_when_provenance_bound(self) -> None:
        common = dict(
            entry_index=0,
            entry_text="same lyric",
            raw_time=10.0,
            spans=(),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=0.9,
            acoustic_onset_time=10.0,
            acoustic_source_artifact={"pair": 1},
            acoustic_evidence_producer="mutual-recurrence-search",
            acoustic_evidence_kind="recurrence-onset",
            acoustic_evidence_independent=True,
            direct_onset_time=10.0,
            direct_onset_source_artifact={"pair": 1},
            direct_onset_evidence_producer="mutual-recurrence-search",
            direct_onset_evidence_kind="recurrence-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"pair": 1},
        )
        unbound = auto_lrc.make_timing_candidate(source="mutual-acoustic-recurrence", **common)
        bound = auto_lrc.make_timing_candidate(
            source="mutual-acoustic-recurrence", independent_content_identity=True, **common
        )
        self.assertFalse(auto_lrc.candidate_supplies_independent_content_identity(unbound))
        self.assertTrue(auto_lrc.candidate_supplies_independent_content_identity(bound))

    def test_collector_rejects_fake_mutual_source_label_without_reciprocal_provenance(self) -> None:
        entries = [LyricEntry(["same lyric"]), LyricEntry(["same lyric"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 10.0,
                    "timing_repair": "ctc",
                    "score": 0.9,
                    "alignment_hypotheses": [{
                        "source": "mutual-acoustic-recurrence",
                        "stage": "occurrence-level-mutual-recurrence-search",
                        "raw_candidate_time": 10.2,
                        "authoritative": False,
                        "mode": "reciprocal-exact-duplicate-pitchclass-recurrence",
                        "parent": {"source": "reciprocal-untrusted-duplicate-pair", "raw_candidate_time": 10.0},
                        "window": {},
                    }],
                },
                {"entry": 2, "timestamp": 30.0, "timing_repair": "ctc", "score": 0.9},
            ],
        }
        sets, _diagnostics = auto_lrc.collect_central_timing_candidates(
            Path(__file__), entries, [10.0, 30.0], report, 40.0, type("Args", (), {})(),
            independent_evidence_audio_path=Path(__file__),
        )
        self.assertNotIn("mutual-acoustic-recurrence", {candidate.source for candidate in sets[0]})

    def test_phonetic_onset_provenance_requires_coherent_ctc_and_exact_audio(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "isolated-vocal.wav"
            audio.write_bytes(b"isolated-vocal-runtime-artifact")
            features, assignment, hypothesis = self._phonetic_runtime_fixture(audio)
            with split_patch.patch("analyze_vocal_onsets", return_value=features):
                self.assertTrue(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, assignment, audio
                ))
                wrong_audio = copy.deepcopy(hypothesis)
                wrong_audio["window"]["audio_sha256"] = "0" * 64  # type: ignore[index]
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    wrong_audio, assignment, audio
                ))
                empty_spans = copy.deepcopy(assignment)
                empty_spans["ctc_token_spans"] = []
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, empty_spans, audio
                ))
                fractured = copy.deepcopy(assignment)
                fractured["ctc_token_spans"] = [
                    {"char": "a", "start": 10.0, "end": 10.02, "score": 0.8},
                    {"char": "b", "start": 10.08, "end": 10.10, "score": 0.8},
                    {"char": "c", "start": 13.2, "end": 13.22, "score": 0.8},
                    {"char": "d", "start": 13.28, "end": 13.30, "score": 0.8},
                ]
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, fractured, audio
                ))

    def test_phonetic_onset_fusion_remains_diagnostic_only_in_central_collector(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "isolated-vocal.wav"
            audio.write_bytes(b"isolated-vocal-runtime-artifact")
            features, assignment, hypothesis = self._phonetic_runtime_fixture(audio)
            assignment["alignment_hypotheses"] = [hypothesis]
            entries = [LyricEntry(["provenance-check"])]
            report: dict[str, object] = {
                "backend": "ctc",
                "assignments": [assignment],
            }
            with split_patch.patch("analyze_vocal_onsets", return_value=features):
                sets, _diagnostics = auto_lrc.collect_central_timing_candidates(
                    audio,
                    entries,
                    [10.003],
                    report,
                    20.0,
                    type("Args", (), {})(),
                    independent_evidence_audio_path=audio,
                )
        self.assertNotIn(
            "ctc-phonetic-onset-fusion",
            {candidate.source for candidate in sets[0]},
        )

    def test_bound_local_occurrence_identity_rejects_mutated_hypothesis_record(self) -> None:
        assignment: dict[str, object] = {"entry": 1, "timestamp": 10.0, "score": 0.5}
        auto_lrc.record_alignment_hypothesis(
            assignment,
            source="ctc-local-window",
            stage="ctc-local-window-realign",
            raw_time=10.2,
            confidence=0.8,
            spans=[
                {"char": chr(97 + index), "start": 10.2 + index * 0.08, "end": 10.22 + index * 0.08, "score": 0.3}
                for index in range(6)
            ],
            identity_support="supported",
            sequence_support="supported",
            parent_source="ctc-global-mms",
            parent_time=10.0,
            mode="local-window-independent-candidate",
        )
        self.assertIsNotNone(auto_lrc._bound_local_occurrence_identity(assignment))
        hypothesis = assignment["alignment_hypotheses"][0]  # type: ignore[index]
        hypothesis["raw_candidate_time"] = 10.9  # type: ignore[index]
        self.assertIsNone(auto_lrc._bound_local_occurrence_identity(assignment))

    def test_mutual_recurrence_provenance_rejects_mutated_peer_identity_evidence(self) -> None:
        entries, assignments, left, _right = self._mutual_pair_fixture()
        self.assertTrue(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            left, entries[0], Path(__file__), assignments, entries, 0
        ))
        peer_hypotheses = assignments[1]["alignment_hypotheses"]  # type: ignore[index]
        local_peer = next(
            hypothesis for hypothesis in peer_hypotheses
            if isinstance(hypothesis, dict) and hypothesis.get("source") == "ctc-local-window"
        )
        local_peer["confidence"] = 0.01
        self.assertFalse(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            left, entries[0], Path(__file__), assignments, entries, 0
        ))

    def test_new_phase_c_producers_are_candidate_only(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        names = {
            "apply_phonetic_onset_fusion_hypotheses",
            "apply_mutual_acoustic_recurrence_hypotheses",
            "apply_trusted_tail_release_onset_hypotheses",
        }
        functions = {
            node.name: node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }
        self.assertEqual(set(functions), names)
        for name, function in functions.items():
            body = ast.get_source_segment(source, function) or ""
            self.assertNotIn('assignment["timestamp"] =', body, name)
            self.assertNotIn("assignment['timestamp'] =", body, name)
            self.assertNotIn("timestamps[index] =", body, name)
            self.assertIn("record_alignment_hypothesis", body, name)

    def test_no_unbound_independent_tail_evidence_in_functions(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        offenders: list[str] = []
        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            loads = {
                child.id for child in ast.walk(node)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id == "independent_tail_evidence"
            }
            if not loads:
                continue
            bound = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            if node.args.vararg is not None:
                bound.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                bound.add(node.args.kwarg.arg)
            bound.update(
                child.id for child in ast.walk(node)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, (ast.Store, ast.Param))
            )
            if "independent_tail_evidence" not in bound:
                offenders.append(node.name)
        self.assertEqual(offenders, [])

    def test_tail_release_provenance_rejects_missing_cluster_break(self) -> None:
        audio = Path(__file__)
        assignments: list[object] = [
            {"entry": 1, "timestamp": 8.0, "timing_repair": "ctc", "score": 0.9},
            {"entry": 2, "timestamp": 9.5, "timing_repair": "ctc", "score": 0.9},
        ]
        base: dict[str, object] = {
            "source": "trusted-tail-release-onset",
            "stage": "trusted-tail-release-onset-search",
            "raw_candidate_time": 10.5,
            "authoritative": False,
            "mode": "strong-onset-inside-centrally-trusted-tail-release",
            "parent": {"source": "central-trusted-predecessor-tail", "raw_candidate_time": 10.0},
            "window": {
                "audio_sha256": auto_lrc._path_sha256(audio),
                "feature_revision": "trusted-tail-release-cluster-onset-v3",
                "previous_candidate_id": "previous",
                "previous_tail_revision": "tail",
                "previous_tail_end": 10.0,
                "previous_tail_uncertainty": 0.9,
                "search_start": 10.0,
                "search_end": 11.0,
                "onset_strength": 0.95,
                "onset_prominence": 0.70,
                "previous_onset_time": 10.2,
                "cluster_gap": 0.3,
                "context_start": 9.2,
                "old_candidate_raw_time": 9.5,
                "old_candidate_centiseconds": 950,
                "old_candidate_time": 9.5,
            },
        }
        base["hypothesis_id"] = auto_lrc._timing_content_digest(base)
        self.assertTrue(auto_lrc.trusted_tail_release_provenance_is_valid(
            base, audio, assignments, 1
        ))
        weak = copy.deepcopy(base)
        weak["window"]["cluster_gap"] = 0.10  # type: ignore[index]
        self.assertFalse(auto_lrc.trusted_tail_release_provenance_is_valid(
            weak, audio, assignments, 1
        ))


    def test_phonetic_provenance_binds_raw_assignment_and_canonical_written_time(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "isolated-vocal.wav"
            audio.write_bytes(b"isolated-vocal-runtime-artifact")
            features, assignment, hypothesis = self._phonetic_runtime_fixture(audio)
            with split_patch.patch("analyze_vocal_onsets", return_value=features):
                self.assertTrue(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, assignment, audio
                ))
                wrong_raw = copy.deepcopy(assignment)
                wrong_raw["timestamp"] = 10.013
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, wrong_raw, audio
                ))
                stale_revision = copy.deepcopy(hypothesis)
                stale_revision["window"]["feature_revision"] = "stale"  # type: ignore[index]
                stale_revision["hypothesis_id"] = auto_lrc._timing_content_digest(
                    {key: value for key, value in stale_revision.items() if key != "hypothesis_id"}
                )
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    stale_revision, assignment, audio
                ))

    def test_tail_release_provenance_binds_raw_assignment_and_canonical_written_time(self) -> None:
        audio = Path(__file__)
        assignments: list[object] = [
            {"entry": 1, "timestamp": 8.0, "timing_repair": "ctc", "score": 0.9},
            {"entry": 2, "timestamp": 9.503, "timing_repair": "ctc", "score": 0.9},
        ]
        base: dict[str, object] = {
            "source": "trusted-tail-release-onset",
            "stage": "trusted-tail-release-onset-search",
            "raw_candidate_time": 10.5,
            "authoritative": False,
            "mode": "strong-onset-inside-centrally-trusted-tail-release",
            "parent": {"source": "central-trusted-predecessor-tail", "raw_candidate_time": 10.0},
            "window": {
                "audio_sha256": auto_lrc._path_sha256(audio),
                "feature_revision": "trusted-tail-release-cluster-onset-v3",
                "previous_candidate_id": "previous",
                "previous_tail_revision": "tail",
                "previous_tail_end": 10.0,
                "previous_tail_uncertainty": 0.9,
                "search_start": 10.0,
                "search_end": 11.0,
                "onset_strength": 0.95,
                "onset_prominence": 0.70,
                "previous_onset_time": 10.2,
                "cluster_gap": 0.3,
                "context_start": 9.2,
                "old_candidate_raw_time": 9.503,
                "old_candidate_centiseconds": 950,
                "old_candidate_time": 9.5,
            },
        }
        base["hypothesis_id"] = auto_lrc._timing_content_digest(base)
        self.assertTrue(auto_lrc.trusted_tail_release_provenance_is_valid(base, audio, assignments, 1))
        wrong_raw = copy.deepcopy(assignments)
        wrong_raw[1]["timestamp"] = 9.513  # type: ignore[index]
        self.assertFalse(auto_lrc.trusted_tail_release_provenance_is_valid(base, audio, wrong_raw, 1))

    def test_update_metrics_preserves_central_candidate_identity_trust(self) -> None:
        source_sha = auto_lrc.validated_loaded_source_sha256()
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            "timing_state_schema_version": 3,
            "algorithm_source_sha256": source_sha,
            "candidate_selection": {
                "status": "selected",
                "metric_stage": "central-evaluated",
                "selected_backend": "ctc",
            },
            "central_timing_state": [{
                "entry": 1,
                "timestamp": 10.0,
                "canonical_written_centiseconds": 1000,
                "candidate_id": "independent",
                "candidate_source": "global-acoustic-recurrence",
                "timestamp_status": "selected_valid",
                "timing_trusted": True,
            }],
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "timing_repair": "ctc",
                "score": 0.2,
                "canonical_written_centiseconds": 1000,
                "timestamp_status": "selected_valid",
                "selection_revision": 1,
                "timestamp_revision": 1,
                "timing_evidence_revision": 1,
                "selected_candidate_id": "independent",
                "provisional_candidate_id": None,
                "candidate_source": "global-acoustic-recurrence",
                "candidate_provenance": {
                    "candidate_id": "independent",
                    "source": "global-acoustic-recurrence",
                    "canonical_written_centiseconds": 1000,
                },
                "normalized_timing_evidence": {
                    "computed_for_centiseconds": 1000,
                    "structural_failures": [],
                    "minimum_proof_satisfied": True,
                },
                "content_trusted": True,
                "content_trust_source": "selected-candidate-independent-identity",
                "timing_trusted": True,
                "timing_audit_trusted": True,
                "overall_trusted": True,
                "review_required": False,
            }],
        }
        self.assertTrue(auto_lrc.report_has_verified_central_commit(report))
        auto_lrc.update_report_confidence_metrics(report)
        self.assertEqual(report["content_trusted_entries"], 1)
        self.assertEqual(report["timing_trusted_entries"], 1)
        self.assertEqual(report["overall_trusted_entries"], 1)
        self.assertEqual(report["overall_trusted_percent"], 100.0)
        assignment = report["assignments"][0]  # type: ignore[index]
        self.assertTrue(assignment["content_trusted"])
        self.assertTrue(assignment["overall_trusted"])

    def test_update_metrics_refuses_unbound_fake_central_content_trust(self) -> None:
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            # A field name alone must not grant central authority.
            "central_timing_state": [{
                "entry": 1,
                "timestamp": 10.0,
                "canonical_written_centiseconds": 1000,
                "candidate_id": "forged",
                "candidate_source": "global-acoustic-recurrence",
                "timestamp_status": "selected_valid",
                "timing_trusted": True,
            }],
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "timing_repair": "ctc",
                "score": 0.2,
                "content_trusted": True,
                "timing_trusted": True,
                "review_required": False,
            }],
        }
        self.assertFalse(auto_lrc.report_has_verified_central_commit(report))
        auto_lrc.update_report_confidence_metrics(report)
        self.assertEqual(report["content_trusted_entries"], 0)
        self.assertEqual(report["overall_trusted_entries"], 0)
        assignment = report["assignments"][0]  # type: ignore[index]
        self.assertFalse(assignment["content_trusted"])
        self.assertFalse(assignment["overall_trusted"])


class BoundaryRecoveryTests(unittest.TestCase):
    def span(self, start: float, score: float, token: str) -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(self, entry: int, time: float, spans, *, source='ctc-current', current=False):
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f'line {entry}',
            source=source,
            raw_time=time,
            spans=tuple(spans),
            confidence=0.8,
            identity_support='supported',
            sequence_support='supported',
            current=current,
            source_artifact={'entry': entry, 'time': time, 'source': source},
        )

    def capability(self):
        payload = {
            'kind': 'mms-known-lyric-ctc',
            'backend_lineage': 'ctc',
            'helper_path': 'ctc_align.py',
            'helper_sha256': 'helper-sha',
            'helper_protocol_revision': 'protocol-revision',
            'alignment_audio_path': 'audio.flac',
            'alignment_audio_sha256': 'audio-sha',
            'alignment_audio_source': 'vocal-stem',
            'sample_rate': auto_lrc.SAMPLE_RATE,
            'model_identity': 'MMS_FA:known-lyric',
            'device': 'cpu',
        }
        return auto_lrc.BoundedRetryCapability(
            **payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(payload)
            ),
        )

    def test_weak_initial_consonant_strong_vowel_can_prove_first_mora(self):
        c = self.candidate(0, 45.745, (
            self.span(45.745, 0.041866, 'k'),
            self.span(45.845, 0.586112, 'i'),
            self.span(45.925, 0.20, 'b'),
        ))
        region = auto_lrc.analyze_candidate_regions(c)
        self.assertAlmostEqual(region.onset_proof_end or 0.0, 45.865, places=3)
        self.assertEqual(region.temporal_coherence, 'coherent')

    def test_global_serial_candidate_cannot_override_tail_conflict(self):
        previous = self.candidate(0, 9.74, (
            self.span(9.74, .2, 'l'), self.span(9.82, .2, 'a'), self.span(9.90, .2, 'l')
        ), current=True)
        current = self.candidate(1, 9.90, (
            self.span(9.90, .2, 's'), self.span(9.98, .4, 'o'), self.span(10.06, .2, 'n')
        ), source='ctc-current')
        ev = auto_lrc.evaluate_timing_candidate(current, previous)
        self.assertFalse(ev.valid)
        self.assertIn('previous-tail-overlap', ev.rejection_reasons)

    def test_current_line_opening_requires_bound_disjoint_evidence(self):
        previous = self.candidate(0, 9.74, (
            self.span(9.74, .2, 'l'), self.span(9.82, .2, 'a'), self.span(9.90, .2, 'l')
        ), current=True)
        current = self.candidate(1, 9.90, (
            self.span(9.90, .2, 's'), self.span(9.98, .4, 'o'), self.span(10.06, .2, 'n')
        ), source='ctc-opening-local-retry')
        ev = auto_lrc.evaluate_timing_candidate(current, previous)
        self.assertFalse(ev.valid)
        self.assertEqual(ev.evidence.ownership.status, 'violation')
        self.assertIn('previous-tail-overlap', ev.rejection_reasons)

        previous = self.candidate(0, 9.74, (
            self.span(9.74, .2, 'l'), self.span(9.82, .2, 'a'), self.span(9.90, .2, 'l')
        ), current=True)
        previous_region = auto_lrc.analyze_candidate_regions(previous)
        previous_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id=previous.candidate_id,
            previous_generation_revision=previous.generation_revision,
            previous_region_revision=previous_region.region_revision,
            tail_evidence_revision='external-tail-observation',
            validity='valid',
            reliable_tail_start=9.74,
            reliable_tail_end=9.92,
            uncertainty_seconds=.18,
            confidence=.9,
            reason='runtime-external-tail',
            authority_backend_lineage='ctc',
            authority_producer='ctc-assignment-terminal-row',
            authority_observation_revision='external-observation-revision',
        )
        occurrence_entries = [LyricEntry(['line 0']), LyricEntry(['line 1'])]
        occurrence_binding = auto_lrc._entry_occurrence_binding(
            occurrence_entries,
            [
                {'timestamp': 9.74, 'score': .95, 'segment': 1},
                {'timestamp': 9.90, 'score': .95, 'segment': 2},
            ],
            1,
        )
        occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            occurrence_entries,
            1,
            producer='raw-asr-consensus',
            occurrence_binding=occurrence_binding,
            upstream_evidence_revision='boundary-direct-upstream',
        )
        current = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text='line 1',
            source='raw-vocal-independent-fusion',
            raw_time=9.90,
            spans=(
                self.span(9.90, .2, 's'),
                self.span(9.98, .4, 'o'),
                self.span(10.06, .2, 'n'),
            ),
            confidence=.8,
            identity_support='supported',
            sequence_support='supported',
            direct_onset_support='supported',
            direct_onset_time=9.90,
            direct_onset_source_artifact={'entry': 1, 'observation': 'external-onset'},
            direct_onset_evidence_producer='raw-asr-consensus',
            direct_onset_evidence_kind='cross-backend-onset',
            direct_onset_evidence_independent=True,
            occurrence_evidence=occurrence_evidence,
            current=True,
        )
        self.assertEqual(auto_lrc._candidate_support_families(current), {'whisper-family'})
        self.assertTrue(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(current))
        evaluation = auto_lrc.evaluate_timing_candidate(
            current, previous, previous_tail=previous_tail
        )
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.ownership.status, 'violation')
        self.assertTrue(evaluation.evidence.ownership_independent)
        self.assertEqual(
            evaluation.evidence.ownership.previous_tail_authority_backend_lineage,
            'ctc',
        )
        self.assertEqual(
            evaluation.evidence.ownership.previous_tail_authority_producer,
            'ctc-assignment-terminal-row',
        )

    def test_unknown_previous_tail_still_schedules_opening_retry(self):
        entries = [LyricEntry(['prev']), LyricEntry(['current'])]
        previous = self.candidate(0, 9.0, (
            self.span(9.0, .2, 'x'), self.span(9.08, .01, 'x')
        ), current=True)
        current = self.candidate(1, 10.0, (
            self.span(10.0, .01, 'x'), self.span(10.08, .01, 'x'), self.span(10.16, .01, 'x')
        ), current=True)
        sets = ((previous,), (current,))
        report = {
            'backend': 'ctc', 'timing_entries': 2,
            'assignments': [
                {'entry': 1, 'timestamp': 9.0, 'timing_repair': 'ctc', 'score': .9},
                {'entry': 2, 'timestamp': 10.0, 'timing_repair': 'ctc', 'score': .9},
            ],
        }
        cap = self.capability()
        epoch = auto_lrc.freeze_base_retry_epoch(
            'ctc', report, sets, auto_lrc.inactive_legacy_timing_diagnostics(), cap
        )
        requests = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, sets, 30.0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].previous_tail_validity, 'unknown')
        self.assertEqual(requests[0].context_entry_indexes, (1,))

    def test_terminal_line_retry_has_no_next_bound_and_is_bounded(self):
        entries = [LyricEntry(['prev']), LyricEntry(['current'])]
        previous = self.candidate(0, 197.6, (
            self.span(197.6, .2, 'x'), self.span(197.68, .01, 'x')
        ), current=True)
        current = self.candidate(1, 206.5, (
            self.span(206.5, .01, 's'), self.span(206.6, .01, 'o'), self.span(206.7, .01, 'n')
        ), current=True)
        sets = ((previous,), (current,))
        report = {
            'backend': 'ctc', 'timing_entries': 2,
            'assignments': [
                {'entry': 1, 'timestamp': 197.6, 'timing_repair': 'ctc', 'score': .9},
                {'entry': 2, 'timestamp': 206.5, 'timing_repair': 'ctc', 'score': .9},
            ],
        }
        cap = self.capability()
        epoch = auto_lrc.freeze_base_retry_epoch(
            'ctc', report, sets, auto_lrc.inactive_legacy_timing_diagnostics(), cap
        )
        request = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, sets, 239.4)[0]
        self.assertIsNone(request.next_bound)
        self.assertEqual(request.context_entry_indexes, (1,))
        self.assertLessEqual(request.window_end - request.window_start, 20.0)
        self.assertLess(request.window_start, 206.5)

    def test_isolated_token_before_large_fracture_is_not_opening_proof(self):
        c = self.candidate(0, 10.0, (
            self.span(10.0, .5, 'a'), self.span(13.5, .5, 'b'), self.span(13.58, .5, 'c')
        ), source='ctc-opening-local-retry')
        ev = auto_lrc.evaluate_timing_candidate(c)
        self.assertFalse(ev.valid)
        self.assertIn('temporal-incoherence', ev.rejection_reasons)


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
