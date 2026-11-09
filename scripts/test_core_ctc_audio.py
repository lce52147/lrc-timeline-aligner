#!/usr/bin/env python3
"""CTC backtrack, vocal onset, whisperx boundary, audit export tests."""

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


class CtcAcousticBacktrackTests(unittest.TestCase):
    def test_crossline_initial_recovery_rejects_a_prior_line_tail_token(self) -> None:
        timestamps = [39.46, 43.74, 52.91]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 39.46,
                    "ctc_token_spans": [{"char": "y", "start": 43.70, "end": 43.72, "score": 0.3}],
                },
                {
                    "entry": 2,
                    "timestamp": 43.74,
                    "ctc_token_spans": [
                        {"char": "i", "start": 43.74, "end": 43.76, "score": 0.19},
                        {"char": "l", "start": 46.32, "end": 46.34, "score": 0.2},
                    ],
                },
                {"entry": 3, "timestamp": 52.91, "ctc_token_spans": []},
            ]
        }

        recovered, updated, changes = apply_ctc_crossline_initial_recovery(timestamps, report, 60.0)

        self.assertAlmostEqual(recovered[1], 46.32, places=3)
        self.assertEqual(len(changes), 1)
        self.assertTrue(updated["assignments"][1]["ctc_crossline_initial_recovery"])  # type: ignore[index]

    def test_crossline_initial_recovery_skips_a_multi_token_ghost_prefix(self) -> None:
        timestamps = [129.82, 137.287]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 129.82,
                    "ctc_token_spans": [
                        {"char": "e", "start": 137.187, "end": 137.207, "score": 0.019817}
                    ],
                },
                {
                    "entry": 2,
                    "timestamp": 137.287,
                    "ctc_score": 0.373806,
                    "ctc_token_spans": [
                        {"char": "a", "start": 137.287, "end": 137.307, "score": 0.019517},
                        {"char": "r", "start": 137.947, "end": 137.967, "score": 0.002573},
                        {"char": "e", "start": 138.627, "end": 138.647, "score": 0.049344},
                        {"char": "y", "start": 139.768, "end": 139.788, "score": 0.625119},
                        {"char": "o", "start": 139.948, "end": 139.968, "score": 0.600449},
                    ],
                },
            ]
        }

        recovered, updated, changes = apply_ctc_crossline_initial_recovery(
            timestamps, report, 151.813
        )

        self.assertAlmostEqual(recovered[1], 138.627, places=3)
        self.assertEqual(len(changes), 1)
        assignment = updated["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["ctc_effective_token_start_index"], 2)
        self.assertEqual(
            assignment["ctc_crossline_initial_recovery_mode"],
            "multi-token-ghost-prefix",
        )
        self.assertTrue(assignment["ctc_crossline_initial_recovery_trusted"])

    def test_candidate_scoring_keeps_a_trusted_crossline_recovery(self) -> None:
        entries = [
            LyricEntry(["previous"]),
            LyricEntry(["Are you checking this synthetic boundary?"]),
        ]
        timestamps = [129.803, 138.627]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 129.803,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [
                        {"char": "e", "start": 137.247, "end": 137.267, "score": 0.022157}
                    ],
                },
                {
                    "entry": 2,
                    "timestamp": 138.627,
                    "score": 0.9,
                    "ctc_score": 0.373806,
                    "timing_repair": "ctc-forced-align",
                    "ctc_effective_token_start_index": 2,
                    "ctc_crossline_initial_recovery": True,
                    "ctc_crossline_initial_recovery_trusted": True,
                    "ctc_token_spans": [
                        {"char": "a", "start": 137.287, "end": 137.307, "score": 0.019517},
                        {"char": "r", "start": 137.947, "end": 137.967, "score": 0.002573},
                        {"char": "e", "start": 138.627, "end": 138.647, "score": 0.049344},
                        {"char": "y", "start": 139.768, "end": 139.788, "score": 0.625119},
                        {"char": "o", "start": 139.948, "end": 139.968, "score": 0.600449},
                    ],
                    "ctc_first_token_candidates": [],
                },
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, timestamps, report, 151.813)

        self.assertAlmostEqual(result[1], 138.627, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(assignment["review_required"])
        self.assertGreaterEqual(assignment["confidence"], 0.95)
        self.assertNotIn("candidate_disagreement", assignment["flags"])
        self.assertNotIn(
            "ctc_detached_initial_token_recovery",
            [item["source"] for item in assignment["candidates"]],
        )

    def test_candidate_scoring_restores_a_detached_crossline_recovery_after_window_realign(self) -> None:
        entries = [
            LyricEntry(["Synthetic signals gather near dawn"]),
            LyricEntry(["Underwater test signals surround the marker"]),
            LyricEntry(["Now this synthetic case is complete"]),
        ]
        timestamps = [81.44, 85.992, 93.955]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 81.44,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [
                        {"char": "s", "start": 84.104, "end": 84.124, "score": 0.107459}
                    ],
                },
                {
                    "entry": 2,
                    "timestamp": 85.992,
                    "score": 0.9,
                    "ctc_score": 0.227128,
                    "timing_repair": "ctc-forced-align",
                    "ctc_effective_token_start_index": 1,
                    "ctc_crossline_initial_recovery": True,
                    "ctc_crossline_initial_recovery_trusted": False,
                    "ctc_crossline_initial_recovery_evidence": {
                        "ghost_span_seconds": 3.501,
                    },
                    "ctc_token_spans": [
                        {"char": "u", "start": 85.992, "end": 86.012, "score": 0.006262},
                        {"char": "n", "start": 89.493, "end": 89.513, "score": 0.602427},
                        {"char": "d", "start": 89.533, "end": 89.553, "score": 0.111421},
                        {"char": "e", "start": 89.573, "end": 89.593, "score": 0.195719},
                    ],
                    "ctc_first_token_candidates": [],
                },
                {
                    "entry": 3,
                    "timestamp": 93.955,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [],
                },
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, timestamps, report, 110.0)

        self.assertAlmostEqual(result[1], 85.992, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertAlmostEqual(float(assignment["timestamp"]), 85.992, places=3)
        self.assertTrue(assignment["ctc_crossline_initial_recovery_candidate_generated"])
        self.assertFalse(assignment["ctc_crossline_initial_recovery_trusted"])
        hypotheses = assignment.get("alignment_hypotheses", [])
        restored = [
            item
            for item in hypotheses
            if isinstance(item, dict) and item.get("source") == "ctc-legacy-crossline-restored"
        ]
        self.assertEqual(len(restored), 1)
        self.assertAlmostEqual(float(restored[0]["raw_candidate_time"]), 89.493, places=3)
        self.assertFalse(restored[0]["authoritative"])

    def test_forward_supported_prefix_candidate_rejects_continuous_weak_prefix(self) -> None:
        candidate = ctc_forward_supported_prefix_candidate(
            {
                "ctc_token_spans": [
                    {"start": 20.00, "end": 20.02, "score": 0.006},
                    {"start": 20.10, "end": 20.12, "score": 0.004},
                    {"start": 20.20, "end": 20.22, "score": 0.008},
                    {"start": 20.30, "end": 20.32, "score": 0.003},
                    {"start": 20.40, "end": 20.42, "score": 0.012},
                    {"start": 20.50, "end": 20.52, "score": 0.004},
                    {"start": 21.20, "end": 21.22, "score": 0.12},
                    {"start": 21.30, "end": 21.32, "score": 0.15},
                    {"start": 21.40, "end": 21.42, "score": 0.09},
                ]
            }
        )

        self.assertIsNone(candidate)

    def test_forward_supported_prefix_recovery_does_not_skip_continuous_prefix(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 21.2))[0][0]] = 1.0
        features = AudioFeatures(
            duration=30.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["The bubbles"]), LyricEntry(["next"])]
            timestamps = [15.0, 20.0, 25.0]
            report: dict[str, object] = {
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 15.0,
                        "ctc_token_spans": [
                            {"char": "x", "start": 18.0, "end": 18.02, "score": 0.2}
                        ],
                    },
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "ctc_score": 0.04,
                        "ctc_token_spans": [
                            {"char": "t", "start": 20.0, "end": 20.02, "score": 0.006},
                            {"char": "h", "start": 20.1, "end": 20.12, "score": 0.004},
                            {"char": "e", "start": 20.2, "end": 20.22, "score": 0.008},
                            {"char": "b", "start": 20.3, "end": 20.32, "score": 0.003},
                            {"char": "u", "start": 20.4, "end": 20.42, "score": 0.012},
                            {"char": "b", "start": 20.5, "end": 20.52, "score": 0.004},
                            {"char": "i", "start": 21.5, "end": 21.52, "score": 0.12},
                            {"char": "n", "start": 21.6, "end": 21.62, "score": 0.15},
                            {"char": "g", "start": 21.7, "end": 21.72, "score": 0.09},
                        ],
                    },
                    {"entry": 3, "timestamp": 25.0, "ctc_token_spans": []},
                ]
            }

            changes = apply_ctc_forward_supported_prefix_recovery(
                Path("dummy.flac"), entries, timestamps, report, 30.0
            )

            self.assertEqual(changes, [])
            self.assertAlmostEqual(timestamps[1], 20.0, places=3)
            assignment = report["assignments"][1]  # type: ignore[index]
            self.assertNotIn("ctc_forward_supported_prefix_recovery_trusted", assignment)
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_crossline_initial_recovery_keeps_a_normal_initial_cluster(self) -> None:
        timestamps = [39.46, 46.15, 52.91]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 39.46,
                    "ctc_token_spans": [{"char": "y", "start": 43.70, "end": 43.72, "score": 0.3}],
                },
                {
                    "entry": 2,
                    "timestamp": 46.15,
                    "ctc_token_spans": [
                        {"char": "i", "start": 46.15, "end": 46.17, "score": 0.2},
                        {"char": "l", "start": 46.32, "end": 46.34, "score": 0.2},
                    ],
                },
                {"entry": 3, "timestamp": 52.91, "ctc_token_spans": []},
            ]
        }

        recovered, _, changes = apply_ctc_crossline_initial_recovery(timestamps, report, 60.0)

        self.assertEqual(recovered, timestamps)
        self.assertEqual(changes, [])

    def test_weak_ctc_prefix_recovers_strong_earlier_first_token(self) -> None:
        timestamps = [195.0, 202.16, 207.03]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 195.0, "ctc_score": 0.5},
                {
                    "entry": 2,
                    "timestamp": 202.16,
                    "ctc_score": 0.08,
                    "ctc_token_spans": [{"char": "h", "start": 202.16, "score": 0.01}],
                    "ctc_first_token_candidates": [
                        {"time": 200.09, "score": 0.19},
                        {"time": 201.00, "score": 0.03},
                    ],
                },
                {"entry": 3, "timestamp": 207.03, "ctc_score": 0.5},
            ]
        }

        recovered, updated, changes = apply_ctc_weak_prefix_recovery(timestamps, report, 220.0)

        self.assertAlmostEqual(recovered[1], 200.09, places=3)
        self.assertEqual(len(changes), 1)
        self.assertTrue(updated["assignments"][1]["ctc_weak_prefix_recovery"])  # type: ignore[index]

    def test_weak_prefix_does_not_use_an_unconvincing_peak(self) -> None:
        timestamps = [100.0, 105.0, 110.0]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 100.0, "ctc_score": 0.5},
                {
                    "entry": 2,
                    "timestamp": 105.0,
                    "ctc_score": 0.1,
                    "ctc_token_spans": [{"char": "h", "start": 105.0, "score": 0.01}],
                    "ctc_first_token_candidates": [{"time": 103.5, "score": 0.04}],
                },
                {"entry": 3, "timestamp": 110.0, "ctc_score": 0.5},
            ]
        }

        recovered, _, changes = apply_ctc_weak_prefix_recovery(timestamps, report, 120.0)

        self.assertEqual(recovered, timestamps)
        self.assertEqual(changes, [])

    def test_weak_prefix_never_reuses_the_previous_line_tail(self) -> None:
        timestamps = [195.0, 202.16, 207.03]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 195.0,
                    "ctc_score": 0.5,
                    "ctc_token_spans": [{"char": "a", "end": 200.45, "score": 0.8}],
                },
                {
                    "entry": 2,
                    "timestamp": 202.16,
                    "ctc_score": 0.08,
                    "ctc_token_spans": [{"char": "h", "start": 202.16, "score": 0.01}],
                    "ctc_first_token_candidates": [{"time": 200.09, "score": 0.19}],
                },
                {"entry": 3, "timestamp": 207.03, "ctc_score": 0.5},
            ]
        }

        recovered, _, changes = apply_ctc_weak_prefix_recovery(timestamps, report, 220.0)

        self.assertEqual(recovered, timestamps)
        self.assertEqual(changes, [])

    def test_backtracks_low_confidence_ctc_when_acoustic_onset_is_clear(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 19.5))[0][0]] = 1.0
        features = AudioFeatures(
            duration=30.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["理"]), LyricEntry(["next"])]
            timestamps = [10.0, 20.0, 25.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.04,
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 30.0)

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 20.0, places=3)
            self.assertEqual(report["assignments"][1]["timestamp"], 20.0)  # type: ignore[index]
            self.assertEqual(report["ctc_acoustic_backtrack_count"], 1)
            self.assertFalse(report["ctc_acoustic_backtrack_authoritative"])
            hypotheses = report["assignments"][1]["alignment_hypotheses"]  # type: ignore[index]
            self.assertTrue(any(item["source"] == "ctc-acoustic-backtrack" and item["raw_candidate_time"] == 19.5 for item in hypotheses))

            entries = [LyricEntry(["previous"]), LyricEntry(["その"]), LyricEntry(["next"])]
            timestamps = [10.0, 20.0, 25.0]
            report = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.10,
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 30.0)

            self.assertEqual(changes, [])
            self.assertEqual(timestamps[1], 20.0)
            self.assertEqual(report["ctc_acoustic_backtrack_count"], 0)
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]


class VocalOnsetTiebreakTests(unittest.TestCase):
    def test_boundary_evidence_requires_a_weak_multi_token_prefix(self) -> None:
        weak = {
            "ctc_score": 0.02,
            "ctc_token_spans": [
                {"start": 10.00 + index * 0.05, "end": 10.02 + index * 0.05, "score": score}
                for index, score in enumerate([0.002, 0.003, 0.001, 0.004, 0.008, 0.002, 0.01, 0.04])
            ],
        }
        strong = {
            "ctc_score": 0.35,
            "ctc_token_spans": [
                {"start": 10.00 + index * 0.08, "end": 10.04 + index * 0.08, "score": score}
                for index, score in enumerate([0.31, 0.24, 0.18, 0.29, 0.15, 0.22, 0.17, 0.20])
            ],
        }

        weak_evidence = ctc_prefix_boundary_evidence(weak, 0.0)
        strong_evidence = ctc_prefix_boundary_evidence(strong, 0.0)

        self.assertTrue(weak_evidence["realign_required"])
        self.assertFalse(strong_evidence["realign_required"])

    def test_boundary_evidence_detects_a_nonzero_gap_ghost_prefix(self) -> None:
        evidence = ctc_prefix_boundary_evidence(
            {
                "ctc_score": 0.362048,
                "ctc_token_spans": [
                    {"start": 137.367, "end": 137.387, "score": 0.018011},
                    {"start": 137.747, "end": 137.767, "score": 0.002650},
                    {"start": 138.627, "end": 138.647, "score": 0.041703},
                    {"start": 139.788, "end": 139.808, "score": 0.649345},
                    {"start": 139.928, "end": 139.948, "score": 0.521000},
                    {"start": 140.128, "end": 140.148, "score": 0.410000},
                    {"start": 140.268, "end": 140.288, "score": 0.380000},
                    {"start": 140.408, "end": 140.428, "score": 0.330000},
                ],
            },
            0.160,
        )

        self.assertTrue(evidence["ghost_weak_prefix"])
        self.assertTrue(evidence["realign_required"])
        self.assertAlmostEqual(evidence["reliable_token_delay_seconds"], 2.421, places=3)

    def test_boundary_realign_requires_next_reliable_token_evidence(self) -> None:
        current = ctc_prefix_boundary_evidence(
            {
                "ctc_score": 0.02,
                "ctc_previous_token_score": 0.145,
                "ctc_token_spans": [
                    {"start": 10.00 + index * 0.05, "end": 10.02 + index * 0.05, "score": 0.003}
                    for index in range(8)
                ],
            },
            0.02,
        )
        local = ctc_prefix_boundary_evidence(
            {
                "ctc_score": 0.18,
                "ctc_previous_token_score": 0.676,
                "ctc_token_spans": [
                    {"start": 12.00 + index * 0.09, "end": 12.04 + index * 0.09, "score": score}
                    for index, score in enumerate([0.04, 0.06, 0.12, 0.10, 0.09, 0.11, 0.08, 0.13])
                ],
            },
            0.12,
        )
        accepted, reasons = should_accept_ctc_boundary_realign(
            current, local, 0.02, 0.18, 10.0, 12.0, 11.94
        )
        self.assertTrue(accepted)
        self.assertIn("next-reliable-token-support", reasons)
        self.assertIn("previous-line-tail-evidence-improved", reasons)

        local_without_reliable = dict(local)
        local_without_reliable["next_reliable_token_index"] = None
        local_without_reliable["next_reliable_token_score"] = None
        local_without_reliable["reliable_token_delay_seconds"] = None
        accepted, reasons = should_accept_ctc_boundary_realign(
            current, local_without_reliable, 0.02, 0.18, 10.0, 12.0, 11.94
        )
        self.assertFalse(accepted)
        self.assertIn("no-nearby-reliable-token-evidence", reasons)

    def test_unresolved_boundary_remains_review_required_after_candidate_scoring(self) -> None:
        entries = [LyricEntry(["previous"]), LyricEntry(["current"]), LyricEntry(["next"])]
        timestamps = [5.0, 10.0, 15.0]
        evidence = {
            "available": True,
            "realign_required": True,
            "zero_boundary_gap": True,
            "low_confidence_prefix": True,
            "abnormal_prefix_compression": True,
        }
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 5.0, "score": 0.9},
                {
                    "entry": 2,
                    "timestamp": 10.0,
                    "score": 0.9,
                    "ctc_score": 0.02,
                    "ctc_boundary_evidence": evidence,
                },
                {"entry": 3, "timestamp": 15.0, "score": 0.9},
            ],
            "suspicious_alignments": [],
        }

        unresolved = flag_unresolved_ctc_boundary_issues(entries, report)
        score_line_timing_candidates(entries, timestamps, report, 20.0)
        assignment = report["assignments"][1]  # type: ignore[index]

        self.assertEqual(len(unresolved), 1)
        self.assertTrue(assignment["review_required"])
        self.assertTrue(assignment["ctc_unresolved_boundary"])
        self.assertIn("ctc_unresolved_boundary", assignment["flags"])

    def test_zero_gap_compressed_prefix_is_not_a_normal_connected_boundary(self) -> None:
        assignment = {
            "ctc_score": 0.07,
            "ctc_token_spans": [
                {"start": 10.00 + index * 0.03, "end": 10.02 + index * 0.03, "score": score}
                for index, score in enumerate([0.002, 0.003, 0.001, 0.004, 0.008, 0.002, 0.01, 0.04])
            ],
        }
        self.assertTrue(has_ctc_compressed_prefix(assignment, 0.0))
        self.assertFalse(has_ctc_compressed_prefix(assignment, 0.12))

    def test_zero_gap_boundary_realign_requires_a_stronger_late_local_path(self) -> None:
        self.assertTrue(should_accept_zero_gap_boundary_realign(0.0, 0.20, 10.0, 10.75, 0.24))
        self.assertFalse(should_accept_zero_gap_boundary_realign(0.12, 0.20, 10.0, 10.75, 0.24))
        self.assertFalse(should_accept_zero_gap_boundary_realign(0.0, 0.20, 10.0, 10.20, 0.24))
        self.assertFalse(should_accept_zero_gap_boundary_realign(0.0, 0.20, 10.0, 10.75, 0.20))

    def test_prefers_ctc_only_when_vocal_onset_evidence_is_clear(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.02), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 18.8))[0][0]] = 0.95
        onset_strength[np.where(np.isclose(frame_times, 20.2))[0][0]] = 0.40
        features = AudioFeatures(30.0, frame_times, np.full_like(frame_times, -20.0), onset_strength, [])
        original_features = auto_lrc.vocal_onset_features
        ctc_refinement.vocal_onset_features = lambda audio, duration, args: (features, None)  # type: ignore[assignment]
        try:
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0},
                    {"entry": 2, "timestamp": 20.0},
                    {"entry": 3, "timestamp": 25.0},
                ]
            }
            ctc_report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0},
                    {"entry": 2, "timestamp": 18.8},
                    {"entry": 3, "timestamp": 25.0},
                ]
            }
            timestamps, result, changes = apply_vocal_onset_tiebreak(
                [10.0, 20.0, 25.0], report, ctc_report, Path("dummy.flac"), 30.0, object()  # type: ignore[arg-type]
            )
            self.assertEqual(timestamps[0], 10.0)
            self.assertAlmostEqual(timestamps[1], 18.8, places=3)
            self.assertEqual(timestamps[2], 25.0)
            self.assertEqual(changes[0]["reason"], "demucs-vocal-onset-prefers-ctc")
            self.assertEqual(result["vocal_onset_refinement"]["change_count"], 1)  # type: ignore[index]
        finally:
            ctc_refinement.vocal_onset_features = original_features  # type: ignore[assignment]

    def test_backtracks_from_ctc_first_token_candidate_for_weak_prefix(self) -> None:
        features = AudioFeatures(
            duration=40.0,
            frame_times=np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32),
            rms_db=np.full(400, -20.0, dtype=np.float32),
            onset_strength=np.zeros(400, dtype=np.float32),
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["target"]), LyricEntry(["next"])]
            timestamps = [10.0, 20.0, 30.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.12,
                        "ctc_first_token_candidates": [
                            {"time": 19.7, "score": 0.008},
                            {"time": 18.6, "score": 0.10},
                        ],
                    },
                    {"entry": 3, "timestamp": 30.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 20.0, places=3)
            self.assertEqual(report["assignments"][1]["timestamp"], 20.0)  # type: ignore[index]
            self.assertEqual(changes[0]["mode"], "ctc-first-token-posterior")
            self.assertEqual(report["assignments"][1]["ctc_acoustic_backtrack_mode"], "ctc-first-token-posterior")  # type: ignore[index]
            hypotheses = report["assignments"][1]["alignment_hypotheses"]  # type: ignore[index]
            self.assertTrue(any(item["raw_candidate_time"] == 18.6 and item["mode"] == "ctc-first-token-posterior" for item in hypotheses))
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_rejects_first_token_candidate_inside_previous_reliable_tail(self) -> None:
        features = AudioFeatures(
            duration=40.0,
            frame_times=np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32),
            rms_db=np.full(400, -20.0, dtype=np.float32),
            onset_strength=np.zeros(400, dtype=np.float32),
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["target"]), LyricEntry(["next"])]
            timestamps = [10.0, 20.0, 30.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.12,
                        "ctc_previous_token_end": 19.65,
                        "ctc_first_token_candidates": [
                            {"time": 19.7, "score": 0.10},
                        ],
                    },
                    {"entry": 3, "timestamp": 30.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(changes, [])
            self.assertEqual(timestamps[1], 20.0)
            self.assertEqual(report["ctc_acoustic_backtrack_count"], 0)
            self.assertEqual(report["ctc_acoustic_backtrack_rejection_count"], 1)
            rejection = report["ctc_acoustic_backtrack_rejections"][0]
            self.assertEqual(rejection["reason"], "candidate-borrows-previous-line-tail")
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_short_acoustic_backtrack_rejects_an_unsupported_weak_prefix(self) -> None:
        frame_times = np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 19.6))[0][0]] = 1.0
        features = AudioFeatures(
            duration=40.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["The bubbles"]), LyricEntry(["next"])]
            timestamps = [15.0, 20.0, 25.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 15.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.04,
                        "romaji": "thebubbles",
                        "ctc_boundary_gap_seconds": 5.0,
                        "ctc_token_spans": [
                            {"char": char, "start": 20.0 + index * 0.08, "end": 20.02 + index * 0.08, "score": 0.006}
                            for index, char in enumerate("thebubbl")
                        ],
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(changes, [])
            self.assertEqual(timestamps[1], 20.0)
            self.assertEqual(report["ctc_acoustic_backtrack_rejection_count"], 1)
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_short_low_ctc_snap_uses_nearby_acoustic_onset(self) -> None:
        frame_times = np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 19.6))[0][0]] = 1.0
        features = AudioFeatures(
            duration=40.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["target"]), LyricEntry(["next"])]
            timestamps = [15.0, 20.0, 25.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 15.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.04,
                        "romaji": "sen",
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 20.0, places=3)
            self.assertEqual(report["assignments"][1]["timestamp"], 20.0)  # type: ignore[index]
            self.assertEqual(changes[0]["mode"], "ctc-short-acoustic-onset")
            hypotheses = report["assignments"][1]["alignment_hypotheses"]  # type: ignore[index]
            self.assertTrue(any(item["raw_candidate_time"] == 19.6 and item["direct_onset_support"] == "supported" for item in hypotheses))
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_short_acoustic_snap_keeps_credible_ctc_start_when_backtrack_harms_pacing(self) -> None:
        frame_times = np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 19.6))[0][0]] = 1.0
        features = AudioFeatures(
            duration=40.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous phrase"]), LyricEntry(["target phrase"]), LyricEntry(["next phrase"])]
            timestamps = [15.0, 20.0, 25.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 15.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.04,
                        "romaji": "sen",
                        "ctc_token_spans": [{"char": "s", "start": 20.0, "score": 0.08}],
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(changes, [])
            self.assertEqual(timestamps[1], 20.0)
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]


class WhisperxCandidateFusionTests(unittest.TestCase):
    def test_fused_report_keeps_forced_rows_from_same_decode_as_assignment(self) -> None:
        normal_report = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {"entry": 2, "timestamp": 20.0, "score": 0.9},
            ],
            "whisperx_forced_rows": [
                {"entry": 1, "first_char_time": 10.1, "decode": "normal"},
                {"entry": 2, "first_char_time": 20.1, "decode": "normal"},
            ],
        }
        sns_report = {
            "assignments": [
                {"entry": 1, "timestamp": 11.0, "score": 0.9},
                {"entry": 2, "timestamp": 21.0, "score": 0.9},
            ],
            "whisperx_forced_rows": [
                {"entry": 1, "first_char_time": 11.1, "decode": "sns"},
                {"entry": 2, "first_char_time": 21.1, "decode": "sns"},
            ],
        }

        fused = auto_lrc.fused_candidate_report(
            normal_report,
            sns_report,
            [10.0, 21.0],
            collapse_start=1,
            adopted_sns_entries=[],
        )

        self.assertEqual(
            [row["decode"] for row in fused["whisperx_forced_rows"]],
            ["normal", "sns"],
        )


class WhisperxAcousticBoundaryTests(unittest.TestCase):
    def test_candidate_disagreement_backtracks_to_local_onset(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 19.5))[0][0]] = 1.0
        features = AudioFeatures(
            duration=30.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            report: dict[str, object] = {
                "timing_entries": 3,
                "assignments": [
                    {"entry": 1, "score": 0.96, "timestamp": 10.0},
                    {"entry": 2, "score": 0.96, "timestamp": 20.0},
                    {"entry": 3, "score": 0.96, "timestamp": 25.0},
                ],
                "suspicious_alignments": [
                    {
                        "entry": 2,
                        "flags": ["candidate_disagreement"],
                        "severity": "low",
                        "review_required": False,
                        "candidate_timestamps": {"output": 20.0},
                    }
                ],
                "review_required_count": 0,
            }

            timestamps, refined_report, changes = apply_whisperx_acoustic_boundary_refinement(
                Path("dummy.flac"),
                [10.0, 20.0, 25.0],
                report,
                30.0,
            )

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 19.5, places=3)
            assignments = refined_report["assignments"]  # type: ignore[assignment]
            self.assertFalse(assignments[1]["timing_trusted"])  # type: ignore[index]
            suspicious = refined_report["suspicious_alignments"]  # type: ignore[assignment]
            self.assertEqual(suspicious[0]["severity"], "low")  # type: ignore[index]
            self.assertTrue(suspicious[0]["review_required"])  # type: ignore[index]
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_close_neighbor_can_snap_forward_to_next_strong_onset(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 20.5))[0][0]] = 1.0
        features = AudioFeatures(
            duration=30.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            report: dict[str, object] = {
                "timing_entries": 3,
                "assignments": [
                    {"entry": 1, "score": 0.96, "timestamp": 15.0},
                    {"entry": 2, "score": 0.96, "timestamp": 20.0},
                    {"entry": 3, "score": 0.96, "timestamp": 25.0},
                ],
                "suspicious_alignments": [
                    {
                        "entry": 2,
                        "text": "long enough lyric line",
                        "flags": ["close_neighbor_onset_uncertain"],
                        "severity": "medium",
                        "review_required": True,
                        "candidate_timestamps": {"output": 20.0},
                    }
                ],
                "review_required_count": 1,
            }

            timestamps, refined_report, changes = apply_whisperx_acoustic_boundary_refinement(
                Path("dummy.flac"),
                [15.0, 20.0, 25.0],
                report,
                30.0,
            )

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 20.5, places=3)
            self.assertEqual(changes[0]["reason"], "lead-in-acoustic-forward-onset")
            self.assertEqual(refined_report["review_required_count"], 0)
            assignments = refined_report["assignments"]  # type: ignore[assignment]
            self.assertFalse(assignments[1]["timing_trusted"])  # type: ignore[index]
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]


class AuditExportTests(unittest.TestCase):
    def test_audit_rows_expose_timing_trust_reason_and_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lrc_path = tmp_path / "song.lrc"
            lrc_path.write_text("[00:10.00]first\n[00:22.50]second\n", encoding="utf-8")
            report: dict[str, object] = {
                "backend": "hybrid",
                "strategy": "unit-test",
                "timing_entries": 2,
                "trusted_percent": 100.0,
                "timing_trusted_entries": 1,
                "review_required_count": 0,
                "assignments": [
                    {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                    {
                        "entry": 2,
                        "segment": 2,
                        "score": 0.55,
                        "timestamp": 22.5,
                        "chosen_time": 22.5,
                        "confidence": 0.81,
                        "reasons": ["raw-asr-lyric-match"],
                        "penalties": [{"kind": "candidate_disagreement", "value": 0.12}],
                        "split_suggestion": {"suggested_after_text": "second"},
                        "timing_trusted": True,
                        "timing_trusted_reason": "multi-backend-time-consensus",
                        "timing_trusted_sources": ["ctc", "raw"],
                        "ctc_token_spans": [{"char": "s", "start": 22.5, "end": 22.6, "score": 0.8}],
                        "timing_trusted_candidate_times": {
                            "selected": 22.5,
                            "whisperx": 20.0,
                            "ctc": 22.5,
                            "raw": 22.3,
                        },
                    },
                ],
                "suspicious_alignments": [],
            }

            rows = build_rows(lrc_path, report)
            output = tmp_path / "audit.md"
            write_markdown(rows, report, output)
            content = output.read_text(encoding="utf-8")

        self.assertEqual(rows[1]["timing_trusted"], "yes")
        self.assertEqual(rows[1]["timing_trust_sources"], "ctc;raw")
        self.assertEqual(rows[1]["ctc_tokens"], "s@00:22.50/0.800")
        self.assertIn("multi-backend-time-consensus", content)
        self.assertIn("ctc;raw", content)
        self.assertIn("ctc-tok s@00:22.50/0.800", content)
        self.assertIn("0.810", content)
        self.assertEqual(rows[1]["chosen_time"], "00:22.50")
        self.assertEqual(rows[1]["decision_penalties"], "candidate_disagreement:0.12")
        self.assertEqual(rows[1]["split_suggestion"], "second")
        self.assertIn("timing_trusted", content)

    def test_audit_rows_do_not_label_precentral_scoring_as_central_decision_reason(self) -> None:
        with TemporaryDirectory() as tmp:
            lrc_path = Path(tmp) / "song.lrc"
            lrc_path.write_text("[00:10.00]line\n", encoding="utf-8")
            report: dict[str, object] = {
                "assignments": [{
                    "entry": 1,
                    "timestamp": 10.0,
                    "chosen_time": 10.0,
                    "candidate_source": "ctc-current",
                    "reasons": ["legacy-precentral-score"],
                    "penalties": [{"kind": "legacy-penalty", "value": 0.4}],
                }],
                "suspicious_alignments": [],
            }
            rows = build_rows(lrc_path, report)
        self.assertEqual(rows[0]["decision_reasons"], "central-selected:ctc-current")
        self.assertEqual(rows[0]["decision_penalties"], "")


    def test_forward_supported_prefix_candidate_accepts_strong_second_token_with_broad_support(self) -> None:
        candidate = ctc_forward_supported_prefix_candidate(
            {
                "ctc_token_spans": [
                    {"start": 10.00, "end": 10.02, "score": 0.005},
                    {"start": 11.58, "end": 11.60, "score": 0.72},
                    {"start": 11.62, "end": 11.64, "score": 0.02},
                    {"start": 11.68, "end": 11.70, "score": 0.02},
                    {"start": 11.72, "end": 11.74, "score": 0.001},
                    {"start": 11.80, "end": 11.82, "score": 0.06},
                    {"start": 11.88, "end": 11.90, "score": 0.02},
                    {"start": 11.94, "end": 11.96, "score": 0.001},
                    {"start": 12.00, "end": 12.02, "score": 0.002},
                    {"start": 12.08, "end": 12.10, "score": 0.43},
                    {"start": 12.14, "end": 12.16, "score": 0.002},
                    {"start": 12.20, "end": 12.22, "score": 0.66},
                    {"start": 12.28, "end": 12.30, "score": 0.63},
                ]
            }
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["token_index"], 1)  # type: ignore[index]
        self.assertEqual(candidate["support_mode"], "strong-detached-initial-token")  # type: ignore[index]

    def test_forward_supported_prefix_candidate_rejects_distant_midline_cluster(self) -> None:
        spans = [
            {"start": 10.0 + index * 0.10, "end": 10.02 + index * 0.10, "score": 0.005}
            for index in range(12)
        ]
        spans.extend(
            [
                {"start": 12.60, "end": 12.62, "score": 0.10},
                {"start": 12.70, "end": 12.72, "score": 0.12},
                {"start": 12.80, "end": 12.82, "score": 0.11},
            ]
        )

        self.assertIsNone(ctc_forward_supported_prefix_candidate({"ctc_token_spans": spans}))

    def test_strong_detached_initial_recovery_sets_effective_token_index(self) -> None:
        frame_times = np.round(np.arange(0.0, 20.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 11.5))[0][0]] = 1.0
        features = AudioFeatures(
            duration=20.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda audio_path, duration: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["Underwater"]), LyricEntry(["next"])]
            timestamps = [5.0, 10.0, 15.0]
            report: dict[str, object] = {
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 5.0,
                        "ctc_token_spans": [
                            {"char": "x", "start": 8.0, "end": 8.02, "score": 0.2}
                        ],
                    },
                    {
                        "entry": 2,
                        "timestamp": 10.0,
                        "timing_repair": "ctc-forced-align",
                        "ctc_token_spans": [
                            {"char": "u", "start": 10.0, "end": 10.02, "score": 0.005},
                            {"char": "n", "start": 11.58, "end": 11.60, "score": 0.72},
                            {"char": "d", "start": 11.62, "end": 11.64, "score": 0.02},
                            {"char": "e", "start": 11.68, "end": 11.70, "score": 0.02},
                            {"char": "r", "start": 11.72, "end": 11.74, "score": 0.001},
                            {"char": "w", "start": 11.80, "end": 11.82, "score": 0.06},
                            {"char": "a", "start": 11.88, "end": 11.90, "score": 0.02},
                            {"char": "t", "start": 11.94, "end": 11.96, "score": 0.001},
                            {"char": "e", "start": 12.00, "end": 12.02, "score": 0.002},
                            {"char": "r", "start": 12.08, "end": 12.10, "score": 0.43},
                            {"char": "w", "start": 12.14, "end": 12.16, "score": 0.002},
                            {"char": "e", "start": 12.20, "end": 12.22, "score": 0.66},
                            {"char": "r", "start": 12.28, "end": 12.30, "score": 0.63},
                        ],
                    },
                    {"entry": 3, "timestamp": 15.0, "ctc_token_spans": []},
                ]
            }

            changes = apply_ctc_forward_supported_prefix_recovery(
                Path("dummy.flac"), entries, timestamps, report, 20.0
            )

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 11.5, places=3)
            assignment = report["assignments"][1]  # type: ignore[index]
            self.assertEqual(assignment["ctc_effective_token_start_index"], 1)
            self.assertTrue(assignment["ctc_forward_supported_prefix_recovery_trusted"])
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_provisional_boundary_detection_does_not_lower_trust(self) -> None:
        entries = [LyricEntry(["previous"]), LyricEntry(["current"])]
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {"entry": 1, "timestamp": 5.0, "score": 0.9, "timing_repair": "ctc-forced-align"},
                {
                    "entry": 2,
                    "timestamp": 10.0,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_boundary_evidence": {
                        "realign_required": True,
                        "zero_boundary_gap": True,
                        "low_confidence_prefix": True,
                    },
                },
            ],
            "suspicious_alignments": [],
            "review_required_count": 0,
        }

        unresolved = flag_unresolved_ctc_boundary_issues(entries, report, persist=False)

        self.assertEqual(len(unresolved), 1)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(bool(assignment.get("review_required")))
        self.assertNotIn("ctc_unresolved_boundary", assignment)
        self.assertEqual(report["suspicious_alignments"], [])

    def test_final_boundary_validation_removes_resolved_provisional_review(self) -> None:
        entries = [LyricEntry(["previous"]), LyricEntry(["current"])]
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {"entry": 1, "timestamp": 5.0, "score": 0.9, "timing_repair": "ctc-forced-align"},
                {
                    "entry": 2,
                    "timestamp": 10.0,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "review_required": True,
                    "ctc_unresolved_boundary": True,
                    "flags": ["ctc_unresolved_boundary", "ctc_low_confidence_prefix"],
                    "ctc_boundary_evidence": {"realign_required": False},
                },
            ],
            "suspicious_alignments": [
                {
                    "entry": 2,
                    "text": "current",
                    "flags": ["ctc_unresolved_boundary", "ctc_low_confidence_prefix"],
                    "severity": "medium",
                    "review_required": True,
                }
            ],
            "review_required_count": 1,
        }

        unresolved = flag_unresolved_ctc_boundary_issues(entries, report, persist=True)

        self.assertEqual(unresolved, [])
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(assignment["review_required"])
        self.assertNotIn("ctc_unresolved_boundary", assignment)
        self.assertEqual(report["suspicious_alignments"], [])
        self.assertEqual(report["review_required_count"], 0)


class AudioAuditV7Tests(unittest.TestCase):
    def test_detached_initial_recovery_stays_near_supported_token(self) -> None:
        entries = [
            LyricEntry(["previous"]),
            LyricEntry(["Underwater test signals surround the marker"]),
            LyricEntry(["next"]),
        ]
        timestamps = [5.0, 10.0, 20.0]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 5.0, "ctc_token_spans": [{"char": "x", "start": 5.0, "end": 8.0, "score": 0.5}]},
                {
                    "entry": 2,
                    "timestamp": 10.0,
                    "ctc_token_spans": [
                        {"char": "u", "start": 10.0, "end": 10.02, "score": 0.004},
                        {"char": "n", "start": 13.56, "end": 13.58, "score": 0.49},
                        {"char": "d", "start": 13.60, "end": 13.62, "score": 0.12},
                        {"char": "e", "start": 13.66, "end": 13.68, "score": 0.14},
                        {"char": "r", "start": 13.72, "end": 13.74, "score": 0.10},
                        {"char": "w", "start": 13.80, "end": 13.82, "score": 0.11},
                    ],
                },
                {"entry": 3, "timestamp": 20.0, "ctc_token_spans": [{"char": "x", "start": 20.0, "end": 21.0, "score": 0.5}]},
            ]
        }
        frame_times = np.arange(0.0, 25.0, 0.02, dtype=np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.argmin(np.abs(frame_times - 12.80))] = 0.95
        onset_strength[np.argmin(np.abs(frame_times - 13.56))] = 0.80
        features = AudioFeatures(25.0, frame_times, np.zeros_like(frame_times), onset_strength, [])
        original = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda *_args, **_kwargs: features)
        try:
            changes = apply_ctc_forward_supported_prefix_recovery(
                Path("dummy.wav"), entries, timestamps, report, 25.0
            )
        finally:
            split_patch.assign("analyze_audio", original)

        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(timestamps[1], 13.56, places=2)
        self.assertGreater(timestamps[1], 13.30)

    def test_duplicate_lyric_offset_consensus_detects_stable_section(self) -> None:
        entries = [
            LyricEntry(["A"]),
            LyricEntry(["B"]),
            LyricEntry(["C"]),
            LyricEntry(["D"]),
            LyricEntry(["A"]),
            LyricEntry(["B"]),
            LyricEntry(["C"]),
            LyricEntry(["D"]),
        ]
        timestamps = [1.0, 3.0, 5.0, 7.0, 101.0, 103.02, 105.01, 107.0]
        result = auto_lrc.duplicate_lyric_offset_consensus(entries, timestamps)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(float(result["median_delta_seconds"]), 100.0, places=2)


class GeorgetteBoundaryRegressionTests(unittest.TestCase):
    def test_backtracked_timestamp_does_not_impersonate_first_ctc_token(self) -> None:
        assignments: list[object] = [
            {
                "timestamp": 46.556,
                "ctc_token_spans": [
                    {"char": "o", "start": 55.282, "end": 55.302, "score": 0.008932},
                ],
            },
            {
                "timestamp": 55.342,
                "ctc_score": 0.104232,
                "ctc_token_spans": [
                    {"char": "m", "start": 57.002, "end": 57.022, "score": 0.576660},
                    {"char": "o", "start": 57.102, "end": 57.122, "score": 0.249099},
                    {"char": "t", "start": 57.582, "end": 57.603, "score": 0.002623},
                    {"char": "s", "start": 57.643, "end": 57.663, "score": 0.246038},
                    {"char": "u", "start": 57.783, "end": 57.803, "score": 0.252842},
                    {"char": "r", "start": 59.503, "end": 59.523, "score": 0.002233},
                    {"char": "e", "start": 59.563, "end": 59.583, "score": 0.010809},
                    {"char": "t", "start": 59.683, "end": 59.703, "score": 0.098325},
                ],
            },
        ]

        annotate_ctc_boundary_evidence(assignments)

        current = assignments[1]
        self.assertAlmostEqual(current["ctc_boundary_start"], 55.342, places=3)  # type: ignore[index]
        self.assertAlmostEqual(current["ctc_first_token_start"], 57.002, places=3)  # type: ignore[index]
        self.assertAlmostEqual(current["ctc_boundary_gap_seconds"], 0.040, places=3)  # type: ignore[index]
        evidence = current["ctc_boundary_evidence"]  # type: ignore[index]
        self.assertAlmostEqual(evidence["ctc_token_lead_seconds"], 1.660, places=3)
        self.assertTrue(evidence["timestamp_precedes_ctc_support"])

    def test_0055_posterior_backtrack_cannot_borrow_suruno_tail(self) -> None:
        features = AudioFeatures(
            duration=80.0,
            frame_times=np.round(np.arange(0.0, 80.0, 0.1), 3).astype(np.float32),
            rms_db=np.full(800, -20.0, dtype=np.float32),
            onset_strength=np.zeros(800, dtype=np.float32),
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda *_args, **_kwargs: features)  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["前の合成試験行"]),
                LyricEntry(["もつれた　テストを　まわる"]),
                LyricEntry(["次の合成試験行"]),
            ]
            timestamps = [46.556, 57.002, 67.223]
            report: dict[str, object] = {
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 46.556,
                        "timing_repair": "ctc-forced-align",
                        "ctc_score": 0.564761,
                    },
                    {
                        "entry": 2,
                        "timestamp": 57.002,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.104232,
                        "romaji": "motsuretatestwomawaru",
                        "ctc_previous_token_end": 55.302,
                        "ctc_first_token_candidates": [
                            {"time": 55.762, "score": 0.009981},
                            {"time": 55.342, "score": 0.001064},
                        ],
                    },
                    {
                        "entry": 3,
                        "timestamp": 67.223,
                        "timing_repair": "ctc-forced-align",
                        "ctc_score": 0.136584,
                    },
                ]
            }

            changes = apply_ctc_acoustic_backtrack(
                Path("dummy.flac"), entries, timestamps, report, 80.0
            )

            self.assertEqual(changes, [])
            self.assertAlmostEqual(timestamps[1], 57.002, places=3)
            self.assertEqual(report["ctc_acoustic_backtrack_rejection_count"], 1)
            rejection = report["ctc_acoustic_backtrack_rejections"][0]  # type: ignore[index]
            self.assertEqual(rejection["reason"], "candidate-borrows-previous-line-tail")
            self.assertAlmostEqual(rejection["candidate_gap_seconds"], 0.040, places=3)
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_0344_weak_zero_gap_after_again_stays_review_required(self) -> None:
        entries = [LyricEntry(["... again"]), LyricEntry(["Everlasting test pattern ever ever ever"])]
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 215.013,
                    "ctc_score": 0.158220,
                    "ctc_token_spans": [
                        {"char": "g", "start": 224.036, "end": 224.056, "score": 0.007160},
                        {"char": "a", "start": 224.056, "end": 224.076, "score": 0.001813},
                        {"char": "i", "start": 224.136, "end": 224.156, "score": 0.290717},
                        {"char": "n", "start": 224.196, "end": 224.216, "score": 0.013697},
                    ],
                },
                {
                    "entry": 2,
                    "timestamp": 224.236,
                    "score": 0.9,
                    "ctc_score": 0.140417,
                    "ctc_token_spans": [
                        {"char": "e", "start": 224.236, "end": 224.256, "score": 0.008757},
                        {"char": "v", "start": 224.276, "end": 224.296, "score": 0.000918},
                        {"char": "e", "start": 224.356, "end": 224.376, "score": 0.157146},
                        {"char": "r", "start": 224.496, "end": 224.516, "score": 0.353693},
                        {"char": "l", "start": 224.576, "end": 224.596, "score": 0.001283},
                        {"char": "a", "start": 224.596, "end": 224.616, "score": 0.645580},
                        {"char": "s", "start": 224.656, "end": 224.676, "score": 0.000583},
                        {"char": "t", "start": 224.896, "end": 224.916, "score": 0.051476},
                    ],
                },
            ],
            "suspicious_alignments": [],
        }

        annotate_ctc_boundary_evidence(report["assignments"])  # type: ignore[arg-type]
        current = report["assignments"][1]  # type: ignore[index]
        evidence = current["ctc_boundary_evidence"]
        self.assertTrue(evidence["borrowed_tail_prefix"])
        self.assertTrue(evidence["realign_required"])

        timestamps = [215.013, 224.236]
        timestamps, report, changes = apply_ctc_crossline_initial_recovery(
            timestamps, report, 234.680
        )
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["mode"], "short-borrowed-tail-prefix")
        self.assertAlmostEqual(timestamps[1], 224.356, places=3)
        self.assertEqual(current["ctc_effective_token_start_index"], 2)
        self.assertTrue(current["ctc_crossline_initial_recovery_trusted"])

        annotate_ctc_boundary_evidence(report["assignments"])  # type: ignore[arg-type]
        unresolved = flag_unresolved_ctc_boundary_issues(entries, report, persist=True)
        self.assertEqual(unresolved, [])
        self.assertFalse(bool(current.get("review_required")))


    def test_trusted_percent_requires_timing_boundary_trust(self) -> None:
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {
                    "entry": 1,
                    "timing_repair": "ctc-forced-align",
                    "score": 0.9,
                    "timestamp": 10.0,
                    "ctc_token_spans": [
                        {"char": "a", "start": 19.98, "end": 20.0, "score": 0.8},
                    ],
                },
                {
                    "entry": 2,
                    "timing_repair": "ctc-forced-align",
                    "score": 0.9,
                    "timestamp": 20.04,
                    "ctc_previous_token_end": 20.0,
                    "ctc_previous_token_score": 0.01,
                    "ctc_token_spans": [
                        {"char": "e", "start": 20.04, "end": 20.06, "score": 0.02},
                        {"char": "v", "start": 20.10, "end": 20.12, "score": 0.01},
                        {"char": "e", "start": 20.16, "end": 20.18, "score": 0.02},
                        {"char": "r", "start": 21.10, "end": 21.12, "score": 0.20},
                        {"char": "x", "start": 21.20, "end": 21.22, "score": 0.20},
                        {"char": "x", "start": 21.30, "end": 21.32, "score": 0.20},
                        {"char": "x", "start": 21.40, "end": 21.42, "score": 0.20},
                        {"char": "x", "start": 21.50, "end": 21.52, "score": 0.20},
                    ],
                },
            ],
        }
        annotate_ctc_boundary_evidence(report["assignments"])  # type: ignore[arg-type]
        update_report_confidence_metrics(report)
        self.assertEqual(report["content_trusted_percent"], 100.0)
        self.assertEqual(report["timing_trusted_percent"], 50.0)
        self.assertEqual(report["overall_trusted_percent"], 50.0)
        self.assertEqual(report["trusted_percent"], 50.0)
        self.assertFalse(assignment_timing_is_trusted(report["assignments"][1]))  # type: ignore[index]

    def test_final_guard_moves_again_boundary_to_its_own_acoustic_onset(self) -> None:
        frame_times = np.round(np.arange(220.0, 231.0, 0.01), 3).astype(np.float32)
        onset = np.zeros(len(frame_times), dtype=np.float32)
        onset[int(round((226.16 - 220.0) / 0.01))] = 1.0
        features = AudioFeatures(
            duration=234.68,
            frame_times=frame_times,
            rms_db=np.full(len(frame_times), -20.0, dtype=np.float32),
            onset_strength=onset,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda *_args, **_kwargs: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["... again"]), LyricEntry(["everlastin ever 離れぬように"])]
            report: dict[str, object] = {
                "backend": "ctc",
                "timing_entries": 2,
                "assignments": [
                    {
                        "entry": 1,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 205.57,
                        "ctc_token_spans": [
                            {"char": "g", "start": 225.557, "end": 225.577, "score": 0.01},
                            {"char": "a", "start": 225.617, "end": 225.637, "score": 0.01},
                            {"char": "i", "start": 225.677, "end": 225.697, "score": 0.29},
                            {"char": "n", "start": 225.717, "end": 225.737, "score": 0.015},
                        ],
                    },
                    {
                        "entry": 2,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 225.777,
                        "ctc_score": 0.131604,
                        "ctc_token_spans": [
                            {"char": "e", "start": 225.777, "end": 225.797, "score": 0.035797},
                            {"char": "v", "start": 225.937, "end": 225.957, "score": 0.000460},
                            {"char": "e", "start": 226.037, "end": 226.057, "score": 0.060586},
                            {"char": "r", "start": 226.777, "end": 226.797, "score": 0.002959},
                            {"char": "l", "start": 226.797, "end": 226.817, "score": 0.000898},
                            {"char": "a", "start": 226.877, "end": 226.897, "score": 0.254915},
                            {"char": "s", "start": 227.057, "end": 227.077, "score": 0.000907},
                            {"char": "t", "start": 227.257, "end": 227.277, "score": 0.005328},
                        ],
                    },
                ],
                "suspicious_alignments": [],
            }
            timestamps, report, changes = apply_final_ctc_timing_guard(
                Path("dummy.flac"), entries, [205.57, 225.777], report, 234.68
            )
            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 226.16, places=2)
            current = report["assignments"][1]  # type: ignore[index]
            self.assertTrue(current["ctc_final_timing_recovery_trusted"])
            self.assertTrue(current["timing_audit_trusted"])
            self.assertFalse(bool(current.get("review_required")))
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_final_guard_catches_80ms_again_leak_with_isolated_first_token(self) -> None:
        """Regression for the v2 miss: token 0 looked usable, the prefix did not."""
        frame_times = np.round(np.arange(223.0, 229.0, 0.01), 3).astype(np.float32)
        onset = np.zeros(len(frame_times), dtype=np.float32)
        onset[int(round((226.16 - 223.0) / 0.01))] = 1.0
        features = AudioFeatures(
            duration=234.68,
            frame_times=frame_times,
            rms_db=np.full(len(frame_times), -20.0, dtype=np.float32),
            onset_strength=onset,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda *_args, **_kwargs: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["... again"]), LyricEntry(["everlastin ever 離れぬように"])]
            report: dict[str, object] = {
                "backend": "ctc",
                "timing_entries": 2,
                "assignments": [
                    {
                        "entry": 1,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 215.013,
                        "ctc_token_spans": [
                            {"char": "g", "start": 224.496, "end": 224.516, "score": 0.016454},
                            {"char": "a", "start": 224.596, "end": 224.616, "score": 0.644287},
                            {"char": "i", "start": 224.756, "end": 224.776, "score": 0.001853},
                            {"char": "n", "start": 224.896, "end": 224.916, "score": 0.010721},
                        ],
                    },
                    {
                        "entry": 2,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 224.996,
                        "ctc_score": 0.124389,
                        "ctc_token_spans": [
                            {"char": "e", "start": 224.996, "end": 225.016, "score": 0.203845},
                            {"char": "v", "start": 225.257, "end": 225.277, "score": 0.000450},
                            {"char": "e", "start": 225.337, "end": 225.357, "score": 0.048510},
                            {"char": "r", "start": 225.517, "end": 225.537, "score": 0.018484},
                            {"char": "l", "start": 225.537, "end": 225.557, "score": 0.006313},
                            {"char": "a", "start": 226.877, "end": 226.897, "score": 0.226242},
                            {"char": "s", "start": 227.057, "end": 227.077, "score": 0.001424},
                            {"char": "t", "start": 227.257, "end": 227.277, "score": 0.008473},
                        ],
                    },
                ],
                "suspicious_alignments": [],
            }
            annotate_ctc_boundary_evidence(report["assignments"])  # type: ignore[arg-type]
            evidence = report["assignments"][1]["ctc_boundary_evidence"]  # type: ignore[index]
            self.assertTrue(evidence["near_gap_weak_prefix"])
            self.assertFalse(assignment_timing_is_trusted(report["assignments"][1]))  # type: ignore[index]
            timestamps, report, changes = apply_final_ctc_timing_guard(
                Path("dummy.flac"), entries, [215.013, 224.996], report, 234.68
            )
            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 226.16, places=2)
            self.assertTrue(report["assignments"][1]["timing_audit_trusted"])  # type: ignore[index]
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_final_guard_salvages_onset_from_rejected_right_edge_row(self) -> None:
        frame_times = np.round(np.arange(143.0, 149.0, 0.01), 3).astype(np.float32)
        onset = np.zeros(len(frame_times), dtype=np.float32)
        onset[int(round((147.42 - 143.0) / 0.01))] = 0.8
        features = AudioFeatures(
            duration=185.0,
            frame_times=frame_times,
            rms_db=np.full(len(frame_times), -20.0, dtype=np.float32),
            onset_strength=onset,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        split_patch.assign("analyze_audio", lambda *_args, **_kwargs: features)  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["もつれたまま"]), LyricEntry(["next"])]
            report: dict[str, object] = {
                "backend": "ctc",
                "timing_entries": 3,
                "assignments": [
                    {
                        "entry": 1,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 136.93,
                        "ctc_token_spans": [{"char": "x", "start": 144.806, "end": 144.826, "score": 0.01}],
                    },
                    {
                        "entry": 2,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.35,
                        "timestamp": 144.866,
                        "ctc_score": 0.0245,
                        "ctc_token_spans": [
                            {"char": "m", "start": 144.866, "end": 144.886, "score": 0.00129},
                            {"char": "o", "start": 144.966, "end": 144.986, "score": 0.003849},
                            {"char": "t", "start": 144.986, "end": 145.006, "score": 0.079064},
                            {"char": "s", "start": 145.006, "end": 145.026, "score": 0.261354},
                            {"char": "u", "start": 145.186, "end": 145.206, "score": 0.351242},
                            {"char": "r", "start": 145.406, "end": 145.426, "score": 0.002466},
                            {"char": "e", "start": 146.046, "end": 146.066, "score": 0.003242},
                            {"char": "t", "start": 146.066, "end": 146.086, "score": 0.014562},
                        ],
                    },
                    {
                        "entry": 3,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 167.588,
                        "ctc_token_spans": [{"char": "n", "start": 167.588, "end": 167.608, "score": 0.6}],
                    },
                ],
                "suspicious_alignments": [],
                "ctc_local_window_realign": {
                    "status": "applied",
                    "rejected_rows": [
                        {
                            "entry": 2,
                            "candidate": 147.42,
                            "reason": "local-window-right-edge-pileup",
                            "ctc_score": 0.014457,
                        }
                    ],
                },
            }
            timestamps, report, changes = apply_final_ctc_timing_guard(
                Path("dummy.flac"), entries, [136.93, 144.866, 167.588], report, 185.0
            )
            # A rejected full-line row no longer donates its onset.  Without an
            # independent prefix probe the line must remain unresolved.
            self.assertEqual(len(changes), 0)
            self.assertAlmostEqual(timestamps[1], 144.866, places=3)
            self.assertTrue(report["assignments"][1]["ctc_final_timing_guard_unresolved"])  # type: ignore[index]
            self.assertFalse(report["assignments"][1]["timing_audit_trusted"])  # type: ignore[index]
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_path_fracture_detects_disconnected_ctc_islands(self) -> None:
        assignment = {
            "ctc_token_spans": [
                {"start": 10.0, "end": 10.1, "score": 0.2},
                {"start": 10.2, "end": 10.3, "score": 0.2},
                {"start": 10.4, "end": 10.5, "score": 0.2},
                {"start": 17.0, "end": 17.1, "score": 0.2},
            ]
        }
        evidence = ctc_path_fracture_evidence(assignment)
        self.assertTrue(evidence["fractured"])
        self.assertAlmostEqual(evidence["max_gap_seconds"], 6.5, places=3)

    def test_duplicate_block_offset_projects_boundary_without_using_manual_timestamp(self) -> None:
        entries = [
            LyricEntry(["repeat A"]),
            LyricEntry(["repeat B"]),
            LyricEntry(["middle"]),
            LyricEntry(["repeat A"]),
            LyricEntry(["repeat B"]),
        ]
        assignments: list[dict[str, object]] = [
            {
                "timestamp": 10.0,
                "ctc_token_spans": [
                    {"char": "a", "start": 10.0, "score": 0.2},
                    {"char": "b", "start": 10.5, "score": 0.2},
                    {"char": "c", "start": 11.0, "score": 0.2},
                ],
            },
            {"timestamp": 20.0, "ctc_token_spans": [{"char": "x", "start": 20.0, "score": 0.01}]},
            {"timestamp": 50.0, "ctc_token_spans": [{"char": "m", "start": 50.0, "score": 0.2}]},
            {
                "timestamp": 110.0,
                "ctc_token_spans": [
                    {"char": "a", "start": 110.02, "score": 0.2},
                    {"char": "b", "start": 110.52, "score": 0.2},
                    {"char": "c", "start": 111.02, "score": 0.2},
                ],
            },
            {"timestamp": 121.4, "ctc_token_spans": [{"char": "x", "start": 121.4, "score": 0.01}]},
        ]
        evidence = duplicate_block_offset_evidence(entries, assignments, 1)
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertAlmostEqual(float(evidence["offset_seconds"]), 100.02, places=2)
        # The projection comes from the adjacent repeated line's acoustic/CTC
        # offset, not from a manually supplied expected boundary.
        self.assertAlmostEqual(float(evidence["projected_time"]), 21.38, places=2)


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
