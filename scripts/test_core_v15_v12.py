#!/usr/bin/env python3
"""V15.1-V15.2 probing/convergence, V12 hubert/node19 and demucs/backend participation tests."""

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
from _core_test_support import _TempModelCacheMixin


class V151MonotonicLocalProbingRegressionTests(_TempModelCacheMixin, unittest.TestCase):
    def test_context_prefix_onset_crosses_segments_without_full_exact_line(self):
        segments = [
            auto_lrc.AsrSegment(1.0, 1.5, "今日の", chars=[("今", 1.02), ("日", 1.20), ("の", 1.42)]),
            auto_lrc.AsrSegment(1.5, 2.1, "天気は", chars=[("天", 1.52), ("気", 1.70), ("は", 1.90)]),
            auto_lrc.AsrSegment(2.1, 2.5, "別語", chars=[("別", 2.12), ("語", 2.30)]),
        ]
        # Full immutable line is intentionally absent; its first six normalized
        # characters are a directly forced, cross-segment observation.
        matches = auto_lrc._aligned_target_prefix_matches("今日の天気は晴れ", segments)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(float(matches[0]["onset_time"]), 1.02, places=6)
        self.assertEqual(matches[0]["start_segment_index"], 0)
        self.assertEqual(matches[0]["end_segment_index"], 1)

    def test_context_prefix_selector_fails_closed_when_two_matches_survive(self):
        entries = [
            LyricEntry(["left anchor"]), LyricEntry(["今日の天気は晴れ"]), LyricEntry(["right anchor"]),
        ]
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .95, "segment": 2, "borrowed": False},
            {"timestamp": 5.0, "score": .95, "segment": 4, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        segments = [
            auto_lrc.AsrSegment(1.5, 2.2, "今日の天気は", chars=[(c, 1.5 + i*.05) for i,c in enumerate("今日の天気は")]),
            auto_lrc.AsrSegment(3.0, 3.7, "今日の天気は", chars=[(c, 3.0 + i*.05) for i,c in enumerate("今日の天気は")]),
        ]
        selected = auto_lrc._select_unique_aligned_target_prefix_match(
            "今日の天気は晴れ", segments,
            occurrence_binding=binding, entry_index=1, preferred_segment_number=None,
        )
        self.assertIsNone(selected)

    def test_runtime_merge_prefers_direct_onset_over_higher_score_occurrence_only(self):
        entries = [LyricEntry(["left"]), LyricEntry(["target"]), LyricEntry(["right"])]
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .96, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        def req(name, start, end):
            payload = dict(
                entry_index=1, capability_revision="capability", audio_revision="audio",
                context_entry_indexes=(0,1,2), context_candidate_ids=("a","b","c"),
                transcript_digests=("a","b","c"), transcript_digest="abc",
                window_start=start, window_end=end, window_definition_revision=name,
            )
            return auto_lrc.LocalWhisperRequest(**payload, request_revision=name)
        narrow=req("narrow",0.0,4.0)
        broad=req("broad",0.0,5.0)
        occurrence_only=auto_lrc.LocalWhisperOutcome(
            entry_index=1, status="evidence", reason="occurrence-only", onset_time=None,
            score=.99, evidence_revision="ignored", occurrence_binding=binding,
            raw_transcript_revision="raw-broad", aligned_transcript_revision="aligned-broad",
            request_revision="broad", outcome_revision="out-broad",
        )
        with_onset=auto_lrc.LocalWhisperOutcome(
            entry_index=1, status="evidence", reason="with-onset", onset_time=2.1,
            score=.70, evidence_revision="ignored", occurrence_binding=binding,
            raw_transcript_revision="raw-narrow", aligned_transcript_revision="aligned-narrow",
            request_revision="narrow", outcome_revision="out-narrow",
        )
        report={}
        auto_lrc.bind_runtime_local_whisper_evidence(
            report, entries, capability, (broad,narrow), (occurrence_only,with_onset)
        )
        evidence=auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 1)
        self.assertIsNotNone(evidence)
        self.assertAlmostEqual(float(evidence["onset_time"]),2.1,places=6)
        self.assertAlmostEqual(float(evidence["score"]),.70,places=6)

    def test_weak_target_assignment_can_recover_from_context_bound_forced_prefix(self):
        entries = [
            LyricEntry(["anchor alpha"]), LyricEntry(["repeat target"]),
            LyricEntry(["shared middle"]), LyricEntry(["branch beta"]),
            LyricEntry(["separator"]), LyricEntry(["anchor alpha"]),
            LyricEntry(["repeat target"]), LyricEntry(["shared middle"]),
            LyricEntry(["branch gamma"]),
        ]
        request = auto_lrc.LocalWhisperRequest(
            entry_index=1, capability_revision="capability", audio_revision="audio",
            context_entry_indexes=(0, 1, 2, 3),
            context_candidate_ids=("a", "b", "c", "d"),
            transcript_digests=("ta", "tb", "tc", "td"), transcript_digest="all",
            window_start=10.0, window_end=20.0,
            window_definition_revision="window", request_revision="request-prefix",
        )
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        raw_segments = [auto_lrc.AsrSegment(0.0, 1.0, "raw")]
        # The target's full text is deliberately absent.  Only its forced opening
        # prefix is present, while the surrounding rows are enough to prove the
        # immutable lyric occurrence.
        aligned_segments = [
            auto_lrc.AsrSegment(.4, .9, "anchoralpha", chars=[(c, .4 + i*.02) for i,c in enumerate("anchoralpha")]),
            auto_lrc.AsrSegment(1.2, 1.7, "repeat", chars=[(c, 1.2 + i*.02) for i,c in enumerate("repeat")]),
            auto_lrc.AsrSegment(2.4, 3.1, "sharedmiddle", chars=[(c, 2.4 + i*.02) for i,c in enumerate("sharedmiddle")]),
            auto_lrc.AsrSegment(3.5, 4.2, "branchbeta", chars=[(c, 3.5 + i*.02) for i,c in enumerate("branchbeta")]),
        ]
        aligned_payload = {
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text,
                 "chars": [{"char": c, "start": t} for c,t in seg.chars]}
                for seg in aligned_segments
            ]
        }
        local_report = {"assignments": [
            {"timestamp": .4, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 1.2, "score": .40, "segment": 2, "borrowed": False},
            {"timestamp": 2.4, "score": .20, "segment": 3, "borrowed": False},
            {"timestamp": 3.5, "score": .94, "segment": 4, "borrowed": False},
        ]}
        with split_patch.patch("_stable_file_sha256", return_value="audio"), \
             mock.patch.object(local_whisper_retry, "decode_temp_wav_window", return_value=Path("window.wav")), \
             split_patch.patch("_run_whispercpp_wav", return_value=raw_segments), \
             split_patch.patch("run_whisperx_alignment", return_value=aligned_payload), \
             split_patch.patch("match_whisper_segments", return_value=([.4, 1.2, 2.4, 3.5], local_report)):
            outcome = auto_lrc.execute_local_whisper_request(
                request, capability, entries, mock.Mock(model_cache=True, model_cache_only=False), cache={}, runtime_stats={}
            )
        self.assertEqual(outcome.status, "evidence")
        self.assertEqual(outcome.reason, "trusted-local-whisper-context-prefix-recovery-and-onset")
        self.assertAlmostEqual(float(outcome.onset_time or 0.0), 11.2, places=6)
        self.assertEqual(outcome.occurrence_binding["mode"], "sparse-sequence-bracket")


class V152BoundaryContentConvergenceRegressionTests(_TempModelCacheMixin, unittest.TestCase):
    def test_one_sided_sparse_fingerprint_cannot_claim_bracket_authority(self):
        entries = [
            LyricEntry(["anchor alpha"]), LyricEntry(["branch beta"]), LyricEntry(["repeat target"]),
            LyricEntry(["weak after"]), LyricEntry(["separator"]),
            LyricEntry(["anchor alpha"]), LyricEntry(["branch gamma"]), LyricEntry(["repeat target"]),
            LyricEntry(["weak after"]),
        ]
        assignments: list[object] = [None] * len(entries)
        assignments[0] = {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False}
        assignments[1] = {"timestamp": 2.0, "score": .95, "segment": 2, "borrowed": False}
        assignments[2] = {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False}
        binding = auto_lrc._entry_occurrence_binding(
            entries, assignments, 2, require_context=True
        )
        self.assertIsNone(binding)

    def test_exact_phonetic_identity_recovers_content_without_inventing_onset(self):
        entries = [LyricEntry(["left anchor"]), LyricEntry(["東京"]), LyricEntry(["right anchor"])]
        request = auto_lrc.LocalWhisperRequest(
            entry_index=1, capability_revision="capability", audio_revision="audio",
            context_entry_indexes=(0, 1, 2), context_candidate_ids=("a", "b", "c"),
            transcript_digests=("ta", "tb", "tc"), transcript_digest="all",
            window_start=10.0, window_end=20.0,
            window_definition_revision="window", request_revision="request-phonetic",
        )
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        raw_segments = [auto_lrc.AsrSegment(0.0, 1.0, "raw")]
        aligned_segments = [
            auto_lrc.AsrSegment(.4, .9, "leftanchor", chars=[(c, .4 + i*.02) for i,c in enumerate("leftanchor")]),
            auto_lrc.AsrSegment(1.2, 1.8, "とうきょう", chars=[(c, 1.2 + i*.05) for i,c in enumerate("とうきょう")]),
            auto_lrc.AsrSegment(2.4, 3.1, "rightanchor", chars=[(c, 2.4 + i*.02) for i,c in enumerate("rightanchor")]),
        ]
        aligned_payload = {
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text,
                 "chars": [{"char": c, "start": t} for c,t in seg.chars]}
                for seg in aligned_segments
            ]
        }
        local_report = {"assignments": [
            {"timestamp": .4, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 1.2, "score": .40, "segment": 2, "borrowed": False},
            {"timestamp": 2.4, "score": .95, "segment": 3, "borrowed": False},
        ]}

        real_romaji = auto_lrc.japanese_romaji
        def fake_romaji(text: str):
            if text in {"東京", "とうきょう"}:
                return "tokyo"
            return real_romaji(text) or text

        with split_patch.patch("_stable_file_sha256", return_value="audio"), \
             mock.patch.object(local_whisper_retry, "decode_temp_wav_window", return_value=Path("window.wav")), \
             split_patch.patch("_run_whispercpp_wav", return_value=raw_segments), \
             split_patch.patch("run_whisperx_alignment", return_value=aligned_payload), \
             split_patch.patch("match_whisper_segments", return_value=([.4, 1.2, 2.4], local_report)), \
             split_patch.patch("japanese_romaji", side_effect=fake_romaji):
            outcome = auto_lrc.execute_local_whisper_request(
                request, capability, entries, mock.Mock(model_cache=True, model_cache_only=False), cache={}, runtime_stats={}
            )
        self.assertEqual(outcome.status, "evidence")
        self.assertEqual(outcome.reason, "trusted-local-whisper-exact-phonetic-occurrence-only")
        self.assertIsNone(outcome.onset_time)
        self.assertIsNotNone(outcome.occurrence_binding)
        self.assertEqual(outcome.occurrence_binding["mode"], "sequence-bracket")

    def test_exact_phonetic_identity_ambiguity_fails_closed_end_to_end(self):
        entries = [LyricEntry(["left anchor"]), LyricEntry(["東京"]), LyricEntry(["right anchor"])]
        request = auto_lrc.LocalWhisperRequest(
            entry_index=1, capability_revision="capability", audio_revision="audio",
            context_entry_indexes=(0, 1, 2), context_candidate_ids=("a", "b", "c"),
            transcript_digests=("ta", "tb", "tc"), transcript_digest="all",
            window_start=10.0, window_end=20.0,
            window_definition_revision="window", request_revision="request-phonetic-ambiguous",
        )
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        raw_segments = [auto_lrc.AsrSegment(0.0, 1.0, "raw")]
        aligned_segments = [
            auto_lrc.AsrSegment(.4, .9, "leftanchor", chars=[(c, .4 + i*.02) for i,c in enumerate("leftanchor")]),
            auto_lrc.AsrSegment(1.2, 1.8, "とうきょう", chars=[(c, 1.2 + i*.05) for i,c in enumerate("とうきょう")]),
            auto_lrc.AsrSegment(2.0, 2.6, "とうきょう", chars=[(c, 2.0 + i*.05) for i,c in enumerate("とうきょう")]),
            auto_lrc.AsrSegment(3.0, 3.7, "rightanchor", chars=[(c, 3.0 + i*.02) for i,c in enumerate("rightanchor")]),
        ]
        aligned_payload = {
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text,
                 "chars": [{"char": c, "start": t} for c,t in seg.chars]}
                for seg in aligned_segments
            ]
        }
        local_report = {"assignments": [
            {"timestamp": .4, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 1.2, "score": .40, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 4, "borrowed": False},
        ]}
        real_romaji = auto_lrc.japanese_romaji

        def fake_romaji(text: str):
            if text in {"東京", "とうきょう"}:
                return "tokyo"
            return real_romaji(text) or text

        with split_patch.patch("_stable_file_sha256", return_value="audio"), \
             mock.patch.object(local_whisper_retry, "decode_temp_wav_window", return_value=Path("window.wav")), \
             split_patch.patch("_run_whispercpp_wav", return_value=raw_segments), \
             split_patch.patch("run_whisperx_alignment", return_value=aligned_payload), \
             split_patch.patch("whisperx_segments", return_value=aligned_segments), \
             split_patch.patch("match_whisper_segments", return_value=([.4, 1.2, 3.0], local_report)), \
             split_patch.patch("japanese_romaji", side_effect=fake_romaji):
            outcome = auto_lrc.execute_local_whisper_request(
                request, capability, entries, mock.Mock(model_cache=True, model_cache_only=False), cache={}, runtime_stats={}
            )
        self.assertEqual(outcome.status, "invalid")
        self.assertEqual(outcome.reason, "target-phonetic-occurrence-ambiguous")
        self.assertIsNone(outcome.occurrence_binding)

    def test_phonetic_identity_fails_closed_when_two_occurrences_bind(self):
        segments = [
            auto_lrc.AsrSegment(0.0, .5, "とうきょう"),
            auto_lrc.AsrSegment(1.0, 1.5, "とうきょう"),
        ]
        real_romaji = auto_lrc.japanese_romaji
        def fake_romaji(text: str):
            if text in {"東京", "とうきょう"}:
                return "tokyo"
            return real_romaji(text) or text
        with split_patch.patch("japanese_romaji", side_effect=fake_romaji):
            matches = auto_lrc._aligned_exact_phonetic_target_matches("東京", segments)
        self.assertEqual(len(matches), 2)
        self.assertNotEqual(matches[0]["match_revision"], matches[1]["match_revision"])

    def test_occurrence_only_local_whisper_vocal_peak_is_not_independent_onset(self):
        entries = [LyricEntry(["left"]), LyricEntry(["target"]), LyricEntry(["right"])]
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .90, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        assert binding is not None
        local_evidence = {
            "evidence_revision": "local-evidence",
            "onset_time": None,
            "score": .90,
            "occurrence_binding": binding,
        }
        features = AudioFeatures(
            duration=4.0,
            frame_times=np.asarray([1.8, 2.0, 2.1, 2.2]),
            rms_db=np.zeros(4), onset_strength=np.asarray([.1, .2, .95, .2]),
            segments=[(1.9, 2.8)],
        )
        with mock.patch.object(local_whisper_retry, "_strong_vocal_onset_near", return_value=(2.1, .95, .80)):
            candidate = auto_lrc._local_whisper_vocal_fusion_candidate(
                entry_index=1, entry=entries[1], local_evidence=local_evidence,
                vocal_onset_features=features, audio_revision="audio-sha",
                occurrence_evidence=auto_lrc.candidate_occurrence_evidence_from_binding(
                    entries,
                    1,
                    producer="local-whisper-vocal-fusion",
                    occurrence_binding=binding,
                    upstream_evidence_revision="local-evidence",
                ),
            )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.source, "local-whisper-vocal-independent-fusion")
        self.assertIsNotNone(candidate.direct_onset_evidence)
        assert candidate.direct_onset_evidence is not None
        self.assertEqual(candidate.direct_onset_evidence.kind, "vocal-onset")
        self.assertFalse(candidate.direct_onset_evidence.independent)
        self.assertFalse(auto_lrc.candidate_has_bound_direct_onset_evidence(candidate))
        self.assertFalse(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(candidate))
        self.assertTrue(candidate.independent_content_identity)
        self.assertAlmostEqual(candidate.written_time.seconds, 2.1, places=6)

    def test_complementary_local_whisper_evidence_merges_context_and_direct_onset(self):
        entries = [LyricEntry(["left"]), LyricEntry(["target"]), LyricEntry(["right"])]
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .96, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False},
        ]
        context_binding = auto_lrc._entry_occurrence_binding(
            entries, assignments, 1, require_context=True
        )
        global_binding = auto_lrc._entry_occurrence_binding(
            entries, assignments, 1, require_context=False
        )
        self.assertIsNotNone(context_binding)
        self.assertIsNotNone(global_binding)
        assert context_binding is not None and global_binding is not None
        self.assertNotEqual(context_binding["mode"], global_binding["mode"])
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        def req(rev: str, end: float):
            return auto_lrc.LocalWhisperRequest(
                entry_index=1, capability_revision="capability", audio_revision="audio",
                context_entry_indexes=(0, 1, 2), context_candidate_ids=("a", "b", "c"),
                transcript_digests=("a", "b", "c"), transcript_digest="abc",
                window_start=0.0, window_end=end, window_definition_revision=rev,
                request_revision=rev,
            )
        narrow, broad = req("narrow", 4.0), req("broad", 5.0)
        onset = auto_lrc.LocalWhisperOutcome(
            entry_index=1, status="evidence", reason="onset", onset_time=2.05,
            score=.96, evidence_revision="unused", occurrence_binding=global_binding,
            raw_transcript_revision="raw-narrow", aligned_transcript_revision="aligned-narrow",
            request_revision="narrow", outcome_revision="out-narrow",
        )
        context = auto_lrc.LocalWhisperOutcome(
            entry_index=1, status="evidence", reason="context", onset_time=None,
            score=.90, evidence_revision="unused", occurrence_binding=context_binding,
            raw_transcript_revision="raw-broad", aligned_transcript_revision="aligned-broad",
            request_revision="broad", outcome_revision="out-broad",
        )
        report = {}
        auto_lrc.bind_runtime_local_whisper_evidence(
            report, entries, capability, (narrow, broad), (onset, context)
        )
        evidence = auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 1)
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertAlmostEqual(float(evidence["onset_time"]), 2.05, places=6)
        self.assertTrue(auto_lrc._occurrence_binding_has_strong_context(evidence["occurrence_binding"]))
        self.assertIsNotNone(evidence.get("composite_onset_evidence_revision"))
        self.assertIsNotNone(evidence.get("composite_context_evidence_revision"))

    def test_local_whisper_vocal_ctc_consensus_is_evidence_additive(self):
        entries = [LyricEntry(["left"]), LyricEntry(["target"]), LyricEntry(["right"])]
        parent = auto_lrc.make_timing_candidate(
            entry_index=1, entry_text="target", source="ctc-current", raw_time=2.0,
            confidence=.90, identity_support="supported", sequence_support="supported",
            current=True, source_artifact={"producer": "ctc"},
        )
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .90, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(
            entries, assignments, 1, require_context=True
        )
        self.assertIsNotNone(binding)
        assert binding is not None
        local_evidence = {
            "evidence_revision": "local", "onset_time": None, "score": .90,
            "occurrence_binding": binding,
        }
        features = AudioFeatures(
            duration=4.0, frame_times=np.asarray([1.9, 2.0, 2.1]),
            rms_db=np.zeros(3), onset_strength=np.asarray([.1, .2, .95]),
            segments=[(1.8, 2.8)],
        )
        with split_patch.patch("runtime_local_whisper_evidence_for_entry", return_value=local_evidence
        ), mock.patch.object(local_whisper_retry, "_strong_vocal_onset_near", return_value=(2.1, .95, .80)
        ):
            augmented = auto_lrc._augment_candidates_with_local_whisper_vocal_consensus(
                ((), (parent,), ()), {}, entries, features, "audio-sha"
            )
        sources = {item.source: item for item in augmented[1]}
        self.assertIn("local-whisper-vocal-independent-fusion", sources)
        self.assertIn("ctc-local-whisper-vocal-independent-consensus", sources)
        composite = sources["ctc-local-whisper-vocal-independent-consensus"]
        self.assertAlmostEqual(composite.written_time.seconds, 2.0, places=6)
        self.assertIsNotNone(composite.direct_onset_evidence)
        self.assertAlmostEqual(composite.direct_onset_evidence.onset_time, 2.1, places=6)
        self.assertTrue(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(composite))
        self.assertTrue(auto_lrc.candidate_has_independent_backend_identity(composite))


class ZeroReferenceAutoIsolationTests(unittest.TestCase):
    def test_auto_timestamp_stripping_preserves_text_revision(self) -> None:
        timestamped = [
            LyricEntry(["one", "translation"], 100),
            LyricEntry(["two", "translation two"], 250),
        ]
        stripped = auto_lrc.strip_lyric_timestamps(timestamped)
        self.assertEqual([entry.lines for entry in stripped], [entry.lines for entry in timestamped])
        self.assertEqual([entry.source_time_cs for entry in stripped], [None, None])
        self.assertEqual(
            auto_lrc.lyric_identity_revision(stripped),
            auto_lrc.lyric_identity_revision(timestamped),
        )

    def test_primary_lyrics_discovery_excludes_checked_candidates_by_default(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "Music" / "album" / "song.flac"
            checked = root / "Music" / "LRC tools checked references" / "song.lrc"
            audio.parent.mkdir(parents=True)
            checked.parent.mkdir(parents=True)
            audio.write_bytes(b"audio")
            checked.write_text("[00:01.00]one\n", encoding="utf-8")
            with self.assertRaises(LrcError):
                auto_lrc.find_lyrics(audio)
            self.assertEqual(auto_lrc.find_lyrics(audio, include_checked=True), checked)

    def test_auto_anchor_discovery_requires_explicit_path(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "song.flac"
            anchor = root / "song.anchors.lrc"
            audio.write_bytes(b"audio")
            anchor.write_text("[00:01.00]one\n", encoding="utf-8")
            args = build_parser().parse_args([str(audio)])
            self.assertIsNone(auto_lrc.find_anchor_hints(audio, args))
            explicit = build_parser().parse_args([str(audio), "--anchor-hints", str(anchor)])
            self.assertEqual(auto_lrc.find_anchor_hints(audio, explicit), anchor.resolve())

    def test_inference_metadata_distinguishes_independent_and_assisted_modes(self) -> None:
        independent = auto_lrc.inference_mode_metadata("auto", "backend_competition", None)
        self.assertEqual(independent["inference_mode"], "independent-auto")
        self.assertTrue(independent["independent_auto"])
        self.assertFalse(independent["reference_used_before_inference"])
        self.assertFalse(independent["experience_artifact_used"])
        checked = auto_lrc.inference_mode_metadata("checked", "checked_lrc", None)
        self.assertEqual(checked["inference_mode"], "assisted-checked")
        self.assertFalse(checked["independent_auto"])
        self.assertTrue(checked["reference_used_before_inference"])
        self.assertTrue(checked["experience_artifact_used"])
        anchor = auto_lrc.inference_mode_metadata("auto", "backend_competition", Path("manual.anchors.lrc"))
        self.assertEqual(anchor["inference_mode"], "assisted-auto-anchor")
        self.assertFalse(anchor["independent_auto"])
        self.assertFalse(anchor["reference_used_before_inference"])
        self.assertTrue(anchor["experience_artifact_used"])

    def test_explicit_checked_mode_remains_assisted(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "song.flac"
            lyrics = root / "song.lrc"
            audio.write_bytes(b"audio")
            lyrics.write_text("[00:01.00]one\n[00:02.00]two\n", encoding="utf-8")
            args = build_parser().parse_args([
                str(audio), "--lyrics", str(lyrics), "--timing-source", "checked",
                "--output", str(root / "result.lrc"), "--report-dir", str(root / "reports"), "--overwrite",
            ])
            with split_patch.patch("probe_duration", return_value=20.0):
                auto_lrc.process_audio(audio, args)
            report_path = next((root / "reports").glob("*.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["inference_mode"], "assisted-checked")
            self.assertFalse(report["independent_auto"])
            self.assertTrue(report["reference_used_before_inference"])
            self.assertTrue(report["experience_artifact_used"])

    def test_explicit_lyrics_mode_remains_assisted(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "song.flac"
            lyrics = root / "song.lrc"
            audio.write_bytes(b"audio")
            lyrics.write_text("[00:01.00]one\n[00:02.00]two\n", encoding="utf-8")
            args = build_parser().parse_args([
                str(audio), "--lyrics", str(lyrics), "--timing-source", "lyrics",
                "--output", str(root / "result.lrc"), "--report-dir", str(root / "reports"), "--overwrite",
            ])
            with split_patch.patch("probe_duration", return_value=20.0):
                auto_lrc.process_audio(audio, args)
            report_path = next((root / "reports").glob("*.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["inference_mode"], "assisted-lyrics")
            self.assertFalse(report["independent_auto"])
            self.assertTrue(report["reference_used_before_inference"])
            self.assertTrue(report["experience_artifact_used"])

    def test_auto_filesystem_metamorphic_and_timestamped_equivalence(self) -> None:
        def run_case(root: Path, lyric_name: str, *, timestamped: bool) -> dict[str, object]:
            music = root / "Music" / "album"
            music.mkdir(parents=True)
            audio = music / "song.flac"
            audio.write_bytes(b"same-audio")
            lyric = music / lyric_name
            lyric.write_text(
                "[00:01.00]one\n[00:02.00]two\n" if timestamped else "one\ntwo\n",
                encoding="utf-8",
            )
            if root.name == "decoy":
                (music / "song.anchors.lrc").write_text("[00:01.00]wrong\n", encoding="utf-8")
                (music / "song.lrc").write_text("[00:99.00]one\n[00:99.50]two\n", encoding="utf-8")
                (music / "song.align-report.json").write_text("{\"timestamp\":99.0}\n", encoding="utf-8")
                (music / "song.previous-trust.json").write_text("{\"overall_trusted\":true}\n", encoding="utf-8")
                (music / "timing-authority").mkdir()
                (music / "timing-authority" / "candidate.json").write_text("{\"timestamp\":99.0}\n", encoding="utf-8")
            spans = lambda start: tuple(
                auto_lrc.TimingTokenSpan(start + index * 0.06, start + index * 0.06 + 0.04, 0.9, chr(97 + index))
                for index in range(3)
            )
            candidates = tuple(
                auto_lrc.make_timing_candidate(
                    entry_index=index, entry_text=text, source="ctc-current", raw_time=time,
                    spans=spans(time), confidence=0.9, identity_support="supported",
                    sequence_support="supported", current=True,
                )
                for index, (text, time) in enumerate((("one", 2.0), ("two", 5.0)))
            )
            report = {
                "backend": "ctc", "timing_entries": 2,
                "assignments": [{"entry": 1, "timestamp": 2.0, "score": 0.9}, {"entry": 2, "timestamp": 5.0, "score": 0.9}],
                "suspicious_alignments": [],
            }
            evaluated = auto_lrc.evaluate_backend_timing("ctc", tuple((candidate,) for candidate in candidates), report, {})
            selection = auto_lrc.select_evaluated_backend((evaluated,))

            def competition(*_args: object, **_kwargs: object):
                return [2.0, 5.0], report, "ctc", selection

            args = build_parser().parse_args([
                str(audio), "--lyrics", str(lyric), "--output", str(root / "result.lrc"),
                "--report-dir", str(root / "reports"), "--overwrite",
            ])
            with split_patch.patch("probe_duration", return_value=20.0), \
                mock.patch.object(auto_lrc, "default_ctc_ready", return_value=True), \
                mock.patch.object(auto_lrc, "default_whisperx_ready", return_value=False), \
                mock.patch.object(auto_lrc, "run_auto_backend_competition", side_effect=competition), \
                mock.patch.object(lyrics_io, "audio_track_labels", return_value=("", "generic")), \
                mock.patch.object(auto_lrc, "checked_lrc_timing_hint", side_effect=AssertionError("checked hint read")):
                auto_lrc.process_audio(audio, args)
            report_path = next((root / "reports").glob("*.json"))
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            return {
                "lrc": (root / "result.lrc").read_bytes(),
                "assignments": persisted["assignments"],
                "metadata": {key: persisted.get(key) for key in (
                    "inference_mode", "independent_auto", "reference_used_before_inference", "experience_artifact_used"
                )},
            }

        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            plain = run_case(root / "plain", "untimed.txt", timestamped=False)
            timestamped = run_case(root / "timestamped", "source.lrc", timestamped=True)
            decoy = run_case(root / "decoy", "untimed.txt", timestamped=False)
        for result in (timestamped, decoy):
            self.assertEqual(result["lrc"], plain["lrc"])
            self.assertEqual(result["assignments"], plain["assignments"])
            self.assertEqual(result["metadata"], plain["metadata"])
        self.assertEqual(plain["metadata"]["inference_mode"], "independent-auto")
        self.assertTrue(plain["metadata"]["independent_auto"])
        self.assertFalse(plain["metadata"]["reference_used_before_inference"])


class Cycle33OccurrenceEvidenceContractTests(unittest.TestCase):
    def occurrence_evidence(
        self,
        entries: list[LyricEntry],
        *,
        producer: str = "raw-asr-vocal-fusion",
    ) -> auto_lrc.CandidateOccurrenceEvidence:
        assignments = [
            {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
            for index in range(len(entries))
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 0)
        evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            entries,
            0,
            producer=producer,
            occurrence_binding=binding,
            upstream_evidence_revision="cycle33-upstream",
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        return evidence

    def direct_candidate(
        self,
        evidence: auto_lrc.CandidateOccurrenceEvidence,
        *,
        producer: str = "raw-asr-vocal-fusion",
    ) -> auto_lrc.TimingCandidate:
        spans = tuple(
            auto_lrc.TimingTokenSpan(1.0 + index * .08, 1.04 + index * .08, .95, str(index))
            for index in range(4)
        )
        return auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="cycle33 target",
            source="raw-vocal-independent-fusion",
            raw_time=1.0,
            spans=spans,
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported",
            direct_onset_time=1.0,
            direct_onset_source_artifact={"producer": producer},
            direct_onset_evidence_producer=producer,
            direct_onset_evidence_kind="cross-backend-onset",
            direct_onset_evidence_independent=True,
            occurrence_evidence=evidence,
            independent_content_identity=True,
        )

    def test_direct_valid_occurrence_invalid_fails_minimum_proof(self) -> None:
        entries = [LyricEntry(["cycle33 target"])]
        valid = self.occurrence_evidence(entries)
        stale = replace(valid, upstream_evidence_revision="stale-upstream")
        candidate = self.direct_candidate(stale)
        evaluation = auto_lrc.evaluate_timing_candidate(
            candidate, occurrence_entries=entries
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("occurrence-evidence-invalid", evaluation.rejection_reasons)
        self.assertFalse(evaluation.evidence.minimum_proof_satisfied)

    def test_occurrence_binding_unknown_path_field_fails_closed(self) -> None:
        entries = [LyricEntry(["cycle33 target"])]
        valid = self.occurrence_evidence(entries)
        binding = json.loads(valid.binding_payload_json)
        binding["path"] = "C:/private/audio.wav"
        binding_payload = {key: value for key, value in binding.items() if key != "binding_revision"}
        binding["binding_revision"] = auto_lrc._timing_content_digest(binding_payload)
        self.assertFalse(auto_lrc._runtime_occurrence_binding_is_valid(entries, 0, binding))
        self.assertIsNone(
            auto_lrc.candidate_occurrence_evidence_from_binding(
                entries,
                0,
                producer="raw-asr-vocal-fusion",
                occurrence_binding=binding,
                upstream_evidence_revision="cycle33-upstream",
            )
        )

    def test_occurrence_producer_mismatch_fails_closed(self) -> None:
        entries = [LyricEntry(["cycle33 target"])]
        evidence = self.occurrence_evidence(entries)
        candidate = self.direct_candidate(evidence, producer="whisperx-vocal-fusion")
        self.assertFalse(
            auto_lrc.candidate_has_bound_direct_onset_evidence(candidate, entries)
        )
        self.assertFalse(
            auto_lrc.candidate_has_independent_backend_identity(candidate, entries)
        )


class V12HubertPairTailOccurrenceRegressionTests(unittest.TestCase):
    def test_current_evidence_scope_preserves_strict_gate_for_valid_predecessor(self) -> None:
        self.assertEqual(
            auto_lrc._known_lyric_hubert_pair_tail_current_evidence_scope("valid"),
            "strict-final-associated-onset",
        )
        self.assertEqual(
            auto_lrc._known_lyric_hubert_pair_tail_current_evidence_scope(
                "first-token-mismatch"
            ),
            "hypothesis-neutral-boundary",
        )
        self.assertEqual(
            auto_lrc._known_lyric_hubert_pair_tail_current_evidence_scope(
                "envelope-mismatch"
            ),
            "reject",
        )

    def test_unique_pair_current_opening_can_bind_tail_when_old_predecessor_onset_disagrees(self) -> None:
        entries = [
            LyricEntry(["あさ"]),
            LyricEntry(["ひる"]),
            LyricEntry(["よる"]),
        ]
        legacy = auto_lrc._known_lyric_hubert_predecessor_occurrence_status(
            4.08,
            5.46,
            620,
            18,
        )
        self.assertEqual(legacy, "first-token-mismatch")
        status = auto_lrc._known_lyric_hubert_pair_tail_occurrence_status(
            entries,
            0,
            1,
            4.08,
            5.46,
            10.08,
            10.10,
            18,
        )
        self.assertEqual(status, "valid")

    def test_repeated_adjacent_reading_pair_stays_unbound(self) -> None:
        entries = [
            LyricEntry(["あさ"]),
            LyricEntry(["ひる"]),
            LyricEntry(["あさ"]),
            LyricEntry(["ひる"]),
        ]
        status = auto_lrc._known_lyric_hubert_pair_tail_occurrence_status(
            entries,
            0,
            1,
            4.0,
            5.0,
            6.0,
            6.0,
            18,
        )
        self.assertEqual(status, "pair-reading-not-unique")

    def test_pair_boundary_must_match_neutral_current_opening(self) -> None:
        entries = [LyricEntry(["あさ"]), LyricEntry(["ひる"])]
        status = auto_lrc._known_lyric_hubert_pair_tail_occurrence_status(
            entries,
            0,
            1,
            4.0,
            5.0,
            6.0,
            6.40,
            18,
        )
        self.assertEqual(status, "current-boundary-mismatch")


class V12HubertPredecessorPlannerRegressionTests(unittest.TestCase):
    def state(
        self,
        entries: list[LyricEntry],
        times: list[float],
    ) -> auto_lrc.FinalTimingState:
        decisions = []
        previous = None
        for index, (entry, timestamp) in enumerate(zip(entries, times, strict=True)):
            candidate = auto_lrc.make_timing_candidate(
                entry_index=index,
                entry_text=auto_lrc.entry_sung_text(entry),
                source="ctc-current",
                raw_time=timestamp,
                spans=(
                    auto_lrc.TimingTokenSpan(
                        timestamp,
                        timestamp + 0.08,
                        0.9,
                        "a",
                    ),
                    auto_lrc.TimingTokenSpan(
                        timestamp + 0.10,
                        timestamp + 0.18,
                        0.9,
                        "b",
                    ),
                ),
                confidence=0.9,
                identity_support="supported",
                sequence_support="supported",
                current=True,
                source_artifact={"planner-test": index},
            )
            decision = auto_lrc.select_timing_decision(
                entry_index=index,
                candidates=(candidate,),
                previous_candidate=previous,
                selection_revision=1,
            )
            decisions.append(decision)
            previous = decision.candidate
        decisions_tuple = tuple(decisions)
        return auto_lrc.FinalTimingState(
            decisions=decisions_tuple,
            written_times=tuple(item.written_time for item in decisions_tuple),
            audits=auto_lrc.audit_final_timing_decisions(decisions_tuple),
        )

    def boundary(
        self,
        state: auto_lrc.FinalTimingState,
        entry_index: int,
        onset_time: float,
    ) -> auto_lrc.IndependentCurrentOnsetEvidence:
        owner = state.decisions[entry_index].candidate
        evidence = auto_lrc._make_known_lyric_hubert_current_onset(
            entry_index=entry_index,
            onset_time=onset_time,
            opening_proof_end=onset_time + 0.20,
            opening_region_revision=f"region-{entry_index}",
            identity_owner_candidate_id=owner.candidate_id,
            identity_owner_generation_revision=owner.generation_revision,
            model_identity="planner-test-model",
            helper_revision="planner-test-helper",
            capability_revision="planner-test-capability",
            audio_revision="planner-test-audio",
            transcript_revision=f"transcript-{entry_index}",
            row_revision=f"row-{entry_index}",
            request_revision=f"request-{entry_index}",
            hypothesis_set_revision=f"hypothesis-{entry_index}",
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertTrue(auto_lrc._known_lyric_hubert_current_onset_is_valid(evidence))
        return evidence

    def test_generalized_planner_uses_neutral_boundary_and_global_predecessor_uniqueness(self) -> None:
        entries = [
            LyricEntry(["あ"]),
            LyricEntry(["い"]),
            LyricEntry(["う"]),
            LyricEntry(["え"]),
            LyricEntry(["お"]),
            LyricEntry(["い"]),
            LyricEntry(["か"]),
        ]
        state = self.state(entries, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        unique_boundary = self.boundary(state, 3, 4.20)
        repeated_boundary = self.boundary(state, 6, 7.20)

        requests = auto_lrc._postcentral_plan_known_lyric_hubert_predecessor_only_requests(
            {
                3: unique_boundary,
                6: repeated_boundary,
            },
            state,
            entries,
            20.0,
        )

        self.assertEqual(
            [int(item["conflict_entry"]) for item in requests],
            [4],
        )
        self.assertEqual(int(requests[0]["previous_entry_index"]), 2)
        self.assertEqual(int(requests[0]["current_entry_index"]), 3)
        self.assertEqual(
            requests[0]["current_boundary_observation_revision"],
            unique_boundary.observation_revision,
        )
        self.assertAlmostEqual(float(requests[0]["window_end"]), 4.20, places=6)


class V12Node19ExistingPeerRecoveryEligibilityTests(unittest.TestCase):
    def candidate(
        self,
        *,
        entry_index: int,
        source: str,
        raw_time: float,
        direct: bool,
        tail_end: float | None = None,
    ) -> auto_lrc.TimingCandidate:
        if tail_end is None:
            starts = [raw_time, raw_time + 0.30]
            starts.extend(raw_time + 0.70 + index * 0.42 for index in range(8))
        else:
            starts = [
                raw_time + (tail_end - raw_time - 0.05) * index / 5.0
                for index in range(6)
            ]
        spans = tuple(
            auto_lrc.TimingTokenSpan(start, start + 0.04, 0.96, chr(97 + index))
            for index, start in enumerate(starts)
        )
        kwargs: dict[str, object] = {}
        if direct:
            kwargs.update({
                "acoustic_support": "supported",
                "direct_onset_support": "supported",
                "acoustic_strength": 0.95,
                "acoustic_onset_time": raw_time,
                "acoustic_source_artifact": {"node19": "independent-vocal-onset"},
                "acoustic_evidence_producer": "consonant-onset-detector",
                "acoustic_evidence_kind": "vocal-onset",
                "acoustic_evidence_independent": True,
                "direct_onset_time": raw_time,
                "direct_onset_source_artifact": {"node19": "independent-vocal-onset"},
                "direct_onset_evidence_producer": "consonant-onset-detector",
                "direct_onset_evidence_kind": "vocal-onset",
                "direct_onset_evidence_independent": True,
            })
        return auto_lrc.make_timing_candidate(
            entry_index=entry_index,
            entry_text=f"node19-{entry_index}",
            source=source,
            raw_time=raw_time,
            spans=spans,
            confidence=0.95,
            identity_support="supported",
            sequence_support="supported",
            source_artifact={"node19": source},
            **kwargs,
        )

    def fixture(
        self,
        *,
        evidence_state: str = "independent",
        cluster_families: list[str] | None = None,
        direct: bool = True,
        trusted_selected: bool = False,
        predecessor_tail_end: float = 29.80,
        multiple_clusters: bool = False,
    ) -> tuple[
        dict[str, object],
        auto_lrc.TimingDecision,
        auto_lrc.TimingDecision,
        dict[tuple[str, str], auto_lrc.TimingCandidate],
    ]:
        previous = self.candidate(
            entry_index=0,
            source="ctc-hubert-known-lyric-boundary-consensus",
            raw_time=29.0,
            direct=True,
            tail_end=predecessor_tail_end,
        )
        previous_decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(previous,),
            previous_candidate=None,
            selection_revision=1,
        )
        weak = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="node19-1",
            source="whispercpp-current",
            raw_time=30.0,
            spans=(),
            confidence=0.6,
            identity_support="unavailable",
            sequence_support="supported",
            source_artifact={"node19": "selected-weak"},
        )
        selected = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(weak,),
            previous_candidate=previous,
            selection_revision=1,
        )
        if trusted_selected:
            selected = replace(
                selected,
                status="selected_valid",
                selected_candidate_id=selected.candidate.candidate_id,
                provisional_candidate_id=None,
            )
        alternative = self.candidate(
            entry_index=1,
            source="ctc-opening-vocal-fusion",
            raw_time=30.88,
            direct=direct,
        )
        cluster = {
            "cluster_revision": "node19-cluster",
            "anchor_time": 30.88,
            "member_candidate_ids": [alternative.candidate_id],
            "member_generation_revisions": [alternative.generation_revision],
            "support_families": cluster_families or ["ctc"],
            "representative_candidate_id": alternative.candidate_id,
            "relation": "distant",
            "evidence_state": evidence_state,
            "review_required": True,
        }
        clusters: list[dict[str, object]] = [cluster]
        if multiple_clusters:
            clusters.append({**cluster, "cluster_revision": "node19-cluster-2", "anchor_time": 31.2})
        row = {
            "entry": 2,
            "postcentral_replay_status": "REBOUND",
            "postcentral_replay_reason": "postcentral-existing-conflict-only-canonical-replay",
            "final_selected_candidate_id": selected.candidate.candidate_id,
            "final_selected_centiseconds": selected.written_time.centiseconds,
            "clusters": clusters,
        }
        lookup = {(alternative.candidate_id, alternative.generation_revision): alternative}
        return row, selected, previous_decision, lookup

    def resolve(self, **kwargs: object) -> tuple[auto_lrc.TimingCandidate | None, str]:
        row, selected, previous, lookup = self.fixture(**kwargs)
        return auto_lrc._existing_peer_recovery_representative(
            row=row,
            final_decision=selected,
            previous_decision=previous,
            bundle_lookup=lookup,
            occurrence_entries=None,
        )

    def test_independent_complete_existing_candidate_is_eligible(self) -> None:
        row, selected, previous, lookup = self.fixture()
        probe = next(iter(lookup.values()))
        evaluation = auto_lrc.evaluate_timing_candidate(
            probe, previous_candidate=previous.candidate
        )
        self.assertTrue(
            evaluation.valid,
            (evaluation.rejection_reasons, evaluation.evidence),
        )
        candidate, reason = auto_lrc._existing_peer_recovery_representative(
            row=row,
            final_decision=selected,
            previous_decision=previous,
            bundle_lookup=lookup,
            occurrence_entries=None,
        )
        self.assertIsNotNone(candidate, reason)
        self.assertEqual(reason, "eligible-independent-existing-peer-candidate")
        assert candidate is not None
        self.assertEqual(candidate.written_time.centiseconds, 3088)

    def test_same_lineage_cluster_is_rejected(self) -> None:
        candidate, reason = self.resolve(cluster_families=["whisper-family"])
        self.assertIsNone(candidate)
        self.assertEqual(reason, "cluster-not-independent-from-selected-lineage")

    def test_mixed_cluster_is_rejected(self) -> None:
        candidate, reason = self.resolve(evidence_state="mixed")
        self.assertIsNone(candidate)
        self.assertEqual(reason, "cluster-evidence-not-independent")

    def test_trusted_selected_boundary_is_not_recovered(self) -> None:
        candidate, reason = self.resolve(trusted_selected=True)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "selected-boundary-not-provisional")

    def test_candidate_without_bound_direct_onset_is_rejected(self) -> None:
        candidate, reason = self.resolve(direct=False)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "representative-lacks-independent-direct-onset")

    def test_multiple_distant_clusters_are_rejected(self) -> None:
        candidate, reason = self.resolve(multiple_clusters=True)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "material-alternative-not-unique")

    def test_representative_still_overlapping_predecessor_is_rejected(self) -> None:
        candidate, reason = self.resolve(predecessor_tail_end=31.10)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "representative-not-central-valid-in-selected-context")

    def test_peer_recovery_runs_after_hubert_predecessor_recovery(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        process = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_audio"
        )
        process_calls = {
            getattr(node.func, "id", None): node.lineno
            for node in ast.walk(process)
            if isinstance(node, ast.Call)
        }
        self.assertLess(
            process_calls["_precommit_hubert_existing_candidate_recovery"],
            process_calls["_precommit_independent_peer_existing_candidate_recovery"],
        )


class V12Node19DuplicateOwnerRecomputeTests(unittest.TestCase):
    def fixture(
        self,
        *,
        corrected_second_a: float,
    ) -> tuple[list[LyricEntry], dict[str, object], auto_lrc.FinalTimingState]:
        entries = [
            LyricEntry(["A"]), LyricEntry(["B"]), LyricEntry(["C"]), LyricEntry(["D"]),
            LyricEntry(["A"]), LyricEntry(["B"]), LyricEntry(["C"]), LyricEntry(["D"]),
        ]
        original_times = [1.0, 3.0, 5.0, 7.0, 84.0, 85.0, 87.0, 89.0]
        final_times = [1.0, 3.0, 5.0, 7.0, corrected_second_a, 85.0, 87.0, 89.0]
        assignments: list[dict[str, object]] = [
            {"entry": index + 1, "timestamp": value, "score": 0.9, "flags": []}
            for index, value in enumerate(original_times)
        ]
        assignments[0]["ctc_forward_supported_prefix_recovery"] = True
        assignments[0]["flags"] = ["duplicate_occurrence_offset_mismatch"]
        assignments[0]["duplicate_occurrence_offset_mismatch"] = {
            "entries": [1, 5],
            "observed_delta_seconds": 83.0,
            "median_delta_seconds": 82.0,
            "error_seconds": 1.0,
        }
        report: dict[str, object] = {
            "assignments": assignments,
            "suspicious_alignments": [{
                "entry": 1,
                "text": "A",
                "flags": ["duplicate_occurrence_offset_mismatch"],
                "severity": "high",
                "review_required": True,
                "duplicate_occurrence_offset_mismatch": dict(
                    assignments[0]["duplicate_occurrence_offset_mismatch"]  # type: ignore[arg-type]
                ),
            }],
            "duplicate_occurrence_offset_outliers": [{
                "entry": 1,
                "entries": [1, 5],
                "observed_delta_seconds": 83.0,
                "median_delta_seconds": 82.0,
                "error_seconds": 1.0,
            }],
            "duplicate_occurrence_offset_outlier_count": 1,
        }
        decisions = []
        previous = None
        for index, value in enumerate(final_times):
            candidate = auto_lrc.make_timing_candidate(
                entry_index=index,
                entry_text=auto_lrc.entry_sung_text(entries[index]),
                source="ctc-current",
                raw_time=value,
                spans=tuple(
                    auto_lrc.TimingTokenSpan(
                        value + offset,
                        value + offset + 0.04,
                        0.95,
                        chr(97 + span_index),
                    )
                    for span_index, offset in enumerate((0.0, 0.08, 0.16, 0.24, 0.32, 0.40))
                ),
                confidence=0.95,
                identity_support="supported",
                sequence_support="supported",
                current=True,
                source_artifact={"node19-duplicate": index},
            )
            decision = auto_lrc.select_timing_decision(
                entry_index=index,
                candidates=(candidate,),
                previous_candidate=previous,
                selection_revision=1,
            )
            decisions.append(decision)
            previous = decision.candidate
        decisions_tuple = tuple(decisions)
        state = auto_lrc.FinalTimingState(
            decisions=decisions_tuple,
            written_times=tuple(item.written_time for item in decisions_tuple),
            audits=auto_lrc.audit_final_timing_decisions(decisions_tuple),
        )
        return entries, report, state

    def test_recomputed_final_delta_removes_stale_duplicate_mismatch(self) -> None:
        entries, report, state = self.fixture(corrected_second_a=83.12)
        payload = auto_lrc._recompute_duplicate_occurrence_offset_diagnostics(
            entries, report, state
        )
        self.assertEqual(payload["outlier_count"], 0)
        self.assertEqual(report["duplicate_occurrence_offset_outliers"], [])
        first = report["assignments"][0]  # type: ignore[index]
        self.assertNotIn("duplicate_occurrence_offset_mismatch", first["flags"])
        self.assertNotIn("duplicate_occurrence_offset_mismatch", first)
        risk = report["suspicious_alignments"][0]  # type: ignore[index]
        self.assertNotIn("duplicate_occurrence_offset_mismatch", risk["flags"])
        self.assertNotIn("duplicate_occurrence_offset_mismatch", risk)

    def test_recomputed_final_delta_retains_genuine_duplicate_mismatch(self) -> None:
        entries, report, state = self.fixture(corrected_second_a=84.0)
        payload = auto_lrc._recompute_duplicate_occurrence_offset_diagnostics(
            entries, report, state
        )
        self.assertEqual(payload["outlier_count"], 1)
        outlier = report["duplicate_occurrence_offset_outliers"][0]  # type: ignore[index]
        self.assertEqual(outlier["entry"], 1)
        self.assertAlmostEqual(float(outlier["error_seconds"]), 1.0, places=3)
        first = report["assignments"][0]  # type: ignore[index]
        self.assertIn("duplicate_occurrence_offset_mismatch", first["flags"])

    def test_duplicate_review_refresh_is_postcommit_only(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        private_replay = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_peer_recovery_project_selection"
        )
        replay_calls = {
            getattr(node.func, "id", None): node.lineno
            for node in ast.walk(private_replay)
            if isinstance(node, ast.Call)
        }
        self.assertLess(
            replay_calls["_apply_evaluated_backend_projection"],
            replay_calls["_postcommit_recompute_duplicate_review"],
        )

        process = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_audio"
        )
        process_calls = {
            getattr(node.func, "id", None): node.lineno
            for node in ast.walk(process)
            if isinstance(node, ast.Call)
        }
        self.assertLess(
            process_calls["commit_evaluated_backend"],
            process_calls["_postcommit_recompute_duplicate_review"],
        )


class DemucsCacheIsolationTests(unittest.TestCase):
    def setUp(self):
        self.td=TemporaryDirectory(prefix="demucs_io_test_")
        self.addCleanup(self.td.cleanup)
        self.root=Path(self.td.name)
        self.audio=self.root/"03.ぼうやの夢よ .flac"
        self.audio.write_bytes(b"source-bytes-never-change")
        self.python=self.root/"python.exe"
        self.python.write_bytes(b"")
        self.cache_root=self.root/"cache"
        self.key="isolated-test"
        self.args=type("Args",(),{"vocal_ctc":True})()
        for patch in (
            split_patch.patch("default_ctc_python",return_value=self.python),
            split_patch.patch("vocal_cache_identity",return_value={"vocal_cache_key":self.key}),
            mock.patch.object(audio_features, "DEFAULT_VOCAL_CACHE_DIR",self.cache_root),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.cached=self.cache_root/self.key/"vocals.mp3"

    def run_fake_demucs(self,*,generate=b"NEW_VALID_STEM"*1000,stderr=None):
        def runner(command,**kwargs):
            if stderr is not None:
                return subprocess.CompletedProcess(command,1,"",stderr)
            dest=Path(command[command.index("-o")+1])/"htdemucs"/"current"/"vocals.mp3"
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(generate)
            return subprocess.CompletedProcess(command,0,"","")
        return mock.patch.object(auto_lrc.subprocess,"run",side_effect=runner)

    def test_existing_stale_stem_does_not_override_new_output(self):
        stale=self.cache_root/self.key/"htdemucs"/"00_failed"/"vocals.mp3"
        stale.parent.mkdir(parents=True,exist_ok=True)
        stale.write_bytes(b"STALE"*2000)
        with self.run_fake_demucs(),mock.patch.object(audio_features, "_demucs_vocal_stem_valid",return_value=True):
            path,kind,status=auto_lrc.prepare_vocal_ctc_audio(self.audio,self.args)
        self.assertEqual((path,kind,status),(self.cached,"vocal-stem","generated"))
        self.assertEqual(path.read_bytes(),b"NEW_VALID_STEM"*1000)
        self.assertTrue(stale.exists())

    def test_one_byte_corrupt_cache_is_rebuilt_not_hit(self):
        self.cached.parent.mkdir(parents=True,exist_ok=True)
        self.cached.write_bytes(b"X")
        def validity(path,source):
            self.assertEqual(source,self.audio)
            return path.stat().st_size>4096
        with self.run_fake_demucs(),mock.patch.object(audio_features, "_demucs_vocal_stem_valid",side_effect=validity):
            path,kind,status=auto_lrc.prepare_vocal_ctc_audio(self.audio,self.args)
        self.assertEqual((kind,status),("vocal-stem","generated"))
        self.assertEqual(path.read_bytes(),b"NEW_VALID_STEM"*1000)

    def test_demucs_error_keeps_full_traceback_tail_and_log(self):
        error="Log progress "+"x"*800+"\nTraceback (most recent call last):\nFileNotFoundError: sentinel-root-cause"
        with self.run_fake_demucs(stderr=error),mock.patch.object(audio_features, "_demucs_vocal_stem_valid",return_value=True):
            path,kind,status=auto_lrc.prepare_vocal_ctc_audio(self.audio,self.args)
        self.assertEqual((path,kind),(self.audio,"mix"))
        self.assertIn("FileNotFoundError: sentinel-root-cause",status)
        self.assertEqual((self.cached.parent/"demucs_last_error.log").read_text(encoding="utf-8"),error)

    def test_valid_cache_is_reused_without_demucs(self):
        self.cached.parent.mkdir(parents=True,exist_ok=True)
        self.cached.write_bytes(b"VALID"*2400)
        with mock.patch.object(audio_features, "_demucs_vocal_stem_valid",return_value=True), \
             mock.patch.object(auto_lrc.subprocess,"run",side_effect=AssertionError("unexpected Demucs")):
            path,kind,status=auto_lrc.prepare_vocal_ctc_audio(self.audio,self.args)
        self.assertEqual((path,kind,status),(self.cached,"vocal-stem","cache-hit"))

    def test_generated_invalid_stem_is_rejected(self):
        with self.run_fake_demucs(generate=b"X"),mock.patch.object(audio_features, "_demucs_vocal_stem_valid",return_value=False
        ):
            path,kind,status=auto_lrc.prepare_vocal_ctc_audio(self.audio,self.args)
        self.assertEqual((path,kind,status),(self.audio,"mix","demucs-invalid-generated-vocal-stem"))
        self.assertFalse(self.cached.exists())

    def test_ffprobe_validator_checks_codec_duration_and_channels(self):
        vocal=self.root/"vocals.mp3"
        vocal.write_bytes(b"M"*6000)
        source_duration=200.0
        sample={"streams":[{"codec_name":"mp3","sample_rate":"44100","channels":2}],
                "format":{"duration":"199.8"}}
        def runner(command,**kwargs):
            self.assertEqual(command[0],"ffprobe")
            return subprocess.CompletedProcess(command,0,json.dumps(sample),"")
        with mock.patch.object(auto_lrc.subprocess,"run",side_effect=runner), \
             split_patch.patch("probe_duration",return_value=source_duration):
            self.assertTrue(auto_lrc._demucs_vocal_stem_valid(vocal,self.audio))
            sample["format"]["duration"]="10"
            self.assertFalse(auto_lrc._demucs_vocal_stem_valid(vocal,self.audio))
            sample["format"]["duration"]="199.8"
            sample["streams"][0]["codec_name"]="invalid"
            self.assertFalse(auto_lrc._demucs_vocal_stem_valid(vocal,self.audio))
            sample["streams"][0]["codec_name"]="mp3"
            sample["streams"][0]["channels"]=0
            self.assertFalse(auto_lrc._demucs_vocal_stem_valid(vocal,self.audio))


class BackendParticipationTests(unittest.TestCase):
    def test_writer_projects_backend_participation_and_hides_runtime_state(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            report = {
                "backend": "ctc",
                "assignments": [],
                auto_lrc._RUNTIME_BACKEND_PARTICIPATION_KEY: {
                    "attempted": ["ctc"],
                    "available": ["ctc"],
                    "unavailable": [{"name": "whisperx", "reason": "missing"}],
                    "evidence_producers_failed": [],
                    "demucs_device": "cuda",
                    "vocal_source": "vocal-stem",
                },
            }
            path = root / "report.json"
            auto_lrc.write_alignment_report(path, report)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["backend_participation"]["available"], ["ctc"])
        self.assertNotIn(auto_lrc._RUNTIME_BACKEND_PARTICIPATION_KEY, payload)

    def test_evidence_producer_failure_is_recorded(self) -> None:
        report = {"assignments": [{}]}
        entries = [LyricEntry(["test"])]
        with split_patch.patch("analyze_audio", side_effect=auto_lrc.LrcError("sentinel")):
            result = auto_lrc.apply_vocal_activity_reentry_hypotheses(
                Path("missing.flac"), entries, [0.0], report, 1.0
            )
        self.assertEqual(result, [])
        failures = report[auto_lrc._RUNTIME_BACKEND_PARTICIPATION_KEY]["evidence_producers_failed"]
        self.assertEqual(failures[0]["producer"], "apply_vocal_activity_reentry_hypotheses")


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
