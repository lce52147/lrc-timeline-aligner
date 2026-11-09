#!/usr/bin/env python3
"""V14.6-V15 local whisper evidence, arbitration and convergence tests."""

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


class V146LocalWhisperEvidenceRegressionTests(_TempModelCacheMixin, unittest.TestCase):
    def span(self, start: float, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, 0.5, token)

    def parent(self, entry: int, time: float, *, source: str = "ctc-local-window") -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"line {entry}",
            source=source,
            raw_time=time,
            spans=(self.span(time, "a"), self.span(time + 0.08, "b"), self.span(time + 0.16, "c")),
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            current=False,
            source_artifact={"source": source, "time": time},
        )

    def local_evidence(self, report, entries, entry, onset, score=0.95):
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[entry]))
        binding_payload = {
            "mode": "global-unique",
            "entry_index": entry,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "segment": 1,
            "timestamp": round(onset, 6),
            "score": round(score, 6),
        }
        binding = {
            **binding_payload,
            "binding_revision": auto_lrc._timing_content_digest(binding_payload),
        }
        payload = {
            "entry_index": entry,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "onset_time": round(onset, 6),
            "score": round(score, 6),
            "occurrence_binding": binding,
            "request_revision": "request",
            "capability_revision": "capability",
            "audio_revision": "audio",
            "window_start": round(onset - 2.0, 6),
            "window_end": round(onset + 2.0, 6),
            "raw_transcript_revision": "raw-transcript",
            "aligned_transcript_revision": "aligned-transcript",
            "producer": "local-whisper-aligner",
        }
        report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY] = {
            entry: {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}
        }

    def local_capability_request(
        self,
        audio: Path,
        entries: list[LyricEntry],
    ) -> tuple[auto_lrc.LocalWhisperCapability, auto_lrc.LocalWhisperRequest]:
        audio_sha = auto_lrc._path_sha256(audio)
        cap_payload = {
            "kind": "whispercpp-whisperx-local",
            "backend_lineage": "whisper-family",
            "audio_path": str(audio.resolve()),
            "audio_sha256": audio_sha,
            "whisper_cli_path": "cli",
            "whisper_cli_sha256": "cli-sha",
            "whisper_model_path": "model",
            "whisper_model_sha256": "model-sha",
            "whisperx_python_path": "python",
            "whisperx_helper_path": "helper",
            "whisperx_helper_sha256": "helper-sha",
            "language": "ja",
            "device": "cuda",
            "protocol_revision": auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
        }
        capability = auto_lrc.LocalWhisperCapability(
            **cap_payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._local_whisper_capability_semantic_payload(cap_payload)
            ),
        )
        request_payload = {
            "entry_index": 0,
            "capability_revision": capability.capability_revision,
            "audio_revision": audio_sha,
            "context_entry_indexes": (0,),
            "context_candidate_ids": ("candidate",),
            "transcript_digests": tuple(
                auto_lrc._timing_content_digest(auto_lrc.entry_sung_text(entry))
                for entry in entries
            ),
            "transcript_digest": auto_lrc._timing_content_digest(
                tuple(auto_lrc.entry_sung_text(entry) for entry in entries)
            ),
            "window_start": 10.0,
            "window_end": 20.0,
            "window_definition_revision": "window",
        }
        request = auto_lrc.LocalWhisperRequest(
            **request_payload,
            request_revision=auto_lrc._timing_content_digest({
                **request_payload,
                "context_entry_indexes": [0],
                "context_candidate_ids": ["candidate"],
                "transcript_digests": list(request_payload["transcript_digests"]),
            }),
        )
        return capability, request

    def test_segment_peer_whisper_outlier_is_diagnostic_only(self) -> None:
        from segment_peer_outlier import exclude_uncorroborated_whisperx_fusion

        def candidate(source: str, time_value: float) -> object:
            return types.SimpleNamespace(
                source=source,
                acoustic_evidence=object(),
                direct_onset_evidence=object(),
                written_time=types.SimpleNamespace(seconds=time_value),
            )

        ctc = candidate("ctc-vocal-onset-confirmed", 10.0)
        raw = candidate("raw-vocal-independent-fusion", 10.1)
        whisper = candidate("whisperx-vocal-independent-fusion", 12.0)
        kept, flagged = exclude_uncorroborated_whisperx_fusion((ctc, raw, whisper))
        self.assertEqual(kept, (ctc, raw, whisper))
        self.assertEqual(flagged, (whisper,))

    def test_transient_local_whisper_error_is_not_cached(self) -> None:
        entries = [LyricEntry(["unique retry target"])]
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "audio.wav"
            audio.write_bytes(b"audio")
            capability, request = self.local_capability_request(audio, entries)
            raw = [auto_lrc.AsrSegment(0.0, 3.0, "raw")]
            aligned = [
                auto_lrc.AsrSegment(
                    1.0,
                    2.0,
                    "unique retry target",
                    chars=[
                        (char, 1.0 + index * 0.03)
                        for index, char in enumerate("uniqueretrytarget")
                    ],
                )
            ]
            local_report = {
                "assignments": [{
                    "entry": 1,
                    "segment": 1,
                    "score": 0.95,
                    "borrowed": False,
                    "timestamp": 1.0,
                }]
            }
            cache: dict[str, tuple[str, object]] = {}
            runtime_stats: dict[str, int] = {}
            args = auto_lrc.argparse.Namespace(
                whisper_cli=None,
                whisper_model=None,
                whisper_language="ja",
                whisper_suppress_nst=False,
                whisperx_python=None,
                whisperx_device="cpu",
            )
            with (
                mock.patch.object(local_whisper_retry, "decode_temp_wav_window",
                    side_effect=[LrcError("transient"), audio],
                ),
                split_patch.patch("_run_whispercpp_wav", return_value=raw),
                split_patch.patch("run_whisperx_alignment", return_value={}),
                split_patch.patch("whisperx_segments", return_value=aligned),
                split_patch.patch("match_whisper_segments",
                    return_value=([1.0], local_report),
                ),
            ):
                first = auto_lrc.execute_local_whisper_request(
                    request,
                    capability,
                    entries,
                    args,
                    cache=cache,
                    runtime_stats=runtime_stats,
                )
                second = auto_lrc.execute_local_whisper_request(
                    request,
                    capability,
                    entries,
                    args,
                    cache=cache,
                    runtime_stats=runtime_stats,
                )
        self.assertEqual(first.status, "error")
        self.assertEqual(second.status, "evidence")
        self.assertEqual(runtime_stats["cache_misses"], 2)
        self.assertEqual(runtime_stats.get("cache_hits", 0), 0)

    def test_local_whisper_consensus_preserves_parent_timestamp(self) -> None:
        entries = [LyricEntry(["unique lyric local alpha"])]
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_evidence(report, entries, 0, 10.08)
        parent = self.parent(0, 10.00)
        augmented = auto_lrc._augment_candidates_with_local_whisper_consensus(
            ((parent,),), report, entries
        )
        composites = [c for c in augmented[0] if c.source == "ctc-local-whisper-independent-consensus"]
        self.assertEqual(len(composites), 1)
        composite = composites[0]
        self.assertEqual(composite.written_time.centiseconds, 1000)
        self.assertTrue(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(composite))
        self.assertTrue(auto_lrc.candidate_has_independent_backend_identity(composite))
        assert composite.direct_onset_evidence is not None
        self.assertAlmostEqual(composite.direct_onset_evidence.onset_time, 10.08, places=6)

    def test_distant_local_whisper_observation_cannot_fuse(self) -> None:
        entries = [LyricEntry(["unique lyric local beta"])]
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_evidence(report, entries, 0, 10.40)
        parent = self.parent(0, 10.00)
        augmented = auto_lrc._augment_candidates_with_local_whisper_consensus(
            ((parent,),), report, entries
        )
        self.assertFalse(any("local-whisper" in c.source for c in augmented[0]))

    def test_local_whisper_family_is_one_lineage_not_raw_plus_whisperx(self) -> None:
        entries = [LyricEntry(["unique lyric local gamma"])]
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_evidence(report, entries, 0, 10.05)
        parent = self.parent(0, 10.00)
        composite = next(
            c for c in auto_lrc._augment_candidates_with_local_whisper_consensus(
                ((parent,),), report, entries
            )[0]
            if "local-whisper" in c.source
        )
        self.assertEqual(auto_lrc._candidate_support_families(composite), frozenset({"ctc", "whisper-family"}))
        raw = auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="unique lyric local gamma",
            source="raw-vocal-independent-fusion", raw_time=12.0, confidence=0.95,
            identity_support="supported", sequence_support="supported",
            acoustic_support="supported", direct_onset_support="supported",
            acoustic_strength=1.0, acoustic_onset_time=12.0,
            acoustic_source_artifact={"raw": True},
            acoustic_evidence_producer="raw-asr-vocal-fusion",
            acoustic_evidence_independent=True, direct_onset_time=12.0,
            direct_onset_source_artifact={"raw": True},
            direct_onset_evidence_producer="raw-asr-vocal-fusion",
            direct_onset_evidence_independent=True, independent_content_identity=True,
            source_artifact={"raw": True},
        )
        self.assertEqual(auto_lrc._candidate_support_families(raw), frozenset({"whisper-family"}))

    def test_local_whisper_runtime_identity_is_orthogonal_to_timing(self) -> None:
        entries = [LyricEntry(["unique lyric local delta"])]
        report = {
            "backend": "ctc",
            "timing_entries": 1,
            "assignments": [{"entry": 1, "timestamp": 10.0, "timing_repair": "ctc", "score": 0.35}],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
        }
        self.local_evidence(report, entries, 0, 14.0)
        self.assertEqual(
            auto_lrc.runtime_independent_content_identity_sources_for_entry(report, 0),
            ("local-whisper-aligner",),
        )
        candidate = self.parent(0, 10.0)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        summary = auto_lrc.summarize_central_timing_state(report, state)
        self.assertEqual(summary.content_trusted_entries, 1)
        self.assertEqual(state.decisions[0].written_time.centiseconds, 1000)

    def test_local_whisper_runtime_evidence_tamper_fails_closed(self) -> None:
        entries = [LyricEntry(["unique lyric local epsilon"])]
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_evidence(report, entries, 0, 10.0)
        self.assertIsNotNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))
        runtime = report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY]
        runtime[0]["onset_time"] = 10.10
        self.assertIsNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))
        self.local_evidence(report, entries, 0, 10.0)
        runtime = report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY]
        runtime[0]["raw_transcript_revision"] = "tampered-transcript"
        self.assertIsNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))

    def test_local_whisper_execute_uses_direct_aligned_char_onset_not_match_heuristic(self) -> None:
        entries = [
            LyricEntry(["unique left local"]),
            LyricEntry(["unique target local"]),
            LyricEntry(["unique right local"]),
        ]
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "audio.wav"
            audio.write_bytes(b"audio")
            audio_sha = auto_lrc._path_sha256(audio)
            cap_payload = {
                "kind": "whispercpp-whisperx-local",
                "backend_lineage": "whisper-family",
                "audio_path": str(audio.resolve()),
                "audio_sha256": audio_sha,
                "whisper_cli_path": "cli",
                "whisper_cli_sha256": "cli-sha",
                "whisper_model_path": "model",
                "whisper_model_sha256": "model-sha",
                "whisperx_python_path": "python",
                "whisperx_helper_path": "helper",
                "whisperx_helper_sha256": "helper-sha",
                "language": "ja",
                "device": "cuda",
                "protocol_revision": auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            }
            cap = auto_lrc.LocalWhisperCapability(
                **cap_payload,
                capability_revision=auto_lrc._canonical_json_digest(
                    auto_lrc._local_whisper_capability_semantic_payload(cap_payload)
                ),
            )
            request_payload = {
                "entry_index": 1,
                "capability_revision": cap.capability_revision,
                "audio_revision": audio_sha,
                "context_entry_indexes": (0, 1, 2),
                "context_candidate_ids": ("a", "b", "c"),
                "transcript_digests": tuple(auto_lrc._timing_content_digest(auto_lrc.entry_sung_text(e)) for e in entries),
                "transcript_digest": auto_lrc._timing_content_digest(tuple(auto_lrc.entry_sung_text(e) for e in entries)),
                "window_start": 10.0,
                "window_end": 20.0,
                "window_definition_revision": "window",
            }
            request = auto_lrc.LocalWhisperRequest(
                **request_payload,
                request_revision=auto_lrc._timing_content_digest({
                    **request_payload,
                    "context_entry_indexes": [0, 1, 2],
                    "context_candidate_ids": ["a", "b", "c"],
                    "transcript_digests": list(request_payload["transcript_digests"]),
                }),
            )
            raw = [auto_lrc.AsrSegment(0.0, 9.0, "raw")]
            aligned = [
                auto_lrc.AsrSegment(0.5, 2.5, "unique left local", chars=[(c, 0.5 + i * 0.03) for i, c in enumerate("uniqueleftlocal")]),
                auto_lrc.AsrSegment(3.0, 5.0, "unique target local", chars=[(c, 3.0 + i * 0.03) for i, c in enumerate("uniquetargetlocal")]),
                auto_lrc.AsrSegment(6.0, 8.0, "unique right local", chars=[(c, 6.0 + i * 0.03) for i, c in enumerate("uniquerightlocal")]),
            ]
            local_report = {
                "assignments": [
                    {"entry": 1, "segment": 1, "score": 0.95, "borrowed": False, "timestamp": 0.5},
                    {"entry": 2, "segment": 2, "score": 0.95, "borrowed": False, "timestamp": 3.0},
                    {"entry": 3, "segment": 3, "score": 0.95, "borrowed": False, "timestamp": 6.0},
                ]
            }
            # Deliberately return a bogus heuristic time (4.5 s). The evidence
            # must use the aligned target character onset (3.0 s) instead.
            with mock.patch.object(local_whisper_retry, "decode_temp_wav_window", return_value=audio), \
                 split_patch.patch("_run_whispercpp_wav", return_value=raw), \
                 split_patch.patch("run_whisperx_alignment", return_value={}), \
                 split_patch.patch("whisperx_segments", return_value=aligned), \
                 split_patch.patch("match_whisper_segments", return_value=([0.5, 4.5, 6.0], local_report)):
                outcome = auto_lrc.execute_local_whisper_request(
                    request, cap, entries, auto_lrc.argparse.Namespace(whisper_cli=None, whisper_model=None, whisper_language="ja", whisper_suppress_nst=False, whisperx_python=None, whisperx_device="cpu"), cache={}
                )
            self.assertEqual(outcome.status, "evidence")
            self.assertAlmostEqual(outcome.onset_time or 0.0, 13.0, places=6)

    def test_content_only_local_whisper_runtime_evidence_is_valid(self) -> None:
        entries = [LyricEntry(["unique content-only local"])]
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[0]))
        binding_payload = {
            "mode": "global-unique", "entry_index": 0,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "segment": 1, "timestamp": 10.0, "score": 0.95,
        }
        binding = {**binding_payload, "binding_revision": auto_lrc._timing_content_digest(binding_payload)}
        payload = {
            "entry_index": 0,
            "lyric_sha256": binding_payload["lyric_sha256"],
            "onset_time": None,
            "score": 0.95,
            "occurrence_binding": binding,
            "request_revision": "request",
            "capability_revision": "capability",
            "audio_revision": "audio",
            "window_start": 8.0,
            "window_end": 12.0,
            "raw_transcript_revision": "raw-transcript",
            "aligned_transcript_revision": "aligned-transcript",
            "producer": "local-whisper-aligner",
        }
        report = {
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY: {
                0: {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}
            },
        }
        self.assertIsNotNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))
        self.assertEqual(
            auto_lrc.runtime_independent_content_identity_sources_for_entry(report, 0),
            ("local-whisper-aligner",),
        )

    def test_local_whisper_planner_targets_content_gap_without_timing_gap(self) -> None:
        entries = [
            LyricEntry(["unique left anchor"]),
            LyricEntry(["unique target line"]),
            LyricEntry(["unique right anchor"]),
        ]
        candidate_sets = tuple((self.parent(i, 10.0 + i * 3.0),) for i in range(3))
        report = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc", "score": 0.9},
                {"entry": 2, "timestamp": 13.0, "timing_repair": "ctc", "score": 0.35},
                {"entry": 3, "timestamp": 16.0, "timing_repair": "ctc", "score": 0.9},
            ],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
        }
        cap = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        requests = auto_lrc.plan_local_whisper_requests(
            report, cap, entries, candidate_sets, 30.0
        )
        self.assertEqual([r.entry_index for r in requests], [1])
        self.assertEqual(requests[0].context_entry_indexes, (0, 1, 2))


class V147HypothesisArbitrationRegressionTests(unittest.TestCase):
    def span(self, start: float, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, 0.8, token)

    def entries(self):
        return [
            LyricEntry(["unique left context anchor"]),
            LyricEntry(["unique target hypothesis lyric"]),
            LyricEntry(["unique right context anchor"]),
        ]

    def local_sequence_evidence(self, report, entries, onset: float, score: float = 0.96):
        assignments = [
            {"timestamp": onset - 3.0, "score": 0.95, "segment": 1, "borrowed": False},
            {"timestamp": onset, "score": score, "segment": 2, "borrowed": False},
            {"timestamp": onset + 3.0, "score": 0.95, "segment": 3, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[1]))
        payload = {
            "entry_index": 1,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "onset_time": round(onset, 6), "score": round(score, 6),
            "occurrence_binding": binding, "request_revision": "local-request",
            "capability_revision": "local-cap", "audio_revision": "audio",
            "window_start": round(onset - 5.0, 6), "window_end": round(onset + 5.0, 6),
            "raw_transcript_revision": "local-raw",
            "aligned_transcript_revision": "local-aligned",
            "producer": "local-whisper-aligner",
        }
        report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY] = {
            1: {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}
        }

    def raw_global_identity(self, report, entries, raw_time: float, score: float = 0.95):
        raw_report = {"assignments": [
            {"timestamp": raw_time - 3.0, "score": 0.95, "segment": 1, "borrowed": False},
            {"timestamp": raw_time, "score": score, "segment": 2, "borrowed": False},
            {"timestamp": raw_time + 3.0, "score": 0.95, "segment": 3, "borrowed": False},
        ]}
        auto_lrc.bind_runtime_raw_content_identity(report, entries, raw_report)

    def ctc_parent(self, time: float):
        return auto_lrc.make_timing_candidate(
            entry_index=1, entry_text="unique target hypothesis lyric",
            source="ctc-local-window", raw_time=time,
            spans=(self.span(time, "a"), self.span(time + .08, "b"), self.span(time + .16, "c")),
            confidence=.8, identity_support="supported", sequence_support="supported",
            source_artifact={"ctc": time},
        )

    def raw_fusion(self, time: float):
        artifact = {"producer": "raw-asr-vocal-fusion", "time": time}
        return auto_lrc.make_timing_candidate(
            entry_index=1, entry_text="unique target hypothesis lyric",
            source="raw-vocal-independent-fusion", raw_time=time, confidence=.95,
            identity_support="supported", sequence_support="supported",
            acoustic_support="supported", direct_onset_support="supported",
            acoustic_strength=1.0, acoustic_onset_time=time,
            acoustic_source_artifact=artifact, acoustic_evidence_producer="raw-asr-vocal-fusion",
            acoustic_evidence_independent=True, direct_onset_time=time,
            direct_onset_source_artifact=artifact, direct_onset_evidence_producer="raw-asr-vocal-fusion",
            direct_onset_evidence_kind="cross-backend-onset", direct_onset_evidence_independent=True,
            independent_content_identity=True, source_artifact=artifact,
        )

    def test_context_bound_local_plus_ctc_can_falsify_weaker_global_raw_occurrence(self):
        entries = self.entries()
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_sequence_evidence(report, entries, 10.05)
        self.raw_global_identity(report, entries, 14.00)
        base = ((self.ctc_parent(7.0),), (self.ctc_parent(10.00), self.raw_fusion(14.02)), (self.ctc_parent(17.0),))
        augmented = auto_lrc._augment_candidates_with_local_whisper_consensus(base, report, entries)
        revised, diagnostics = auto_lrc._apply_context_conditioned_whisper_hypothesis_arbitration(augmented, report, entries)
        self.assertEqual(len(diagnostics), 1)
        self.assertFalse(any(c.source == "raw-vocal-independent-fusion" for c in revised[1]))
        self.assertTrue(any(c.source == "ctc-local-whisper-independent-consensus" for c in revised[1]))
        self.assertIn("bounded-sequence-context", diagnostics[0]["reason"])

    def test_global_unique_local_binding_alone_cannot_falsify(self):
        entries = self.entries()
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        # Reuse v14.6-style weaker global-unique local evidence.
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[1]))
        binding_payload = {"mode":"global-unique","entry_index":1,"lyric_sha256":auto_lrc.hashlib.sha256(target.encode()).hexdigest(),"segment":2,"timestamp":10.05,"score":0.96}
        binding={**binding_payload,"binding_revision":auto_lrc._timing_content_digest(binding_payload)}
        payload={"entry_index":1,"lyric_sha256":binding_payload["lyric_sha256"],"onset_time":10.05,"score":0.96,"occurrence_binding":binding,"request_revision":"r","capability_revision":"c","audio_revision":"a","window_start":5.0,"window_end":15.0,"raw_transcript_revision":"rr","aligned_transcript_revision":"ar","producer":"local-whisper-aligner"}
        report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY]={1:{**payload,"evidence_revision":auto_lrc._timing_content_digest(payload)}}
        self.raw_global_identity(report, entries, 14.0)
        base=((self.ctc_parent(7.0),),(self.ctc_parent(10.0),self.raw_fusion(14.02)),(self.ctc_parent(17.0),))
        augmented=auto_lrc._augment_candidates_with_local_whisper_consensus(base,report,entries)
        revised, diagnostics=auto_lrc._apply_context_conditioned_whisper_hypothesis_arbitration(augmented,report,entries)
        self.assertFalse(diagnostics)
        self.assertTrue(any(c.source=="raw-vocal-independent-fusion" for c in revised[1]))

    def test_independent_ctc_support_at_raw_cluster_blocks_falsification(self):
        entries=self.entries(); report={auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY:tuple(entries)}
        self.local_sequence_evidence(report,entries,10.05); self.raw_global_identity(report,entries,14.0)
        raw_ctc=self.ctc_parent(14.02)
        base=((self.ctc_parent(7.0),),(self.ctc_parent(10.0),self.raw_fusion(14.02),raw_ctc),(self.ctc_parent(17.0),))
        augmented=auto_lrc._augment_candidates_with_local_whisper_consensus(base,report,entries)
        revised, diagnostics=auto_lrc._apply_context_conditioned_whisper_hypothesis_arbitration(augmented,report,entries)
        self.assertFalse(diagnostics)
        self.assertTrue(any(c.source=="raw-vocal-independent-fusion" for c in revised[1]))

    def test_raw_and_local_whisper_are_same_support_family(self):
        raw=self.raw_fusion(10.0)
        local=auto_lrc.make_timing_candidate(
            entry_index=1,entry_text="unique target hypothesis lyric",source="local-whisper-independent-consensus",raw_time=10.0,confidence=.9,identity_support="supported",sequence_support="supported",direct_onset_support="supported",direct_onset_time=10.0,direct_onset_source_artifact={"local":1},direct_onset_evidence_producer="local-whisper-aligner",direct_onset_evidence_independent=True,independent_content_identity=True,source_artifact={"local":1})
        self.assertEqual(auto_lrc._candidate_support_families(raw),frozenset({"whisper-family"}))
        self.assertEqual(auto_lrc._candidate_support_families(local),frozenset({"whisper-family"}))


class V148SequenceFingerprintRegressionTests(unittest.TestCase):
    def span(self, start: float, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, 0.8, token)

    def valid_candidate(self, entry: int, text: str, time: float) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=text, source="ctc-local-window", raw_time=time,
            spans=(self.span(time, "a"), self.span(time + .08, "b"), self.span(time + .16, "c")),
            confidence=.8, identity_support="supported", sequence_support="supported",
            source_artifact={"entry": entry, "time": time},
        )

    def provisional_candidate(self, entry: int, text: str, time: float) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=text, source="ctc-current", raw_time=time,
            spans=(), confidence=.7, identity_support="supported", sequence_support="supported",
            source_artifact={"entry": entry, "time": time},
        )

    def entries(self):
        return [
            LyricEntry(["unique intro"]),
            LyricEntry(["repeat left"]),
            LyricEntry(["repeat target"]),
            LyricEntry(["unique branch alpha"]),
            LyricEntry(["unique middle"]),
            LyricEntry(["repeat left"]),
            LyricEntry(["repeat target"]),
            LyricEntry(["unique branch beta"]),
        ]

    def capability(self):
        return auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )

    def test_short_unique_sequence_fingerprint_recovers_repeated_target(self):
        entries = self.entries()
        self.assertFalse(auto_lrc._entry_identity_is_unique(entries, 2))
        self.assertFalse(auto_lrc._entry_identity_is_unique(entries, 1))
        self.assertEqual(auto_lrc._minimal_unique_sequence_context(entries, 2, max_items=9), (1, 3))
        self.assertEqual(auto_lrc._minimal_unique_sequence_context(entries, 6, max_items=9), (5, 7))

    def test_sequence_bracket_binding_does_not_require_individually_unique_neighbors(self):
        entries = self.entries()
        assignments = [None] * len(entries)
        assignments[1] = {"timestamp": 10.0, "score": .95, "segment": 1, "borrowed": False}
        assignments[2] = {"timestamp": 12.0, "score": .96, "segment": 2, "borrowed": False}
        assignments[3] = {"timestamp": 14.0, "score": .94, "segment": 3, "borrowed": False}
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 2, require_context=True)
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding["mode"], "sequence-bracket")
        self.assertEqual(binding["context_entry_indexes"], [1, 2, 3])
        self.assertTrue(auto_lrc._runtime_occurrence_binding_is_valid(entries, 2, binding))

    def test_planner_uses_short_sequence_fingerprint_instead_of_distant_unique_anchors(self):
        entries = self.entries()
        candidate_sets = []
        for i, entry in enumerate(entries):
            text = auto_lrc.entry_sung_text(entry)
            candidate = (
                self.provisional_candidate(i, text, 10.0 + i * 3.0)
                if i == 2 else self.valid_candidate(i, text, 10.0 + i * 3.0)
            )
            candidate_sets.append((candidate,))
        report = {
            "backend": "ctc",
            "assignments": [
                {"entry": i + 1, "timestamp": 10.0 + i * 3.0, "timing_repair": "ctc", "score": .9}
                for i in range(len(entries))
            ],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
        }
        requests = auto_lrc.plan_local_whisper_requests(
            report, self.capability(), entries, tuple(candidate_sets), 60.0
        )
        target = [request for request in requests if request.entry_index == 2]
        self.assertGreaterEqual(len(target), 1)
        self.assertTrue(all(len(request.context_entry_indexes) <= 9 for request in target))
        self.assertTrue(any({1, 2, 3}.issubset(set(request.context_entry_indexes)) for request in target))


class V1410MaterialHypothesisReviewRegressionTests(unittest.TestCase):
    def capability(self) -> auto_lrc.LocalWhisperCapability:
        return auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )

    def runtime_evidence(self, entries, entry=0, onset=10.0, score=.96):
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[entry]))
        binding_payload = {
            "mode": "global-unique",
            "entry_index": entry,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "segment": 1,
            "timestamp": round(onset, 6),
            "score": round(score, 6),
        }
        binding = {
            **binding_payload,
            "binding_revision": auto_lrc._timing_content_digest(binding_payload),
        }
        payload = {
            "entry_index": entry,
            "lyric_sha256": binding_payload["lyric_sha256"],
            "onset_time": round(onset, 6),
            "score": round(score, 6),
            "occurrence_binding": binding,
            "request_revision": "request-r1",
            "capability_revision": "capability",
            "audio_revision": "audio",
            "window_start": round(onset - 2.0, 6),
            "window_end": round(onset + 2.0, 6),
            "raw_transcript_revision": "raw-r1",
            "aligned_transcript_revision": "aligned-r1",
            "producer": "local-whisper-aligner",
        }
        return {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}

    def test_fixed_point_recollect_retains_prior_valid_local_whisper_evidence(self):
        entries = [LyricEntry(["fixed point unique lyric"])]
        evidence = self.runtime_evidence(entries)
        report = {
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY: {0: evidence},
        }
        auto_lrc.bind_runtime_local_whisper_evidence(
            report, entries, self.capability(), (), ()
        )
        retained = auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0)
        self.assertIsNotNone(retained)
        self.assertEqual(report["local_whisper_evidence"]["candidate_count"], 1)
        self.assertEqual(report["local_whisper_evidence"]["retained_from_prior_collect_count"], 1)
        self.assertEqual(report["local_whisper_evidence"]["entries"], [1])

    def test_fixed_point_recollect_drops_tampered_prior_runtime_evidence(self):
        entries = [LyricEntry(["fixed point unique lyric tamper"])]
        evidence = self.runtime_evidence(entries)
        evidence["raw_transcript_revision"] = "tampered"
        report = {
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY: {0: evidence},
        }
        auto_lrc.bind_runtime_local_whisper_evidence(
            report, entries, self.capability(), (), ()
        )
        self.assertIsNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))
        self.assertEqual(report["local_whisper_evidence"]["candidate_count"], 0)

    def valid_selected(self, time: float) -> auto_lrc.TimingCandidate:
        artifact = {"producer": "whisperx-vocal-fusion", "time": time}
        occurrence_entries = [LyricEntry(["review invariant lyric"])]
        occurrence_binding = auto_lrc._entry_occurrence_binding(
            occurrence_entries,
            [{"timestamp": time, "score": .96, "segment": 1}],
            0,
        )
        occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            occurrence_entries,
            0,
            producer="whisperx-vocal-fusion",
            occurrence_binding=occurrence_binding,
            upstream_evidence_revision="v1410-direct-upstream",
        )
        return auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="review invariant lyric",
            source="whisperx-vocal-independent-fusion", raw_time=time,
            confidence=.96, identity_support="supported", sequence_support="supported",
            acoustic_support="supported", direct_onset_support="supported",
            acoustic_strength=1.0, acoustic_onset_time=time,
            acoustic_source_artifact=artifact,
            acoustic_evidence_producer="whisperx-vocal-fusion",
            acoustic_evidence_independent=True,
            direct_onset_time=time, direct_onset_source_artifact=artifact,
            direct_onset_evidence_producer="whisperx-vocal-fusion",
            direct_onset_evidence_kind="cross-backend-onset",
            direct_onset_evidence_independent=True,
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=True,
            source_artifact=artifact,
        )

    def invalid_distant_raw(self, time: float) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="review invariant lyric",
            source="raw-asr", raw_time=time, confidence=.78,
            identity_support="supported", sequence_support="supported",
            source_artifact={"raw": time},
        )

    def review_fixture(self):
        selected = self.valid_selected(10.0)
        distant = self.invalid_distant_raw(11.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(selected, distant),
            previous_candidate=None, selection_revision=1,
        )
        audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
        row = {
            "entry": 1,
            "flags": ["unresolved_raw_ctc_disagreement"],
            "severity": "high",
            "review_required": True,
            "candidate_timestamps": {"output": 10.0, "raw_asr": 11.0},
        }
        return decision, audit, row

    def test_diagnostic_only_raw_disagreement_keeps_review_when_distant(self):
        decision, audit, row = self.review_fixture()
        report = {"hypothesis_arbitration": {"entries": [], "falsification_count": 0}}
        self.assertTrue(audit.timing_trusted)
        self.assertFalse(
            auto_lrc._central_decision_supersedes_suspicious_row(
                decision, audit, row, report
            )
        )

    def test_material_distant_direct_onset_hypothesis_keeps_review_open(self):
        selected = self.valid_selected(10.0)
        material_artifact = {"producer": "vocal-activity-detector", "time": 11.0}
        material = auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="review invariant lyric",
            source="vocal-activity-reentry", raw_time=11.0, confidence=.9,
            identity_support="supported", sequence_support="supported",
            acoustic_support="supported", direct_onset_support="supported",
            acoustic_strength=.9, acoustic_onset_time=11.0,
            acoustic_source_artifact=material_artifact,
            acoustic_evidence_producer="vocal-activity-detector",
            acoustic_evidence_kind="vocal-activity-reentry",
            acoustic_evidence_independent=True,
            direct_onset_time=11.0, direct_onset_source_artifact=material_artifact,
            direct_onset_evidence_producer="vocal-activity-detector",
            direct_onset_evidence_kind="vocal-activity-reentry",
            direct_onset_evidence_independent=True,
            current=False, window_truncated=True, source_artifact=material_artifact,
        )
        diagnostic_raw = self.invalid_distant_raw(11.05)
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(selected, material, diagnostic_raw),
            previous_candidate=None, selection_revision=1,
        )
        audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
        row = {
            "entry": 1,
            "flags": ["unresolved_raw_ctc_disagreement"],
            "severity": "high",
            "review_required": True,
            "candidate_timestamps": {"output": 10.0, "raw_asr": 11.05},
        }
        report = {"hypothesis_arbitration": {"entries": [], "falsification_count": 0}}
        self.assertTrue(audit.timing_trusted)
        self.assertTrue(any(
            auto_lrc.candidate_has_bound_direct_onset_evidence(item.candidate)
            and abs(item.candidate.written_time.seconds - 10.0) > auto_lrc.RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            for item in decision.rejected_candidates
        ))
        self.assertFalse(
            auto_lrc._central_decision_supersedes_suspicious_row(
                decision, audit, row, report
            )
        )

    def test_unknown_ownership_only_blocker_keeps_supported_opening_review(self):
        previous = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="previous lyric",
            source="ctc-current",
            raw_time=8.0,
            spans=(
                auto_lrc.TimingTokenSpan(8.0, 8.08, .8, "a"),
                auto_lrc.TimingTokenSpan(8.10, 8.18, .8, "b"),
                auto_lrc.TimingTokenSpan(8.20, 8.28, .8, "c"),
            ),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"generic": "previous"},
        )
        selected = self.valid_selected(10.0)
        distant = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="review invariant lyric",
            source="ctc-current",
            raw_time=11.0,
            spans=(
                auto_lrc.TimingTokenSpan(11.0, 11.02, .01, "k"),
                auto_lrc.TimingTokenSpan(11.04, 11.12, .12, "a"),
                auto_lrc.TimingTokenSpan(11.14, 11.16, .01, "n"),
            ),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            current=False,
            source_artifact={"generic": "distant-opening"},
        )
        unknown_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id=previous.candidate_id,
            previous_generation_revision=previous.generation_revision,
            previous_region_revision=auto_lrc.analyze_candidate_regions(previous).region_revision,
            tail_evidence_revision="unknown-tail-revision",
            validity="unknown",
            reliable_tail_start=None,
            reliable_tail_end=None,
            uncertainty_seconds=None,
            confidence=None,
            reason="independent-tail-unavailable",
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(selected, distant),
            previous_candidate=previous,
            previous_tail=unknown_tail,
            selection_revision=2,
        )
        self.assertTrue(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )
        distant_evaluation = next(
            item for item in decision.rejected_candidates
            if item.candidate.candidate_id == distant.candidate_id
        )
        self.assertLess(distant_evaluation.evidence.reliable_support_count, 3)
        self.assertIsNotNone(distant_evaluation.evidence.onset_proof_end)
        audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
        row = {
            "entry": 2,
            "flags": ["unresolved_raw_ctc_disagreement"],
            "severity": "high",
            "review_required": True,
            "candidate_timestamps": {"output": 10.0, "raw_asr": 11.0},
        }
        self.assertFalse(
            auto_lrc._central_decision_supersedes_suspicious_row(
                decision, audit, row, {"hypothesis_arbitration": {"entries": []}}
            )
        )

    def test_revision_bound_ownership_violation_still_falsifies_alternative(self):
        previous = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="previous lyric",
            source="ctc-current",
            raw_time=8.0,
            spans=(
                auto_lrc.TimingTokenSpan(8.0, 8.30, .8, "a"),
                auto_lrc.TimingTokenSpan(8.40, 8.70, .8, "b"),
                auto_lrc.TimingTokenSpan(8.80, 9.50, .8, "c"),
            ),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"generic": "previous-tail"},
        )
        valid_tail = auto_lrc.derive_reliable_tail(previous)
        self.assertEqual(valid_tail.validity, "valid")
        selected = self.valid_selected(10.2)
        invading = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="review invariant lyric",
            source="ctc-current",
            raw_time=8.9,
            spans=(
                auto_lrc.TimingTokenSpan(8.9, 8.98, .8, "a"),
                auto_lrc.TimingTokenSpan(9.0, 9.08, .8, "b"),
                auto_lrc.TimingTokenSpan(9.1, 9.18, .8, "c"),
            ),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            current=False,
            source_artifact={"generic": "invading"},
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(selected, invading),
            previous_candidate=previous,
            previous_tail=valid_tail,
            selection_revision=2,
        )
        self.assertFalse(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_explicit_context_falsification_keeps_distant_review(self):
        decision, audit, row = self.review_fixture()
        report = {
            "hypothesis_arbitration": {
                "entries": [{
                    "entry": 1,
                    "raw_identity_time": 11.0,
                    "local_onset": 10.0,
                    "falsified_candidate_ids": ["raw-hypothesis"],
                }],
                "falsification_count": 1,
            }
        }
        self.assertFalse(
            auto_lrc._central_decision_supersedes_suspicious_row(
                decision, audit, row, report
            )
        )


class V15EvidenceConvergenceRegressionTests(_TempModelCacheMixin, unittest.TestCase):
    def repeated_entries(self):
        return [
            LyricEntry(["anchor alpha"]),
            LyricEntry(["repeat target"]),
            LyricEntry(["shared middle"]),
            LyricEntry(["branch beta"]),
            LyricEntry(["separator"]),
            LyricEntry(["anchor alpha"]),
            LyricEntry(["repeat target"]),
            LyricEntry(["shared middle"]),
            LyricEntry(["branch gamma"]),
        ]

    def test_sparse_sequence_binding_ignores_untrusted_middle_rows_without_inventing_them(self):
        entries = self.repeated_entries()
        assignments = [None] * len(entries)
        assignments[0] = {"timestamp": 10.0, "score": .95, "segment": 1, "borrowed": False}
        assignments[1] = {"timestamp": 12.0, "score": .96, "segment": 2, "borrowed": False}
        assignments[2] = {"timestamp": 13.0, "score": .20, "segment": 3, "borrowed": False}
        assignments[3] = {"timestamp": 15.0, "score": .94, "segment": 4, "borrowed": False}
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding["mode"], "sparse-sequence-bracket")
        self.assertEqual(binding["anchor_entry_indexes"], [0, 1, 3])
        self.assertTrue(auto_lrc._runtime_occurrence_binding_is_valid(entries, 1, binding))

    def test_sparse_binding_revalidation_fails_closed_after_anchor_tamper(self):
        entries = self.repeated_entries()
        assignments = [None] * len(entries)
        assignments[0] = {"timestamp": 10.0, "score": .95, "segment": 1, "borrowed": False}
        assignments[1] = {"timestamp": 12.0, "score": .96, "segment": 2, "borrowed": False}
        assignments[3] = {"timestamp": 15.0, "score": .94, "segment": 4, "borrowed": False}
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        assert binding is not None
        tampered = copy.deepcopy(binding)
        tampered["anchor_timestamps"][0] = 9.5
        self.assertFalse(auto_lrc._runtime_occurrence_binding_is_valid(entries, 1, tampered))

    def test_exact_aligned_char_stream_recovers_cross_segment_onset(self):
        segments = [
            auto_lrc.AsrSegment(1.0, 1.5, "今日", chars=[("今", 1.02), ("日", 1.20)]),
            auto_lrc.AsrSegment(1.5, 2.1, "の天気", chars=[("の", 1.52), ("天", 1.62), ("気", 1.82)]),
        ]
        matches = auto_lrc._aligned_exact_target_matches("今日の天気", segments)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(float(matches[0]["onset_time"]), 1.02, places=6)
        self.assertEqual(matches[0]["start_segment_index"], 0)
        self.assertEqual(matches[0]["end_segment_index"], 1)

    def test_planner_adds_bounded_redundant_neighbors_for_sparse_fallback(self):
        entries = [LyricEntry([f"unique row {i}"]) for i in range(7)]
        candidate_sets = []
        for i, entry in enumerate(entries):
            text = auto_lrc.entry_sung_text(entry)
            candidate_sets.append((auto_lrc.make_timing_candidate(
                entry_index=i, entry_text=text, source="ctc-current",
                raw_time=10.0 + i * 2.0, confidence=.8,
                identity_support="supported", sequence_support="supported",
                source_artifact={"entry": i},
            ),))
        report = {
            "backend": "ctc",
            "assignments": [
                {"entry": i + 1, "timestamp": 10.0 + i * 2.0, "score": .9}
                for i in range(len(entries))
            ],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
        }
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
        requests = auto_lrc.plan_local_whisper_requests(
            report, capability, entries, tuple(candidate_sets), 60.0
        )
        target = [request for request in requests if request.entry_index == 3]
        self.assertGreaterEqual(len(target), 2)
        self.assertLessEqual(len(target), 3)
        contexts = [request.context_entry_indexes for request in target]
        self.assertIn((2, 3, 4), contexts)
        self.assertIn((1, 2, 3, 4, 5), contexts)
        self.assertTrue(all(len(context) <= 9 for context in contexts))

    def test_execute_local_whisper_recovers_weak_target_only_from_exact_chars_and_sparse_context(self):
        entries = self.repeated_entries()
        request = auto_lrc.LocalWhisperRequest(
            entry_index=1, capability_revision="capability", audio_revision="audio",
            context_entry_indexes=(0, 1, 2, 3),
            context_candidate_ids=("a", "b", "c", "d"),
            transcript_digests=("ta", "tb", "tc", "td"), transcript_digest="all",
            window_start=10.0, window_end=20.0,
            window_definition_revision="window", request_revision="request",
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
            auto_lrc.AsrSegment(.4, .9, "anchoralpha", chars=[(c, .4 + i*.02) for i,c in enumerate("anchoralpha")]),
            auto_lrc.AsrSegment(1.2, 2.0, "repeattarget", chars=[(c, 1.2 + i*.02) for i,c in enumerate("repeattarget")]),
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
        self.assertEqual(outcome.reason, "trusted-local-whisper-exact-char-recovery-and-onset")
        self.assertAlmostEqual(float(outcome.onset_time or 0.0), 11.2, places=6)
        self.assertEqual(outcome.occurrence_binding["mode"], "sparse-sequence-bracket")

    def test_sparse_context_can_drive_explicit_whisper_hypothesis_falsification(self):
        entries = [
            LyricEntry(["far left"]), LyricEntry(["weak left"]),
            LyricEntry(["unique target"]), LyricEntry(["weak right"]), LyricEntry(["far right"]),
        ]
        local_assignments = [
            {"timestamp": 12.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 13.0, "score": .20, "segment": 2, "borrowed": False},
            {"timestamp": 15.0, "score": .96, "segment": 3, "borrowed": False},
            {"timestamp": 16.0, "score": .20, "segment": 4, "borrowed": False},
            {"timestamp": 18.0, "score": .95, "segment": 5, "borrowed": False},
        ]
        local_binding = auto_lrc._entry_occurrence_binding(entries, local_assignments, 2, require_context=True)
        self.assertIsNotNone(local_binding)
        assert local_binding is not None
        self.assertEqual(local_binding["mode"], "sparse-sequence-bracket")
        raw_assignments = [None] * len(entries)
        raw_assignments[2] = {"timestamp": 10.0, "score": .90, "segment": 1, "borrowed": False}
        raw_binding = auto_lrc._entry_occurrence_binding(entries, raw_assignments, 2)
        self.assertEqual(raw_binding["mode"], "global-unique")

        lyric_sha = auto_lrc.hashlib.sha256(auto_lrc.normalize_match_text("unique target").encode("utf-8")).hexdigest()
        local_payload = {
            "entry_index": 2, "lyric_sha256": lyric_sha, "onset_time": 15.0, "score": .96,
            "occurrence_binding": local_binding, "request_revision": "req",
            "capability_revision": "cap", "audio_revision": "audio",
            "window_start": 11.0, "window_end": 19.0,
            "raw_transcript_revision": "raw-local", "aligned_transcript_revision": "aligned-local",
            "producer": "local-whisper-aligner",
        }
        raw_payload = {
            "entry_index": 2, "lyric_sha256": lyric_sha, "raw_time": 10.0, "raw_score": .90,
            "segment": 1, "occurrence_binding": raw_binding, "producer": "raw-asr-content-identity",
        }
        report = {
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY: {2: {**local_payload, "evidence_revision": auto_lrc._timing_content_digest(local_payload)}},
            auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY: {2: {**raw_payload, "evidence_revision": auto_lrc._timing_content_digest(raw_payload)}},
        }
        raw_artifact = {"producer": "raw-asr-vocal-fusion"}
        raw_candidate = auto_lrc.make_timing_candidate(
            entry_index=2, entry_text="unique target", source="raw-vocal-independent-fusion",
            raw_time=10.0, confidence=.96, identity_support="supported", sequence_support="supported",
            direct_onset_support="supported", direct_onset_time=10.0,
            direct_onset_source_artifact=raw_artifact,
            direct_onset_evidence_producer="raw-asr-vocal-fusion",
            direct_onset_evidence_kind="cross-backend-onset", direct_onset_evidence_independent=True,
            occurrence_evidence=auto_lrc.candidate_occurrence_evidence_from_binding(
                entries, 2, producer="raw-asr-vocal-fusion",
                occurrence_binding=raw_binding,
                upstream_evidence_revision=auto_lrc._timing_content_digest(raw_payload),
            ),
            independent_content_identity=True, source_artifact=raw_artifact,
        )
        ctc_artifact = {"producer": "local-whisper-aligner"}
        ctc_local = auto_lrc.make_timing_candidate(
            entry_index=2, entry_text="unique target", source="ctc-local-whisper-independent-consensus",
            raw_time=15.0, confidence=.9, identity_support="supported", sequence_support="supported",
            direct_onset_support="supported", direct_onset_time=15.0,
            direct_onset_source_artifact=ctc_artifact,
            direct_onset_evidence_producer="local-whisper-aligner",
            direct_onset_evidence_kind="cross-backend-onset", direct_onset_evidence_independent=True,
            occurrence_evidence=auto_lrc.candidate_occurrence_evidence_from_binding(
                entries, 2, producer="local-whisper-aligner",
                occurrence_binding=local_binding,
                upstream_evidence_revision=auto_lrc._timing_content_digest(local_payload),
            ),
            independent_content_identity=True, source_artifact=ctc_artifact,
        )
        candidate_sets = tuple((ctc_local,) if i != 2 else (raw_candidate, ctc_local) for i in range(len(entries)))
        revised, diagnostics = auto_lrc._apply_context_conditioned_whisper_hypothesis_arbitration(
            candidate_sets, report, entries
        )
        self.assertEqual(len(diagnostics), 1)
        self.assertNotIn(raw_candidate.candidate_id, {item.candidate_id for item in revised[2]})
        self.assertIn(ctc_local.candidate_id, {item.candidate_id for item in revised[2]})


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
