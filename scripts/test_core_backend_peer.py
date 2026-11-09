#!/usr/bin/env python3
"""Backend peer simulation, lineage, candidate cleanup and tail-aware retry tests."""

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


class BackendPeerSimulationTests(unittest.TestCase):
    def span(self, start: float, token: str = "x") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.04, 0.95, token)

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        current: bool = False,
        confidence: float = 0.80,
        direct: bool = False,
        independent: bool = False,
    ) -> auto_lrc.TimingCandidate:
        spans = tuple(self.span(time + index * 0.04, f"t{index}") for index in range(4))
        occurrence_evidence = None
        if direct:
            occurrence_entries = [
                LyricEntry([f"peer line {index}"])
                for index in range(max(1, entry + 1))
            ]
            occurrence_assignments = [
                {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
                for index in range(len(occurrence_entries))
            ]
            occurrence_binding = auto_lrc._entry_occurrence_binding(
                occurrence_entries, occurrence_assignments, entry
            )
            occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
                occurrence_entries,
                entry,
                producer="raw-asr-vocal-fusion",
                occurrence_binding=occurrence_binding,
                upstream_evidence_revision="peer-direct-upstream",
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"peer line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=confidence,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported" if direct else "unavailable",
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct else None,
            direct_onset_source_artifact={"entry": entry, "time": time} if direct else None,
            direct_onset_evidence_producer=("raw-asr-vocal-fusion" if direct else None),
            direct_onset_evidence_kind=("cross-backend-onset" if direct else "generic-onset"),
            direct_onset_evidence_independent=bool(direct),
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=independent,
            current=current,
            source_artifact={"entry": entry, "time": time, "source": source},
        )

    def report(self, count: int, backend: str = "ctc") -> dict[str, object]:
        return {
            "backend": backend,
            "assignments": [
                {"entry": index + 1, "timestamp": 1.0 + index, "flags": []}
                for index in range(count)
            ],
        }

    def report_with_entries(
        self,
        count: int,
        backend: str = "ctc",
    ) -> tuple[dict[str, object], list[LyricEntry]]:
        report = self.report(count, backend)
        entries = [LyricEntry([f"peer line {index}"]) for index in range(count)]
        report[auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY] = tuple(entries)
        report["lyric_entry_revision"] = auto_lrc.lyric_identity_revision(entries)
        return report, entries

    def evaluated_bundle(
        self,
        root: Path,
        candidate_sets: tuple[tuple[auto_lrc.TimingCandidate, ...], ...],
        *,
        backend: str = "ctc",
        report: dict[str, object] | None = None,
    ) -> auto_lrc.EvaluatedBackendBundle:
        report = report or self.report(len(candidate_sets), backend)
        evaluated = auto_lrc.evaluate_backend_timing(
            backend,
            candidate_sets,
            report,
            {},
        )
        audio = root / "peer.flac"
        audio.write_bytes(b"peer-audio")
        args = build_parser().parse_args([str(audio)])
        return auto_lrc._capture_evaluated_backend_bundle(
            evaluated,
            report,
            audio_path=audio,
            args=args,
        )

    def test_peer_cluster_anchor_boundary_and_distant_same_lineage_review(self) -> None:
        candidates = (
            self.candidate(0, 1.00, source="ctc-current", direct=True, confidence=0.95),
            self.candidate(0, 1.30, source="ctc-opening-vocal-fusion", direct=True),
            self.candidate(0, 1.40, source="ctc-vocal-activity-reentry", direct=True),
            self.candidate(0, 1.60, source="ctc-vocal-onset-confirmed", direct=True),
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=candidates,
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        clusters = auto_lrc._peer_observation_cluster_summaries(
            decision, decision.written_time.seconds
        )
        self.assertEqual(
            [round(item.anchor_time, 2) for item in clusters],
            [1.00, 1.30, 1.60],
        )
        self.assertTrue(all(item.review_required for item in clusters[1:]))
        self.assertEqual(clusters[1].member_candidate_ids.__len__(), 2)

    def test_peer_tight_cluster_has_no_material_conflict(self) -> None:
        candidates = (
            self.candidate(0, 1.00, source="ctc-current", direct=True, confidence=0.95),
            self.candidate(0, 1.17, source="ctc-opening-vocal-fusion", direct=True),
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=candidates,
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        clusters = auto_lrc._peer_observation_cluster_summaries(
            decision, decision.written_time.seconds
        )
        self.assertEqual(len(clusters), 1)
        self.assertFalse(clusters[0].review_required)
        self.assertEqual(
            auto_lrc._material_distant_timing_alternatives(
                decision, decision.written_time.seconds
            ),
            (),
        )

        boundary_candidates = (
            self.candidate(0, 1.00, source="ctc-current", direct=True, confidence=0.95),
            self.candidate(0, 1.18, source="ctc-opening-vocal-fusion", direct=True),
        )
        boundary_decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=boundary_candidates,
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        boundary_clusters = auto_lrc._peer_observation_cluster_summaries(
            boundary_decision, boundary_decision.written_time.seconds
        )
        self.assertEqual(len(boundary_clusters), 1)
        self.assertFalse(boundary_clusters[0].review_required)

    def test_peer_full_scaffold_preserves_suspicious_history_and_report_immutability(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            selected_report, _entries = self.report_with_entries(1, "ctc")
            nonselected_report, _entries = self.report_with_entries(1, "whispercpp")
            selected_report["suspicious_alignments"] = [{
                "entry": 1,
                "flags": ["candidate_disagreement"],
                "candidate_timestamps": {"ctc": 1.0, "raw_asr": 2.0},
                "review_required": True,
                "severity": "high",
            }]
            selected_report["timing_review_entries"] = [1]
            selected_report["legacy_timing_diagnostics"] = {
                "history": [{"entry": 1, "reason": "retained"}]
            }
            ctc = self.candidate(0, 1.0, current=True, source="ctc-current")
            whisper = self.candidate(
                0,
                2.0,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )
            first = self.evaluated_bundle(
                root, ((ctc,),), backend="ctc", report=selected_report
            )
            second = self.evaluated_bundle(
                root, ((whisper,),), backend="whispercpp", report=nonselected_report
            )
            before = copy.deepcopy(selected_report)
            result = auto_lrc._simulate_cross_backend_peer_graph(
                (first, second), selected_backend="ctc"
            )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["authority"])
        self.assertEqual(result["scaffold_binding"]["selected_backend"], "ctc")  # type: ignore[index]
        self.assertEqual(result["summary"]["review_required_count"], 1)  # type: ignore[index]
        self.assertTrue(result["rows"][0]["review_required"])  # type: ignore[index]
        self.assertEqual(selected_report, before)
        self.assertNotIn("commit", json.dumps(result).lower())
        self.assertNotIn("candidate_disagreement", json.dumps(result["bundles"]).lower())

    def test_current_is_identity_bound_but_peer_selection_neutralizes_only_policy(self) -> None:
        current = self.candidate(0, 1.0, current=True, source="ctc-current", direct=True)
        neutral = self.candidate(0, 1.0, current=False, source="ctc-current", direct=True)
        self.assertNotEqual(current.candidate_id, neutral.candidate_id)
        self.assertNotEqual(current.generation_revision, neutral.generation_revision)
        self.assertNotEqual(
            auto_lrc.evaluate_timing_candidate(current).evidence.candidate_revision,
            auto_lrc.evaluate_timing_candidate(neutral).evidence.candidate_revision,
        )
        normal = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, self.candidate(
                0,
                2.0,
                current=True,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )),
            previous_candidate=None,
            selection_revision=1,
        )
        peer = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, self.candidate(
                0,
                2.0,
                current=True,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        self.assertEqual(normal.status, "selected_valid")
        self.assertEqual(peer.status, "provisional_unresolved")
        self.assertIsNone(peer.selected_candidate_id)
        self.assertIsNone(peer.recovery)
        self.assertTrue(current.current)
        self.assertFalse(neutral.current)

    def test_peer_simulation_respects_ctc_first_timing_policy(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            ctc_report, _entries = self.report_with_entries(1, "ctc")
            whisper_report, _entries = self.report_with_entries(1, "whispercpp")
            ctc = self.candidate(0, 1.0, current=True, source="ctc-current", direct=True)
            raw = self.candidate(
                0,
                2.0,
                current=True,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )
            first = self.evaluated_bundle(root, ((ctc,),), backend="ctc", report=ctc_report)
            second = self.evaluated_bundle(
                root, ((raw,),), backend="whispercpp", report=whisper_report
            )
            default = auto_lrc._simulate_cross_backend_peer_graph(
                (first, second), selected_backend="ctc"
            )
            ctc_first = auto_lrc._simulate_cross_backend_peer_graph(
                (first, second),
                selected_backend="ctc",
                timing_source_policy="ctc-first",
            )
        self.assertEqual(default["rows"][0]["status"], "provisional_unresolved")
        self.assertEqual(ctc_first["rows"][0]["status"], "selected_valid")
        self.assertEqual(ctc_first["rows"][0]["candidate_source"], "ctc-current")
        self.assertEqual(ctc_first["rows"][0]["hypothetical_centiseconds"], 100)

    def test_backend_projection_passes_selected_timing_policy_to_peer_simulation(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "projection-policy.flac"
            audio.write_bytes(b"projection-policy-audio")
            args = build_parser().parse_args([str(audio)])
            report, _entries = self.report_with_entries(1, "ctc")
            candidate = self.candidate(
                0, 1.0, current=True, source="ctc-current", direct=True
            )
            evaluated = auto_lrc.evaluate_backend_timing(
                "ctc",
                ((candidate,),),
                report,
                {},
                timing_source_policy="ctc-first",
            )
            selection = auto_lrc.select_evaluated_backend((evaluated,))
            generation = [
                {
                    "backend": "ctc",
                    "report": report,
                    "timestamps": [1.0],
                    "error": None,
                }
            ]
            preview = [
                {"timestamps": [1.0], "report": report, "evaluated": evaluated}
            ]
            with split_patch.patch("_simulate_cross_backend_peer_graph",
                wraps=auto_lrc._simulate_cross_backend_peer_graph,
            ) as simulated:
                auto_lrc._build_backend_provenance_projection_seed(
                    audio, args, generation, preview, selection
                )
        self.assertEqual(
            simulated.call_args.kwargs["timing_source_policy"], "ctc-first"
        )

    def test_peer_unresolved_fallback_keeps_current_clock_independent_of_candidate_id(self) -> None:
        current = self.candidate(0, 10.0, current=True, source="ctc-current")
        alternative = self.candidate(0, 11.0, current=False, source="ctc-current")
        first = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(
                replace(current, candidate_id="aaa"),
                replace(alternative, candidate_id="zzz"),
            ),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
            timing_source_policy="ctc-first",
        )
        second = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(
                replace(current, candidate_id="zzz"),
                replace(alternative, candidate_id="aaa"),
            ),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
            timing_source_policy="ctc-first",
        )
        self.assertEqual(first.status, "provisional_unresolved")
        self.assertEqual(second.status, "provisional_unresolved")
        self.assertEqual(first.written_time.seconds, 10.0)
        self.assertEqual(second.written_time.seconds, 10.0)
        self.assertTrue(first.candidate.current)
        self.assertTrue(second.candidate.current)

    def test_peer_dominance_and_consensus_use_existing_evidence_only(self) -> None:
        weak_current = self.candidate(0, 1.0, current=True, confidence=0.20)
        strong = self.candidate(
            0,
            2.0,
            source="raw-vocal-independent-fusion",
            direct=True,
            independent=True,
            confidence=0.95,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(weak_current, strong),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertEqual(decision.candidate.candidate_id, strong.candidate_id)
        self.assertIsNone(decision.recovery)

        ctc = self.candidate(0, 1.00, source="ctc-opening-vocal-fusion")
        whisper = self.candidate(
            0,
            1.10,
            source="raw-vocal-independent-fusion",
            direct=True,
            independent=True,
        )
        cluster = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(ctc, whisper),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        self.assertEqual(cluster.status, "selected_valid")
        self.assertIn(cluster.candidate.candidate_id, {ctc.candidate_id, whisper.candidate_id})

        same_lineage = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(
                self.candidate(0, 1.00, source="ctc-opening-vocal-fusion"),
                self.candidate(0, 1.10, source="ctc-vocal-activity-reentry"),
            ),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        self.assertEqual(same_lineage.status, "provisional_unresolved")

    def test_bundle_freezes_scaffold_and_exact_candidate_union(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            candidate = self.candidate(0, 1.0, current=True)
            report = self.report(1)
            bundle = self.evaluated_bundle(root, ((candidate,),), report=report)
            digest_before = bundle.bundle_revision
            report["assignments"][0]["entry"] = 99  # type: ignore[index]
            self.assertEqual(auto_lrc._validate_evaluated_backend_bundle(bundle)["valid"], True)
            self.assertEqual(bundle.bundle_revision, digest_before)
            union, error = auto_lrc._union_peer_candidate_sets((bundle, bundle))
            self.assertIsNone(error)
            self.assertIsNotNone(union)
            assert union is not None
            self.assertEqual(len(union[0]), 1)
            self.assertNotIn("path", bundle.capability_projection_json.lower())

    def test_missing_tail_context_is_diagnostic_only_and_simulation_never_authoritative(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            candidate = self.candidate(0, 1.0, current=True)
            report = self.report(1)
            report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = {
                0: auto_lrc.IndependentTailEvidence(
                    entry_index=0,
                    tail_start=0.5,
                    tail_end=0.9,
                    uncertainty_seconds=0.1,
                    confidence=0.9,
                    producer="ctc-assignment-terminal-row",
                    source_revision="missing-provenance",
                    provenance=None,
                )
            }
            bundle = self.evaluated_bundle(root, ((candidate,),), report=report)
            result = auto_lrc._simulate_cross_backend_peer_graph((bundle,))
        self.assertFalse(result["authority"])
        self.assertIn(result["status"], {"SUCCESS", "BLOCKED"})
        self.assertEqual(result["context_validation"]["diagnostic_tail_row_count"], 1)  # type: ignore[index]

    def test_stale_or_transplanted_content_cannot_enter_peer_observations(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            report, entries = self.report_with_entries(1)
            stale_lyric = hashlib.sha256(
                auto_lrc.normalize_match_text("stale lyric").encode("utf-8")
            ).hexdigest()
            report[auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY] = {
                0: {
                    "producer": "raw-asr-content-identity",
                    "lyric_sha256": stale_lyric,
                    "occurrence_binding": {"entry": 2},
                    "evidence_revision": "self-consistent-stale-revision",
                }
            }
            candidate = self.candidate(0, 1.0, current=True)
            bundle = self.evaluated_bundle(root, ((candidate,),), report=report)
            bound, diagnostic = auto_lrc._bundle_content_observations_bound_to_scaffold(bundle)
            report[auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY] = {
                1: {
                    "producer": "raw-asr-content-identity",
                    "lyric_sha256": hashlib.sha256(
                        auto_lrc.normalize_match_text(entries[0].lines[0]).encode("utf-8")
                    ).hexdigest(),
                    "occurrence_binding": {"entry": 1},
                    "evidence_revision": "wrong-entry-revision",
                }
            }
            transplanted = self.evaluated_bundle(root, ((candidate,),), report=report)
            transplanted_bound, transplanted_diagnostic = (
                auto_lrc._bundle_content_observations_bound_to_scaffold(transplanted)
            )
        self.assertEqual(bound, ())
        self.assertEqual(diagnostic, 1)
        self.assertEqual(transplanted_bound, ())
        self.assertEqual(transplanted_diagnostic, 1)
        self.assertEqual(entries[0].lines[0], "peer line 0")

    def test_peer_success_is_non_authoritative_with_current_conflict_diagnostic(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            ctc_report, _entries = self.report_with_entries(1, "ctc")
            whisper_report, _entries = self.report_with_entries(1, "whispercpp")
            ctc = self.candidate(0, 1.0, current=True, source="ctc-current")
            whisper = self.candidate(
                0,
                2.0,
                current=True,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )
            first = self.evaluated_bundle(root, ((ctc,),), backend="ctc", report=ctc_report)
            second = self.evaluated_bundle(
                root,
                ((whisper,),),
                backend="whispercpp",
                report=whisper_report,
            )
            before = copy.deepcopy(ctc_report)
            result = auto_lrc._simulate_cross_backend_peer_graph((first, second))
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["authority"])
        self.assertEqual(result["current_conflict_entries"], [1])
        self.assertNotIn("multiple-current-candidates", result.get("blocked_reasons", []))
        self.assertEqual(result["rows"][0]["current_baseline_conflict"], True)  # type: ignore[index]
        self.assertEqual(ctc_report, before)

    def test_material_owner_details_match_existing_boolean_predicate(self) -> None:
        current = self.candidate(0, 1.0, current=True, confidence=0.95)
        alternative = self.candidate(
            0,
            3.0,
            source="raw-vocal-independent-fusion",
            direct=True,
            independent=True,
            confidence=0.20,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, alternative),
            previous_candidate=None,
            selection_revision=1,
        )
        details = auto_lrc._material_distant_timing_alternatives(
            decision, decision.written_time.seconds
        )
        self.assertEqual(
            bool(details),
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            ),
        )
        self.assertTrue(details)
        self.assertFalse(details[0]["explicit_falsification"])
        self.assertEqual(details[0]["direct_onset_evidence"]["producer"], "raw-asr-vocal-fusion")  # type: ignore[index]
        self.assertNotIn("path", json.dumps(details).lower())

    def test_peer_simulation_diagnoses_distinct_current_baselines_without_blocking(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            ctc_report, _entries = self.report_with_entries(1, "ctc")
            whisper_report, _entries = self.report_with_entries(1, "whispercpp")
            ctc = self.candidate(0, 1.0, current=True, source="ctc-current")
            whisper = self.candidate(
                0,
                2.0,
                current=True,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )
            first = self.evaluated_bundle(root, ((ctc,),), backend="ctc", report=ctc_report)
            second = self.evaluated_bundle(
                root,
                ((whisper,),),
                backend="whispercpp",
                report=whisper_report,
            )
            result = auto_lrc._simulate_cross_backend_peer_graph((first, second))
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["authority"])
        self.assertIn(1, result["current_conflict_entries"])
        self.assertNotIn("multiple-current-candidates", result.get("blocked_reasons", []))

    def test_review_only_projection_filters_tight_and_falsified_clusters(self) -> None:
        def cluster(relation: str, review: bool, falsified: bool, revision: str) -> dict[str, object]:
            return {
                "cluster_revision": revision,
                "relation": relation,
                "review_required": review,
                "explicit_falsification": falsified,
                "member_candidate_ids": [revision],
                "member_generation_revisions": [revision],
                "support_families": ["ctc"],
            }
        peer = {
            "status": "SUCCESS",
            "authority": False,
            "simulation_revision": "peer-revision",
            "bundles": [
                {
                    "valid": True,
                    "scaffold_lyric_valid": True,
                    "bundle_revision": "bundle-a",
                }
            ],
            "context_validation": {"valid_context_count": 1},
            "rows": [
                {
                    "entry": 1,
                    "material_revision": "mat-1",
                    "cluster_revision": "clusters-1",
                    "cluster_count": 2,
                    "clusters": [
                        cluster("baseline", False, False, "base-1"),
                        cluster("distant", True, False, "distant-1"),
                    ],
                },
                {
                    "entry": 2,
                    "material_revision": "mat-2",
                    "cluster_revision": "clusters-2",
                    "cluster_count": 1,
                    "clusters": [cluster("baseline", False, False, "base-2")],
                },
                {
                    "entry": 3,
                    "material_revision": "mat-3",
                    "cluster_revision": "clusters-3",
                    "cluster_count": 2,
                    "clusters": [
                        cluster("baseline", False, False, "base-3"),
                        cluster("distant", True, True, "falsified-3"),
                    ],
                },
                {
                    "entry": 4,
                    "material_revision": "mat-4",
                    "cluster_revision": "clusters-4",
                    "cluster_count": 2,
                    "clusters": [
                        cluster("baseline", False, False, "base-4"),
                        cluster("distant", True, False, "same-4"),
                    ],
                },
            ],
        }
        projection = auto_lrc._peer_review_only_conflict_projection(peer)
        self.assertEqual(projection["status"], "RETRIEVED")
        self.assertEqual([item["entry"] for item in projection["entries"]], [1, 4])  # type: ignore[index]
        self.assertEqual(projection["authority"], "review-only")
        blocked = auto_lrc._peer_review_only_conflict_projection({
            "status": "BLOCKED",
            "authority": False,
        })
        self.assertEqual(blocked["status"], "ABSENT")
        self.assertEqual(blocked["entries"], [])

    def test_review_only_guard_changes_review_only_state_not_time_content_or_timing(self) -> None:
        candidate = self.candidate(0, 1.0, current=True, direct=True)
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [{"entry": 1, "segment": 1, "score": 0.9, "flags": []}],
        }
        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", ((candidate,),), report, {}
        )
        report[auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY] = frozenset({1})
        before = {
            "timestamp": evaluated.state.decisions[0].written_time.seconds,
            "content": report["assignments"][0].get("content_trusted"),  # type: ignore[index]
        }
        committed = auto_lrc.commit_final_timing_state(
            evaluated.state,
            report,
            {},
            finding_states=evaluated.finding_states,
        )
        assignment = report["assignments"][0]  # type: ignore[index]
        self.assertEqual(committed.decisions[0].written_time.seconds, before["timestamp"])
        self.assertTrue(assignment["timing_trusted"])
        self.assertTrue(assignment["content_trusted"])
        self.assertTrue(assignment["review_required"])
        self.assertFalse(assignment["overall_trusted"])
        self.assertEqual(assignment["verification_status"], "review-required")
        self.assertEqual(assignment["timestamp"], before["timestamp"])

    def test_review_only_runtime_metadata_is_writer_private(self) -> None:
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [],
            auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY: frozenset({1}),
            auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY: {
                "authority": "review-only",
                "entries": [{"entry": 1}],
            },
        }
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "review-only.json"
            auto_lrc.write_alignment_report(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn(auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY, text)
        self.assertNotIn(auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY, text)


class Node18SelectedLineageRegressionTests(unittest.TestCase):
    def candidate(
        self,
        time: float,
        source: str,
        *,
        current: bool = False,
        direct: bool = False,
    ) -> auto_lrc.TimingCandidate:
        spans = tuple(
            auto_lrc.TimingTokenSpan(
                time + index * 0.04,
                time + index * 0.04 + 0.04,
                0.95,
                f"t{index}",
            )
            for index in range(4)
        )
        selected_baseline = current and source == "whispercpp-current"
        return auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="node18 lineage probe",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.90,
            identity_support="unavailable" if selected_baseline else "supported",
            sequence_support="supported",
            acoustic_support="supported" if direct else "unavailable",
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct else None,
            direct_onset_source_artifact=(
                {"source": source, "time": time} if direct else None
            ),
            direct_onset_evidence_producer=(
                "consonant-onset-detector" if direct else None
            ),
            direct_onset_evidence_kind="vocal-onset" if direct else "generic-onset",
            direct_onset_evidence_independent=direct,
            current=current,
            source_artifact={"source": source, "time": time},
        )

    def decision(
        self,
        selected: auto_lrc.TimingCandidate,
        *alternatives: auto_lrc.TimingCandidate,
    ) -> auto_lrc.TimingDecision:
        selected_eval = auto_lrc.evaluate_timing_candidate(selected)
        rejected = tuple(
            auto_lrc.evaluate_timing_candidate(candidate)
            for candidate in alternatives
        )
        return auto_lrc.TimingDecision(
            entry_index=0,
            selection_revision=1,
            status="selected_valid",
            selected_candidate_id=selected.candidate_id,
            provisional_candidate_id=None,
            written_time=selected.written_time,
            timestamp_revision=1,
            timing_evidence_revision=1,
            evidence=selected_eval.evidence,
            candidate=selected,
            rejected_candidates=rejected,
            recovery=None,
        )

    def distant_cluster(self, *alternatives: auto_lrc.TimingCandidate):
        selected = self.candidate(1.00, "whispercpp-current", current=True)
        decision = self.decision(selected, *alternatives)
        clusters = auto_lrc._peer_observation_cluster_summaries(
            decision,
            decision.written_time.seconds,
        )
        distant = [cluster for cluster in clusters if cluster.relation == "distant"]
        self.assertEqual(len(distant), 1)
        self.assertTrue(distant[0].review_required)
        return distant[0]

    def test_explicit_whispercpp_selected_lineage_allows_proven_ctc_independence(self):
        cluster = self.distant_cluster(
            self.candidate(1.40, "ctc-opening-vocal-fusion", direct=True)
        )
        self.assertEqual(cluster.evidence_state, "independent")

    def test_explicit_whispercpp_selected_lineage_blocks_same_whisper_lineage(self):
        cluster = self.distant_cluster(
            self.candidate(
                1.40,
                "local-whisper-vocal-independent-fusion",
                direct=True,
            )
        )
        self.assertEqual(cluster.evidence_state, "same-lineage")

    def test_ctc_without_direct_onset_remains_unavailable(self):
        selected = self.candidate(1.00, "whispercpp-current", current=True)
        alternative = self.candidate(1.40, "ctc-local-window", direct=False)
        evaluation = auto_lrc.evaluate_timing_candidate(alternative)
        state = auto_lrc._peer_cluster_evidence_state(
            (evaluation,),
            auto_lrc._neutral_arbitration_support_families(selected),
        )
        self.assertEqual(state, "unavailable")

    def test_mixed_independent_and_same_lineage_members_remain_mixed(self):
        cluster = self.distant_cluster(
            self.candidate(1.40, "ctc-opening-vocal-fusion", direct=True),
            self.candidate(
                1.45,
                "local-whisper-vocal-independent-fusion",
                direct=True,
            ),
        )
        self.assertEqual(cluster.evidence_state, "mixed")


class UnresolvedEvidenceRequirementsProjectionTests(unittest.TestCase):
    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        identity: str = "supported",
        sequence: str = "supported",
        direct: bool = False,
        current: bool = False,
    ) -> auto_lrc.TimingCandidate:
        spans = tuple(
            auto_lrc.TimingTokenSpan(
                time + index * 0.08,
                time + index * 0.08 + 0.04,
                0.95,
                f"t{index}",
            )
            for index in range(4)
        )
        occurrence_evidence = None
        if direct:
            occurrence_entries = [
                LyricEntry([f"projection line {index}"])
                for index in range(max(1, entry + 1))
            ]
            occurrence_assignments = [
                {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
                for index in range(len(occurrence_entries))
            ]
            occurrence_binding = auto_lrc._entry_occurrence_binding(
                occurrence_entries, occurrence_assignments, entry
            )
            occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
                occurrence_entries,
                entry,
                producer="raw-asr-vocal-fusion",
                occurrence_binding=occurrence_binding,
                upstream_evidence_revision="projection-direct-upstream",
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"projection line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.9,
            identity_support=identity,
            sequence_support=sequence,
            acoustic_support="supported" if direct else "unavailable",
            acoustic_strength=0.9 if direct else None,
            acoustic_onset_time=time if direct else None,
            acoustic_source_artifact={"source": "synthetic-acoustic", "time": time} if direct else None,
            acoustic_evidence_producer="synthetic-acoustic" if direct else None,
            acoustic_evidence_kind="vocal-onset" if direct else "generic-acoustic",
            acoustic_evidence_independent=bool(direct),
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct else None,
            direct_onset_source_artifact={"source": "synthetic-direct", "time": time} if direct else None,
            direct_onset_evidence_producer="raw-asr-vocal-fusion" if direct else None,
            direct_onset_evidence_kind="cross-backend-onset" if direct else "generic-onset",
            direct_onset_evidence_independent=bool(direct),
            occurrence_evidence=occurrence_evidence,
            current=current,
            source_artifact={"source": source, "time": time},
        )

    def report_state(
        self,
        candidates: tuple[auto_lrc.TimingCandidate, ...],
        *,
        review: bool = True,
        content: bool = True,
        timing: bool = True,
        overall: bool = False,
    ) -> tuple[dict[str, object], auto_lrc.EvaluatedBackendTiming]:
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            "assignments": [{"entry": 1, "timestamp": 1.0, "flags": []}],
            "lyric_entry_revision": "lyric-revision",
            "source_audio_sha256": "audio-revision",
            "vocal_stem_sha256": "stem-revision",
            "algorithm_revision": auto_lrc.LRC_ALGORITHM_REVISION,
        }
        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", (candidates,), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        auto_lrc.commit_evaluated_backend(selection, report)
        assignment = report["assignments"][0]
        assignment.update({
            "content_trusted": content,
            "timing_trusted": timing,
            "overall_trusted": overall,
            "review_required": review,
            "verification_status": "review-required" if review else "unverified",
        })
        report["candidate_selection"] = {
            "selection_revision": selection.selection_revision,
        }
        return report, evaluated

    def project(self, report: dict[str, object], evaluated: auto_lrc.EvaluatedBackendTiming) -> dict[str, object]:
        projection = auto_lrc._build_unresolved_evidence_requirements_projection(
            report, evaluated.state
        )
        report[auto_lrc._UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY] = projection
        auto_lrc._validate_unresolved_evidence_requirements_projection(report)
        return projection

    def test_projection_preserves_material_direct_producer_and_state(self) -> None:
        current = self.candidate(0, 1.0, current=True)
        alternative = self.candidate(
            0,
            3.0,
            source="raw-vocal-independent-fusion",
            direct=True,
        )
        report, evaluated = self.report_state((current, alternative))
        before = copy.deepcopy(report["assignments"])
        projection = self.project(report, evaluated)
        row = projection["rows"][0]
        self.assertEqual(projection["authority"], "diagnostic-only")
        self.assertTrue(row["material_alternatives"])
        alternative_projection = row["material_alternatives"][0]
        self.assertEqual(
            alternative_projection["direct_onset_evidence"]["producer"],
            "raw-asr-vocal-fusion",
        )
        self.assertEqual(
            alternative_projection["direct_onset_evidence"]["audio_revision"]["value"],
            "stem-revision",
        )
        self.assertIn("ALTERNATIVE_FALSIFICATION", row["missing_proof_categories"])
        self.assertIn("OWNERSHIP_INDEPENDENCE", row["missing_proof_categories"])
        self.assertEqual(report["assignments"], before)
        self.assertEqual(report["assignments"][0]["timestamp"], 1.0)
        self.assertTrue(report["assignments"][0]["review_required"])
        self.assertFalse(report["assignments"][0]["overall_trusted"])

    def test_projection_deduplicates_wrapper_cluster_only(self) -> None:
        current = self.candidate(0, 1.0, current=True)
        wrapper_a = self.candidate(0, 3.0, source="ctc-opening-vocal-fusion", direct=True)
        wrapper_b = self.candidate(0, 3.1, source="ctc-vocal-onset-confirmed", direct=True)
        report, evaluated = self.report_state((current, wrapper_a, wrapper_b))
        projection = self.project(report, evaluated)
        clusters = projection["rows"][0]["material_clusters"]
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]["member_candidate_ids"]), 2)
        self.assertEqual(
            len(set(clusters[0]["member_candidate_ids"])),
            len(clusters[0]["member_candidate_ids"]),
        )
        self.assertTrue(clusters[0]["review_required"])

    def test_projection_covers_content_opening_and_suspicious_only(self) -> None:
        content_candidate = self.candidate(
            0,
            1.0,
            current=True,
            identity="unavailable",
            sequence="unavailable",
        )
        content_report, content_evaluated = self.report_state(
            (content_candidate,), review=False, content=False, timing=True, overall=False
        )
        content_projection = self.project(content_report, content_evaluated)
        categories = set(content_projection["rows"][0]["missing_proof_categories"])
        self.assertIn("CONTENT_IDENTITY", categories)
        self.assertIn("OCCURRENCE", categories)
        self.assertIn("CURRENT_OPENING", categories)

        suspicious_candidate = self.candidate(0, 1.0, current=True, direct=True)
        suspicious_report, suspicious_evaluated = self.report_state(
            (suspicious_candidate,), review=True, content=True, timing=True, overall=False
        )
        suspicious_report["suspicious_alignments"] = [{
            "entry": 1,
            "flags": ["unresolved_raw_ctc_disagreement"],
            "source_row_revision": "suspicious-revision",
            "review_required": True,
        }]
        suspicious_projection = self.project(suspicious_report, suspicious_evaluated)
        suspicious_row = suspicious_projection["rows"][0]
        self.assertEqual(
            suspicious_row["suspicious_actionable_flags"],
            ["unresolved_raw_ctc_disagreement"],
        )
        self.assertIn("SUSPICIOUS_CONFLICT", suspicious_row["missing_proof_categories"])
        self.assertEqual(suspicious_row["material_alternatives"], [])

    def test_projection_tamper_stale_binding_and_private_stripping_fail_closed(self) -> None:
        current = self.candidate(0, 1.0, current=True)
        report, evaluated = self.report_state((current,))
        projection = self.project(report, evaluated)
        projection["rows"][0]["selected"]["selected_time"] = 99.0
        with self.assertRaises(LrcError):
            auto_lrc._validate_unresolved_evidence_requirements_projection(report)

        report, evaluated = self.report_state((current,))
        self.project(report, evaluated)
        report["source_audio_sha256"] = "stale-audio"
        with self.assertRaises(LrcError):
            auto_lrc._validate_unresolved_evidence_requirements_projection(report)

        report, evaluated = self.report_state((current,))
        self.project(report, evaluated)
        report[auto_lrc._RUNTIME_WHISPERX_TEMP_PATHS_KEY] = ("C:/private/temp.wav",)
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "projection.json"
            auto_lrc.write_alignment_report(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertIn(auto_lrc._UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY, text)
        self.assertNotIn("private/temp.wav", text)
        self.assertNotIn("_runtime_", text)


class CandidateGenerationCleanupTests(unittest.TestCase):
    def test_active_automatic_precentral_stages_do_not_write_assignment_timestamp(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        names = {
            "apply_ctc_acoustic_backtrack",
            "apply_ctc_local_window_realign",
            "score_line_timing_candidates",
            "apply_short_duplicate_acoustic_recovery",
            "apply_post_score_audio_validation",
            "apply_acoustic_late_onset_refinement",
        }
        functions = {
            node.name: node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }
        self.assertEqual(set(functions), names)
        for name, function in functions.items():
            body = ast.get_source_segment(source, function) or ""
            self.assertNotIn('assignment["timestamp"] =', body, name)
            self.assertNotIn("assignment['timestamp'] =", body, name)
        acoustic_body = ast.get_source_segment(source, functions["apply_ctc_acoustic_backtrack"]) or ""
        self.assertNotIn("timestamps[index] =", acoustic_body)
        local_body = ast.get_source_segment(source, functions["apply_ctc_local_window_realign"]) or ""
        self.assertNotIn("refined[index] = timestamp", local_body)
        duplicate_body = ast.get_source_segment(source, functions["apply_short_duplicate_acoustic_recovery"]) or ""
        self.assertNotIn("recovered[index] = candidate", duplicate_body)

    def test_structured_hypothesis_is_separate_from_current_backend_candidate(self) -> None:
        entries = [LyricEntry(["target"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 10.0,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [
                        {"char": "x", "start": 10.0, "end": 10.02, "score": 0.01},
                        {"char": "y", "start": 10.04, "end": 10.06, "score": 0.01},
                    ],
                }
            ],
            "suspicious_alignments": [],
        }
        assignment = report["assignments"][0]  # type: ignore[index]
        auto_lrc.record_alignment_hypothesis(
            assignment,
            source="ctc-local-window",
            stage="ctc-local-window-realign",
            raw_time=12.0,
            confidence=0.95,
            spans=[
                {"char": "t", "start": 12.0, "end": 12.02, "score": 0.9},
                {"char": "a", "start": 12.04, "end": 12.06, "score": 0.9},
                {"char": "r", "start": 12.08, "end": 12.10, "score": 0.9},
            ],
            identity_support="supported",
            sequence_support="supported",
            parent_source="ctc-global-mms",
            parent_time=10.0,
        )

        candidate_sets, diagnostics = auto_lrc.collect_central_timing_candidates(
            Path("dummy.flac"),
            entries,
            [10.0],
            report,
            20.0,
            type("Args", (), {})(),
        )
        self.assertEqual(assignment["timestamp"], 10.0)
        sources = {candidate.source for candidate in candidate_sets[0]}
        self.assertIn("ctc-current", sources)
        self.assertIn("ctc-local-window", sources)
        self.assertFalse(diagnostics["alignment_hypothesis_layer"]["authoritative_precentral_mutation"])  # type: ignore[index]

        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", candidate_sets, report, diagnostics
        )
        self.assertEqual(evaluated.state.decisions[0].candidate.source, "ctc-local-window")
        self.assertEqual(assignment["timestamp"], 10.0)
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        auto_lrc.commit_evaluated_backend(selection, report)
        self.assertEqual(assignment["timestamp"], 12.0)
        self.assertEqual(assignment["candidate_source"], "ctc-local-window")
        self.assertTrue(report["alignment_hypothesis_layer"]["central_commit_only"])  # type: ignore[index]

    def test_score_recommendation_cannot_control_review_or_move_backend_baseline(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["target"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {"entry": 2, "timestamp": 20.0, "score": 0.35, "review_required": True},
                {"entry": 3, "timestamp": 25.0, "score": 0.9},
            ],
            "suspicious_alignments": [
                {
                    "entry": 2,
                    "review_required": True,
                    "candidate_timestamps": {"raw_asr": 20.15},
                    "raw_asr_score": 0.96,
                }
            ],
        }
        result = score_line_timing_candidates(entries, [10.0, 20.0, 25.0], report, 30.0)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(result, [10.0, 20.0, 25.0])
        self.assertEqual(assignment["timestamp"], 20.0)
        self.assertFalse(assignment["review_required"])
        self.assertEqual(report["review_required_count"], 0)
        self.assertFalse(assignment["precentral_candidate_recommendation"]["authoritative"])

        actionable = copy.deepcopy(report)
        actionable["suspicious_alignments"][0]["flags"] = ["unresolved_raw_ctc_disagreement"]  # type: ignore[index]
        score_line_timing_candidates(entries, [10.0, 20.0, 25.0], actionable, 30.0)
        self.assertTrue(actionable["assignments"][1]["review_required"])  # type: ignore[index]
        self.assertEqual(actionable["review_required_count"], 1)

    def test_explicit_whisperx_late_onset_is_candidate_only(self) -> None:
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
        split_patch.assign("analyze_audio", lambda *_args, **_kwargs: features)  # type: ignore[assignment]
        try:
            report: dict[str, object] = {
                "backend": "whisperx",
                "whisperx_forced_first_times": [19.0, 24.0],
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 20.0,
                        "score": 0.75,
                        "timing_repair": "whisperx-word",
                    },
                    {
                        "entry": 2,
                        "timestamp": 25.0,
                        "score": 0.9,
                        "timing_repair": "whisperx-word",
                    },
                ],
            }
            original = [20.0, 25.0]
            result, changes = auto_lrc.apply_acoustic_late_onset_refinement(
                Path("dummy.flac"), 30.0, original, report
            )
            self.assertEqual(result, original)
            self.assertEqual(report["assignments"][0]["timestamp"], 20.0)  # type: ignore[index]
            self.assertEqual(len(changes), 1)
            self.assertFalse(changes[0]["authoritative"])
            hypotheses = report["assignments"][0]["alignment_hypotheses"]  # type: ignore[index]
            self.assertEqual(hypotheses[0]["source"], "whisperx-acoustic-late-onset")
            self.assertEqual(hypotheses[0]["raw_candidate_time"], 20.5)
            self.assertFalse(hypotheses[0]["authoritative"])
            self.assertFalse(report["acoustic_late_onset_refinement_authoritative"])
            self.assertFalse(report["acoustic_late_onset_refinement_mutates_baseline"])
        finally:
            split_patch.assign("analyze_audio", original_analyze_audio)  # type: ignore[assignment]

    def test_hybrid_ctc_derivation_preserves_base_candidate_by_copy(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_auto_backend_competition"
        )
        body = ast.get_source_segment(source, function) or ""
        self.assertIn("ctc_candidate = copy.deepcopy(base_ctc_candidate)", body)


class TailAwareCrossLineRetryTests(unittest.TestCase):
    def span(self, start: float, score: float = 0.20, token: str = "x") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        current: bool = True,
        direct: bool = False,
        direct_independent: bool = False,
        pileup: bool = False,
        spans: tuple[auto_lrc.TimingTokenSpan, ...] | None = None,
    ) -> auto_lrc.TimingCandidate:
        spans = spans or tuple(self.span(time + offset) for offset in (0.0, 0.08, 0.16))
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"line {entry}",
            source="ctc-current",
            raw_time=time,
            spans=spans,
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct else None,
            direct_onset_source_artifact={"entry": entry, "time": time} if direct else None,
            direct_onset_evidence_producer=(
                "phonetic-onset-fusion" if direct and direct_independent else None
            ),
            direct_onset_evidence_kind=(
                "phonetic-onset" if direct and direct_independent else "generic-onset"
            ),
            direct_onset_evidence_independent=bool(direct and direct_independent),
            current=current,
            right_edge_pileup=pileup,
            source_artifact={"entry": entry, "time": time},
        )

    def capability(self, backend: str = "ctc") -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": backend,
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

    def fixture(
        self,
        current_time: float = 9.90,
        *,
        next_time: float = 12.0,
    ) -> tuple[
        list[LyricEntry],
        tuple[tuple[auto_lrc.TimingCandidate, ...], ...],
        dict[str, object],
        auto_lrc.BoundedRetryCapability,
        auto_lrc.BaseRetryEpoch,
        tuple[auto_lrc.CrossLineRetryRequest, ...],
    ]:
        entries = [LyricEntry(["line zero"]), LyricEntry(["line one"]), LyricEntry(["line two"])]
        previous = self.candidate(0, 9.74)
        current = self.candidate(1, current_time)
        following = self.candidate(2, next_time)
        candidate_sets = ((previous,), (current,), (following,))
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 3,
            "assignments": [
                {"entry": index + 1, "timestamp": candidate.written_time.seconds, "timing_repair": "ctc", "score": 0.9}
                for index, candidate in enumerate((previous, current, following))
            ],
        }
        diagnostics = auto_lrc.inactive_legacy_timing_diagnostics()
        capability = self.capability()
        epoch = auto_lrc.freeze_base_retry_epoch(
            "ctc", report, candidate_sets, diagnostics, capability
        )
        requests = auto_lrc.plan_crossline_retry_requests(
            epoch, capability, entries, candidate_sets, 20.0
        )
        return entries, candidate_sets, report, capability, epoch, requests

    def helper_payload(
        self,
        request: auto_lrc.CrossLineRetryRequest,
        entries: list[LyricEntry],
        *,
        target_starts: tuple[float, ...] = (10.40, 10.48, 10.56),
        target_scores: tuple[float, ...] | None = None,
    ) -> dict[str, object]:
        target_scores = target_scores or tuple(0.20 for _ in target_starts)
        row_starts = {
            request.context_entry_indexes[0]: (request.window_start, request.window_start + 0.08, request.window_start + 0.16),
            request.entry_index: target_starts,
        }
        if request.context_entry_indexes[-1] != request.entry_index:
            assert request.next_bound is not None
            next_time = request.next_bound.canonical_centiseconds / 100.0
            row_starts[request.context_entry_indexes[-1]] = (next_time, next_time + 0.08, next_time + 0.16)
        rows: list[dict[str, object]] = []
        for local, global_index in enumerate(request.context_entry_indexes):
            starts = row_starts[global_index]
            scores = target_scores if global_index == request.entry_index else tuple(0.2 for _ in starts)
            spans = [
                {"char": chr(97 + index), "start": start, "end": start + 0.02, "score": scores[index]}
                for index, start in enumerate(starts)
            ]
            rows.append({
                "entry": local + 1,
                "text": auto_lrc.entry_sung_text(entries[global_index]),
                "romaji": "".join(item["char"] for item in spans),
                "start": spans[0]["start"],
                "end": spans[-1]["end"],
                "ctc_score": sum(scores) / len(scores),
                "tokens": len(spans),
                "token_spans": spans,
            })
        return {
            "window_start": round(request.window_start, 3),
            "window_end": round(request.window_end, 3),
            "entries": rows,
        }

    def test_01_clear_after_release_is_valid_and_never_retries(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture(10.50)
        decision = auto_lrc.build_final_timing_state(sets).decisions[1]
        self.assertEqual(decision.evidence.ownership.status, "clear")
        self.assertEqual(requests, ())

    def test_02_exact_tail_end_is_violation_and_plans_once(self) -> None:
        entries, sets, _report, capability, epoch, _ = self.fixture(9.92)
        requests = auto_lrc.plan_crossline_retry_requests(epoch, capability, entries, sets, 20.0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].ordinal, 1)

    def test_03_after_tail_inside_release_uncertainty_plans_bounded_retry(self) -> None:
        entries, sets, _report, _capability, _epoch, requests = self.fixture(10.00)
        decision = auto_lrc.build_final_timing_state(sets).decisions[1]
        self.assertEqual(decision.evidence.ownership.status, "unknown")
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(len(requests), 1)

    def test_04_strong_valid_current_is_protected_without_retry(self) -> None:
        entries, sets, _report, _capability, _epoch, requests = self.fixture(10.50)
        self.assertEqual(auto_lrc.build_final_timing_state(sets).decisions[1].selected_candidate_id, sets[1][0].candidate_id)
        self.assertEqual(requests, ())

    def test_05_unknown_with_bound_direct_onset_can_pass_without_retry(self) -> None:
        previous = self.candidate(0, 9.74, pileup=True)
        current = self.candidate(1, 12.0, direct=True, direct_independent=True)
        evaluation = auto_lrc.evaluate_timing_candidate(current, previous)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertTrue(evaluation.valid)

    def test_06_unknown_sequence_only_gets_recovery_attempt_not_trust_bypass(self) -> None:
        entries, sets, _report, _capability, _epoch, requests = self.fixture(10.00)
        self.assertEqual(auto_lrc.build_final_timing_state(sets).decisions[1].status, "provisional_unresolved")
        self.assertEqual(len(requests), 1)

    def test_07_overlap_request_is_content_derived_and_repeatable(self) -> None:
        entries, sets, _report, capability, epoch, first = self.fixture()
        second = auto_lrc.plan_crossline_retry_requests(epoch, capability, entries, sets, 20.0)
        self.assertEqual(first, second)
        self.assertEqual(first[0].request_revision, second[0].request_revision)

    def test_07b_candidate_permutation_preserves_epoch_request_calls_and_final_state(self) -> None:
        entries, sets, report, capability, epoch, first = self.fixture()
        extra = self.candidate(1, 9.80, current=False)
        first_sets = (sets[0], (*sets[1], extra), sets[2])
        second_sets = (sets[0], (extra, *sets[1]), sets[2])
        first_epoch = auto_lrc.freeze_base_retry_epoch("ctc", report, first_sets, auto_lrc.inactive_legacy_timing_diagnostics(), capability)
        second_epoch = auto_lrc.freeze_base_retry_epoch("ctc", report, second_sets, auto_lrc.inactive_legacy_timing_diagnostics(), capability)
        self.assertEqual(first_epoch, second_epoch)
        first_requests = auto_lrc.plan_crossline_retry_requests(first_epoch, capability, entries, first_sets, 20.0)
        second_requests = auto_lrc.plan_crossline_retry_requests(second_epoch, capability, entries, second_sets, 20.0)
        self.assertEqual(first_requests, second_requests)
        self.assertEqual(auto_lrc.build_final_timing_state(first_sets), auto_lrc.build_final_timing_state(second_sets))

    def test_08_retry_epoch_is_distinct_from_final_evaluation_input_revision(self) -> None:
        entries, sets, report, capability, epoch, requests = self.fixture()
        outcome = auto_lrc.parse_crossline_retry_output(requests[0], capability, entries, self.helper_payload(requests[0], entries))
        augmented = (sets[0], (*sets[1], outcome.candidate), sets[2])  # type: ignore[arg-type]
        evaluated = auto_lrc.evaluate_backend_timing("ctc", augmented, report, {"retry_epoch": epoch.retry_epoch_revision})
        self.assertNotEqual(epoch.retry_epoch_revision, evaluated.input_revision)

    def test_09_duplicate_and_reverse_schedule_invoke_at_most_once(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture()
        calls: list[str] = []
        original = auto_lrc.invoke_crossline_retry
        def fake(request, _capability, _entries):
            calls.append(request.request_revision)
            return auto_lrc.parse_crossline_retry_output(request, capability, entries, self.helper_payload(request, entries))
        split_patch.assign("invoke_crossline_retry", fake)  # type: ignore[assignment]
        try:
            outcomes = auto_lrc.execute_crossline_retry_epoch(epoch, capability, entries, sets, 20.0, tuple(reversed((*requests, *requests))))
        finally:
            split_patch.assign("invoke_crossline_retry", original)  # type: ignore[assignment]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(outcomes), 1)

    def test_10_stale_core_relation_dependencies_fail_closed(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture()
        request = requests[0]
        for stale in (
            replace(request, retry_epoch_revision="stale"),
            replace(request, current=replace(request.current, generation_revision="stale")),
            replace(request, previous=replace(request.previous, candidate_id="stale")),
            replace(request, previous_tail_evidence_revision="stale"),
            replace(request, ownership_relation_revision="stale"),
        ):
            self.assertFalse(auto_lrc.crossline_retry_request_is_current(stale, epoch, capability, entries, sets, 20.0))

    def test_11_stale_context_transcript_mapping_and_window_fail_closed(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture()
        request = requests[0]
        for stale in (
            replace(request, next_bound=replace(request.next_bound, candidate_id="stale")),  # type: ignore[arg-type]
            replace(request, context_candidate_ids=("stale",)),
            replace(request, transcript_digest="stale"),
            replace(request, row_mapping_revision="stale"),
            replace(request, window_end=request.window_end + 0.1),
        ):
            self.assertFalse(auto_lrc.crossline_retry_request_is_current(stale, epoch, capability, entries, sets, 20.0))

    def test_12_capability_is_runtime_authorized_for_ctc_and_carried_hybrid_only(self) -> None:
        with TemporaryDirectory() as name:
            path = Path(name) / "audio.bin"
            path.write_bytes(b"audio")
            args = auto_lrc.argparse.Namespace(whisperx_device="cpu")
            helper = Path(auto_lrc.__file__).resolve().parent / "ctc_align.py"
            seed = auto_lrc.create_bounded_retry_capability_seed(
                path, "vocal-stem", helper, sample_rate=16000, device="cpu"
            )
            metadata = {
                "ctc_backend": "mms_ctc",
                "ctc_audio_path": str(path),
                "ctc_audio_source": "vocal-stem",
                "ctc_sample_rate": 16000,
                "ctc_device": "cpu",
            }
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(
                    {"backend": "ctc", **metadata}, path, args
                )
            )
            # A serialized/dict seed is provenance only.  Metadata must not be
            # sufficient to mint execution capability, even when all hashes match.
            metadata_only_ctc: dict[str, object] = {
                "backend": "ctc",
                "ctc_bounded_retry_capability_seed": copy.deepcopy(seed),
            }
            metadata_only_hybrid: dict[str, object] = {
                "backend": "hybrid",
                "ctc_bounded_retry_capability_seed": copy.deepcopy(seed),
            }
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(metadata_only_ctc, path, args)
            )
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(metadata_only_hybrid, path, args)
            )

            ctc_report: dict[str, object] = {
                "backend": "ctc",
                "ctc_bounded_retry_capability_seed": copy.deepcopy(seed),
            }
            auto_lrc._attach_bounded_retry_runtime_capability(ctc_report, seed)
            self.assertIsNotNone(
                auto_lrc.build_bounded_retry_capability(ctc_report, path, args)
            )

            hybrid_report: dict[str, object] = {"backend": "hybrid"}
            auto_lrc.carry_bounded_retry_capability_seed(hybrid_report, ctc_report)
            self.assertIsNotNone(
                auto_lrc.build_bounded_retry_capability(hybrid_report, path, args)
            )
            tampered = copy.deepcopy(hybrid_report)
            tampered["ctc_bounded_retry_capability_seed"]["alignment_audio_sha256"] = "fake"  # type: ignore[index]
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(tampered, path, args)
            )
            for field_name, bad_value in (
                ("helper_protocol_revision", ""),
                ("helper_protocol_revision", "self-consistent-but-wrong"),
                ("model_identity", ""),
                ("model_identity", "other-model"),
                ("device", ""),
            ):
                incomplete_seed = copy.deepcopy(seed)
                incomplete_seed[field_name] = bad_value
                seed_payload = auto_lrc._bounded_retry_capability_semantic_payload(
                    incomplete_seed
                )
                incomplete_seed["capability_revision"] = auto_lrc._canonical_json_digest(seed_payload)
                incomplete_report: dict[str, object] = {
                    "backend": "ctc",
                    "ctc_bounded_retry_capability_seed": incomplete_seed,
                }
                auto_lrc._attach_bounded_retry_runtime_capability(
                    incomplete_report, incomplete_seed
                )
                self.assertIsNone(
                    auto_lrc.build_bounded_retry_capability(
                        incomplete_report, path, args
                    ),
                    field_name,
                )
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(
                    {
                        "backend": "whisperx",
                        "ctc_bounded_retry_capability_seed": copy.deepcopy(seed),
                        auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: ctc_report[
                            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY
                        ],
                    },
                    path,
                    args,
                )
            )

    def test_13_helper_wrong_rows_mapping_and_nonmonotonic_output_fail(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        variants = []
        missing = self.helper_payload(request, entries); missing["entries"] = missing["entries"][:-1]; variants.append(missing)
        wrong = self.helper_payload(request, entries); wrong["entries"][0]["entry"] = 2; variants.append(wrong)  # type: ignore[index]
        nonmonotonic = self.helper_payload(request, entries); nonmonotonic["entries"][0]["token_spans"][1]["start"] = 9.0; variants.append(nonmonotonic)  # type: ignore[index]
        for payload in variants:
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, payload).status, "invalid")

    def test_13b_same_length_token_substitution_and_reorder_fail(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        for mode in ("substitute", "reorder"):
            payload = self.helper_payload(request, entries)
            row = payload["entries"][0]  # type: ignore[index]
            if mode == "substitute":
                row["token_spans"][1]["char"] = "z"
            else:
                row["token_spans"][0]["char"], row["token_spans"][1]["char"] = row["token_spans"][1]["char"], row["token_spans"][0]["char"]
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, payload).status, "invalid", mode)

    def test_13c_nonfinite_score_and_nested_values_fail_closed(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        for value in (float("nan"), float("inf"), float("-inf")):
            payload = self.helper_payload(request, entries)
            payload["entries"][0]["ctc_score"] = value  # type: ignore[index]
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, payload).status, "invalid")
            nested = self.helper_payload(request, entries)
            nested["diagnostic"] = {"bad": value}
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, nested).status, "invalid")

    def test_13d_nonfinite_candidate_confidence_never_promotes_to_one(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            self.assertEqual(auto_lrc._candidate_confidence({"score": value}), 0.0)
            candidate = auto_lrc.make_timing_candidate(
                entry_index=0,
                entry_text="generic",
                source="synthetic-nonfinite",
                raw_time=12.0,
                spans=(self.span(12.0), self.span(12.08)),
                confidence=value,
                identity_support="supported",
                sequence_support="supported",
            )
            self.assertEqual(candidate.confidence, 0.0)

    def test_14_partial_target_row_prefix_suffix_interior_and_empty_fail(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        for mode in ("prefix", "suffix", "interior", "empty"):
            payload = self.helper_payload(request, entries)
            row = payload["entries"][0]  # type: ignore[index]
            spans = row["token_spans"]
            if mode == "empty":
                row["token_spans"] = []
            elif mode == "prefix":
                row["token_spans"] = spans[1:]
            elif mode == "suffix":
                row["token_spans"] = spans[:-1]
            else:
                row["token_spans"] = [spans[0], spans[-1]]
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, payload).status, "invalid", mode)

    def test_15_complete_target_row_touching_edge_is_truncated_and_rejected(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        starts = (request.window_end - 0.09, request.window_end - 0.06, request.window_end - 0.03)
        payload = self.helper_payload(request, entries, target_starts=starts)
        payload["entries"] = payload["entries"][:2]
        request = replace(request, context_entry_indexes=request.context_entry_indexes[:2], context_candidate_ids=request.context_candidate_ids[:2], transcript_digests=request.transcript_digests[:2], row_mapping=request.row_mapping[:2])
        outcome = auto_lrc.parse_crossline_retry_output(request, capability, entries, payload)
        self.assertEqual(outcome.status, "candidate")
        self.assertTrue(outcome.candidate.window_truncated)  # type: ignore[union-attr]
        self.assertFalse(auto_lrc.evaluate_timing_candidate(outcome.candidate).valid)  # type: ignore[arg-type]

    def test_16_right_edge_pileup_is_hard_rejected(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        starts = tuple(request.window_end - 0.09 + index * 0.01 for index in range(8))
        payload = self.helper_payload(request, entries, target_starts=starts, target_scores=tuple(0.001 for _ in starts))
        row = payload["entries"][0]  # type: ignore[index]
        for span in row["token_spans"]:
            span["end"] = span["start"] + 0.005
        row["end"] = row["token_spans"][-1]["end"]
        outcome = auto_lrc.parse_crossline_retry_output(request, capability, entries, payload)
        self.assertTrue(outcome.candidate.right_edge_pileup)  # type: ignore[union-attr]
        self.assertFalse(auto_lrc.evaluate_timing_candidate(outcome.candidate).valid)  # type: ignore[arg-type]

    def test_17_retry_still_overlapping_tail_is_rejected_without_second_attempt(self) -> None:
        entries, sets, _report, _capability, _epoch, requests = self.fixture()
        retry = self.candidate(1, 9.90, current=False)
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(*sets[1], retry),
            previous_candidate=sets[0][0],
            selection_revision=2,
        )
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(len(requests), 1)

    def test_18_retry_with_opening_fracture_is_rejected(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture(next_time=15.0)
        payload = self.helper_payload(requests[0], entries, target_starts=(10.4, 13.6, 13.68))
        outcome = auto_lrc.parse_crossline_retry_output(requests[0], capability, entries, payload)
        evaluation = auto_lrc.evaluate_timing_candidate(outcome.candidate)  # type: ignore[arg-type]
        self.assertFalse(evaluation.valid)
        self.assertIn("temporal-incoherence", evaluation.rejection_reasons)

    def test_18b_retry_strong_opening_late_tail_fracture_keeps_onset_not_tail(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture(next_time=15.0)
        payload = self.helper_payload(requests[0], entries, target_starts=(10.4, 10.48, 10.56, 13.8, 13.88))
        outcome = auto_lrc.parse_crossline_retry_output(requests[0], capability, entries, payload)
        evaluation = auto_lrc.evaluate_timing_candidate(outcome.candidate)  # type: ignore[arg-type]
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.fracture_region, "middle")
        self.assertEqual(auto_lrc.derive_reliable_tail(outcome.candidate).validity, "invalid")  # type: ignore[arg-type]

    def test_18c_missing_helper_returns_error_outcome_and_no_second_attempt(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture()
        original = auto_lrc.run_command
        split_patch.assign("run_command", lambda *_args, **_kwargs: (_ for _ in ()).throw(LrcError("missing python")))  # type: ignore[assignment]
        try:
            first = auto_lrc.execute_crossline_retry_epoch(epoch, capability, entries, sets, 20.0, requests)
        finally:
            split_patch.assign("run_command", original)  # type: ignore[assignment]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].status, "error")
        self.assertEqual(auto_lrc.build_final_timing_state(sets).decisions[1].status, "provisional_unresolved")

    def test_19_coherent_retry_after_release_can_replace_invalid_current(self) -> None:
        entries, sets, _report, capability, _epoch, requests = self.fixture()
        outcome = auto_lrc.parse_crossline_retry_output(requests[0], capability, entries, self.helper_payload(requests[0], entries))
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(*sets[1], outcome.candidate),  # type: ignore[arg-type]
            previous_candidate=sets[0][0],
            selection_revision=2,
        )
        self.assertEqual(decision.selected_candidate_id, outcome.candidate.candidate_id)  # type: ignore[union-attr]

    def test_20_late_tail_fracture_preserves_onset_but_removes_tail_authority(self) -> None:
        candidate = self.candidate(0, 10.0, spans=(self.span(10.0), self.span(10.08), self.span(10.16), self.span(14.0), self.span(14.08)))
        self.assertTrue(auto_lrc.evaluate_timing_candidate(candidate).valid)
        self.assertEqual(auto_lrc.derive_reliable_tail(candidate).validity, "invalid")

    def test_21_adjacent_violations_are_planned_once_from_frozen_epoch(self) -> None:
        entries = [LyricEntry([f"line {index}"]) for index in range(3)]
        sets = ((self.candidate(0, 9.74),), (self.candidate(1, 9.90),), (self.candidate(2, 10.02),))
        report = {"backend": "ctc", "assignments": [{"timing_repair": "ctc", "score": .9} for _ in entries]}
        cap = self.capability(); diagnostics = auto_lrc.inactive_legacy_timing_diagnostics()
        epoch = auto_lrc.freeze_base_retry_epoch("ctc", report, sets, diagnostics, cap)
        first = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, sets, 20.0)
        second = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, sets, 20.0)
        self.assertEqual(first, second)
        self.assertEqual(len({(item.retry_epoch_revision, item.entry_index) for item in first}), len(first))

    def test_22_legacy_stages_are_inactive_and_generic_contamination_stays_unknown(self) -> None:
        diagnostics = auto_lrc.inactive_legacy_timing_diagnostics()
        self.assertFalse(diagnostics["legacy_boundary_mutation_pipeline"]["executed"])  # type: ignore[index]
        self.assertFalse(diagnostics["legacy_prefix_probe_candidate_generation"]["authoritative"])  # type: ignore[index]
        contaminated = self.candidate(0, 10.0, pileup=True)
        current = self.candidate(1, 12.0)
        evaluation = auto_lrc.evaluate_timing_candidate(current, contaminated)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertFalse(evaluation.valid)

    def test_23_collect_does_not_execute_legacy_final_guard(self) -> None:
        entries = [LyricEntry(["generic"])]
        report: dict[str, object] = {"backend": "whisperx", "assignments": [{"timestamp": 12.0, "timing_repair": "word", "score": 0.9}]}
        original = auto_lrc.apply_final_ctc_timing_guard
        calls = 0
        def forbidden(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("inactive legacy final guard executed")
        split_patch.assign("apply_final_ctc_timing_guard", forbidden)  # type: ignore[assignment]
        try:
            _sets, diagnostics = auto_lrc.collect_central_timing_candidates(Path("missing.wav"), entries, [12.0], report, 20.0, auto_lrc.argparse.Namespace(whisperx_device="cpu"))
        finally:
            split_patch.assign("apply_final_ctc_timing_guard", original)  # type: ignore[assignment]
        self.assertEqual(calls, 0)
        self.assertEqual(diagnostics["legacy_prefix_probe_candidate_generation"]["active_candidate_count"], 0)  # type: ignore[index]

    def test_23b_collector_has_no_dormant_prefix_probe_candidate_construction(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        collector = next(item for item in module.body if isinstance(item, ast.FunctionDef) and item.name == "collect_central_timing_candidates")
        collector_source = ast.unparse(collector)
        self.assertNotIn("ctc-prefix-probe", collector_source)
        self.assertNotIn("probes_by_entry", collector_source)
        self.assertNotIn("prefix_assignment_items", collector_source)

    def test_24_production_source_has_zero_legacy_boundary_stage_callers(self) -> None:
        source = auto_lrc.algorithm_source_text()
        module = ast.parse(source)
        callers = []
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None)
            if name in {"apply_ctc_boundary_validation_stage", "apply_final_ctc_timing_guard"}:
                parent_function = next(
                    (item.name for item in module.body if isinstance(item, ast.FunctionDef) and node in ast.walk(item)),
                    "module",
                )
                callers.append(parent_function)
        self.assertEqual(callers, [])

    def test_25_inactive_legacy_console_never_claims_forward_prefix_was_applied(self) -> None:
        source = auto_lrc.algorithm_source_text()
        self.assertNotIn("CTC forward weak-prefix recovery applied", source)
        self.assertIn("CTC legacy forward weak-prefix evidence (inactive)", source)

    def test_26_runtime_retry_authority_is_not_serialized(self) -> None:
        with TemporaryDirectory() as name:
            report_path = Path(name) / "report.json"
            report: dict[str, object] = {
                "backend": "ctc",
                auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: self.capability(),
            }
            auto_lrc.write_alignment_report(report_path, report)
            serialized = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertNotIn(auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY, serialized)
            self.assertIn(auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY, report)


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
