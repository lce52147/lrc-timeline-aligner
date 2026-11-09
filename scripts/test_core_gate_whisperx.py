#!/usr/bin/env python3
"""Gate A, contract A, whisperx provenance/refine, capability identity and V14.5 tests."""

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


class GateASemanticFixedPointTests(unittest.TestCase):
    def capability(self) -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": "ctc",
            "helper_path": "ctc_align.py",
            "helper_sha256": "gate-a-helper",
            "helper_protocol_revision": "gate-a-protocol",
            "alignment_audio_path": "gate-a-audio.flac",
            "alignment_audio_sha256": "gate-a-audio-sha",
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

    def fixture(self, *, valid_previous: bool = False):
        entries = [LyricEntry(["generic predecessor"]), LyricEntry(["generic current"])]
        previous_spans = (
            (
                auto_lrc.TimingTokenSpan(9.0, 9.08, 0.8, "a"),
                auto_lrc.TimingTokenSpan(9.10, 9.18, 0.8, "b"),
                auto_lrc.TimingTokenSpan(9.20, 9.28, 0.8, "c"),
            )
            if valid_previous else ()
        )
        previous = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="generic predecessor",
            source="ctc-current" if valid_previous else "raw-vocal-independent-fusion",
            raw_time=9.0,
            spans=previous_spans,
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"gate": "A", "entry": 0},
        )
        current = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="generic current",
            # Keep the Gate-A fixed-point fixture's current opening disjoint
            # from the injected CTC tail; same-lineage ownership is covered by
            # Contract-A lineage tests below.
            source="raw-vocal-independent-fusion",
            raw_time=10.0,
            spans=(
                auto_lrc.TimingTokenSpan(10.0, 10.08, 0.8, "a"),
                auto_lrc.TimingTokenSpan(10.10, 10.18, 0.8, "b"),
            ),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"gate": "A", "entry": 1},
        )
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 9.0, "score": 0.9},
                {"entry": 2, "timestamp": 10.0, "score": 0.9},
            ],
        }
        capability = self.capability()
        report[auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY] = capability
        helper_revision = auto_lrc._tail_helper_revision(
            capability.helper_sha256,
            capability.helper_protocol_revision,
        )
        report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY] = auto_lrc.TailEvidenceContext(
            backend_lineage=capability.backend_lineage,
            model_identity=capability.model_identity,
            helper_revision=helper_revision,
            capability_revision=capability.capability_revision,
            audio_revision=capability.alignment_audio_sha256,
            transcript_revisions=("gate-a-transcript",),
            row_revisions=("gate-a-row", "gate-a-weaker"),
            scopes=(auto_lrc.TailEvidenceScope(
                entry_index=0,
                producer="joint-ctc-boundary-previous-tail",
                backend_lineage=capability.backend_lineage,
                model_identity=capability.model_identity,
                helper_revision=helper_revision,
                capability_revision=capability.capability_revision,
                audio_revision=capability.alignment_audio_sha256,
                transcript_revisions=("gate-a-transcript",),
                row_revisions=("gate-a-row", "gate-a-weaker"),
            ),),
        )
        candidate_sets = ((previous,), (current,))
        return entries, candidate_sets, report, capability

    def tail(self, *, tail_end: float = 9.5, row_revision: str = "gate-a-row"):
        cap = self.capability()
        provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=8.9,
            tail_end=tail_end,
            uncertainty_seconds=0.05,
            confidence=0.9,
            producer="joint-ctc-boundary-previous-tail",
            backend_lineage="ctc",
            model_identity=cap.model_identity,
            helper_revision=auto_lrc._tail_helper_revision(
                cap.helper_sha256,
                cap.helper_protocol_revision,
            ),
            capability_revision=cap.capability_revision,
            audio_revision=cap.alignment_audio_sha256,
            transcript_revision="gate-a-transcript",
            row_revision=row_revision,
        )
        assert provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=8.9,
            tail_end=tail_end,
            uncertainty_seconds=0.05,
            confidence=0.9,
            producer="joint-ctc-boundary-previous-tail",
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )

    def run_synthetic(self, *, inject_tail=False, weaker=False):
        entries, candidate_sets, report, capability = self.fixture()
        diagnostics = auto_lrc.inactive_legacy_timing_diagnostics()
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, len(entries), context)
        passes: list[dict[str, object]] = []
        for pass_index in range(1, 5):
            before = auto_lrc._retry_semantic_state_revision(candidate_sets, tails, context)
            epoch = auto_lrc.freeze_base_retry_epoch(
                "ctc", report, candidate_sets, diagnostics, capability
            )
            requests = auto_lrc.plan_crossline_retry_requests(
                epoch, capability, entries, candidate_sets, 20.0, tails, context
            )
            if not requests:
                passes.append({"pass": pass_index, "requests": 0, "stable": True})
                break
            if inject_tail and pass_index == 1:
                auto_lrc.merge_runtime_independent_tail_evidence(
                    report, self.tail()
                )
                if weaker:
                    auto_lrc.merge_runtime_independent_tail_evidence(
                        report, self.tail(tail_end=9.2, row_revision="gate-a-weaker")
                    )
                tails = auto_lrc.runtime_independent_tail_evidence(report, len(entries), context)
            after = auto_lrc._retry_semantic_state_revision(candidate_sets, tails, context)
            stable = before == after
            passes.append({"pass": pass_index, "requests": len(requests), "stable": stable})
            if stable:
                break
        return passes, tails

    def test_a_tail_change_re_evaluates_without_candidate(self) -> None:
        passes, tails = self.run_synthetic(inject_tail=True)
        self.assertEqual([item["requests"] for item in passes], [1, 0])
        self.assertIsNotNone(tails[0])

    def test_b_tail_change_then_no_consequence_converges(self) -> None:
        passes, _tails = self.run_synthetic(inject_tail=True)
        self.assertEqual(passes[-1]["stable"], True)
        self.assertLessEqual(len(passes), 2)

    def test_c_zero_candidate_zero_tail_is_immediate(self) -> None:
        passes, _tails = self.run_synthetic(inject_tail=False)
        # The synthetic unresolved boundary has one request, but no evidence
        # mutation is allowed to claim a fixed point before its planner pass.
        self.assertEqual(len(passes), 1)
        self.assertEqual(passes[0]["stable"], True)

    def test_d_identical_tail_revision_does_not_loop(self) -> None:
        passes, _tails = self.run_synthetic(inject_tail=True)
        self.assertLessEqual(len(passes), 2)

    def test_e_weaker_tail_does_not_churn_authoritative_state(self) -> None:
        passes, _tails = self.run_synthetic(inject_tail=True, weaker=True)
        self.assertEqual([item["requests"] for item in passes], [1, 0])


class ContractATailLineageTests(unittest.TestCase):
    def capability(self) -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": "ctc",
            "helper_path": "ctc_align.py",
            "helper_sha256": "contract-a-helper",
            "helper_protocol_revision": "contract-a-protocol",
            "alignment_audio_path": "contract-a-audio.flac",
            "alignment_audio_sha256": "contract-a-audio-sha",
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

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        spans: tuple[auto_lrc.TimingTokenSpan, ...] | None = None,
        current: bool = True,
    ) -> auto_lrc.TimingCandidate:
        if spans is None:
            spans = () if source.startswith("raw-") else (
                auto_lrc.TimingTokenSpan(time, time + 0.08, 0.8, "a"),
                auto_lrc.TimingTokenSpan(time + 0.10, time + 0.18, 0.8, "b"),
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"generic contract line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=current,
            source_artifact={"contract": "A", "entry": entry, "source": source},
        )

    def observation(
        self,
        *,
        producer: str = "joint-ctc-boundary-previous-tail",
        backend_lineage: str = "ctc",
        tail_end: float = 9.5,
        uncertainty: float = 0.05,
        model_identity: str | None = None,
        helper_revision: str | None = None,
        capability_revision: str | None = None,
        audio_revision: str | None = None,
        transcript_revision: str = "contract-transcript",
        row_revision: str = "contract-row",
    ) -> auto_lrc.IndependentTailEvidence:
        cap = self.capability()
        provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=8.9,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=0.9,
            producer=producer,
            backend_lineage=backend_lineage,
            model_identity=model_identity or cap.model_identity,
            helper_revision=helper_revision or auto_lrc._tail_helper_revision(
                cap.helper_sha256,
                cap.helper_protocol_revision,
            ),
            capability_revision=capability_revision or cap.capability_revision,
            audio_revision=audio_revision or cap.alignment_audio_sha256,
            transcript_revision=transcript_revision,
            row_revision=row_revision,
        )
        assert provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=8.9,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=0.9,
            producer=producer,
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )

    def report(self, observations=()) -> dict[str, object]:
        cap = self.capability()
        helper_revision = auto_lrc._tail_helper_revision(
            cap.helper_sha256,
            cap.helper_protocol_revision,
        )
        report: dict[str, object] = {
            "backend": "ctc",
            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: cap,
            auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY: auto_lrc.TailEvidenceContext(
                backend_lineage=cap.backend_lineage,
                model_identity=cap.model_identity,
                helper_revision=helper_revision,
                capability_revision=cap.capability_revision,
                audio_revision=cap.alignment_audio_sha256,
                transcript_revisions=("contract-transcript",),
                row_revisions=("contract-row",),
                producer_contexts=(
                    (
                        "whisper-family",
                        cap.model_identity,
                        helper_revision,
                        cap.capability_revision,
                        cap.alignment_audio_sha256,
                    ),
                ),
                scopes=(
                    auto_lrc.TailEvidenceScope(
                        entry_index=0,
                        producer="joint-ctc-boundary-previous-tail",
                        backend_lineage=cap.backend_lineage,
                        model_identity=cap.model_identity,
                        helper_revision=helper_revision,
                        capability_revision=cap.capability_revision,
                        audio_revision=cap.alignment_audio_sha256,
                        transcript_revisions=("contract-transcript",),
                        row_revisions=("contract-row",),
                    ),
                    auto_lrc.TailEvidenceScope(
                        entry_index=0,
                        producer="whisperx-forced-row",
                        backend_lineage="whisper-family",
                        model_identity=cap.model_identity,
                        helper_revision=helper_revision,
                        capability_revision=cap.capability_revision,
                        audio_revision=cap.alignment_audio_sha256,
                        transcript_revisions=("contract-transcript",),
                        row_revisions=("contract-row",),
                    ),
                ),
            ),
        }
        if observations:
            report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = {0: tuple(observations)}
        return report

    def state(
        self,
        predecessor: auto_lrc.TimingCandidate,
        observations=(),
        current: auto_lrc.TimingCandidate | None = None,
    ):
        current = current or self.candidate(1, 10.0)
        report = self.report(observations)
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
        return auto_lrc.build_final_timing_state(
            ((predecessor,), (current,)),
            independent_tail_evidence=tails,
            tail_evidence_context=context,
        )

    def direct_candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str,
        producer: str,
        kind: str = "cross-backend-onset",
        current: bool = True,
    ) -> auto_lrc.TimingCandidate:
        occurrence_entries = [
            LyricEntry([f"direct line {index}"])
            for index in range(max(2, entry + 1))
        ]
        occurrence_assignments = [
            {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
            for index in range(len(occurrence_entries))
        ]
        binding = auto_lrc._entry_occurrence_binding(
            occurrence_entries, occurrence_assignments, entry
        )
        occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            occurrence_entries,
            entry,
            producer=producer,
            occurrence_binding=binding,
            upstream_evidence_revision="contract-direct-upstream",
        )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"direct line {entry}",
            source=source,
            raw_time=time,
            spans=(),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported",
            direct_onset_time=time,
            direct_onset_source_artifact={
                "entry": entry,
                "time": time,
                "producer": producer,
            },
            direct_onset_evidence_producer=producer,
            direct_onset_evidence_kind=kind,
            direct_onset_evidence_independent=True,
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=True,
            current=current,
            source_artifact={"entry": entry, "time": time, "source": source},
        )

    def test_a_same_lineage_is_not_authoritative(self) -> None:
        predecessor = self.candidate(0, 9.0, spans=())
        observation = self.observation()
        state = self.state(predecessor, (observation,))
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")
        self.assertFalse(state.audits[1].timing_trusted)

    def test_b_disjoint_lineage_is_authoritative(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        current = self.direct_candidate(
            1,
            10.0,
            source="raw-vocal-independent-fusion",
            producer="raw-asr-vocal-fusion",
        )
        state = self.state(predecessor, (observation,), current=current)
        self.assertEqual(state.decisions[1].evidence.ownership.status, "clear")
        self.assertTrue(state.audits[1].timing_trusted)

    def test_c_missing_provenance_is_diagnostic_only(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        missing = auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=8.9,
            tail_end=9.5,
            uncertainty_seconds=0.05,
            confidence=0.9,
            producer="joint-ctc-boundary-previous-tail",
            source_revision="missing-provenance",
        )
        state = self.state(predecessor, (missing,))
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")
        self.assertFalse(state.audits[1].timing_trusted)

    def test_d_stale_provenance_revisions_fail_closed(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        cap = self.capability()
        helper_revision = auto_lrc._tail_helper_revision(
            cap.helper_sha256,
            cap.helper_protocol_revision,
        )
        for field, value in (
            ("model_identity", "stale-model"),
            ("helper_revision", "stale-helper"),
            ("capability_revision", "stale-capability"),
            ("audio_revision", "stale-audio"),
        ):
            kwargs = {field: value}
            if field == "helper_revision":
                kwargs[field] = str(value)
            stale = self.observation(**kwargs)
            state = self.state(predecessor, (stale,))
            self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown", field)
            self.assertFalse(state.audits[1].timing_trusted, field)
        for field, value in (
            ("transcript_revision", "stale-transcript"),
            ("row_revision", "stale-row"),
        ):
            original = self.observation()
            assert original.provenance is not None
            tampered = replace(original.provenance, **{field: value})
            stale = replace(original, provenance=tampered)
            state = self.state(predecessor, (stale,))
            self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown", field)
            self.assertFalse(state.audits[1].timing_trusted, field)
        self.assertTrue(helper_revision)

    def test_direct_resolver_without_context_fails_closed(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        resolved = auto_lrc.resolve_reliable_tail(predecessor, observation)
        self.assertNotEqual(resolved.reason, "independent-joint-ctc-boundary-previous-tail")
        self.assertEqual(resolved.validity, "unknown")

    def test_self_consistent_stale_ctc_context_fails_runtime_and_direct(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        stale = self.observation(
            transcript_revision="stale-current-transcript",
            row_revision="stale-current-row",
        )
        report = self.report((stale,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        self.assertIsNone(
            auto_lrc.runtime_independent_tail_evidence(report, 2, context)[0]
        )
        resolved = auto_lrc.resolve_reliable_tail(
            predecessor, stale, tail_evidence_context=context
        )
        self.assertEqual(resolved.validity, "unknown")

    def test_self_consistent_non_ctc_context_cannot_auto_pass(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        non_ctc = self.observation(
            producer="whisperx-forced-row",
            backend_lineage="whisper-family",
            model_identity="whisper-model",
            helper_revision="whisper-helper",
            capability_revision="whisper-capability",
            audio_revision="whisper-audio",
        )
        report = self.report((non_ctc,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        self.assertIsNone(
            auto_lrc.runtime_independent_tail_evidence(report, 2, context)[0]
        )
        resolved = auto_lrc.resolve_reliable_tail(
            predecessor, non_ctc, tail_evidence_context=context
        )
        self.assertEqual(resolved.validity, "unknown")

    def test_serialized_retry_diagnostics_cannot_mint_current_tail_context(self) -> None:
        cap = self.capability()
        entries = [LyricEntry(["current immutable lyric"])]
        assignment = {
            "entry": 1,
            "timestamp": 9.0,
            "ctc_token_spans": [
                {"char": "a", "start": 9.0, "end": 9.08, "score": .8},
                {"char": "b", "start": 9.10, "end": 9.18, "score": .8},
                {"char": "c", "start": 9.20, "end": 9.28, "score": .8},
            ],
        }
        stale = self.observation(
            transcript_revision="serialized-stale-transcript",
            row_revision="serialized-stale-row",
        )
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [assignment],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: cap,
            auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY: {0: (stale,)},
            "joint_boundary_retry": {
                "requests": [{
                    "transcript_digest": "serialized-stale-transcript",
                    "row_mapping_revision": "serialized-stale-row",
                }],
            },
        }
        context = auto_lrc._tail_evidence_context_from_report(report)
        self.assertIsNotNone(context)
        assert context is not None
        self.assertNotIn("serialized-stale-transcript", context.transcript_revisions)
        self.assertNotIn("serialized-stale-row", context.row_revisions)
        self.assertIsNone(auto_lrc.runtime_independent_tail_evidence(report, 1, context)[0])

    def test_exact_entry_producer_scope_rejects_transplant_and_wrong_mapping(self) -> None:
        cap = self.capability()
        helper_revision = auto_lrc._tail_helper_revision(
            cap.helper_sha256,
            cap.helper_protocol_revision,
        )
        context = auto_lrc.TailEvidenceContext(
            backend_lineage=cap.backend_lineage,
            model_identity=cap.model_identity,
            helper_revision=helper_revision,
            capability_revision=cap.capability_revision,
            audio_revision=cap.alignment_audio_sha256,
            transcript_revisions=("entry-a-transcript", "entry-b-transcript"),
            row_revisions=("entry-a-row", "entry-b-row"),
            scopes=(
                auto_lrc.TailEvidenceScope(
                    entry_index=0,
                    producer="joint-ctc-boundary-previous-tail",
                    backend_lineage=cap.backend_lineage,
                    model_identity=cap.model_identity,
                    helper_revision=helper_revision,
                    capability_revision=cap.capability_revision,
                    audio_revision=cap.alignment_audio_sha256,
                    transcript_revisions=("entry-a-transcript",),
                    row_revisions=("entry-a-row",),
                ),
                auto_lrc.TailEvidenceScope(
                    entry_index=1,
                    producer="joint-ctc-boundary-previous-tail",
                    backend_lineage=cap.backend_lineage,
                    model_identity=cap.model_identity,
                    helper_revision=helper_revision,
                    capability_revision=cap.capability_revision,
                    audio_revision=cap.alignment_audio_sha256,
                    transcript_revisions=("entry-b-transcript",),
                    row_revisions=("entry-b-row",),
                ),
            ),
        )
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        transplant = self.observation(
            transcript_revision="entry-b-transcript",
            row_revision="entry-b-row",
        )
        wrong_producer = self.observation(
            producer="ctc-assignment-terminal-row",
            transcript_revision="entry-a-transcript",
            row_revision="entry-a-row",
        )
        for observation in (transplant, wrong_producer):
            report = {
                "backend": "ctc",
                auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: cap,
                auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY: context,
                auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY: {0: (observation,)},
            }
            self.assertIsNone(
                auto_lrc.runtime_independent_tail_evidence(report, 1, context)[0]
            )
            resolved = auto_lrc.resolve_reliable_tail(
                predecessor, observation, tail_evidence_context=context
            )
            self.assertEqual(resolved.validity, "unknown")

    def test_e_same_lineage_recomputed_observation_is_rejected(self) -> None:
        predecessor = self.candidate(0, 9.0, spans=())
        first = self.observation(tail_end=9.5, row_revision="recompute-1")
        second = self.observation(tail_end=10.5, row_revision="recompute-2")
        state = self.state(predecessor, (first, second))
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")

    def test_f_only_disjoint_sidecar_can_change_unknown_to_clear(self) -> None:
        raw_predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        ctc_predecessor = self.candidate(0, 9.0, spans=())
        observation = self.observation()
        raw_state = self.state(
            raw_predecessor,
            (observation,),
            current=self.candidate(1, 10.0, source="raw-vocal-independent-fusion", spans=()),
        )
        ctc_state = self.state(ctc_predecessor, (observation,))
        self.assertEqual(raw_state.decisions[1].evidence.ownership.status, "clear")
        self.assertEqual(ctc_state.decisions[1].evidence.ownership.status, "unknown")

    def test_multi_observation_order_and_later_same_lineage_are_stable(self) -> None:
        predecessor = self.candidate(0, 9.0, spans=())
        cross_backend = self.observation(
            producer="whisperx-forced-row",
            backend_lineage="whisper-family",
            tail_end=9.7,
        )
        same_lineage_later = self.observation(tail_end=10.8, row_revision="later-same")
        for ordered in ((cross_backend, same_lineage_later), (same_lineage_later, cross_backend)):
            report = self.report(ordered)
            context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
            tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
            state = auto_lrc.build_final_timing_state(
                ((predecessor,), (self.candidate(1, 10.0),)),
                independent_tail_evidence=tails,
                tail_evidence_context=context,
            )
            self.assertEqual(state.decisions[1].evidence.ownership.status, "clear")
            self.assertAlmostEqual(
                state.decisions[1].evidence.ownership.reliable_tail_end or 0.0,
                9.7,
                places=3,
            )

    def test_cross_backend_tail_cannot_clear_inside_supported_predecessor_continuation(self) -> None:
        predecessor = self.candidate(
            0,
            9.0,
            spans=(
                auto_lrc.TimingTokenSpan(9.00, 9.08, 0.80, "a"),
                auto_lrc.TimingTokenSpan(9.10, 9.18, 0.80, "b"),
                auto_lrc.TimingTokenSpan(9.80, 9.88, 0.90, "c"),
                auto_lrc.TimingTokenSpan(9.92, 10.00, 0.90, "d"),
                auto_lrc.TimingTokenSpan(10.04, 10.12, 0.90, "e"),
                auto_lrc.TimingTokenSpan(10.14, 10.22, 0.01, "f"),
            ),
        )
        cross_backend = self.observation(
            producer="whisperx-forced-row",
            backend_lineage="whisper-family",
            tail_end=9.50,
            uncertainty=0.05,
        )
        current = self.candidate(1, 9.92, source="ctc-opening-local-retry")
        state = self.state(predecessor, (cross_backend,), current=current)
        decision = state.decisions[1]
        self.assertEqual(decision.evidence.ownership.status, "unknown")
        self.assertFalse(state.audits[1].timing_trusted)

    def test_single_attached_weak_terminal_token_keeps_local_window_tail_usable(self) -> None:
        predecessor = self.candidate(
            0,
            1.24,
            source="ctc-local-window",
            spans=(
                auto_lrc.TimingTokenSpan(1.241, 1.261, 0.796901, "s"),
                auto_lrc.TimingTokenSpan(1.321, 1.341, 0.473394, "o"),
                auto_lrc.TimingTokenSpan(1.761, 1.781, 0.843175, "e"),
                auto_lrc.TimingTokenSpan(1.921, 1.941, 0.406651, "v"),
                auto_lrc.TimingTokenSpan(1.941, 1.961, 0.007228, "e"),
                auto_lrc.TimingTokenSpan(1.961, 1.981, 0.924431, "r"),
                auto_lrc.TimingTokenSpan(2.001, 2.021, 0.377052, "y"),
                auto_lrc.TimingTokenSpan(2.081, 2.101, 0.041817, "t"),
                auto_lrc.TimingTokenSpan(2.101, 2.121, 0.029486, "h"),
                auto_lrc.TimingTokenSpan(2.181, 2.201, 0.856136, "i"),
                auto_lrc.TimingTokenSpan(2.321, 2.341, 0.909203, "n"),
                auto_lrc.TimingTokenSpan(2.361, 2.381, 0.480541, "g"),
                auto_lrc.TimingTokenSpan(2.441, 2.461, 0.045536, "t"),
                auto_lrc.TimingTokenSpan(2.481, 2.501, 0.021416, "h"),
                auto_lrc.TimingTokenSpan(2.541, 2.561, 0.944563, "a"),
                auto_lrc.TimingTokenSpan(2.741, 2.761, 0.876551, "t"),
                auto_lrc.TimingTokenSpan(2.821, 2.841, 0.920501, "m"),
                auto_lrc.TimingTokenSpan(2.901, 2.921, 0.909942, "a"),
                auto_lrc.TimingTokenSpan(3.041, 3.061, 0.822256, "k"),
                auto_lrc.TimingTokenSpan(3.082, 3.102, 0.853498, "e"),
                auto_lrc.TimingTokenSpan(3.122, 3.142, 0.831668, "s"),
                auto_lrc.TimingTokenSpan(3.182, 3.202, 0.926002, "m"),
                auto_lrc.TimingTokenSpan(3.262, 3.282, 0.482192, "e"),
                auto_lrc.TimingTokenSpan(3.522, 3.542, 0.012802, "w"),
                auto_lrc.TimingTokenSpan(3.542, 3.562, 0.049645, "h"),
                auto_lrc.TimingTokenSpan(3.662, 3.682, 0.548348, "o"),
                auto_lrc.TimingTokenSpan(3.962, 3.982, 0.900118, "l"),
                auto_lrc.TimingTokenSpan(4.002, 4.022, 0.008684, "e"),
            ),
        )
        tail = auto_lrc.derive_reliable_tail(predecessor)
        self.assertEqual(tail.validity, "valid")
        self.assertLessEqual(
            (tail.reliable_tail_end or 99.0) + (tail.uncertainty_seconds or 0.0),
            4.10,
        )
        current = self.candidate(1, 4.16, source="ctc-local-window")
        evaluation = auto_lrc.evaluate_timing_candidate(
            current,
            predecessor,
            previous_tail=tail,
        )
        self.assertEqual(evaluation.evidence.ownership.status, "clear")

    def test_detached_low_reliability_local_window_terminal_ghost_does_not_poison_tail(self) -> None:
        predecessor = self.candidate(
            0,
            4.16,
            source="ctc-local-window",
            spans=(
                auto_lrc.TimingTokenSpan(4.162, 4.182, 0.705, "i"),
                auto_lrc.TimingTokenSpan(4.422, 4.442, 0.911, "m"),
                auto_lrc.TimingTokenSpan(4.522, 4.542, 0.904, "a"),
                auto_lrc.TimingTokenSpan(4.762, 4.782, 0.514, "k"),
                auto_lrc.TimingTokenSpan(4.782, 4.802, 0.015, "u"),
                auto_lrc.TimingTokenSpan(4.942, 4.963, 0.010, "n"),
                auto_lrc.TimingTokenSpan(5.323, 5.343, 0.283, "n"),
                auto_lrc.TimingTokenSpan(5.423, 5.443, 0.584, "i"),
                auto_lrc.TimingTokenSpan(5.643, 5.663, 0.614, "s"),
                auto_lrc.TimingTokenSpan(5.763, 5.783, 0.931, "a"),
                auto_lrc.TimingTokenSpan(5.983, 6.003, 0.837, "s"),
                auto_lrc.TimingTokenSpan(6.083, 6.103, 0.858, "a"),
                auto_lrc.TimingTokenSpan(6.203, 6.223, 0.697, "g"),
                auto_lrc.TimingTokenSpan(6.303, 6.323, 0.428, "e"),
                auto_lrc.TimingTokenSpan(6.723, 6.743, 0.064, "y"),
                auto_lrc.TimingTokenSpan(6.864, 6.884, 0.210, "o"),
                auto_lrc.TimingTokenSpan(12.046, 12.066, 0.064, "o"),
            ),
        )
        tail = auto_lrc.derive_reliable_tail(predecessor)
        self.assertEqual(tail.validity, "valid")
        self.assertLess(tail.reliable_tail_end or 99.0, 7.0)

        current = self.candidate(1, 12.17, source="ctc-local-window")
        evaluation = auto_lrc.evaluate_timing_candidate(
            current,
            predecessor,
            previous_tail=tail,
        )
        self.assertEqual(evaluation.evidence.ownership.status, "clear")

    def test_candidate_lineage_switch_recomputes_authority(self) -> None:
        observation = self.observation()
        report = self.report((observation,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
        raw_state = auto_lrc.build_final_timing_state(
            (
                (self.candidate(0, 9.0, source="raw-vocal-independent-fusion"),),
                (self.candidate(1, 10.0, source="raw-vocal-independent-fusion", spans=()),),
            ),
            independent_tail_evidence=tails,
            tail_evidence_context=context,
        )
        ctc_state = auto_lrc.build_final_timing_state(
            ((self.candidate(0, 9.0, spans=()),), (self.candidate(1, 10.0),)),
            independent_tail_evidence=tails,
            tail_evidence_context=context,
        )
        self.assertEqual(raw_state.decisions[1].evidence.ownership.status, "clear")
        self.assertEqual(ctc_state.decisions[1].evidence.ownership.status, "unknown")

    def test_unknown_candidate_family_fails_closed(self) -> None:
        predecessor = self.candidate(0, 9.0, source="synthetic-unknown", spans=())
        state = self.state(predecessor, (self.observation(),))
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")

    def test_current_same_lineage_tail_cannot_clear_or_violate(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        later = self.state(predecessor, (observation,))
        self.assertEqual(
            later.decisions[1].evidence.ownership.reason,
            "external-tail-current-lineage-overlap",
        )
        self.assertEqual(later.decisions[1].evidence.ownership.status, "unknown")

        earlier = self.state(
            predecessor,
            (observation,),
            current=self.candidate(1, 9.0, source="ctc-current", current=True),
        )
        self.assertEqual(earlier.decisions[1].evidence.ownership.status, "unknown")
        self.assertEqual(
            earlier.decisions[1].evidence.ownership.reason,
            "external-tail-current-lineage-overlap",
        )

    def test_composite_overlap_requires_disjoint_direct_producer(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        composite = self.state(
            predecessor,
            (observation,),
            current=self.candidate(
                1,
                10.0,
                source="ctc-local-whisper-vocal-independent-consensus",
            ),
        )
        self.assertEqual(composite.decisions[1].evidence.ownership.status, "unknown")
        direct_exception = self.state(
            predecessor,
            (observation,),
            current=self.direct_candidate(
                1,
                10.0,
                source="ctc-local-whisper-vocal-independent-consensus",
                producer="local-whisper-vocal-fusion",
            ),
        )
        self.assertEqual(direct_exception.decisions[1].evidence.ownership.status, "clear")
        self.assertTrue(direct_exception.audits[1].timing_trusted)

    def test_same_lineage_direct_onset_is_not_the_exception(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        state = self.state(
            predecessor,
            (observation,),
            current=self.direct_candidate(
                1,
                10.0,
                source="ctc-current",
                producer="joint-ctc-boundary-consensus",
                kind="joint-lyric-boundary",
            ),
        )
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")
        self.assertEqual(
            state.decisions[1].evidence.ownership.reason,
            "external-tail-current-lineage-overlap",
        )
        self.assertFalse(state.audits[1].timing_trusted)

    def test_59_like_ctc_tail_can_select_disjoint_whisper_opening(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        report = self.report((observation,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
        ctc_current = self.candidate(1, 10.0, source="ctc-current", current=True)
        whisper_opening = self.direct_candidate(
            1,
            10.1,
            source="local-whisper-vocal-independent-fusion",
            producer="local-whisper-vocal-fusion",
            current=False,
        )
        state = auto_lrc.build_final_timing_state(
            ((predecessor,), (ctc_current, whisper_opening)),
            independent_tail_evidence=tails,
            tail_evidence_context=context,
        )
        self.assertEqual(
            next(item for item in state.decisions if item.entry_index == 1)
            .candidate.source,
            "local-whisper-vocal-independent-fusion",
        )
        self.assertEqual(
            next(item for item in state.decisions if item.entry_index == 1)
            .candidate.raw_time,
            10.1,
        )
        self.assertEqual(ctc_current.raw_time, 10.0)

    def test_same_lineage_material_alternative_remains_reviewable(self) -> None:
        predecessor = self.candidate(0, 8.0, source="ctc-current", spans=())
        observation = self.observation(
            producer="whisperx-forced-row",
            backend_lineage="whisper-family",
        )
        report = self.report((observation,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
        current = self.direct_candidate(
            1,
            10.0,
            source="local-whisper-vocal-independent-fusion",
            producer="local-whisper-vocal-fusion",
            current=True,
        )
        distant = self.direct_candidate(
            1,
            9.0,
            source="whisperx-vocal-independent-fusion",
            producer="whisperx-vocal-fusion",
            current=False,
        )
        previous_tail = auto_lrc.resolve_reliable_tail(
            predecessor,
            tails[0],
            tail_evidence_context=context,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(current, distant),
            previous_candidate=predecessor,
            previous_tail=previous_tail,
            selection_revision=2,
        )
        self.assertTrue(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )


class WhisperXRuntimeProvenanceTests(unittest.TestCase):
    def provenance(self, **overrides) -> auto_lrc.WhisperXRuntimeProvenance:
        payload = {
            "backend_lineage": "whisper-family",
            "raw_executable_sha256": "raw-executable",
            "raw_model_sha256": "raw-model",
            "python_sha256": "python",
            "helper_sha256": "helper",
            "helper_protocol_revision": auto_lrc._WHISPERX_HELPER_PROTOCOL_REVISION,
            "whisperx_package_version": "3.8.6",
            "align_model_id": "model-id",
            "align_model_revision": "model-revision",
            "align_model_blob_sha256": "model-blob",
            "source_audio_sha256": "source-audio",
            "alignment_wav_sha256": "alignment-wav",
            "device": "cuda",
            "language": "ja",
            "raw_transcript_revision": "raw-transcript",
            "initial_aligned_output_revision": "initial-output",
            "forced_output_revision": "forced-output",
        }
        payload.update(overrides)
        return auto_lrc.WhisperXRuntimeProvenance(
            **payload,
            capability_revision=auto_lrc._timing_content_digest(payload),
        )

    def scope(self, provenance: auto_lrc.WhisperXRuntimeProvenance) -> auto_lrc.TailEvidenceScope:
        model_identity = (
            f"{provenance.align_model_id}@{provenance.align_model_revision}#"
            f"{provenance.align_model_blob_sha256}"
        )
        helper_revision = auto_lrc._timing_content_digest({
            "helper_sha256": provenance.helper_sha256,
            "helper_protocol_revision": provenance.helper_protocol_revision,
        })
        return auto_lrc.TailEvidenceScope(
            entry_index=0,
            producer="whisperx-forced-row",
            backend_lineage="whisper-family",
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=provenance.capability_revision,
            audio_revision=provenance.alignment_wav_sha256,
            transcript_revisions=("row-transcript",),
            row_revisions=("row-revision",),
        )

    def context(self, provenance: auto_lrc.WhisperXRuntimeProvenance) -> auto_lrc.TailEvidenceContext:
        scope = self.scope(provenance)
        return auto_lrc.TailEvidenceContext(
            backend_lineage="ctc",
            model_identity="ctc-model",
            helper_revision="ctc-helper",
            capability_revision="ctc-capability",
            audio_revision="ctc-audio",
            transcript_revisions=("row-transcript",),
            row_revisions=("row-revision",),
            scopes=(scope,),
        )

    def observation(self, provenance: auto_lrc.WhisperXRuntimeProvenance) -> auto_lrc.IndependentTailEvidence:
        scope = self.scope(provenance)
        tail_provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=8.9,
            tail_end=9.5,
            uncertainty_seconds=.05,
            confidence=.9,
            producer="whisperx-forced-row",
            backend_lineage=scope.backend_lineage,
            model_identity=scope.model_identity,
            helper_revision=scope.helper_revision,
            capability_revision=scope.capability_revision,
            audio_revision=scope.audio_revision,
            transcript_revision="row-transcript",
            row_revision="row-revision",
        )
        assert tail_provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=8.9,
            tail_end=9.5,
            uncertainty_seconds=.05,
            confidence=.9,
            producer="whisperx-forced-row",
            source_revision=tail_provenance.observation_revision,
            provenance=tail_provenance,
        )

    def test_missing_runtime_fields_fail_closed(self) -> None:
        fields = tuple(auto_lrc._whisperx_runtime_provenance_payload(self.provenance()))
        for field in fields:
            self.assertFalse(
                auto_lrc._whisperx_runtime_provenance_is_valid(
                    self.provenance(**{field: ""})
                ),
                field,
            )

    def test_self_consistent_stale_runtime_context_fails_scope_match(self) -> None:
        original = self.provenance()
        for field, value in (
            ("raw_executable_sha256", "stale-executable"),
            ("raw_model_sha256", "stale-model-file"),
            ("python_sha256", "stale-python"),
            ("helper_sha256", "stale-helper"),
            ("align_model_revision", "stale-model-revision"),
            ("align_model_blob_sha256", "stale-model-blob"),
            ("source_audio_sha256", "stale-source-audio"),
            ("alignment_wav_sha256", "stale-alignment-wav"),
            ("raw_transcript_revision", "stale-raw-transcript"),
            ("initial_aligned_output_revision", "stale-initial-output"),
            ("forced_output_revision", "stale-forced-output"),
        ):
            stale_base = replace(original, **{field: value, "capability_revision": ""})
            stale_payload = auto_lrc._whisperx_runtime_provenance_payload(stale_base)
            stale = replace(
                stale_base,
                capability_revision=auto_lrc._timing_content_digest(stale_payload),
            )
            self.assertTrue(auto_lrc._whisperx_runtime_provenance_is_valid(stale), field)
            observation = self.observation(stale)
            self.assertFalse(
                auto_lrc._tail_provenance_matches_context(observation, self.context(original)),
                field,
            )

    def test_binder_populates_exact_scope_without_timestamp_mutation(self) -> None:
        provenance = self.provenance()
        entries = [LyricEntry(["unique target"]), LyricEntry(["next line"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "segment": 1, "score": .96, "timestamp": 9.0, "borrowed": False},
                {"entry": 2, "segment": 2, "score": .96, "timestamp": 10.0, "borrowed": False},
            ],
            "whisperx_forced_rows": [
                {"segment_start": 8.9, "segment_end": 9.5, "first_char_time": 8.9, "last_char_time": 9.45, "char_count": 3},
                {"segment_start": 10.0, "segment_end": 10.5, "first_char_time": 10.0, "last_char_time": 10.45, "char_count": 3},
            ],
            auto_lrc._RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY: provenance,
        }
        before = [item["timestamp"] for item in report["assignments"]]  # type: ignore[index]
        auto_lrc.bind_runtime_whisperx_row_evidence(
            report, entries, [([], report, "synthetic-error")],
        )
        runtime = report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY]
        self.assertIsNotNone(runtime[0][0].provenance)  # type: ignore[index]
        self.assertEqual(
            report[auto_lrc._RUNTIME_WHISPERX_TAIL_SCOPES_KEY][0].producer,  # type: ignore[index]
            "whisperx-forced-row",
        )
        self.assertEqual(
            [item["timestamp"] for item in report["assignments"]],  # type: ignore[index]
            before,
        )

    def test_ambiguous_occurrence_is_diagnostic_only(self) -> None:
        provenance = self.provenance()
        entries = [LyricEntry(["repeat line"]), LyricEntry(["repeat line"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "segment": 1, "score": .96, "timestamp": 9.0, "borrowed": False},
                {"entry": 2, "segment": 1, "score": .96, "timestamp": 9.0, "borrowed": False},
            ],
            "whisperx_forced_rows": [
                {"segment_start": 8.9, "segment_end": 9.5, "first_char_time": 8.9, "last_char_time": 9.45, "char_count": 3},
                {"segment_start": 9.0, "segment_end": 9.6, "first_char_time": 9.0, "last_char_time": 9.55, "char_count": 3},
            ],
            auto_lrc._RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY: provenance,
        }
        auto_lrc.bind_runtime_whisperx_row_evidence(
            report, entries, [([], report, "synthetic-error")]
        )
        runtime = report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY]
        self.assertNotIn(0, runtime)
        self.assertNotIn(1, runtime)

    def test_row_output_and_occurrence_revision_tamper_fails_closed(self) -> None:
        provenance = self.provenance()
        observation = self.observation(provenance)
        assert observation.provenance is not None
        tampered = replace(
            observation,
            provenance=replace(observation.provenance, row_revision="wrong-row"),
        )
        self.assertFalse(auto_lrc._tail_provenance_is_valid(tampered))
        self.assertFalse(
            auto_lrc._tail_provenance_matches_context(tampered, self.context(provenance))
        )

    def test_writer_strips_runtime_provenance_and_temp_artifacts(self) -> None:
        provenance = self.provenance()
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [],
            auto_lrc._RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY: provenance,
            auto_lrc._RUNTIME_WHISPERX_TEMP_PATHS_KEY: ("C:/temp/should-not-serialize",),
        }
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "report.json"
            auto_lrc.write_alignment_report(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn(auto_lrc._RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY, text)
        self.assertNotIn("should-not-serialize", text)


class WhisperXRefineContractTests(unittest.TestCase):
    def _load_helper(self, fake_whisperx: types.ModuleType, fake_hf: types.ModuleType):
        helper_path = Path(__file__).resolve().parent / "whisperx_refine.py"
        spec = importlib.util.spec_from_file_location("_whisperx_refine_contract_test", helper_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {"whisperx": fake_whisperx, "huggingface_hub": fake_hf},
        ):
            spec.loader.exec_module(module)
        return module

    def test_helper_emits_runtime_alignment_provenance_matching_auto_lrc_protocol(self) -> None:
        with TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            snapshot = temp / "models--fake" / "snapshots" / "revision-123"
            snapshot.mkdir(parents=True)
            config_path = snapshot / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"fake-model")

            fake_alignment = types.SimpleNamespace(
                DEFAULT_ALIGN_MODELS_HF={"ja": "fake/alignment-model"},
                DEFAULT_ALIGN_MODELS_TORCH={},
                torchaudio=types.SimpleNamespace(
                    pipelines=types.SimpleNamespace(__all__=()),
                ),
            )
            fake_whisperx = types.ModuleType("whisperx")
            fake_whisperx.alignment = fake_alignment
            fake_whisperx.load_align_model = lambda **_: (object(), {"dictionary": "fake"})
            fake_whisperx.align = lambda transcript, *_args, **_kwargs: {
                "segments": transcript,
            }

            fake_hf = types.ModuleType("huggingface_hub")
            fake_hf.try_to_load_from_cache = lambda *_args, **_kwargs: str(config_path)

            helper = self._load_helper(fake_whisperx, fake_hf)
            audio_path = temp / "input.wav"
            audio_path.write_bytes(b"fake-audio")
            transcript_path = temp / "transcript.json"
            transcript = [{"start": 0.0, "end": 1.0, "text": "test"}]
            transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
            output_path = temp / "output.json"

            argv = [
                str(Path(__file__).resolve().parent / "whisperx_refine.py"),
                "--audio",
                str(audio_path),
                "--transcript",
                str(transcript_path),
                "--output",
                str(output_path),
                "--language",
                "ja",
                "--device",
                "cpu",
            ]
            with mock.patch.dict(
                sys.modules,
                {"whisperx": fake_whisperx, "huggingface_hub": fake_hf},
            ), mock.patch.object(sys, "argv", argv), mock.patch.object(
                importlib.metadata, "version", return_value="fake-whisperx-version"
            ):
                self.assertEqual(helper.main(), 0)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            provenance = payload[auto_lrc._RUNTIME_WHISPERX_ALIGNMENT_METADATA_KEY]
            self.assertEqual(
                provenance["helper_protocol_revision"],
                auto_lrc._WHISPERX_HELPER_PROTOCOL_REVISION,
            )
            self.assertEqual(provenance["align_model_id"], "fake/alignment-model")
            self.assertEqual(provenance["align_model_revision"], "revision-123")
            self.assertEqual(provenance["device"], "cpu")
            self.assertEqual(provenance["language"], "ja")
            self.assertTrue(provenance["helper_sha256"])
            self.assertTrue(provenance["python_sha256"])
            self.assertTrue(provenance["alignment_wav_sha256"])
            self.assertTrue(provenance["raw_transcript_revision"])

    def test_auto_lrc_missing_helper_provenance_fails_closed(self) -> None:
        payload = {"segments": []}
        self.assertIsNone(auto_lrc._whisperx_alignment_metadata(payload))
        with split_patch.patch("_whispercpp_runtime_artifact_hashes",
            return_value=("raw-executable", "raw-model"),
        ):
            self.assertIsNone(
                auto_lrc._build_whisperx_runtime_provenance(
                    Path("unused-audio.wav"),
                    types.SimpleNamespace(),
                    [],
                    payload,
                    payload,
                )
            )


class PathFreeCapabilityIdentityTests(unittest.TestCase):
    def ctc_seed(self, audio: Path) -> dict[str, object]:
        helper = Path(auto_lrc.__file__).resolve().parent / "ctc_align.py"
        return auto_lrc.create_bounded_retry_capability_seed(
            audio,
            "vocal-stem",
            helper,
            sample_rate=auto_lrc.SAMPLE_RATE,
            device="cpu",
        )

    def retry_request(self, capability_revision: str) -> auto_lrc.CrossLineRetryRequest:
        reference = auto_lrc.RetryCandidateReference(
            "candidate", "generation", "region", 100
        )
        return auto_lrc.CrossLineRetryRequest(
            retry_epoch_revision="epoch",
            entry_index=0,
            ordinal=1,
            capability_revision=capability_revision,
            alignment_audio_revision="audio-sha",
            current=reference,
            previous=reference,
            next_bound=None,
            audio_end_seconds=10.0,
            audio_duration_revision="duration",
            previous_tail_evidence_revision=None,
            previous_tail_start=None,
            previous_tail_end=None,
            previous_tail_uncertainty=None,
            previous_tail_validity="unknown",
            ownership_relation_revision="ownership",
            context_entry_indexes=(0,),
            context_candidate_ids=("candidate",),
            transcript_digests=("text",),
            transcript_digest="text",
            row_mapping=((0, 0),),
            row_mapping_revision="row",
            window_start=1.0,
            window_end=2.0,
            window_definition_revision="window",
            request_revision="request",
        )

    def test_same_bytes_different_paths_have_equal_ctc_local_tail_and_cache_identity(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            first = root / "first" / "stem.mp3"
            second = root / "second" / "stem.mp3"
            first.parent.mkdir(); second.parent.mkdir()
            first.write_bytes(b"identical-audio")
            second.write_bytes(b"identical-audio")
            runtime = root / "runtime"
            runtime.mkdir()
            whisper_cli = runtime / "whisper-cli.exe"
            whisper_model = runtime / "ggml-large-v3.bin"
            whisperx_python = runtime / "python.exe"
            whisper_cli.write_bytes(b"whisper-cli")
            whisper_model.write_bytes(b"whisper-model")
            whisperx_python.write_bytes(b"python")
            first_seed = self.ctc_seed(first)
            second_seed = self.ctc_seed(second)
            self.assertNotEqual(first_seed["alignment_audio_path"], second_seed["alignment_audio_path"])
            self.assertEqual(first_seed["capability_revision"], second_seed["capability_revision"])
            first_cap = auto_lrc.BoundedRetryCapability(**first_seed)
            second_cap = auto_lrc.BoundedRetryCapability(**second_seed)
            self.assertEqual(
                auto_lrc._tail_helper_revision(
                    first_cap.helper_sha256, first_cap.helper_protocol_revision
                ),
                auto_lrc._tail_helper_revision(
                    second_cap.helper_sha256, second_cap.helper_protocol_revision
                ),
            )
            self.assertEqual(
                auto_lrc._crossline_retry_helper_cache_key(
                    self.retry_request(first_cap.capability_revision), first_cap
                ),
                auto_lrc._crossline_retry_helper_cache_key(
                    self.retry_request(second_cap.capability_revision), second_cap
                ),
            )
            args_first = auto_lrc.build_parser().parse_args([str(first)])
            args_second = auto_lrc.build_parser().parse_args([str(second)])
            with (
                split_patch.patch("default_whisper_cli", return_value=whisper_cli),
                split_patch.patch("default_whisper_model", return_value=whisper_model),
                split_patch.patch("default_whisperx_python", return_value=whisperx_python),
            ):
                local_first = auto_lrc.build_local_whisper_capability(first, args_first)
                local_second = auto_lrc.build_local_whisper_capability(second, args_second)
            self.assertIsNotNone(local_first)
            self.assertIsNotNone(local_second)
            assert local_first is not None and local_second is not None
            self.assertNotEqual(local_first.audio_path, local_second.audio_path)
            self.assertEqual(local_first.capability_revision, local_second.capability_revision)
            request_first = auto_lrc.LocalWhisperRequest(
                0, local_first.capability_revision, local_first.audio_sha256,
                (0,), ("candidate",), ("text",), "text", 1.0, 2.0, "window", "request"
            )
            request_second = replace(
                request_first,
                capability_revision=local_second.capability_revision,
                audio_revision=local_second.audio_sha256,
            )
            self.assertEqual(
                auto_lrc._local_whisper_cache_key(request_first, local_first),
                auto_lrc._local_whisper_cache_key(request_second, local_second),
            )

    def test_local_whisper_cache_revision_ignores_planner_provenance(self) -> None:
        request = auto_lrc.LocalWhisperRequest(
            3, "capability", "audio", (3,), ("candidate-a",),
            ("text-a",), "text-a", 21.32, 29.54, "window-a", "request-a",
        )
        planner_only_change = replace(
            request,
            context_candidate_ids=("candidate-b",),
            window_definition_revision="window-b",
            request_revision="request-b",
        )
        self.assertEqual(
            auto_lrc._local_whisper_execution_request_revision(request.__dict__),
            auto_lrc._local_whisper_execution_request_revision(
                planner_only_change.__dict__
            ),
        )
        self.assertNotEqual(
            auto_lrc._local_whisper_execution_request_revision(request.__dict__),
            auto_lrc._local_whisper_execution_request_revision(
                replace(request, window_end=29.55).__dict__
            ),
        )

    def test_planner_and_execution_revisions_are_separate(self) -> None:
        payload = {
            "entry_index": 3,
            "capability_revision": "capability",
            "audio_revision": "audio",
            "context_entry_indexes": (3,),
            "context_candidate_ids": ("candidate-a",),
            "transcript_digests": ("text-a",),
            "transcript_digest": "text-a",
            "window_start": 21.32,
            "window_end": 29.54,
            "window_definition_revision": "window-a",
        }

        def make(candidate_id: str, window_revision: str) -> auto_lrc.LocalWhisperRequest:
            request_payload = {
                **payload,
                "context_candidate_ids": (candidate_id,),
                "window_definition_revision": window_revision,
            }
            return auto_lrc.LocalWhisperRequest(
                **request_payload,
                request_revision=auto_lrc._timing_content_digest({
                    **request_payload,
                    "context_entry_indexes": list(request_payload["context_entry_indexes"]),
                    "context_candidate_ids": list(request_payload["context_candidate_ids"]),
                    "transcript_digests": list(request_payload["transcript_digests"]),
                }),
                execution_revision=auto_lrc._local_whisper_execution_request_revision(
                    request_payload
                ),
            )

        first = make("candidate-a", "window-a")
        second = make("candidate-b", "window-b")
        self.assertNotEqual(first.request_revision, second.request_revision)
        self.assertEqual(first.execution_revision, second.execution_revision)
        capability = types.SimpleNamespace(
            capability_revision="capability", protocol_revision="protocol"
        )
        self.assertEqual(
            auto_lrc._local_whisper_cache_key(first, capability),
            auto_lrc._local_whisper_cache_key(second, capability),
        )

    def test_different_bytes_same_path_change_path_free_identity(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "stem.mp3"
            audio.write_bytes(b"first-bytes")
            first = self.ctc_seed(audio)
            audio.write_bytes(b"second-bytes")
            second = self.ctc_seed(audio)
        self.assertNotEqual(first["alignment_audio_sha256"], second["alignment_audio_sha256"])
        self.assertNotEqual(first["capability_revision"], second["capability_revision"])

    def test_missing_artifact_sha_fails_closed_and_paths_do_not_rescue(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "stem.mp3"
            audio.write_bytes(b"audio")
            seed = self.ctc_seed(audio)
            seed["alignment_audio_sha256"] = ""
            seed["capability_revision"] = auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(seed)
            )
            capability = auto_lrc.BoundedRetryCapability(**seed)
            report = {
                "backend": "ctc",
                "ctc_bounded_retry_capability_seed": seed,
                auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: capability,
            }
            args = auto_lrc.build_parser().parse_args([str(audio)])
            self.assertIsNone(auto_lrc.build_bounded_retry_capability(report, audio, args))
            self.assertIsNone(
                auto_lrc.build_local_whisper_capability(
                    Path(temp_name) / "missing.mp3", args
                )
            )

    def test_execution_locators_remain_available_but_not_semantic(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "stem.mp3"
            audio.write_bytes(b"audio")
            seed = self.ctc_seed(audio)
            self.assertEqual(Path(str(seed["alignment_audio_path"])), audio.resolve())
            semantic = auto_lrc._bounded_retry_capability_semantic_payload(seed)
            self.assertNotIn("alignment_audio_path", semantic)
            self.assertNotIn("helper_path", semantic)

    def test_full_diagnostics_integrity_and_semantic_revision_are_separate(self) -> None:
        first = {
            "crossline_retry": {
                "capability": {
                    "helper_path": "C:/one/ctc_align.py",
                    "alignment_audio_path": "C:/one/vocals.mp3",
                    "helper_sha256": "helper-sha",
                    "alignment_audio_sha256": "audio-sha",
                    "status": "candidate",
                },
                "request_revision": "request-1",
                "candidate_id": "candidate-1",
            },
            "custom_path_note": "semantic-value",
        }
        second = copy.deepcopy(first)
        second["crossline_retry"]["capability"]["helper_path"] = "D:/two/ctc_align.py"  # type: ignore[index]
        second["crossline_retry"]["capability"]["alignment_audio_path"] = "D:/two/vocals.mp3"  # type: ignore[index]
        first_json, first_full = auto_lrc.freeze_generation_diagnostics(first)
        second_json, second_full = auto_lrc.freeze_generation_diagnostics(second)
        self.assertNotEqual(first_full, second_full)
        self.assertNotEqual(first_json, second_json)
        self.assertEqual(
            auto_lrc.generation_diagnostics_semantic_revision(first),
            auto_lrc.generation_diagnostics_semantic_revision(second),
        )
        semantic = auto_lrc._generation_diagnostics_semantic_projection(first)
        self.assertIn("custom_path_note", semantic)
        self.assertNotIn("helper_path", semantic["crossline_retry"]["capability"])

        changed = copy.deepcopy(first)
        changed["crossline_retry"]["capability"]["helper_sha256"] = "changed-helper"  # type: ignore[index]
        changed["crossline_retry"]["candidate_id"] = "candidate-2"  # type: ignore[index]
        changed["crossline_retry"]["capability"]["status"] = "error"  # type: ignore[index]
        self.assertNotEqual(
            auto_lrc.generation_diagnostics_semantic_revision(first),
            auto_lrc.generation_diagnostics_semantic_revision(changed),
        )

    def test_evaluation_identity_uses_semantic_diagnostics_but_thaw_uses_full_digest(self) -> None:
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="diagnostic identity",
            source="ctc-current",
            raw_time=1.0,
            spans=(
                auto_lrc.TimingTokenSpan(1.0, 1.04, 0.9, "a"),
                auto_lrc.TimingTokenSpan(1.1, 1.14, 0.9, "b"),
                auto_lrc.TimingTokenSpan(1.2, 1.24, 0.9, "c"),
            ),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"producer": "diagnostic-test"},
        )
        report = {
            "backend": "ctc",
            "assignments": [{"entry": 1, "score": 0.9, "segment": None}],
        }
        first = {
            "crossline_retry": {
                "capability": {
                    "helper_path": "C:/one/ctc_align.py",
                    "alignment_audio_path": "C:/one/vocals.mp3",
                    "helper_sha256": "helper-sha",
                    "alignment_audio_sha256": "audio-sha",
                }
            }
        }
        second = copy.deepcopy(first)
        second["crossline_retry"]["capability"]["alignment_audio_path"] = "D:/two/vocals.mp3"  # type: ignore[index]
        evaluated_first = auto_lrc.evaluate_backend_timing(
            "ctc", ((candidate,),), report, first
        )
        evaluated_second = auto_lrc.evaluate_backend_timing(
            "ctc", ((candidate,),), report, second
        )
        self.assertNotEqual(
            evaluated_first.generation_diagnostics_revision,
            evaluated_second.generation_diagnostics_revision,
        )
        self.assertEqual(
            evaluated_first.generation_diagnostics_semantic_revision,
            evaluated_second.generation_diagnostics_semantic_revision,
        )
        self.assertEqual(evaluated_first.input_revision, evaluated_second.input_revision)
        self.assertEqual(evaluated_first.evaluation_revision, evaluated_second.evaluation_revision)
        tampered = replace(
            evaluated_first,
            generation_diagnostics_json=evaluated_first.generation_diagnostics_json + " ",
        )
        with self.assertRaises(LrcError):
            auto_lrc.thaw_generation_diagnostics(tampered)


class V145ContentTimingSeparationRegressionTests(unittest.TestCase):
    def span(self, start: float, score: float = 0.5, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def structural_candidate(
        self, entry: int, time: float, *, source: str, confidence: float, current: bool = False
    ) -> auto_lrc.TimingCandidate:
        spans = (
            self.span(time, 0.5, "a"),
            self.span(time + 0.08, 0.6, "b"),
            self.span(time + 0.16, 0.7, "c"),
        )
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=f"line {entry}", source=source, raw_time=time,
            spans=spans, confidence=confidence, identity_support="supported",
            sequence_support="supported", current=current,
            source_artifact={"source": source, "time": time},
        )

    def independent_candidate(
        self, entry: int, time: float, *, source: str, producer: str, confidence: float
    ) -> auto_lrc.TimingCandidate:
        artifact = {"source": source, "producer": producer, "time": time}
        occurrence_entries = [
            LyricEntry([f"line {index}"])
            for index in range(max(1, entry + 1))
        ]
        occurrence_assignments = [
            {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
            for index in range(len(occurrence_entries))
        ]
        binding = auto_lrc._entry_occurrence_binding(
            occurrence_entries, occurrence_assignments, entry
        )
        occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            occurrence_entries,
            entry,
            producer=producer,
            occurrence_binding=binding,
            upstream_evidence_revision="v145-direct-upstream",
        )
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=f"line {entry}", source=source, raw_time=time,
            confidence=confidence, identity_support="supported", sequence_support="supported",
            direct_onset_support="supported", direct_onset_time=time,
            direct_onset_source_artifact=artifact, direct_onset_evidence_producer=producer,
            direct_onset_evidence_kind="cross-backend-onset",
            direct_onset_evidence_independent=True,
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=occurrence_evidence is not None,
            current=False, source_artifact=artifact,
        )

    def invalid_current(self, entry: int = 0, time: float = 10.5) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=f"line {entry}", source="ctc-current",
            raw_time=time, confidence=0.8, identity_support="supported",
            sequence_support="supported", current=True, source_artifact={"invalid": True},
        )

    def test_consensus_uses_valid_support_witnesses_before_pareto_pruning(self) -> None:
        current = self.invalid_current()
        raw = self.independent_candidate(
            0, 10.00, source="raw-vocal-independent-fusion",
            producer="raw-asr-vocal-fusion", confidence=0.96,
        )
        ctc = self.structural_candidate(
            0, 10.05, source="ctc-local-window", confidence=0.50
        )
        whisper = self.independent_candidate(
            0, 11.00, source="whisperx-vocal-independent-fusion",
            producer="whisperx-vocal-fusion", confidence=0.96,
        )
        # The CTC witness is individually dominated by RAW, but it supplies the
        # second lineage proving the 10.00 s cluster.  A distant WhisperX
        # singleton must not erase that cluster before consensus is evaluated.
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(current, raw, ctc, whisper),
            previous_candidate=None, selection_revision=1,
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertEqual(decision.written_time.centiseconds, 1000)
        self.assertEqual(decision.candidate.source, "raw-vocal-independent-fusion")

    def test_two_equally_supported_separated_clusters_still_fail_closed(self) -> None:
        current = self.invalid_current()
        raw = self.independent_candidate(
            0, 10.00, source="raw-vocal-independent-fusion",
            producer="raw-asr-vocal-fusion", confidence=0.96,
        )
        ctc_a = self.structural_candidate(0, 10.05, source="ctc-local-window", confidence=0.50)
        whisper = self.independent_candidate(
            0, 11.00, source="whisperx-vocal-independent-fusion",
            producer="whisperx-vocal-fusion", confidence=0.96,
        )
        ctc_b = self.structural_candidate(0, 11.05, source="ctc-opening-local-retry", confidence=0.50)
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(current, raw, ctc_a, whisper, ctc_b),
            previous_candidate=None, selection_revision=1,
        )
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertIsNone(decision.selected_candidate_id)

    def test_runtime_raw_identity_can_trust_content_without_trusting_its_timestamp(self) -> None:
        entries = [LyricEntry(["unique lyric alpha"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            "assignments": [{
                "entry": 1, "timestamp": 10.0, "timing_repair": "ctc",
                "score": 0.35,
            }],
        }
        raw_report = {
            "assignments": [{
                "entry": 1, "timestamp": 14.0, "score": 0.95,
                "segment": 0, "borrowed": False,
            }]
        }
        auto_lrc.bind_runtime_raw_content_identity(report, entries, raw_report)
        report[auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY] = tuple(entries)
        self.assertEqual(
            auto_lrc.runtime_independent_content_identity_sources_for_entry(report, 0),
            ("raw-asr-content-identity",),
        )
        candidate = self.structural_candidate(0, 10.0, source="ctc-local-window", confidence=0.8)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        self.assertTrue(state.audits[0].timing_trusted)
        summary = auto_lrc.summarize_central_timing_state(report, state)
        self.assertEqual(summary.content_trusted_entries, 1)
        self.assertEqual(summary.timing_trusted_entries, 1)
        self.assertEqual(summary.overall_trusted_entries, 1)
        # The independent ASR timestamp is deliberately far away: it proves
        # content identity only and must not alter the selected 10.00 s timing.
        self.assertEqual(state.decisions[0].written_time.centiseconds, 1000)

    def test_runtime_content_identity_revision_is_part_of_central_projection(self) -> None:
        entries = [LyricEntry(["unique lyric beta"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            "assignments": [{"entry": 1, "timestamp": 1.0, "score": 0.35}],
        }
        raw_report = {
            "assignments": [{
                "entry": 1, "timestamp": 2.0, "score": 0.95,
                "segment": 0, "borrowed": False,
            }]
        }
        auto_lrc.bind_runtime_raw_content_identity(report, entries, raw_report)
        report[auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY] = tuple(entries)
        projection = auto_lrc.timing_report_evaluation_projection(report)
        self.assertEqual(len(projection["runtime_content_identity"]), 1)
        before = auto_lrc.timing_report_evaluation_revision(report)
        runtime = report[auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY]
        assert isinstance(runtime, dict) and isinstance(runtime[0], dict)
        runtime[0]["evidence_revision"] = "tampered"
        self.assertEqual(
            auto_lrc.runtime_independent_content_identity_sources_for_entry(report, 0), ()
        )
        after = auto_lrc.timing_report_evaluation_revision(report)
        self.assertNotEqual(before, after)

    def test_runtime_lyric_entries_are_never_serialized(self) -> None:
        source = auto_lrc.algorithm_source_text()
        tree = ast.parse(source)
        writer = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "write_alignment_report"
        )
        segment = ast.get_source_segment(source, writer) or ""
        self.assertIn("_RUNTIME_LYRIC_ENTRIES_KEY", segment)


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
