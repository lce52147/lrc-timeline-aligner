"""Extracted from auto_lrc.py (split step 9): local whisper and crossline retry planning and execution, runtime evidence binding, central timing candidate collection."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Literal
import model_cache
from asr_backends import _aligned_exact_phonetic_target_matches, _aligned_exact_target_matches, _aligned_target_prefix_matches, _occurrence_binding_time_bounds, _run_whispercpp_wav, _run_whispercpp_wav_batch, _select_unique_aligned_target_match, _select_unique_aligned_target_prefix_match, asr_transcript, build_bounded_retry_capability, build_local_whisper_capability, decode_temp_wav_window, inactive_legacy_timing_diagnostics, lyric_segment_score, match_whisper_segments, run_whisperx_alignment, segment_char_time, whisperx_segments
from acoustic_hypotheses import _strong_vocal_onset_near, ctc_opening_vocal_fusion_is_valid, mutual_recurrence_hypothesis_provenance_is_valid, phonetic_onset_fusion_provenance_is_valid, recurrence_hypothesis_provenance_is_valid, trusted_tail_release_provenance_is_valid, vocal_activity_reentry_provenance_is_valid, vocal_onset_confirmation_provenance_is_valid
from ctc_refinement import CTC_WIDE_REGION_RADIUS, RAW_CTC_CONSENSUS_MIN_SCORE, SEGMENT_REALIGN_SOURCE, ctc_right_edge_pileup_evidence
from report_trust import _RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY, _suspicious_entry_requires_review, assignment_content_is_trusted, freeze_generation_diagnostics, runtime_independent_content_identity_sources_for_entry, runtime_local_whisper_evidence_for_entry, runtime_raw_content_identity_for_entry, runtime_whisperx_row_evidence_for_entry, timing_report_evaluation_revision
from timing_core import BaseRetryEpoch, BoundedRetryCapability, CTC_WIDE_REGION_FALLBACK_SOURCE, CandidateOccurrenceEvidence, CandidateRegionEvidence, CrossLineRetryOutcome, CrossLineRetryRequest, IndependentTailEvidence, LocalWhisperCapability, LocalWhisperOutcome, LocalWhisperRequest, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, RetryCandidateReference, RuntimeTailEvidence, TRUSTED_ALIGNMENT_SCORE, TailEvidenceContext, TailRegionEvidence, TimingCandidate, _RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY, _active_legacy_candidate, _candidate_confidence, _candidate_support_families, _entry_identity_is_unique, _entry_occurrence_binding, _finite_unit_confidence, _make_tail_provenance, _minimal_unique_sequence_context, _occurrence_binding_has_strong_context, _tail_evidence_context_from_report, _tail_helper_revision, analyze_candidate_regions, build_final_timing_state, candidate_has_bound_direct_onset_evidence, candidate_has_boundary_grade_direct_onset_evidence, candidate_occurrence_evidence_from_binding, derive_reliable_tail, generation_diagnostics_semantic_revision, make_timing_candidate, record_alignment_hypothesis, resolve_reliable_tail, runtime_independent_tail_evidence
from lyrics_io import AsrSegment, LyricEntry, _RUNTIME_LYRIC_ENTRIES_KEY, assert_lyric_identity_contract, entry_sung_text, first_forced_char_time, normalize_match_text
from audio_features import AudioFeatures, HOP_SECONDS
from runtime_base import DEFAULT_EVIDENCE_WORKERS, LrcError, _canonical_json_digest, _ctc_model_blob_sha256, _model_cache_read, _model_cache_write, _park_persistent_helper_mode, _path_sha256, _run_model_helper_command, _stable_file_sha256, _timing_content_digest, canonicalize_lrc_time, default_ctc_python, progress_log
from runtime_base import TimingTokenSpan, _timing_spans_from_value
from runtime_evidence_binding import (
    _attach_ctc_wide_region_candidates,
    _augment_candidates_with_joint_boundary_consensus,
    _augment_local_ctc_candidates_with_runtime_backend_consensus,
    _candidate_set_semantic_revision,
    _independent_tail_evidence_payload,
    _retry_semantic_state_revision,
    _tail_authority_diagnostics,
    bind_runtime_ctc_assignment_tail_fallback,
    bind_runtime_whisperx_ctc_consensus,
    bind_runtime_whisperx_row_evidence,
    independent_tail_evidence_payload,
    merge_runtime_independent_tail_evidence,
    runtime_raw_ctc_consensus_for_entry,
    runtime_whisperx_ctc_consensus_for_entry,
)


CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS = 0.75


CROSSLINE_RETRY_MAX_WINDOW_EXPANSIONS = 3


LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS = 1.50


LOCAL_WHISPER_MAX_WINDOW_SECONDS = 30.0


def _central_preview_generation_candidates(
    candidates: list[dict[str, object]],
    selected_backend: str,
    selected_timestamps: list[float],
    selected_report: dict[str, object],
) -> list[dict[str, object]]:
    """Replace same-backend/same-timestamp diagnostic clones with the refined seed."""
    selected_payload = {
        "backend": selected_backend,
        "timestamps": list(selected_timestamps),
        "report": selected_report,
        "error": None,
    }
    result: list[dict[str, object]] = []
    inserted = False
    for candidate in candidates:
        if (
            not candidate.get("error")
            and candidate.get("backend") == selected_backend
            and candidate.get("timestamps") == selected_timestamps
        ):
            if not inserted:
                result.append(selected_payload)
                inserted = True
            continue
        result.append(candidate)
    if not inserted:
        result.append(selected_payload)
    return result


def bind_ctc_generation_candidate_evidence(
    generation_candidates: list[dict[str, object]],
    entries: list[LyricEntry],
    whisperx_evidence_candidates: list[tuple[list[float], dict[str, object], str]],
) -> None:
    """Attach existing runtime evidence to each valid CTC generation candidate."""
    for candidate in generation_candidates:
        report = candidate.get("report")
        timestamps = candidate.get("timestamps")
        if (
            candidate.get("backend") != "ctc"
            or candidate.get("error")
            or not isinstance(report, dict)
            or not isinstance(timestamps, list)
            or len(timestamps) != len(entries)
        ):
            continue
        if whisperx_evidence_candidates:
            bind_runtime_whisperx_row_evidence(
                report, entries, whisperx_evidence_candidates
            )
            bind_runtime_whisperx_ctc_consensus(
                report, entries, timestamps, whisperx_evidence_candidates
            )
        bind_runtime_ctc_assignment_tail_fallback(report, entries)


def freeze_base_retry_epoch(
    backend: str,
    report: dict[str, object],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    diagnostics: dict[str, object],
    capability: BoundedRetryCapability | None,
) -> BaseRetryEpoch:
    candidate_revision = _candidate_set_semantic_revision(candidate_sets)
    _diagnostics_json, diagnostics_integrity_revision = freeze_generation_diagnostics(
        diagnostics
    )
    diagnostics_revision = generation_diagnostics_semantic_revision(diagnostics)
    epoch_payload = {
        "backend": backend,
        "report_projection_revision": timing_report_evaluation_revision(report),
        "ordinary_candidate_sets_revision": candidate_revision,
        "ordinary_diagnostics_revision": diagnostics_revision,
        "capability_revision": capability.capability_revision if capability else None,
        "schema_version": 1,
    }
    return BaseRetryEpoch(
        **epoch_payload,
        ordinary_diagnostics_integrity_revision=diagnostics_integrity_revision,
        retry_epoch_revision=_timing_content_digest(epoch_payload),
    )


def _retry_candidate_reference(candidate: TimingCandidate) -> RetryCandidateReference:
    regions = analyze_candidate_regions(candidate)
    return RetryCandidateReference(
        candidate.candidate_id,
        candidate.generation_revision,
        regions.region_revision,
        candidate.written_time.centiseconds,
    )


def plan_crossline_retry_requests(
    epoch: BaseRetryEpoch,
    capability: BoundedRetryCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> tuple[CrossLineRetryRequest, ...]:
    """Plan bounded *current-line opening* recovery.

    The retry target is the current lyric alone.  Previous/next line starts
    bound the search window, while previous-line tail evidence is diagnostic
    only.  This prevents a long backing-vocal/ad-lib tail from forcing the
    current lyric to start after the previous transcript has been serially
    consumed.
    """
    if capability is None or capability.capability_revision != epoch.capability_revision:
        return ()
    if len(entries) != len(candidate_sets) or not math.isfinite(duration) or duration <= 0.0:
        return ()
    state = build_final_timing_state(
        candidate_sets,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        timing_source_policy=timing_source_policy,
    )
    requests: list[CrossLineRetryRequest] = []
    duration_revision = _timing_content_digest({"duration": round(duration, 6)})

    for index in range(1, len(entries)):
        decision = state.decisions[index]
        if decision.status != "provisional_unresolved":
            continue

        previous_decision = state.decisions[index - 1]
        previous_tail = resolve_reliable_tail(
            previous_decision.candidate,
            independent_tail_evidence[index - 1] if independent_tail_evidence else None,
            tail_evidence_context=tail_evidence_context,
        )
        relation = decision.evidence.ownership
        if (
            relation.previous_tail_evidence_revision is not None
            and relation.previous_tail_evidence_revision != previous_tail.tail_evidence_revision
        ):
            continue

        # A current-line-only retry cannot manufacture missing predecessor-tail
        # ownership evidence.  If the current lyric already has coherent intrinsic
        # opening/path proof and the *only* structural blocker is unavailable
        # previous-tail ownership, re-aligning the same current lyric is impossible
        # to change the authority graph.  Preserve retry for ambiguous/incoherent
        # current paths and for valid-tail uncertainty/violation where a changed
        # current boundary can legitimately change the relation.
        structural_failures = set(decision.evidence.structural_failures)
        current_tail = derive_reliable_tail(decision.candidate)
        if (
            relation.status == "unknown"
            and previous_tail.validity != "valid"
            and current_tail.validity == "valid"
            and decision.evidence.temporal_coherence == "coherent"
            and decision.evidence.onset_proof_end is not None
            and decision.evidence.reliable_support_count >= 3
            and structural_failures
            and structural_failures <= {"previous-tail-ownership-unknown"}
        ):
            continue

        current_ref = _retry_candidate_reference(decision.candidate)
        previous_ref = _retry_candidate_reference(previous_decision.candidate)
        next_ref = (
            _retry_candidate_reference(state.decisions[index + 1].candidate)
            if index + 1 < len(entries)
            else None
        )

        # Opening-first recovery aligns only the current lyric.  Do not serially
        # prepend the previous lyric, because that makes overlap impossible.
        context_indexes = (index,)
        context_candidates = (decision.candidate,)
        transcripts = (entry_sung_text(entries[index]),)
        transcript_digests = (_timing_content_digest(transcripts[0]),)
        row_mapping = ((index, 0),)

        current_start = decision.written_time.seconds
        previous_start = previous_decision.written_time.seconds
        previous_spacing = max(HOP_SECONDS, current_start - previous_start)

        # Search several seconds before the provisional start, but never before
        # the previous lyric's own start.  The previous *tail* is not a lower
        # bound: overlapping vocals are legal.
        backtrack = min(4.0, max(1.25, previous_spacing * 0.60))
        window_start = max(
            0.0,
            previous_start + HOP_SECONDS,
            current_start - backtrack,
        )

        if index + 1 < len(entries):
            next_start = state.decisions[index + 1].written_time.seconds
            # A helper row that reaches the right edge is incomplete evidence,
            # not a reason to accept the truncated path.  On the next bounded
            # fixed-point pass, widen by one *existing* 0.75 s overlap unit per
            # observed truncated retry, with a hard finite expansion count.
            truncated_retry_count = sum(
                1
                for candidate in candidate_sets[index]
                if candidate.source == "ctc-opening-local-retry"
                and candidate.window_truncated
            )
            expansion_count = min(
                CROSSLINE_RETRY_MAX_WINDOW_EXPANSIONS, truncated_retry_count
            )
            forward_overlap = CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS * (1 + expansion_count)
            # Give the current line a bounded overlap allowance beyond the next
            # provisional start so ad-libs/backing vocals do not compress its
            # terminal tokens.  Cap the forward horizon because opening
            # recovery must remain local even when the next lyric is far away.
            window_end = min(duration, next_start + forward_overlap, current_start + 14.0)
        else:
            # Terminal line: infer a bounded phrase horizon from the previous
            # inter-line spacing instead of aligning against the entire song.
            forward = min(14.0, max(7.0, previous_spacing * 1.60))
            window_end = min(duration, current_start + forward)

        if not 1.0 <= window_end - window_start <= 20.0:
            continue

        mapping_revision = _timing_content_digest(row_mapping)
        window_revision = _timing_content_digest({
            "mode": "opening-first-single-line",
            "start": round(window_start, 6),
            "end": round(window_end, 6),
            "previous_start": previous_ref.canonical_centiseconds,
            "current": current_ref.candidate_id,
            "next": next_ref.candidate_id if next_ref else None,
            "retry_window_truncation_count": (
                min(CROSSLINE_RETRY_MAX_WINDOW_EXPANSIONS, truncated_retry_count)
                if index + 1 < len(entries) else 0
            ),
            "duration_revision": duration_revision,
        })
        payload = {
            "retry_epoch_revision": epoch.retry_epoch_revision,
            "entry_index": index,
            "ordinal": 1,
            "capability_revision": capability.capability_revision,
            "alignment_audio_revision": capability.alignment_audio_sha256,
            "current": current_ref,
            "previous": previous_ref,
            "next_bound": next_ref,
            "audio_end_seconds": round(duration, 6),
            "audio_duration_revision": duration_revision,
            "previous_tail_evidence_revision": previous_tail.tail_evidence_revision,
            "previous_tail_start": previous_tail.reliable_tail_start,
            "previous_tail_end": previous_tail.reliable_tail_end,
            "previous_tail_uncertainty": previous_tail.uncertainty_seconds,
            "previous_tail_validity": previous_tail.validity,
            "ownership_relation_revision": relation.relation_revision,
            "context_entry_indexes": context_indexes,
            "context_candidate_ids": tuple(item.candidate_id for item in context_candidates),
            "transcript_digests": transcript_digests,
            "transcript_digest": _timing_content_digest(transcripts),
            "row_mapping": row_mapping,
            "row_mapping_revision": mapping_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "window_definition_revision": window_revision,
        }
        revision_payload = {
            **payload,
            "current": current_ref.__dict__,
            "previous": previous_ref.__dict__,
            "next_bound": next_ref.__dict__ if next_ref else None,
        }
        requests.append(CrossLineRetryRequest(
            **payload,
            request_revision=_timing_content_digest(revision_payload),
        ))
    return tuple(sorted(requests, key=lambda item: (item.entry_index, item.request_revision)))


def plan_joint_boundary_retry_requests(
    epoch: BaseRetryEpoch,
    capability: BoundedRetryCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> tuple[CrossLineRetryRequest, ...]:
    """Plan bounded *previous+current lyric* joint boundary observations.

    Current-line-only retry can recover opening geometry, but it cannot directly
    answer which side of an ambiguous acoustic boundary belongs to the previous
    lyric.  This producer therefore aligns the immutable previous and current
    lyric rows *together* inside one bounded window.

    The helper result is observation-only.  It is never eligible for direct
    central selection; a later consensus step must bind it to a separately
    generated current-line candidate within the existing 0.18 s consensus
    tolerance before it may become boundary-grade evidence.
    """
    if capability is None or capability.capability_revision != epoch.capability_revision:
        return ()
    if len(entries) != len(candidate_sets) or not math.isfinite(duration) or duration <= 0.0:
        return ()

    state = build_final_timing_state(
        candidate_sets,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        timing_source_policy=timing_source_policy,
    )
    requests: list[CrossLineRetryRequest] = []
    duration_revision = _timing_content_digest({"duration": round(duration, 6)})

    for index in range(1, len(entries)):
        decision = state.decisions[index]
        if decision.status != "provisional_unresolved":
            continue

        audit_flags = set(decision.evidence.structural_failures)
        lacks_opening = not (
            decision.evidence.onset_proof_end is not None
            or candidate_has_bound_direct_onset_evidence(decision.candidate)
        )
        needs_boundary = bool(
            decision.evidence.ownership.status in {"unknown", "violation"}
            or lacks_opening
            or {
                "previous-tail-ownership-unknown",
                "previous-tail-overlap",
            }.intersection(audit_flags)
        )
        if not needs_boundary:
            continue

        previous_decision = state.decisions[index - 1]
        previous_tail = resolve_reliable_tail(
            previous_decision.candidate,
            independent_tail_evidence[index - 1] if independent_tail_evidence else None,
            tail_evidence_context=tail_evidence_context,
        )
        relation = decision.evidence.ownership
        if (
            relation.previous_tail_evidence_revision is not None
            and relation.previous_tail_evidence_revision != previous_tail.tail_evidence_revision
        ):
            continue

        current_ref = _retry_candidate_reference(decision.candidate)
        previous_ref = _retry_candidate_reference(previous_decision.candidate)
        next_ref = (
            _retry_candidate_reference(state.decisions[index + 1].candidate)
            if index + 1 < len(entries)
            else None
        )

        context_indexes = (index - 1, index)
        context_candidates = (previous_decision.candidate, decision.candidate)
        transcripts = tuple(entry_sung_text(entries[item]) for item in context_indexes)
        transcript_digests = tuple(_timing_content_digest(text) for text in transcripts)
        row_mapping = ((index - 1, 0), (index, 1))

        previous_start = previous_decision.written_time.seconds
        current_start = decision.written_time.seconds
        previous_previous_start = (
            state.decisions[index - 2].written_time.seconds if index >= 2 else 0.0
        )

        # Include enough pre-roll to let the helper observe the previous row's
        # real opening without swallowing the row before it.
        spacing = max(HOP_SECONDS, current_start - previous_start)
        pre_roll = min(3.0, max(1.0, spacing * 0.45))
        lower_neighbor = previous_previous_start + HOP_SECONDS if index >= 2 else 0.0
        window_start = max(lower_neighbor, previous_start - pre_roll)

        if index + 1 < len(entries):
            next_start = state.decisions[index + 1].written_time.seconds
            window_end = min(
                duration,
                next_start + 2.0 * CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
                current_start + 14.0,
            )
        else:
            window_end = min(duration, current_start + min(14.0, max(7.0, spacing * 1.8)))

        if not 2.0 <= window_end - window_start <= 24.0:
            continue

        mapping_revision = _timing_content_digest(row_mapping)
        window_revision = _timing_content_digest({
            "mode": "joint-previous-current-boundary",
            "start": round(window_start, 6),
            "end": round(window_end, 6),
            "previous_previous_start": round(previous_previous_start, 6),
            "previous": previous_ref.candidate_id,
            "current": current_ref.candidate_id,
            "next": next_ref.candidate_id if next_ref else None,
            "duration_revision": duration_revision,
        })
        payload = {
            "retry_epoch_revision": epoch.retry_epoch_revision,
            "entry_index": index,
            "ordinal": 2,
            "capability_revision": capability.capability_revision,
            "alignment_audio_revision": capability.alignment_audio_sha256,
            "current": current_ref,
            "previous": previous_ref,
            "next_bound": next_ref,
            "audio_end_seconds": round(duration, 6),
            "audio_duration_revision": duration_revision,
            "previous_tail_evidence_revision": previous_tail.tail_evidence_revision,
            "previous_tail_start": previous_tail.reliable_tail_start,
            "previous_tail_end": previous_tail.reliable_tail_end,
            "previous_tail_uncertainty": previous_tail.uncertainty_seconds,
            "previous_tail_validity": previous_tail.validity,
            "ownership_relation_revision": relation.relation_revision,
            "context_entry_indexes": context_indexes,
            "context_candidate_ids": tuple(item.candidate_id for item in context_candidates),
            "transcript_digests": transcript_digests,
            "transcript_digest": _timing_content_digest(transcripts),
            "row_mapping": row_mapping,
            "row_mapping_revision": mapping_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "window_definition_revision": window_revision,
        }
        revision_payload = {
            **payload,
            "current": current_ref.__dict__,
            "previous": previous_ref.__dict__,
            "next_bound": next_ref.__dict__ if next_ref else None,
        }
        requests.append(CrossLineRetryRequest(
            **payload,
            request_revision=_timing_content_digest(revision_payload),
        ))
    return tuple(sorted(requests, key=lambda item: (item.entry_index, item.request_revision)))


def crossline_retry_request_is_current(
    request: CrossLineRetryRequest,
    epoch: BaseRetryEpoch,
    capability: BoundedRetryCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> bool:
    planner = (
        plan_joint_boundary_retry_requests
        if len(request.context_entry_indexes) == 2
        else plan_crossline_retry_requests
    )
    return request in planner(
        epoch,
        capability,
        entries,
        candidate_sets,
        duration,
        independent_tail_evidence,
        tail_evidence_context,
        timing_source_policy,
    )


def plan_local_whisper_requests(
    report: dict[str, object],
    capability: LocalWhisperCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> tuple[LocalWhisperRequest, ...]:
    """Plan independent Whisper-family observations only where central evidence is incomplete."""
    if capability is None or len(entries) != len(candidate_sets):
        return ()
    if not math.isfinite(duration) or duration <= 0.0:
        return ()
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return ()
    state = build_final_timing_state(
        candidate_sets,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        timing_source_policy=timing_source_policy,
    )
    requests: list[LocalWhisperRequest] = []
    audio_revision = capability.audio_sha256

    for index, (decision, assignment) in enumerate(zip(state.decisions, assignments)):
        if not isinstance(assignment, dict):
            continue
        timing_need = decision.status == "provisional_unresolved"
        content_need = bool(
            not assignment_content_is_trusted(assignment)
            and not runtime_independent_content_identity_sources_for_entry(report, index)
        )
        review_need = _suspicious_entry_requires_review(report, index + 1)
        if not (timing_need or content_need or review_need):
            continue

        # Build a bounded identity context with one row of redundancy on each
        # available side.  The extra rows are not presumed trusted; they merely
        # give sparse occurrence binding a second anchor when an immediate
        # neighbor is weak.  If the expanded window would exceed the existing
        # 30 s local cap, fall back to the original minimal context.
        if _entry_identity_is_unique(entries, index):
            base_left = max(0, index - 1)
            base_right = min(len(entries) - 1, index + 1)
        else:
            fingerprint = _minimal_unique_sequence_context(entries, index, max_items=9)
            if fingerprint is None:
                continue
            base_left, base_right = fingerprint

        # Preserve the original minimal probe *and* optionally add one broader
        # sparse-context probe.  Broader context is evidence-additive only: it
        # must never replace a narrower request that may expose a cleaner direct
        # onset.  Runtime merging already ranks direct-onset evidence above
        # occurrence-only evidence for the same entry.
        context_options: list[tuple[int, int]] = [(base_left, base_right)]
        # Evaluate a small monotonic context ladder rather than betting the whole
        # observation on one segmentation-sensitive window.  Each broader probe
        # is optional evidence only; the minimal probe is always preserved.
        # Bound the ladder to at most three contexts and nine lyric rows so the
        # helper cost stays finite even on long songs.
        for expansion in (1, 2):
            expanded_left = max(0, base_left - expansion)
            expanded_right = min(len(entries) - 1, base_right + expansion)
            option = (expanded_left, expanded_right)
            if (
                expanded_left < index < expanded_right
                and expanded_right - expanded_left + 1 <= 9
                and option not in context_options
            ):
                context_options.append(option)

        seen_contexts: set[tuple[int, int, int, int]] = set()
        for option_left, option_right in context_options:
            option_left_start = state.decisions[option_left].written_time.seconds
            if option_left > 0:
                option_predecessor = state.decisions[option_left - 1].written_time.seconds
                option_window_start = max(
                    0.0,
                    option_predecessor + HOP_SECONDS,
                    option_left_start - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
                )
            else:
                option_window_start = 0.0

            option_right_start = state.decisions[option_right].written_time.seconds
            if option_right + 1 < len(entries):
                option_successor = state.decisions[option_right + 1].written_time.seconds
                option_window_end = min(
                    duration,
                    option_successor + LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
                )
            else:
                option_window_end = min(duration, option_right_start + 8.0)
            if option_window_end <= option_right_start + 0.5:
                option_window_end = min(duration, option_right_start + 4.0)
            option_length = option_window_end - option_window_start
            if (
                option_left == 0
                and option_length > LOCAL_WHISPER_MAX_WINDOW_SECONDS
            ):
                option_window_start = max(
                    0.0,
                    option_left_start - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
                )
                option_length = option_window_end - option_window_start
            if not 2.0 <= option_length <= LOCAL_WHISPER_MAX_WINDOW_SECONDS:
                continue

            context_indexes = tuple(range(option_left, option_right + 1))
            if not context_indexes or index not in context_indexes or len(context_indexes) > 9:
                continue
            semantic_key = (
                option_left,
                option_right,
                int(round(option_window_start * 1000)),
                int(round(option_window_end * 1000)),
            )
            if semantic_key in seen_contexts:
                continue
            seen_contexts.add(semantic_key)

            context_candidates = tuple(state.decisions[item].candidate for item in context_indexes)
            transcripts = tuple(entry_sung_text(entries[item]) for item in context_indexes)
            transcript_digests = tuple(_timing_content_digest(text) for text in transcripts)

            window_definition = {
                "mode": "unresolved-only-local-whisper",
                "start": round(option_window_start, 6),
                "end": round(option_window_end, 6),
                "context_indexes": list(context_indexes),
                "context_candidate_ids": [item.candidate_id for item in context_candidates],
                "timing_need": timing_need,
                "content_need": content_need,
                "review_need": review_need,
                "capability_revision": capability.capability_revision,
            }
            payload = {
                "entry_index": index,
                "capability_revision": capability.capability_revision,
                "audio_revision": audio_revision,
                "context_entry_indexes": context_indexes,
                "context_candidate_ids": tuple(item.candidate_id for item in context_candidates),
                "transcript_digests": transcript_digests,
                "transcript_digest": _timing_content_digest(transcripts),
                "window_start": round(option_window_start, 6),
                "window_end": round(option_window_end, 6),
                "window_definition_revision": _timing_content_digest(window_definition),
            }
            requests.append(LocalWhisperRequest(
                **payload,
                request_revision=_timing_content_digest({
                    **payload,
                    "context_entry_indexes": list(context_indexes),
                    "context_candidate_ids": list(payload["context_candidate_ids"]),
                    "transcript_digests": list(transcript_digests),
                }),
                execution_revision=_local_whisper_execution_request_revision(payload),
            ))
    # A broken global CTC can collapse an entire verse 10-20 seconds earlier.
    # Local Whisper requests centered on Central's provisional baseline will
    # then listen to the wrong audio, even when S1 has a real bounded CTC path
    # at the correct vocal segment. Add only a capped independent ASR probe
    # around the *untrusted* S1 path. This is evidence acquisition, not timing
    # selection, and does not make S1 direct-onset/identity evidence.
    displaced_segment_rows: list[tuple[float, int, tuple[TimingCandidate, ...]]] = []
    segment_by_row: dict[int, TimingCandidate] = {}
    for index, items in enumerate(candidate_sets):
        proposals = [item for item in items
                     if item.source == "segment-constrained-ctc"
                     and item.spans and not item.window_truncated
                     and not item.right_edge_pileup]
        if not proposals:
            # V2 segment-realign proposals get the same capped independent probe.
            proposals = [item for item in items
                         if item.source == SEGMENT_REALIGN_SOURCE
                         and item.spans and not item.window_truncated
                         and not item.right_edge_pileup]
        if len(proposals) == 1:
            segment_by_row[index] = proposals[0]
            gap = abs(proposals[0].written_time.seconds - state.decisions[index].written_time.seconds)
            if gap >= 3.0 and (
                state.decisions[index].status == "provisional_unresolved"
                or not assignment_content_is_trusted(assignments[index])
                or _suspicious_entry_requires_review(report, index + 1)
            ):
                displaced_segment_rows.append((-gap, index, items))
    for _gap, index, _items in sorted(displaced_segment_rows)[:12]:
        if _entry_identity_is_unique(entries, index):
            first, last = max(0, index - 1), min(len(entries) - 1, index + 1)
        else:
            context = _minimal_unique_sequence_context(entries, index, max_items=9)
            if context is None:
                continue
            first, last = context
        context_indexes = tuple(range(first, last + 1))
        if (
            len(context_indexes) > 9
            or any(j not in segment_by_row for j in context_indexes)
        ):
            continue
        context_candidates = tuple(segment_by_row[j] for j in context_indexes)
        s1_times = tuple(item.written_time.seconds for item in context_candidates)
        if any(a >= b for a, b in zip(s1_times, s1_times[1:])):
            continue
        window_start = max(0.0, s1_times[0] - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS)
        window_end = min(duration, s1_times[-1] + max(4.0, LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS))
        if not 2.0 <= window_end - window_start <= LOCAL_WHISPER_MAX_WINDOW_SECONDS:
            continue
        transcript = tuple(entry_sung_text(entries[j]) for j in context_indexes)
        transcripts_digest = tuple(_timing_content_digest(text) for text in transcript)
        window_definition = {
            "mode": "s1-untrusted-segment-search-not-proof",
            "start": round(window_start, 6),
            "end": round(window_end, 6),
            "context_indexes": list(context_indexes),
            "context_candidate_ids": [x.candidate_id for x in context_candidates],
            "capability_revision": capability.capability_revision,
        }
        payload = {
            "entry_index": index,
            "capability_revision": capability.capability_revision,
            "audio_revision": audio_revision,
            "context_entry_indexes": context_indexes,
            "context_candidate_ids": tuple(x.candidate_id for x in context_candidates),
            "transcript_digests": transcripts_digest,
            "transcript_digest": _timing_content_digest(transcript),
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "window_definition_revision": _timing_content_digest(window_definition),
        }
        requests.append(LocalWhisperRequest(
            **payload,
            request_revision=_timing_content_digest({
                **payload,
                "context_entry_indexes": list(context_indexes),
                "context_candidate_ids": list(payload["context_candidate_ids"]),
                "transcript_digests": list(transcripts_digest),
            }),
            execution_revision=_local_whisper_execution_request_revision(payload),
        ))
    unique = {item.request_revision: item for item in requests}
    return tuple(sorted(
        unique.values(),
        key=lambda item: (item.entry_index, len(item.context_entry_indexes), item.request_revision),
    ))


def _local_whisper_execution_request_revision(
    payload: dict[str, object],
) -> str:
    """Hash only the bounded helper input; planner provenance stays diagnostic."""
    return _timing_content_digest({
        "schema": "local-whisper-execution-request-v2",
        "entry_index": payload["entry_index"],
        "capability_revision": payload["capability_revision"],
        "audio_revision": payload["audio_revision"],
        "context_entry_indexes": list(payload["context_entry_indexes"]),
        "transcript_digests": list(payload["transcript_digests"]),
        "transcript_digest": payload["transcript_digest"],
        "window_start": payload["window_start"],
        "window_end": payload["window_end"],
    })


def _local_whisper_request_execution_revision(request: LocalWhisperRequest) -> str:
    return request.execution_revision or _local_whisper_execution_request_revision(
        request.__dict__
    )


def _local_whisper_cache_key(
    request: LocalWhisperRequest,
    capability: LocalWhisperCapability,
) -> str:
    # Keep the serialized key shape stable so K2 cache entries remain reusable.
    return "local-whisper:" + _timing_content_digest({
        "request_revision": _local_whisper_request_execution_revision(request),
        "capability_revision": capability.capability_revision,
        "protocol_revision": capability.protocol_revision,
    })


def _local_whisper_outcome_for_request(
    outcome: LocalWhisperOutcome,
    request: LocalWhisperRequest,
) -> LocalWhisperOutcome:
    if outcome.request_revision == request.request_revision:
        return outcome
    base = {
        "entry_index": outcome.entry_index,
        "request_revision": request.request_revision,
        "status": outcome.status,
        "reason": outcome.reason,
        "onset_time": outcome.onset_time,
        "score": outcome.score,
        "evidence_revision": outcome.evidence_revision,
        "occurrence_binding": outcome.occurrence_binding,
        "raw_transcript_revision": outcome.raw_transcript_revision,
        "aligned_transcript_revision": outcome.aligned_transcript_revision,
    }
    return LocalWhisperOutcome(
        **base,
        outcome_revision=_timing_content_digest(base),
    )


def _shift_local_assignment_to_global(
    assignment: object,
    offset: float,
) -> dict[str, object] | None:
    if not isinstance(assignment, dict):
        return None
    shifted = copy.deepcopy(assignment)
    try:
        shifted["timestamp"] = round(float(assignment.get("timestamp")) + offset, 6)
    except (TypeError, ValueError):
        return None
    return shifted


def execute_local_whisper_request(
    request: LocalWhisperRequest,
    capability: LocalWhisperCapability,
    entries: list[LyricEntry],
    args: argparse.Namespace,
    *,
    cache: dict[str, tuple[str, object]] | None = None,
    runtime_stats: dict[str, int] | None = None,
    prepared: tuple[list[AsrSegment], dict[str, object]] | None = None,
    preparation_error: str | None = None,
) -> LocalWhisperOutcome:
    """Run one bounded Whisper.cpp -> WhisperX observation; never select a timestamp directly."""
    persistent_cache_key: str | None = None
    persistent_cache_identity: dict[str, object] | None = None
    def finish(
        status: Literal["evidence", "invalid", "error"],
        reason: str,
        *,
        onset_time: float | None = None,
        score: float | None = None,
        evidence_revision: str | None = None,
        occurrence_binding: dict[str, object] | None = None,
        raw_transcript_revision: str | None = None,
        aligned_transcript_revision: str | None = None,
    ) -> LocalWhisperOutcome:
        base = {
            "entry_index": request.entry_index,
            "request_revision": request.request_revision,
            "status": status,
            "reason": reason,
            "onset_time": onset_time,
            "score": score,
            "evidence_revision": evidence_revision,
            "occurrence_binding": occurrence_binding,
            "raw_transcript_revision": raw_transcript_revision,
            "aligned_transcript_revision": aligned_transcript_revision,
        }
        outcome = LocalWhisperOutcome(
            **base,
            outcome_revision=_timing_content_digest(base),
        )
        if (status != "error" or prepared is not None) and persistent_cache_key is not None and persistent_cache_identity is not None:
            _model_cache_write("local-whisper", persistent_cache_key, persistent_cache_identity, {
                "entry_index": outcome.entry_index,
                "request_revision": outcome.request_revision,
                "status": outcome.status,
                "reason": outcome.reason,
                "onset_time": outcome.onset_time,
                "score": outcome.score,
                "evidence_revision": outcome.evidence_revision,
                "occurrence_binding": outcome.occurrence_binding,
                "raw_transcript_revision": outcome.raw_transcript_revision,
                "aligned_transcript_revision": outcome.aligned_transcript_revision,
                "outcome_revision": outcome.outcome_revision,
            }, args)
        return outcome

    if capability.capability_revision != request.capability_revision:
        return finish("invalid", "capability-revision-mismatch")
    if request.audio_revision != capability.audio_sha256:
        return finish("invalid", "audio-revision-mismatch")
    if not 0 <= request.entry_index < len(entries):
        return finish("invalid", "entry-out-of-range")
    source_audio = Path(capability.audio_path)
    try:
        if _stable_file_sha256(source_audio) != capability.audio_sha256:
            return finish("invalid", "source-audio-changed")
    except (OSError, LrcError):
        return finish("invalid", "source-audio-unavailable")

    if cache is None:
        cache = {}
    if runtime_stats is None:
        runtime_stats = {}
    cache_key = _local_whisper_cache_key(request, capability)
    local_identity = model_cache.identity_local_whisper(
        audio_sha256=capability.audio_sha256,
        whisper_model_sha256=capability.whisper_model_sha256,
        protocol_revision=capability.protocol_revision,
        request_revision=_local_whisper_request_execution_revision(request),
    )
    persistent_cache_key = cache_key
    persistent_cache_identity = local_identity
    cached = cache.get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[1], LocalWhisperOutcome):
        runtime_stats["cache_hits"] = runtime_stats.get("cache_hits", 0) + 1
        return _local_whisper_outcome_for_request(cached[1], request)
    persisted = _model_cache_read("local-whisper", cache_key, local_identity, args)
    if isinstance(persisted, dict) and persisted.get("status") == "error":
        persisted = None
    if isinstance(persisted, dict):
        try:
            outcome = LocalWhisperOutcome(**persisted)
        except (TypeError, ValueError):
            outcome = None
        if outcome is not None:
            outcome = _local_whisper_outcome_for_request(outcome, request)
            cache[cache_key] = (outcome.outcome_revision, outcome)
            runtime_stats["cache_hits"] = runtime_stats.get("cache_hits", 0) + 1
            return outcome
    runtime_stats["cache_misses"] = runtime_stats.get("cache_misses", 0) + 1

    try:
        if preparation_error is not None:
            raise LrcError(preparation_error)
        if prepared is None:
            wav_path = decode_temp_wav_window(
                source_audio, request.window_start, request.window_end
            )
            raw_segments = _run_whispercpp_wav(wav_path, args, suppress_nst=False)
            aligned_payload = run_whisperx_alignment(
                wav_path, asr_transcript(raw_segments), args, predecoded_wav=True
            )
        else:
            raw_segments, aligned_payload = prepared
        aligned_segments = whisperx_segments(aligned_payload)
        runtime_stats["helper_invocations"] = runtime_stats.get("helper_invocations", 0) + 1
    except (LrcError, OSError, ValueError, json.JSONDecodeError) as exc:
        outcome = finish("error", f"local-whisper-failed:{type(exc).__name__}")
        return outcome

    context_entries = [entries[index] for index in request.context_entry_indexes]
    local_duration = request.window_end - request.window_start
    try:
        local_times, local_report = match_whisper_segments(
            context_entries, aligned_segments, local_duration
        )
    except (LrcError, ValueError, IndexError) as exc:
        outcome = finish("error", f"local-match-failed:{type(exc).__name__}")
        return outcome
    assignments = local_report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(context_entries):
        outcome = finish("invalid", "local-assignment-shape-mismatch")
        cache[cache_key] = (outcome.outcome_revision, outcome)
        return outcome

    full_assignments: list[object] = [None] * len(entries)
    target_local_index: int | None = None
    for local_index, global_index in enumerate(request.context_entry_indexes):
        shifted = _shift_local_assignment_to_global(assignments[local_index], request.window_start)
        if shifted is not None:
            full_assignments[global_index] = shifted
        if global_index == request.entry_index:
            target_local_index = local_index
    if target_local_index is None:
        outcome = finish("invalid", "target-context-mapping-missing")
        cache[cache_key] = (outcome.outcome_revision, outcome)
        return outcome
    target_text = normalize_match_text(entry_sung_text(entries[request.entry_index]))
    target_assignment = full_assignments[request.entry_index]
    score = 0.0
    segment_number: int | None = None
    target_assignment_trusted = False
    if isinstance(target_assignment, dict):
        try:
            score = float(target_assignment.get("score", 0.0) or 0.0)
            segment_number = int(target_assignment.get("segment"))
        except (TypeError, ValueError):
            score = 0.0
            segment_number = None
        target_assignment_trusted = bool(
            score >= TRUSTED_ALIGNMENT_SCORE
            and not bool(target_assignment.get("borrowed"))
            and segment_number is not None
            and 1 <= segment_number <= len(aligned_segments)
        )

    occurrence_binding: dict[str, object] | None = None
    recovered_exact_match: dict[str, object] | None = None
    recovered_prefix_match: dict[str, object] | None = None
    recovered_phonetic_match: dict[str, object] | None = None
    if not target_assignment_trusted:
        # Do not promote a weak matcher score.  Recover only from real forced
        # WhisperX characters.  A full exact target is strongest; when the full
        # line is absent from WhisperX segmentation, a forced opening prefix may
        # recover the target only if that synthetic onset makes a full/sparse
        # immutable lyric context uniquely bind the occurrence.  No nearest-time
        # choice is allowed, and partial-prefix recovery never falls back to a
        # global-unique-only identity claim.
        direct_text_score = lyric_segment_score(target_text, target_text)

        def recovered_trials(
            matches: tuple[dict[str, object], ...],
            *,
            prefix_only: bool,
        ) -> list[tuple[dict[str, object], dict[str, object], list[object]]]:
            recovered: list[tuple[dict[str, object], dict[str, object], list[object]]] = []
            for match in matches:
                global_onset = float(match["onset_time"]) + request.window_start
                synthetic = {
                    "timestamp": round(global_onset, 6),
                    "score": direct_text_score,
                    "segment": int(match["start_segment_index"]) + 1,
                    "borrowed": False,
                    "direct_exact_char_stream": not prefix_only,
                    "direct_prefix_char_stream": prefix_only,
                    "char_match_revision": match["match_revision"],
                }
                trial_assignments = list(full_assignments)
                trial_assignments[request.entry_index] = synthetic
                binding = _entry_occurrence_binding(
                    entries, trial_assignments, request.entry_index, require_context=True
                )
                if binding is None and not prefix_only:
                    binding = _entry_occurrence_binding(
                        entries, trial_assignments, request.entry_index
                    )
                if binding is None:
                    continue
                if prefix_only and not _occurrence_binding_has_strong_context(binding):
                    continue
                recovered.append((match, binding, trial_assignments))
            return recovered

        exact_matches = _aligned_exact_target_matches(target_text, aligned_segments)
        recovered = recovered_trials(exact_matches, prefix_only=False)
        unique_recovered = {
            (str(item[0].get("match_revision")), str(item[1].get("binding_revision"))): item
            for item in recovered
        }
        if len(unique_recovered) == 1:
            recovered_exact_match, occurrence_binding, full_assignments = next(iter(unique_recovered.values()))
        elif len(unique_recovered) > 1:
            outcome = finish("invalid", "target-exact-char-occurrence-ambiguous")
            cache[cache_key] = (outcome.outcome_revision, outcome)
            return outcome
        else:
            prefix_matches = _aligned_target_prefix_matches(target_text, aligned_segments)
            recovered_prefix = recovered_trials(prefix_matches, prefix_only=True)
            unique_prefix = {
                (str(item[0].get("match_revision")), str(item[1].get("binding_revision"))): item
                for item in recovered_prefix
            }
            if len(unique_prefix) == 1:
                recovered_prefix_match, occurrence_binding, full_assignments = next(iter(unique_prefix.values()))
            elif len(unique_prefix) > 1:
                outcome = finish("invalid", "target-prefix-char-occurrence-ambiguous")
                cache[cache_key] = (outcome.outcome_revision, outcome)
                return outcome
            else:
                # Exact phonetic identity is a separate content proof.  It can
                # recover a weak text assignment caused by kana/kanji spelling
                # variation, but it does not itself claim a line onset.
                phonetic_matches = _aligned_exact_phonetic_target_matches(
                    entry_sung_text(entries[request.entry_index]), aligned_segments
                )
                phonetic_recovered: list[tuple[dict[str, object], dict[str, object], list[object]]] = []
                for match in phonetic_matches:
                    try:
                        start_segment_index = int(match["start_segment_index"])
                        local_reference_time = float(aligned_segments[start_segment_index].start)
                        phonetic_score = float(match["phonetic_score"])
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue
                    if not math.isfinite(local_reference_time) or phonetic_score < TRUSTED_ALIGNMENT_SCORE:
                        continue
                    synthetic = {
                        "timestamp": round(local_reference_time + request.window_start, 6),
                        "score": round(phonetic_score, 6),
                        "segment": start_segment_index + 1,
                        "borrowed": False,
                        "direct_exact_phonetic_identity": True,
                        "phonetic_match_revision": match["match_revision"],
                    }
                    trial_assignments = list(full_assignments)
                    trial_assignments[request.entry_index] = synthetic
                    binding = _entry_occurrence_binding(
                        entries, trial_assignments, request.entry_index, require_context=True
                    )
                    if binding is None:
                        binding = _entry_occurrence_binding(
                            entries, trial_assignments, request.entry_index
                        )
                    if binding is None:
                        continue
                    phonetic_recovered.append((match, binding, trial_assignments))
                unique_phonetic = {
                    (str(item[0].get("match_revision")), str(item[1].get("binding_revision"))): item
                    for item in phonetic_recovered
                }
                if len(unique_phonetic) == 1:
                    recovered_phonetic_match, occurrence_binding, full_assignments = next(
                        iter(unique_phonetic.values())
                    )
                elif len(unique_phonetic) > 1:
                    outcome = finish("invalid", "target-phonetic-occurrence-ambiguous")
                    cache[cache_key] = (outcome.outcome_revision, outcome)
                    return outcome
                else:
                    outcome = finish(
                        "invalid",
                        "target-local-assignment-untrusted"
                        if not exact_matches and not prefix_matches and not phonetic_matches
                        else "target-independent-identity-occurrence-unproven",
                    )
                    cache[cache_key] = (outcome.outcome_revision, outcome)
                    return outcome

        target_assignment = full_assignments[request.entry_index]
        assert isinstance(target_assignment, dict)
        score = float(target_assignment["score"])
        segment_number = int(target_assignment["segment"])
        target_assignment_trusted = True

    if not target_assignment_trusted or segment_number is None:
        outcome = finish("invalid", "target-local-assignment-untrusted")
        cache[cache_key] = (outcome.outcome_revision, outcome)
        return outcome

    # Prefer a full/sparse trusted sequence bracket.  Unique lyrics retain the
    # weaker global-unique fallback for content identity, but only context-bound
    # observations may participate in hypothesis falsification.
    if occurrence_binding is None:
        occurrence_binding = _entry_occurrence_binding(
            entries, full_assignments, request.entry_index, require_context=True
        )
        if occurrence_binding is None:
            occurrence_binding = _entry_occurrence_binding(
                entries, full_assignments, request.entry_index
            )
    if occurrence_binding is None:
        outcome = finish("invalid", "occurrence-binding-unproven")
        cache[cache_key] = (outcome.outcome_revision, outcome)
        return outcome

    # Timing evidence must be an actual forced-character onset.  Search the full
    # aligned char stream first so a lyric split across WhisperX segments remains
    # directly observable.  No interpolation or nearest-candidate choice is used.
    direct_match = recovered_exact_match or recovered_prefix_match or _select_unique_aligned_target_match(
        target_text, aligned_segments,
        occurrence_binding=occurrence_binding,
        entry_index=request.entry_index,
        preferred_segment_number=segment_number,
    )
    used_prefix_match = recovered_prefix_match is not None
    if direct_match is None:
        direct_match = _select_unique_aligned_target_prefix_match(
            target_text,
            aligned_segments,
            occurrence_binding=occurrence_binding,
            entry_index=request.entry_index,
            preferred_segment_number=segment_number,
        )
        used_prefix_match = direct_match is not None
    direct_local_onset = (
        float(direct_match["onset_time"]) if isinstance(direct_match, dict) else None
    )
    if direct_local_onset is None:
        aligned_segment = aligned_segments[segment_number - 1]
        direct_local_onset = segment_char_time(target_text, aligned_segment)
        if direct_local_onset is None:
            aligned_text = normalize_match_text(aligned_segment.text)
            if aligned_text.startswith(target_text) and target_text:
                direct_local_onset = aligned_segment.start
    onset_time = (
        float(direct_local_onset) + request.window_start
        if direct_local_onset is not None else None
    )
    if onset_time is not None and (
        not math.isfinite(onset_time)
        or onset_time < request.window_start - HOP_SECONDS
        or onset_time > request.window_end + HOP_SECONDS
    ):
        onset_time = None

    lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entries[request.entry_index])).encode("utf-8")
    ).hexdigest()
    raw_projection = [
        {
            "start": round(segment.start, 6),
            "end": round(segment.end, 6),
            "text_digest": _timing_content_digest(normalize_match_text(segment.text)),
        }
        for segment in raw_segments
    ]
    aligned_projection = [
        {
            "start": round(segment.start, 6),
            "end": round(segment.end, 6),
            "text_digest": _timing_content_digest(normalize_match_text(segment.text)),
            "first_char": round(first_forced_char_time(segment), 6),
        }
        for segment in aligned_segments
    ]
    raw_transcript_revision = _timing_content_digest(raw_projection)
    aligned_transcript_revision = _timing_content_digest(aligned_projection)
    evidence_payload = {
        "entry_index": request.entry_index,
        "lyric_sha256": lyric_sha,
        "onset_time": (round(onset_time, 6) if onset_time is not None else None),
        "score": round(score, 6),
        "occurrence_binding": occurrence_binding,
        "request_revision": request.request_revision,
        "capability_revision": capability.capability_revision,
        "audio_revision": capability.audio_sha256,
        "window_start": round(request.window_start, 6),
        "window_end": round(request.window_end, 6),
        "raw_transcript_revision": raw_transcript_revision,
        "aligned_transcript_revision": aligned_transcript_revision,
        "producer": "local-whisper-aligner",
    }
    evidence_revision = _timing_content_digest(evidence_payload)
    if recovered_exact_match is not None:
        outcome_reason = (
            "trusted-local-whisper-exact-char-recovery-and-onset"
            if onset_time is not None
            else "trusted-local-whisper-exact-char-recovery-occurrence-only"
        )
    elif recovered_phonetic_match is not None:
        outcome_reason = (
            "trusted-local-whisper-exact-phonetic-and-onset"
            if onset_time is not None
            else "trusted-local-whisper-exact-phonetic-occurrence-only"
        )
    elif recovered_prefix_match is not None and onset_time is not None:
        outcome_reason = "trusted-local-whisper-context-prefix-recovery-and-onset"
    elif used_prefix_match and onset_time is not None:
        outcome_reason = "trusted-local-whisper-context-prefix-onset"
    elif isinstance(occurrence_binding, dict) and occurrence_binding.get("mode") == "sparse-sequence-bracket":
        outcome_reason = (
            "trusted-local-whisper-sparse-occurrence-and-onset"
            if onset_time is not None
            else "trusted-local-whisper-sparse-occurrence-only"
        )
    else:
        outcome_reason = (
            "trusted-local-whisper-occurrence-and-onset"
            if onset_time is not None
            else "trusted-local-whisper-occurrence-only"
        )
    outcome = finish(
        "evidence",
        outcome_reason,
        onset_time=onset_time,
        score=score,
        evidence_revision=evidence_revision,
        occurrence_binding=occurrence_binding,
        raw_transcript_revision=raw_transcript_revision,
        aligned_transcript_revision=aligned_transcript_revision,
    )
    cache[cache_key] = (outcome.outcome_revision, outcome)
    _model_cache_write("local-whisper", cache_key, local_identity, {
        "entry_index": outcome.entry_index,
        "request_revision": outcome.request_revision,
        "status": outcome.status,
        "reason": outcome.reason,
        "onset_time": outcome.onset_time,
        "score": outcome.score,
        "evidence_revision": outcome.evidence_revision,
        "occurrence_binding": outcome.occurrence_binding,
        "raw_transcript_revision": outcome.raw_transcript_revision,
        "aligned_transcript_revision": outcome.aligned_transcript_revision,
        "outcome_revision": outcome.outcome_revision,
    }, args)
    return outcome


def _prepare_local_whisper_batch(
    work: list[tuple[str, int, LocalWhisperRequest]],
    capability: LocalWhisperCapability,
    args: argparse.Namespace,
) -> tuple[
    dict[str, tuple[list[AsrSegment], dict[str, object]]],
    dict[str, str],
]:
    """Prepare a frozen local-Whisper epoch with batched raw ASR and resident WhisperX.

    This function performs no Central mutation.  It only materializes the same
    per-request raw/aligned observations that execute_local_whisper_request would
    otherwise acquire one process at a time.
    """
    prepared: dict[str, tuple[list[AsrSegment], dict[str, object]]] = {}
    errors: dict[str, str] = {}
    if not work:
        return prepared, errors
    source_audio = Path(capability.audio_path)
    decoded: dict[str, Path] = {}

    def decode_one(item: tuple[str, int, LocalWhisperRequest]) -> tuple[str, Path | None, str | None]:
        key, _ordinal, request = item
        try:
            wav_path = decode_temp_wav_window(source_audio, request.window_start, request.window_end)
            return key, wav_path, None
        except (LrcError, OSError, ValueError) as exc:
            return key, None, f"decode:{type(exc).__name__}:{exc}"

    progress_log(f"local Whisper prepare: decode_windows={len(work)} workers={DEFAULT_EVIDENCE_WORKERS}")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(DEFAULT_EVIDENCE_WORKERS, len(work) or 1),
        thread_name_prefix="lrc-local-decode",
    ) as pool:
        for key, wav_path, error in pool.map(decode_one, work):
            if error is not None or wav_path is None:
                errors[key] = error or "decode:unknown"
            else:
                decoded[key] = wav_path

    raw_inputs = [(key, decoded[key]) for key, _ordinal, _request in work if key in decoded]
    # Raw whisper.cpp gets exclusive ownership of the heavy GPU phase; park the
    # resident WhisperX align model on CPU before entering that phase.
    _park_persistent_helper_mode("whisperx", reason="before-local-whispercpp-batch")
    raw_results, raw_errors = _run_whispercpp_wav_batch(raw_inputs, args, suppress_nst=False)
    errors.update(raw_errors)

    align_items = [
        (key, raw_results[key], decoded[key])
        for key, _ordinal, _request in work
        if key in raw_results and key in decoded and key not in errors
    ]
    progress_log(
        f"local Whisper align batch: requests={len(align_items)} "
        f"resident_workers={DEFAULT_EVIDENCE_WORKERS}"
    )

    def align_one(item: tuple[str, list[AsrSegment], Path]) -> tuple[str, dict[str, object] | None, str | None]:
        key, raw_segments, wav_path = item
        try:
            payload = run_whisperx_alignment(
                wav_path,
                asr_transcript(raw_segments),
                args,
                predecoded_wav=True,
            )
            return key, payload, None
        except (LrcError, OSError, ValueError, json.JSONDecodeError) as exc:
            return key, None, f"align:{type(exc).__name__}:{exc}"

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(DEFAULT_EVIDENCE_WORKERS, len(align_items) or 1),
        thread_name_prefix="lrc-local-whisperx",
    ) as pool:
        for key, payload, error in pool.map(align_one, align_items):
            if error is not None or payload is None:
                errors[key] = error or "align:unknown"
            else:
                prepared[key] = (raw_results[key], payload)
    _park_persistent_helper_mode("whisperx", reason="local-whisperx-align-complete")
    progress_log(
        f"local Whisper prepare complete: prepared={len(prepared)} errors={len(errors)}"
    )
    return prepared, errors


def bind_runtime_local_whisper_evidence(
    report: dict[str, object],
    entries: list[LyricEntry],
    capability: LocalWhisperCapability | None,
    requests: tuple[LocalWhisperRequest, ...],
    outcomes: tuple[LocalWhisperOutcome, ...],
) -> None:
    if capability is None:
        return
    request_by_revision = {item.request_revision: item for item in requests}

    # Fixed-point recollection may legitimately stop requesting an entry after a
    # first-round local observation changes the candidate graph.  The observation
    # itself is immutable audio/lyric evidence and must not disappear merely
    # because the next planner round no longer needs to execute that request.
    # Seed from canonical-valid prior runtime evidence, then monotonically merge
    # any new observations.  Tampered/stale evidence fails the canonical accessor
    # and is therefore not retained.
    prior_runtime: dict[int, dict[str, object]] = {}
    previous_runtime = report.get(_RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY)
    if isinstance(previous_runtime, dict):
        for entry_index in sorted(previous_runtime):
            if not isinstance(entry_index, int) or not 0 <= entry_index < len(entries):
                continue
            retained = runtime_local_whisper_evidence_for_entry(report, entries, entry_index)
            if retained is not None:
                prior_runtime[entry_index] = retained
    retained_before_merge = len(prior_runtime)

    observations: dict[int, list[dict[str, object]]] = {
        index: [dict(value)] for index, value in prior_runtime.items()
    }
    for outcome in outcomes:
        if outcome.status != "evidence" or outcome.evidence_revision is None:
            continue
        request = request_by_revision.get(outcome.request_revision)
        if request is None or outcome.score is None:
            continue
        lyric_sha = hashlib.sha256(
            normalize_match_text(entry_sung_text(entries[outcome.entry_index])).encode("utf-8")
        ).hexdigest()
        payload = {
            "entry_index": outcome.entry_index,
            "lyric_sha256": lyric_sha,
            "onset_time": (
                round(float(outcome.onset_time), 6)
                if outcome.onset_time is not None else None
            ),
            "score": round(float(outcome.score), 6),
            "occurrence_binding": outcome.occurrence_binding,
            "request_revision": request.request_revision,
            "capability_revision": capability.capability_revision,
            "audio_revision": capability.audio_sha256,
            "window_start": round(request.window_start, 6),
            "window_end": round(request.window_end, 6),
            "raw_transcript_revision": outcome.raw_transcript_revision,
            "aligned_transcript_revision": outcome.aligned_transcript_revision,
            "producer": "local-whisper-aligner",
        }
        evidence = {
            **payload,
            "evidence_revision": _timing_content_digest(payload),
        }
        observations.setdefault(outcome.entry_index, []).append(evidence)

    composite_count = 0
    # A narrow probe may contain the clean forced-character onset while a wider
    # probe contains the sequence context needed to bind a repeated lyric.  They
    # are complementary observations from the same immutable audio/lyric source,
    # not independent votes.  Merge only when both observations are revision-
    # bound, their target times agree inside the existing association tolerance,
    # and the onset lies inside the context bracket.  The timestamp is never
    # averaged or moved: it remains the original direct onset observation.
    for entry_index, items in list(observations.items()):
        onset_items = [item for item in items if item.get("onset_time") is not None]
        context_items = [
            item for item in items
            if _occurrence_binding_has_strong_context(item.get("occurrence_binding"))
        ]
        composites: list[dict[str, object]] = []
        for onset_item in onset_items:
            try:
                onset_time = float(onset_item.get("onset_time"))
                onset_score = float(onset_item.get("score"))
            except (TypeError, ValueError):
                continue
            for context_item in context_items:
                context_binding = context_item.get("occurrence_binding")
                if context_binding == onset_item.get("occurrence_binding"):
                    continue
                if (
                    onset_item.get("lyric_sha256") != context_item.get("lyric_sha256")
                    or onset_item.get("capability_revision") != context_item.get("capability_revision")
                    or onset_item.get("audio_revision") != context_item.get("audio_revision")
                ):
                    continue
                try:
                    context_time = float(context_binding.get("timestamp"))  # type: ignore[union-attr]
                    context_score = float(context_item.get("score"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if (
                    not math.isfinite(onset_time)
                    or not math.isfinite(context_time)
                    or abs(onset_time - context_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
                ):
                    continue
                left_bound, right_bound = _occurrence_binding_time_bounds(
                    context_binding, entry_index
                )
                if left_bound is not None and onset_time <= left_bound - HOP_SECONDS:
                    continue
                if right_bound is not None and onset_time >= right_bound + HOP_SECONDS:
                    continue
                composite_payload = {
                    "entry_index": entry_index,
                    "lyric_sha256": onset_item.get("lyric_sha256"),
                    "onset_time": round(onset_time, 6),
                    "score": round(min(onset_score, context_score), 6),
                    "occurrence_binding": context_binding,
                    "request_revision": onset_item.get("request_revision"),
                    "capability_revision": onset_item.get("capability_revision"),
                    "audio_revision": onset_item.get("audio_revision"),
                    "window_start": round(float(onset_item.get("window_start")), 6),
                    "window_end": round(float(onset_item.get("window_end")), 6),
                    "raw_transcript_revision": onset_item.get("raw_transcript_revision"),
                    "aligned_transcript_revision": onset_item.get("aligned_transcript_revision"),
                    "producer": "local-whisper-aligner",
                    "composite_onset_evidence_revision": onset_item.get("evidence_revision"),
                    "composite_context_evidence_revision": context_item.get("evidence_revision"),
                }
                composite = {
                    **composite_payload,
                    "evidence_revision": _timing_content_digest(composite_payload),
                }
                composites.append(composite)
        if composites:
            unique = {item["evidence_revision"]: item for item in composites}
            observations[entry_index].extend(unique.values())
            composite_count += len(unique)

    def evidence_rank(item: dict[str, object]) -> tuple[int, int, float, str]:
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return (
            int(item.get("onset_time") is not None),
            int(_occurrence_binding_has_strong_context(item.get("occurrence_binding"))),
            score,
            str(item.get("evidence_revision") or ""),
        )

    runtime: dict[int, dict[str, object]] = {}
    for entry_index, items in observations.items():
        if not items:
            continue
        runtime[entry_index] = max(items, key=evidence_rank)

    report[_RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY] = runtime
    report["local_whisper_evidence"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "protocol_revision": capability.protocol_revision,
        "candidate_count": len(runtime),
        "retained_from_prior_collect_count": retained_before_merge,
        "composite_evidence_count": composite_count,
        "entries": [index + 1 for index in sorted(runtime)],
    }


def run_local_whisper_requests(
    requests: tuple[LocalWhisperRequest, ...],
    capability: LocalWhisperCapability,
    entries: list[LyricEntry],
    args: argparse.Namespace,
    *,
    cache: dict[str, tuple[str, object]],
    stats: dict[str, int],
) -> tuple[LocalWhisperOutcome, ...]:
    """Execute one frozen local-Whisper epoch and return outcomes in request order.

    Multi-request epochs always go through _prepare_local_whisper_batch (one
    whisper.cpp batch, one WhisperX alignment batch), whatever the worker
    count.  With one worker this path used to be skipped, so every request
    ran its own whisper.cpp + WhisperX (70 calls for one song), each taking
    the cross-song GPU lock.
    """
    workers = DEFAULT_EVIDENCE_WORKERS
    # A single request has nothing to batch.  Every larger epoch prepares
    # raw ASR and WhisperX alignment in one batch first; the worker count
    # only sets how many evaluations run at once.
    if len(requests) <= 1:
        outcomes: list[LocalWhisperOutcome] = []
        for ordinal, request in enumerate(requests, start=1):
            progress_log(
                f"local Whisper evidence: {ordinal}/{len(requests)} "
                f"entry={request.entry_index + 1} "
                f"window={request.window_start:.2f}-{request.window_end:.2f}s"
            )
            outcome = execute_local_whisper_request(
                request, capability, entries, args,
                cache=cache, runtime_stats=stats,
            )
            outcomes.append(outcome)
            progress_log(
                f"local Whisper evidence: entry={request.entry_index + 1} "
                f"status={outcome.status} reason={outcome.reason}"
            )
        return tuple(outcomes)
    else:
        # Requests are planned from one frozen pre-local-whisper graph and are
        # bound only after every outcome is collected.  They can therefore be
        # executed concurrently without changing Central selection semantics.
        # Preserve deterministic result ordering and legacy cache statistics.
        grouped: dict[str, list[tuple[int, LocalWhisperRequest]]] = {}
        for ordinal, request in enumerate(requests, start=1):
            key = _local_whisper_cache_key(request, capability)
            grouped.setdefault(key, []).append((ordinal, request))

        outcomes_by_ordinal: dict[int, LocalWhisperOutcome] = {}
        work: list[tuple[str, int, LocalWhisperRequest]] = []
        for key, group in grouped.items():
            cached = cache.get(key)
            if (
                isinstance(cached, tuple)
                and len(cached) == 2
                and isinstance(cached[1], LocalWhisperOutcome)
            ):
                for ordinal, grouped_request in group:
                    outcomes_by_ordinal[ordinal] = _local_whisper_outcome_for_request(
                        cached[1], grouped_request
                    )
                stats["cache_hits"] += len(group)
                continue
            first_ordinal, first_request = group[0]
            work.append((key, first_ordinal, first_request))
            stats["cache_misses"] += 1
            if len(group) > 1:
                stats["cache_hits"] += len(group) - 1

        progress_log(
            f"local Whisper evidence: internal_workers={workers} "
            f"unique_exec={len(work)}"
        )
        prepared_local, preparation_errors = _prepare_local_whisper_batch(
            work, capability, args
        )

        def _run_local_parallel(
            item: tuple[str, int, LocalWhisperRequest],
        ) -> tuple[str, int, LocalWhisperOutcome, dict[str, int], dict[str, tuple[str, object]]]:
            key, ordinal, request = item
            private_cache: dict[str, tuple[str, object]] = {}
            private_stats: dict[str, int] = {}
            progress_log(
                f"local Whisper evidence: start {ordinal}/{len(requests)} "
                f"entry={request.entry_index + 1} "
                f"window={request.window_start:.2f}-{request.window_end:.2f}s"
            )
            outcome = execute_local_whisper_request(
                request, capability, entries, args,
                cache=private_cache, runtime_stats=private_stats,
                prepared=prepared_local.get(key),
                preparation_error=preparation_errors.get(key),
            )
            return key, ordinal, outcome, private_stats, private_cache

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(work) or 1),
            thread_name_prefix="lrc-local-whisper",
        ) as pool:
            future_map = {pool.submit(_run_local_parallel, item): item for item in work}
            for future in concurrent.futures.as_completed(future_map):
                key, ordinal, outcome, private_stats, private_cache = future.result()
                cache.update(private_cache)
                stats["helper_invocations"] += private_stats.get(
                    "helper_invocations", 0
                )
                for grouped_ordinal, grouped_request in grouped[key]:
                    outcomes_by_ordinal[grouped_ordinal] = _local_whisper_outcome_for_request(
                        outcome, grouped_request
                    )
                progress_log(
                    f"local Whisper evidence: done {ordinal}/{len(requests)} "
                    f"entry={outcome.entry_index + 1} "
                    f"status={outcome.status} reason={outcome.reason}"
                )

        return tuple(
            outcomes_by_ordinal[ordinal]
            for ordinal in range(1, len(requests) + 1)
        )


def _local_whisper_vocal_fusion_candidate(
    *,
    entry_index: int,
    entry: LyricEntry,
    local_evidence: dict[str, object] | None,
    vocal_onset_features: AudioFeatures | None,
    audio_revision: str | None,
    occurrence_evidence: CandidateOccurrenceEvidence | None = None,
) -> TimingCandidate | None:
    """Fuse context-bound Local Whisper identity with an independent vocal onset.

    Local Whisper occurrence-only evidence already proves *which lyric row* is
    present, but intentionally does not manufacture an onset when exact forced
    characters are absent.  The existing isolated-vocal detector is a distinct
    acoustic producer.  When a production-grade vocal onset lies near the
    context-bound Local Whisper semantic time, their fusion may create a real
    boundary-grade candidate.  Global-unique-only identity is excluded here so a
    repeated lyric cannot acquire timing authority without contextual binding.
    """
    if (
        not isinstance(local_evidence, dict)
        or local_evidence.get("onset_time") is not None
        or vocal_onset_features is None
        or not audio_revision
    ):
        return None
    binding = local_evidence.get("occurrence_binding")
    if not _occurrence_binding_has_strong_context(binding):
        return None
    try:
        identity_time = float(binding.get("timestamp"))  # type: ignore[union-attr]
        identity_score = float(local_evidence.get("score"))
    except (TypeError, ValueError, AttributeError):
        return None
    if not math.isfinite(identity_time) or identity_score < TRUSTED_ALIGNMENT_SCORE:
        return None
    peak = _strong_vocal_onset_near(vocal_onset_features, identity_time)
    if peak is None:
        return None
    peak_time, peak_strength, peak_prominence = peak
    artifact = {
        "producer": "local-whisper-vocal-fusion",
        "local_whisper_evidence_revision": local_evidence.get("evidence_revision"),
        "occurrence_binding_revision": (
            binding.get("binding_revision") if isinstance(binding, dict) else None
        ),
        "local_identity_time": round(identity_time, 6),
        "local_identity_score": round(identity_score, 6),
        "vocal_onset_time": round(peak_time, 6),
        "vocal_onset_strength": round(peak_strength, 8),
        "vocal_onset_prominence": round(peak_prominence, 8),
        "audio_sha256": audio_revision,
        "feature_revision": "vocal-consonant-onset-v1",
        "runtime_only": True,
    }
    return make_timing_candidate(
        entry_index=entry_index,
        entry_text=entry_sung_text(entry),
        source="local-whisper-vocal-independent-fusion",
        raw_time=peak_time,
        spans=(),
        confidence=min(identity_score, peak_strength),
        identity_support="supported",
        sequence_support="supported",
        acoustic_support="supported",
        direct_onset_support="supported",
        acoustic_strength=peak_strength,
        acoustic_onset_time=peak_time,
        acoustic_source_artifact=artifact,
        acoustic_evidence_producer="local-whisper-vocal-fusion",
        acoustic_evidence_kind="vocal-onset",
        acoustic_evidence_independent=True,
        direct_onset_time=peak_time,
        direct_onset_source_artifact=artifact,
        direct_onset_evidence_producer="local-whisper-vocal-fusion",
        direct_onset_evidence_kind="vocal-onset",
        direct_onset_evidence_independent=False,
        occurrence_evidence=occurrence_evidence,
        independent_content_identity=True,
        current=False,
        source_artifact=artifact,
    )


def _augment_candidates_with_local_whisper_vocal_consensus(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
    vocal_onset_features: AudioFeatures | None,
    audio_revision: str | None,
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Add context-bound Local Whisper/vocal observations and CTC composites.

    This is an evidence-additive producer.  Local Whisper proves occurrence
    identity; the isolated-vocal detector proves a real acoustic onset; a CTC
    parent may contribute token geometry.  No timestamp is averaged: the
    standalone candidate uses the observed vocal onset, while the composite
    keeps the CTC parent's raw time and binds the same observed onset as direct
    onset evidence.  Whisper-derived parents are excluded so one model lineage
    can never count twice.
    """
    if vocal_onset_features is None or not audio_revision:
        return candidate_sets
    augmented = [list(items) for items in candidate_sets]
    for index, parents in enumerate(candidate_sets):
        evidence = runtime_local_whisper_evidence_for_entry(report, entries, index)
        local_vocal = _local_whisper_vocal_fusion_candidate(
            entry_index=index,
            entry=entries[index],
            local_evidence=evidence,
            vocal_onset_features=vocal_onset_features,
            audio_revision=audio_revision,
            occurrence_evidence=(
                candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="local-whisper-vocal-fusion",
                    occurrence_binding=(
                        evidence.get("occurrence_binding")
                        if isinstance(evidence, dict) else None
                    ),
                    upstream_evidence_revision=(
                        evidence.get("evidence_revision")
                        if isinstance(evidence, dict) else None
                    ),
                )
                if isinstance(evidence, dict) else None
            ),
        )
        if local_vocal is None:
            continue
        augmented[index].append(local_vocal)
        onset_evidence = local_vocal.direct_onset_evidence
        if onset_evidence is None:
            continue
        for parent in parents:
            families = _candidate_support_families(parent)
            if "ctc" not in families or "whisper-family" in families:
                continue
            if (
                parent.identity_support != "supported"
                or parent.sequence_support != "supported"
                or parent.right_edge_pileup
                or parent.window_truncated
                or abs(parent.written_time.seconds - local_vocal.written_time.seconds)
                > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            ):
                continue
            artifact = {
                "producer": "local-whisper-vocal-fusion",
                "local_vocal_candidate_id": local_vocal.candidate_id,
                "local_vocal_generation_revision": local_vocal.generation_revision,
                "local_whisper_evidence_revision": (
                    evidence.get("evidence_revision") if isinstance(evidence, dict) else None
                ),
                "parent_candidate_id": parent.candidate_id,
                "parent_generation_revision": parent.generation_revision,
                "parent_source": parent.source,
                "parent_time": round(parent.raw_time, 6),
                "vocal_onset_time": round(onset_evidence.onset_time, 6),
                "audio_sha256": audio_revision,
                "runtime_only": True,
            }
            augmented[index].append(make_timing_candidate(
                entry_index=index,
                entry_text=entry_sung_text(entries[index]),
                source="ctc-local-whisper-vocal-independent-consensus",
                raw_time=parent.raw_time,
                spans=parent.spans,
                confidence=min(parent.confidence, local_vocal.confidence),
                identity_support="supported",
                sequence_support="supported",
                acoustic_support="supported",
                direct_onset_support="supported",
                acoustic_strength=(
                    local_vocal.acoustic_evidence.strength
                    if local_vocal.acoustic_evidence is not None else None
                ),
                acoustic_onset_time=onset_evidence.onset_time,
                acoustic_source_artifact=artifact,
                acoustic_evidence_producer="local-whisper-vocal-fusion",
                acoustic_evidence_kind="vocal-onset",
                acoustic_evidence_independent=True,
                direct_onset_time=onset_evidence.onset_time,
                direct_onset_source_artifact=artifact,
                direct_onset_evidence_producer="local-whisper-vocal-fusion",
                direct_onset_evidence_kind="cross-backend-onset",
                direct_onset_evidence_independent=True,
                occurrence_evidence=candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="local-whisper-vocal-fusion",
                    occurrence_binding=evidence.get("occurrence_binding"),
                    upstream_evidence_revision=evidence.get("evidence_revision"),
                ),
                independent_content_identity=True,
                current=False,
                prefix_scope=parent.prefix_scope,
                right_edge_pileup=False,
                window_truncated=False,
                source_artifact=artifact,
            ))
    return tuple(
        tuple({item.candidate_id: item for item in items}[key] for key in sorted(
            {item.candidate_id: item for item in items}
        ))
        for items in augmented
    )


def _augment_candidates_with_local_whisper_consensus(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Fuse local Whisper onset/identity into existing candidates without moving time."""
    augmented = [list(items) for items in candidate_sets]
    for index, parents in enumerate(candidate_sets):
        evidence = runtime_local_whisper_evidence_for_entry(report, entries, index)
        if evidence is None:
            continue
        raw_onset_time = evidence.get("onset_time")
        if raw_onset_time is None:
            continue
        onset_time = float(raw_onset_time)
        evidence_score = float(evidence["score"])
        for parent in parents:
            families = _candidate_support_families(parent)
            # Local Whisper is one Whisper-family observation. Never use a
            # Whisper-derived parent to manufacture a second independent vote.
            if "whisper-family" in families:
                continue
            if (
                parent.identity_support != "supported"
                or parent.sequence_support != "supported"
                or parent.right_edge_pileup
                or parent.window_truncated
                or abs(parent.written_time.seconds - onset_time) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            ):
                continue
            if "ctc" in families:
                source = "ctc-local-whisper-independent-consensus"
            elif "recurrence" in families:
                source = "recurrence-local-whisper-independent-consensus"
            else:
                source = "local-whisper-independent-consensus"
            artifact = {
                "producer": "local-whisper-aligner",
                "local_whisper_evidence_revision": evidence["evidence_revision"],
                "parent_candidate_id": parent.candidate_id,
                "parent_generation_revision": parent.generation_revision,
                "parent_source": parent.source,
                "parent_time": round(parent.raw_time, 6),
                "onset_time": round(onset_time, 6),
                "runtime_only": True,
            }
            augmented[index].append(make_timing_candidate(
                entry_index=index,
                entry_text=entry_sung_text(entries[index]),
                source=source,
                raw_time=parent.raw_time,
                spans=parent.spans,
                confidence=min(parent.confidence, evidence_score),
                identity_support="supported",
                sequence_support="supported",
                acoustic_support="unavailable",
                direct_onset_support="supported",
                direct_onset_time=onset_time,
                direct_onset_source_artifact=artifact,
                direct_onset_evidence_producer="local-whisper-aligner",
                direct_onset_evidence_kind="cross-backend-onset",
                direct_onset_evidence_independent=True,
                occurrence_evidence=candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="local-whisper-aligner",
                    occurrence_binding=evidence.get("occurrence_binding"),
                    upstream_evidence_revision=evidence.get("evidence_revision"),
                ),
                independent_content_identity=True,
                current=False,
                prefix_scope=parent.prefix_scope,
                right_edge_pileup=False,
                window_truncated=False,
                source_artifact=artifact,
            ))
    return tuple(
        tuple({item.candidate_id: item for item in items}[key] for key in sorted(
            {item.candidate_id: item for item in items}
        ))
        for items in augmented
    )


def _apply_context_conditioned_whisper_hypothesis_arbitration(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
) -> tuple[tuple[tuple[TimingCandidate, ...], ...], tuple[dict[str, object], ...]]:
    """Invalidate weaker whole-song Whisper hypotheses with stronger bound context.

    This is *not* majority voting.  A global raw-ASR occurrence may be removed only
    when all of the following are provenance-bound facts:

    1. bounded local Whisper re-identifies the lyric inside a trusted full or
       sparse left/target/right sequence bracket and exposes either a direct aligned
       onset or an exact revision-bound independent vocal onset;
    2. an independent CTC+local-Whisper composite converges at that onset inside the
       existing 0.18 s tolerance;
    3. the global raw occurrence is only ``global-unique`` (no sequence bracket), is
       distant from the bounded onset, and is not independently corroborated by CTC
       or recurrence near its own time;
    4. the bounded observation is at least as confident as the raw identity row.

    The losing timestamp is not overwritten.  Its candidate is excluded from the
    central valid graph with an explicit diagnostic that binds both evidence
    revisions.  If any premise is absent, nothing is invalidated and fail-closed
    review behavior is preserved.
    """
    revised: list[tuple[TimingCandidate, ...]] = []
    diagnostics: list[dict[str, object]] = []
    tolerance = RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS

    for index, candidates in enumerate(candidate_sets):
        local = runtime_local_whisper_evidence_for_entry(report, entries, index)
        raw_identity = runtime_raw_content_identity_for_entry(report, entries, index)
        if not isinstance(local, dict) or not isinstance(raw_identity, dict):
            revised.append(candidates)
            continue
        local_binding = local.get("occurrence_binding")
        raw_binding = raw_identity.get("occurrence_binding")
        if not (
            _occurrence_binding_has_strong_context(local_binding)
            and isinstance(raw_binding, dict)
            and raw_binding.get("mode") == "global-unique"
        ):
            revised.append(candidates)
            continue
        try:
            local_score = float(local.get("score"))
            raw_time = float(raw_identity.get("raw_time"))
            raw_score = float(raw_identity.get("raw_score"))
        except (TypeError, ValueError):
            revised.append(candidates)
            continue

        # Independent corroboration must already exist as an actual composite;
        # raw CTC proximity or generic acoustic activity is not inferred here.
        raw_local_onset = local.get("onset_time")
        if raw_local_onset is not None:
            try:
                local_onset = float(raw_local_onset)
            except (TypeError, ValueError):
                revised.append(candidates)
                continue
            local_cluster = [
                candidate for candidate in candidates
                if candidate.source == "ctc-local-whisper-independent-consensus"
                and abs(candidate.written_time.seconds - local_onset) <= tolerance
                and candidate.identity_support == "supported"
                and candidate.sequence_support == "supported"
                and candidate_has_bound_direct_onset_evidence(candidate)
            ]
        else:
            # Occurrence-only Local Whisper can already have a boundary-grade
            # onset in the candidate graph via the independent vocal-fusion
            # producer.  Consume only an exact revision/binding match; never
            # synthesize an onset from the occurrence timestamp itself.
            local_revision = local.get("evidence_revision")
            binding_revision = (
                local_binding.get("binding_revision")
                if isinstance(local_binding, dict) else None
            )
            bound_vocal: list[tuple[TimingCandidate, float]] = []
            for candidate in candidates:
                if (
                    candidate.source != "ctc-local-whisper-vocal-independent-consensus"
                    or candidate.identity_support != "supported"
                    or candidate.sequence_support != "supported"
                    or not candidate_has_boundary_grade_direct_onset_evidence(candidate, entries)
                ):
                    continue
                occurrence = candidate.occurrence_evidence
                onset_evidence = candidate.direct_onset_evidence
                if (
                    occurrence is None
                    or onset_evidence is None
                    or occurrence.producer != "local-whisper-vocal-fusion"
                    or occurrence.upstream_evidence_revision != local_revision
                    or occurrence.binding_revision != binding_revision
                    or onset_evidence.producer != "local-whisper-vocal-fusion"
                ):
                    continue
                onset = float(onset_evidence.onset_time)
                if (
                    math.isfinite(onset)
                    and abs(candidate.written_time.seconds - onset) <= tolerance
                ):
                    bound_vocal.append((candidate, onset))
            canonical_onsets = {round(onset, 6) for _, onset in bound_vocal}
            if len(canonical_onsets) != 1:
                revised.append(candidates)
                continue
            local_onset = next(iter(canonical_onsets))
            local_cluster = [candidate for candidate, _ in bound_vocal]

        if (
            not all(math.isfinite(value) for value in (local_onset, local_score, raw_time, raw_score))
            or abs(local_onset - raw_time) <= tolerance
            or local_score < raw_score
            or not local_cluster
        ):
            revised.append(candidates)
            continue

        # If CTC or recurrence independently supports the raw cluster, it is a
        # genuine competing hypothesis and must remain review-required.
        raw_independent_support = any(
            abs(candidate.written_time.seconds - raw_time) <= tolerance
            and (
                "ctc" in _candidate_support_families(candidate)
                or "recurrence" in _candidate_support_families(candidate)
            )
            and "whisper-family" not in _candidate_support_families(candidate)
            for candidate in candidates
        )
        if raw_independent_support:
            revised.append(candidates)
            continue

        falsified = [
            candidate for candidate in candidates
            if "whisper-family" in _candidate_support_families(candidate)
            and "ctc" not in _candidate_support_families(candidate)
            and "recurrence" not in _candidate_support_families(candidate)
            and abs(candidate.written_time.seconds - raw_time) <= tolerance
            and abs(candidate.written_time.seconds - local_onset) > tolerance
        ]
        if not falsified:
            revised.append(candidates)
            continue
        falsified_ids = {candidate.candidate_id for candidate in falsified}
        survivors = tuple(candidate for candidate in candidates if candidate.candidate_id not in falsified_ids)
        if not survivors:
            revised.append(candidates)
            continue
        revised.append(survivors)
        diagnostics.append({
            "entry": index + 1,
            "reason": "bounded-sequence-context-falsifies-weaker-global-whisper-occurrence",
            "local_whisper_evidence_revision": local.get("evidence_revision"),
            "raw_content_identity_revision": raw_identity.get("evidence_revision"),
            "local_onset": round(local_onset, 6),
            "raw_identity_time": round(raw_time, 6),
            "local_score": round(local_score, 6),
            "raw_score": round(raw_score, 6),
            "context_binding_revision": local_binding.get("binding_revision"),
            "raw_binding_revision": raw_binding.get("binding_revision"),
            "corroborating_candidate_ids": sorted(candidate.candidate_id for candidate in local_cluster),
            "falsified_candidate_ids": sorted(falsified_ids),
            "max_delta_seconds": tolerance,
            "authority": "candidate-exclusion-only-central-still-commits",
        })

    return tuple(revised), tuple(diagnostics)


def _joint_previous_tail_evidence(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
    rows: list[object],
    parsed_rows: list[tuple[TimingTokenSpan, ...]],
    previous_probe: TimingCandidate | None,
    previous_pair_tail: TailRegionEvidence | None,
    target_probe: TimingCandidate,
    target_regions: CandidateRegionEvidence,
    output_revision: str,
    pair_gap_seconds: float | None,
    edge: dict[str, object],
    touches_edge: bool,
    touches_left_edge: bool,
    strong_joint_boundary: bool,
) -> IndependentTailEvidence | None:
    """Retain strong joint previous-row tail evidence in the existing sidecar.

    The helper's previous probe is an immutable known-lyric observation.  It is
    independent of the selected predecessor timestamp; request planning may use
    selected timestamps to choose a bounded window, but this evidence revision is
    built only from helper output, immutable lyric/transcript binding, audio and
    the probe's terminal geometry.  Weak/edge/truncated pairs fail closed.
    """
    if (
        not strong_joint_boundary
        or previous_probe is None
        or previous_pair_tail is None
        or previous_pair_tail.validity != "valid"
        or previous_pair_tail.reliable_tail_start is None
        or previous_pair_tail.reliable_tail_end is None
        or not math.isfinite(float(previous_pair_tail.reliable_tail_start))
        or not math.isfinite(float(previous_pair_tail.reliable_tail_end))
        or not math.isfinite(float(pair_gap_seconds or 0.0))
        or touches_edge
        or touches_left_edge
        or bool(edge.get("right_edge_pileup"))
        or previous_probe.right_edge_pileup
        or previous_probe.window_truncated
    ):
        return None
    previous_local_index = dict(request.row_mapping).get(request.entry_index - 1)
    if (
        previous_local_index is None
        or not 0 <= previous_local_index < len(parsed_rows)
        or previous_local_index >= len(rows)
        or not isinstance(rows[previous_local_index], dict)
    ):
        return None
    previous_spans = parsed_rows[previous_local_index]
    previous_span_payload = [
        [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
        for span in previous_spans
    ]
    target_span_payload = [
        [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
        for span in target_probe.spans
    ]
    evidence_payload = {
        "schema": "joint-previous-tail-v1",
        "producer": "joint-ctc-boundary-previous-tail",
        "entry_index": request.entry_index - 1,
        "context_entry_indexes": list(request.context_entry_indexes),
        "row_mapping": [list(pair) for pair in request.row_mapping],
        "row_mapping_revision": request.row_mapping_revision,
        "previous_lyric_digest": (
            request.transcript_digests[previous_local_index]
            if previous_local_index < len(request.transcript_digests)
            else None
        ),
        "transcript_digests": list(request.transcript_digests),
        "transcript_digest": request.transcript_digest,
        "capability_revision": capability.capability_revision,
        "helper_sha256": capability.helper_sha256,
        "helper_protocol_revision": capability.helper_protocol_revision,
        "alignment_audio_revision": request.alignment_audio_revision,
        "retry_epoch_revision": request.retry_epoch_revision,
        "request_revision": request.request_revision,
        "output_revision": output_revision,
        "previous_probe_candidate_id": previous_probe.candidate_id,
        "previous_probe_generation_revision": previous_probe.generation_revision,
        "previous_probe_region_revision": analyze_candidate_regions(previous_probe).region_revision,
        "previous_spans_revision": _timing_content_digest(previous_span_payload),
        "previous_spans": previous_span_payload,
        "target_spans_revision": _timing_content_digest(target_span_payload),
        "target_region_revision": target_regions.region_revision,
        "target_onset_proof_end": target_regions.onset_proof_end,
        "target_temporal_coherence": target_regions.temporal_coherence,
        "pair_gap_seconds": round(float(pair_gap_seconds), 6),
        "previous_pair_tail_revision": previous_pair_tail.tail_evidence_revision,
        "tail": {
            "validity": previous_pair_tail.validity,
            "start": round(float(previous_pair_tail.reliable_tail_start), 6),
            "end": round(float(previous_pair_tail.reliable_tail_end), 6),
            "uncertainty": round(float(previous_pair_tail.uncertainty_seconds or 0.0), 6),
            "confidence": round(float(previous_pair_tail.confidence or 0.0), 6),
            "terminal_span_indexes": list(previous_pair_tail.terminal_span_indexes),
            "local_gap_references": [
                round(float(value), 6)
                for value in previous_pair_tail.local_gap_references
            ],
        },
        "previous_touches_left_edge": bool(touches_left_edge),
        "previous_right_edge_pileup": bool(previous_probe.right_edge_pileup),
        "previous_window_truncated": bool(previous_probe.window_truncated),
        "target_right_edge_pileup": bool(edge.get("right_edge_pileup")),
        "target_touches_edge": bool(touches_edge),
    }
    entry_index = request.entry_index - 1
    tail_start = float(previous_pair_tail.reliable_tail_start)
    tail_end = float(previous_pair_tail.reliable_tail_end)
    uncertainty_seconds = max(0.0, float(previous_pair_tail.uncertainty_seconds or 0.0))
    confidence = _finite_unit_confidence(previous_pair_tail.confidence or 0.0)
    producer = "joint-ctc-boundary-previous-tail"
    provenance = _make_tail_provenance(
        entry_index=entry_index,
        tail_start=tail_start,
        tail_end=tail_end,
        uncertainty_seconds=uncertainty_seconds,
        confidence=confidence,
        producer=producer,
        backend_lineage=capability.backend_lineage,
        model_identity=capability.model_identity,
        helper_revision=_tail_helper_revision(
            capability.helper_sha256,
            capability.helper_protocol_revision,
        ),
        capability_revision=capability.capability_revision,
        audio_revision=request.alignment_audio_revision,
        transcript_revision=request.transcript_digest,
        row_revision=request.row_mapping_revision,
    )
    source_revision = provenance.observation_revision if provenance is not None else _timing_content_digest(evidence_payload)
    return IndependentTailEvidence(
        entry_index=entry_index,
        tail_start=tail_start,
        tail_end=tail_end,
        uncertainty_seconds=uncertainty_seconds,
        confidence=confidence,
        producer=producer,
        source_revision=source_revision,
        provenance=provenance,
    )


def parse_crossline_retry_output(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
    entries: list[LyricEntry],
    payload: object,
) -> CrossLineRetryOutcome:
    def invalid(reason: str) -> CrossLineRetryOutcome:
        base = {
            "retry_epoch_revision": request.retry_epoch_revision,
            "request_revision": request.request_revision,
            "entry_index": request.entry_index,
            "status": "invalid",
            "reason": reason,
            "candidate": None,
            "output_revision": None,
        }
        return CrossLineRetryOutcome(**base, outcome_revision=_timing_content_digest(base))

    if not isinstance(payload, dict):
        return invalid("helper-output-not-object")
    try:
        output_revision = _canonical_json_digest(payload)
    except LrcError:
        return invalid("helper-output-nonfinite-or-noncanonical")
    rows = payload.get("entries")
    if not isinstance(rows, list) or len(rows) != len(request.context_entry_indexes):
        return invalid("helper-row-count-mismatch")
    try:
        payload_window_start = float(payload.get("window_start", -1.0))
        payload_window_end = float(payload.get("window_end", -1.0))
    except (TypeError, ValueError):
        return invalid("helper-window-malformed")
    if round(payload_window_start, 3) != round(request.window_start, 3):
        return invalid("helper-window-start-mismatch")
    if round(payload_window_end, 3) != round(request.window_end, 3):
        return invalid("helper-window-end-mismatch")
    parsed_rows: list[tuple[TimingTokenSpan, ...]] = []
    previous_end = request.window_start
    for local_index, (global_index, row) in enumerate(zip(request.context_entry_indexes, rows)):
        if not isinstance(row, dict) or row.get("entry") != local_index + 1:
            return invalid("helper-row-mapping-mismatch")
        if row.get("text") != entry_sung_text(entries[global_index]):
            return invalid("helper-transcript-mismatch")
        raw_spans = row.get("token_spans")
        romaji = row.get("romaji")
        tokens = row.get("tokens")
        if (
            not isinstance(raw_spans, list)
            or not raw_spans
            or not isinstance(romaji, str)
            or not isinstance(tokens, int)
            or tokens != len(raw_spans)
            or len(romaji) != len(raw_spans)
        ):
            return invalid("helper-row-incomplete")
        spans: list[TimingTokenSpan] = []
        for token_index, raw_span in enumerate(raw_spans):
            if not isinstance(raw_span, dict):
                return invalid("helper-span-malformed")
            try:
                start = float(raw_span.get("start"))
                end = float(raw_span.get("end"))
                score = float(raw_span.get("score"))
            except (TypeError, ValueError):
                return invalid("helper-span-malformed")
            if (
                not all(math.isfinite(value) for value in (start, end, score))
                or start > end
                or start < request.window_start - HOP_SECONDS
                or end > request.window_end + HOP_SECONDS
                or start < previous_end
            ):
                return invalid("helper-span-nonmonotonic-or-outside-window")
            token = str(raw_span.get("char") or "")
            if token != romaji[token_index]:
                return invalid("helper-token-sequence-mismatch")
            spans.append(TimingTokenSpan(start, end, score, token))
            previous_end = end
        try:
            row_start = float(row.get("start"))
            row_end = float(row.get("end"))
        except (TypeError, ValueError):
            return invalid("helper-row-boundary-missing")
        if (
            not math.isfinite(row_start)
            or not math.isfinite(row_end)
            or row_start != spans[0].start
            or row_end != spans[-1].end
        ):
            return invalid("helper-row-boundary-mismatch")
        parsed_rows.append(tuple(spans))
    mapping = dict(request.row_mapping)
    target_local_index = mapping.get(request.entry_index)
    if target_local_index is None or not 0 <= target_local_index < len(parsed_rows):
        return invalid("target-row-mapping-missing")
    target_row = rows[target_local_index]
    assert isinstance(target_row, dict)
    target_spans = parsed_rows[target_local_index]
    joint_pair = bool(
        len(request.context_entry_indexes) == 2
        and request.context_entry_indexes == (request.entry_index - 1, request.entry_index)
        and target_local_index == 1
    )
    local_window = {"end": request.window_end}
    edge = ctc_right_edge_pileup_evidence({
        "ctc_local_window": local_window,
        "ctc_token_spans": [
            {"start": span.start, "end": span.end, "score": span.score}
            for span in target_spans
        ],
    })
    edge_margin = request.window_end - target_spans[-1].end
    touches_edge = -0.03 <= edge_margin <= 0.10
    touches_left_edge = False
    previous_probe: TimingCandidate | None = None
    previous_pair_tail: TailRegionEvidence | None = None
    pair_gap_seconds: float | None = None
    if joint_pair:
        previous_local_index = dict(request.row_mapping).get(request.entry_index - 1)
        if previous_local_index is None or not 0 <= previous_local_index < len(parsed_rows):
            return invalid("joint-previous-row-mapping-missing")
        previous_row = rows[previous_local_index]
        assert isinstance(previous_row, dict)
        previous_spans = parsed_rows[previous_local_index]
        touches_left_edge = (
            -0.03 <= previous_spans[0].start - request.window_start <= 0.10
        )
        try:
            previous_confidence = float(previous_row.get("ctc_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return invalid("helper-previous-score-malformed")
        if not math.isfinite(previous_confidence):
            return invalid("helper-previous-score-nonfinite")
        previous_probe = make_timing_candidate(
            entry_index=request.entry_index - 1,
            entry_text=entry_sung_text(entries[request.entry_index - 1]),
            source="ctc-joint-boundary-previous-probe",
            raw_time=previous_spans[0].start,
            spans=previous_spans,
            confidence=previous_confidence,
            identity_support="supported",
            sequence_support="supported",
            current=False,
            source_artifact={
                "output_revision": output_revision,
                "row_mapping_revision": request.row_mapping_revision,
                "joint_boundary_probe": True,
            },
        )
        previous_pair_tail = derive_reliable_tail(previous_probe)
        pair_gap_seconds = target_spans[0].start - previous_spans[-1].end
    try:
        confidence = float(target_row.get("ctc_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return invalid("helper-target-score-malformed")
    if not math.isfinite(confidence):
        return invalid("helper-target-score-nonfinite")

    probe = make_timing_candidate(
        entry_index=request.entry_index,
        entry_text=entry_sung_text(entries[request.entry_index]),
        source="ctc-joint-boundary-current-probe" if joint_pair else "ctc-opening-local-retry-probe",
        raw_time=target_spans[0].start,
        spans=target_spans,
        confidence=confidence,
        identity_support="supported",
        sequence_support="supported",
        current=False,
        prefix_scope=False,
        right_edge_pileup=bool(edge.get("right_edge_pileup")),
        window_truncated=bool(touches_edge or touches_left_edge),
        source_artifact={"probe": True, "output_revision": output_revision},
    )
    target_regions = analyze_candidate_regions(probe)
    strong_joint_boundary = bool(
        joint_pair
        and previous_pair_tail is not None
        and previous_pair_tail.validity == "valid"
        and previous_pair_tail.reliable_tail_end is not None
        and previous_pair_tail.reliable_tail_end <= target_spans[0].start
        and pair_gap_seconds is not None
        and pair_gap_seconds >= -HOP_SECONDS * 0.51
        and target_regions.temporal_coherence == "coherent"
        and target_regions.onset_proof_end is not None
        and not bool(edge.get("right_edge_pileup"))
        and not touches_edge
        and not touches_left_edge
    )
    source = (
        "ctc-joint-boundary-observation"
        if strong_joint_boundary
        else "ctc-joint-boundary-observation-weak"
        if joint_pair
        else "ctc-opening-local-retry"
    )
    candidate = make_timing_candidate(
        entry_index=request.entry_index,
        entry_text=entry_sung_text(entries[request.entry_index]),
        source=source,
        raw_time=target_spans[0].start,
        spans=target_spans,
        confidence=confidence,
        identity_support="supported",
        sequence_support="supported",
        current=False,
        prefix_scope=False,
        right_edge_pileup=bool(edge.get("right_edge_pileup")),
        window_truncated=bool(touches_edge or touches_left_edge),
        source_artifact={
            "capability_revision": capability.capability_revision,
            "retry_semantic_revision": _crossline_retry_semantic_key(request, capability),
            "helper_input_revision": _crossline_retry_helper_cache_key(request, capability),
            "output_revision": output_revision,
            "transcript_digest": request.transcript_digest,
            "row_mapping_revision": request.row_mapping_revision,
            "joint_boundary": joint_pair,
            "strong_joint_boundary": strong_joint_boundary,
            "pair_gap_seconds": round(pair_gap_seconds, 6) if pair_gap_seconds is not None else None,
            "pair_previous_tail_revision": (
                previous_pair_tail.tail_evidence_revision
                if previous_pair_tail is not None else None
            ),
        },
    )
    previous_tail_evidence = _joint_previous_tail_evidence(
        request,
        capability,
        rows,
        parsed_rows,
        previous_probe,
        previous_pair_tail,
        probe,
        target_regions,
        output_revision,
        pair_gap_seconds,
        edge,
        touches_edge,
        touches_left_edge,
        strong_joint_boundary,
    )
    base = {
        "retry_epoch_revision": request.retry_epoch_revision,
        "request_revision": request.request_revision,
        "entry_index": request.entry_index,
        "status": "candidate",
        "reason": ("complete-joint-boundary-rows" if joint_pair else "complete-known-lyric-target-row"),
        "candidate": candidate,
        "output_revision": output_revision,
        "previous_tail_evidence": previous_tail_evidence,
    }
    return CrossLineRetryOutcome(**base, outcome_revision=_timing_content_digest({
        **base, "candidate": candidate.candidate_id,
        "previous_tail_evidence": (
            {
                "entry_index": previous_tail_evidence.entry_index,
                "tail_start": previous_tail_evidence.tail_start,
                "tail_end": previous_tail_evidence.tail_end,
                "uncertainty_seconds": previous_tail_evidence.uncertainty_seconds,
                "confidence": previous_tail_evidence.confidence,
                "producer": previous_tail_evidence.producer,
                "source_revision": previous_tail_evidence.source_revision,
            }
            if previous_tail_evidence is not None
            else None
        ),
    }))


def _crossline_retry_helper_cache_key(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
) -> str:
    """Digest only the actual helper inputs, excluding retry-epoch bookkeeping."""
    return _timing_content_digest({
        "schema": 1,
        "helper_sha256": capability.helper_sha256,
        "helper_protocol_revision": capability.helper_protocol_revision,
        "alignment_audio_sha256": request.alignment_audio_revision,
        "model_identity": capability.model_identity,
        "model_blob_sha256": _ctc_model_blob_sha256(),
        "device": capability.device,
        "transcript_digest": request.transcript_digest,
        "window_start": request.window_start,
        "window_end": request.window_end,
    })


def _crossline_retry_semantic_key(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
) -> str:
    """Stable retry meaning, excluding epoch/request bookkeeping.

    The helper inference key captures the actual model inputs.  Entry identity and
    row mapping are added here because the same acoustic/helper payload can be
    interpreted for different immutable lyric entries, which must remain distinct
    central candidates.
    """
    return _timing_content_digest({
        "schema": 1,
        "entry_index": request.entry_index,
        "row_mapping_revision": request.row_mapping_revision,
        "helper_input_revision": _crossline_retry_helper_cache_key(request, capability),
    })


def _novel_crossline_retry_requests(
    requests: tuple[CrossLineRetryRequest, ...],
    capability: BoundedRetryCapability,
    processed_semantics: set[str],
) -> tuple[tuple[CrossLineRetryRequest, ...], tuple[str, ...]]:
    """Return only requests that can produce genuinely new retry evidence."""
    novel: list[CrossLineRetryRequest] = []
    semantic_keys: list[str] = []
    for request in sorted(requests, key=lambda item: (item.entry_index, item.request_revision)):
        semantic_key = _crossline_retry_semantic_key(request, capability)
        if semantic_key in processed_semantics:
            continue
        novel.append(request)
        semantic_keys.append(semantic_key)
    return tuple(novel), tuple(semantic_keys)


def invoke_crossline_retry(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
    entries: list[LyricEntry],

    *,
    helper_cache: dict[str, tuple[str, object]] | None = None,
    timeout_seconds: float = 180.0,
    cache_policy: "model_cache.ModelCachePolicy | None" = None,
) -> CrossLineRetryOutcome:
    def error_outcome(reason: str) -> CrossLineRetryOutcome:
        base = {
            "retry_epoch_revision": request.retry_epoch_revision,
            "request_revision": request.request_revision,
            "entry_index": request.entry_index,
            "status": "error",
            "reason": reason,
            "candidate": None,
            "output_revision": None,
        }
        return CrossLineRetryOutcome(**base, outcome_revision=_timing_content_digest(base))

    # R2L performance-only cache fast path: helper payload cache hits are
    # immutable model outputs.  Do not create a temp directory/transcript/command
    # merely to discover that the helper input was already evaluated.  Parsing
    # still runs against the *current* immutable request, preserving all epoch,
    # ownership, and candidate semantics.
    cache_key = _crossline_retry_helper_cache_key(request, capability)
    cached = helper_cache.get(cache_key) if helper_cache is not None else None
    cache_identity = model_cache.identity_crossline_joint_ctc(
        alignment_audio_sha256=request.alignment_audio_revision,
        model_identity=capability.model_identity,
        model_blob_sha256=_ctc_model_blob_sha256(),
        helper_protocol_revision=capability.helper_protocol_revision,
        transcript_digest=request.transcript_digest,
        window_start=request.window_start,
        window_end=request.window_end,
        context_entry_indexes=list(request.context_entry_indexes),
    )
    if cache_policy is None:
        cache_policy = model_cache.policy_from_env_and_args(None)
    if cached is None:
        persisted = _model_cache_read("crossline-joint-ctc", cache_key, cache_identity, cache_policy)
        if persisted is not None and helper_cache is not None:
            if isinstance(persisted, dict) and persisted.get("kind") in {"payload", "error"}:
                helper_cache[cache_key] = (str(persisted["kind"]), persisted.get("value"))
                cached = helper_cache[cache_key]
    if cached is not None:
        result_kind, result_value = cached
        if result_kind == "error":
            return error_outcome(str(result_value))
        return parse_crossline_retry_output(request, capability, entries, result_value)

    with tempfile.TemporaryDirectory(prefix="lrc-crossline-retry-") as temp_name:
        temp_dir = Path(temp_name)
        transcript_path = temp_dir / "transcript.json"
        output_path = temp_dir / "result.json"
        transcript_path.write_text(
            json.dumps(
                [entry_sung_text(entries[index]) for index in request.context_entry_indexes],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        command = [
            str(default_ctc_python()), capability.helper_path,
            "--audio", capability.alignment_audio_path,
            "--transcript", str(transcript_path),
            "--output", str(output_path),
            "--device", capability.device,
            "--start", f"{request.window_start:.3f}",
            "--end", f"{request.window_end:.3f}",
        ]
        try:
            proc = _run_model_helper_command(command, mode="ctc", timeout_seconds=timeout_seconds)
        except subprocess.TimeoutExpired:
            result: tuple[str, object] = ("error", "helper-timeout")
        except (LrcError, OSError, subprocess.SubprocessError):
            result = ("error", "helper-invocation-error")
        else:
            if proc.returncode != 0 or not output_path.is_file():
                result = ("error", "helper-invocation-failed")
            else:
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = None
                result = ("payload", payload)
        # Compare kind/value separately: a successful payload is a dict and
        # therefore unhashable, so tuple-in-set membership would raise.
        transient = (
            result[0] == "error"
            and result[1] in {"helper-timeout", "helper-invocation-error"}
        ) or (result[0] == "payload" and result[1] is None)
        if helper_cache is not None and not transient:
            # R3-6: transient helper failures must not suppress this evidence
            # for every later recollection in the same run.
            helper_cache[cache_key] = result
            _model_cache_write(
                "crossline-joint-ctc", cache_key, cache_identity,
                {"kind": result[0], "value": result[1]},
                cache_policy,
            )
        result_kind, result_value = result
        if result_kind == "error":
            return error_outcome(str(result_value))
        payload = result_value
    return parse_crossline_retry_output(request, capability, entries, payload)


def record_crossline_retry_identity_hypothesis(
    report: dict[str, object],
    outcome: CrossLineRetryOutcome,
) -> dict[str, object] | None:
    """Publish a completed bounded retry as identity-only evidence.

    The retry helper aligned the current immutable lyric row inside a bounded
    audio window.  A complete, non-truncated row can therefore establish local
    occurrence identity for later reciprocal-recurrence reasoning.  It does
    *not* establish an onset by itself and is deliberately excluded from the
    ordinary central timing-candidate collector.
    """
    candidate = outcome.candidate
    if (
        outcome.status != "candidate"
        or candidate is None
        or candidate.source != "ctc-opening-local-retry"
        or candidate.identity_support != "supported"
        or candidate.sequence_support != "supported"
        or len(candidate.spans) < 4
        or candidate.window_truncated
        or candidate.right_edge_pileup
    ):
        return None
    assignments = report.get("assignments")
    if (
        not isinstance(assignments, list)
        or not (0 <= outcome.entry_index < len(assignments))
        or not isinstance(assignments[outcome.entry_index], dict)
    ):
        return None
    assignment = assignments[outcome.entry_index]
    assert isinstance(assignment, dict)
    spans = [
        {
            "char": span.token,
            "start": round(float(span.start), 6),
            "end": round(float(span.end), 6),
            "score": round(float(span.score), 8),
        }
        for span in candidate.spans
    ]
    window = {
        "candidate_id": candidate.candidate_id,
        "generation_revision": candidate.generation_revision,
        "request_revision": outcome.request_revision,
        "retry_epoch_revision": outcome.retry_epoch_revision,
        "output_revision": outcome.output_revision,
        "outcome_revision": outcome.outcome_revision,
        "status": outcome.status,
        "reason": outcome.reason,
        "identity_only": True,
    }
    before = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else 0
    hypothesis = record_alignment_hypothesis(
        assignment,
        source="ctc-opening-local-retry-identity",
        stage="bounded-crossline-retry-identity",
        raw_time=candidate.raw_time,
        confidence=candidate.confidence,
        spans=spans,
        identity_support="supported",
        sequence_support="supported",
        acoustic_support="unavailable",
        direct_onset_support="unavailable",
        parent_source="ctc-opening-local-retry",
        parent_time=candidate.raw_time,
        window=window,
        mode="identity-only-complete-known-lyric-row",
        right_edge_pileup=False,
        window_truncated=False,
    )
    after = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else before
    return hypothesis if after > before else None


def execute_crossline_retry_epoch(
    epoch: BaseRetryEpoch,
    capability: BoundedRetryCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    requests: tuple[CrossLineRetryRequest, ...],
    *,
    helper_cache: dict[str, tuple[str, object]] | None = None,
    timeout_seconds: float = 180.0,
    pass_index: int | str | None = None,
    runtime_stats: dict[str, int] | None = None,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
    worker_count: int = 1,
) -> tuple[CrossLineRetryOutcome, ...]:
    """Execute one frozen retry epoch.

    Performance mode may run independent helper inputs concurrently.  Planning,
    request-currentness checks, cache semantics, result ordering, and all Central
    mutations remain outside the concurrent section and therefore deterministic.
    """
    if capability is None:
        return ()

    seen: set[tuple[str, int]] = set()
    ordered_requests = sorted(requests, key=lambda item: (item.entry_index, item.request_revision))
    pass_label = str(pass_index) if pass_index is not None else "?"
    total = len(ordered_requests)

    # R2L performance-only planning dedup: request currentness is evaluated
    # against a frozen epoch/candidate/tail state.  The legacy per-request
    # helper recomputed the complete planner output for every row even though
    # those inputs cannot change inside this epoch.  Compute each applicable
    # planner exactly once, then preserve the same membership test.
    needs_joint = any(len(item.context_entry_indexes) == 2 for item in ordered_requests)
    needs_regular = any(len(item.context_entry_indexes) != 2 for item in ordered_requests)
    current_joint = set(
        plan_joint_boundary_retry_requests(
            epoch, capability, entries, candidate_sets, duration,
            independent_tail_evidence, tail_evidence_context,
            timing_source_policy,
        )
    ) if needs_joint else set()
    current_regular = set(
        plan_crossline_retry_requests(
            epoch, capability, entries, candidate_sets, duration,
            independent_tail_evidence, tail_evidence_context,
            timing_source_policy,
        )
    ) if needs_regular else set()

    prepared: list[tuple[int, CrossLineRetryRequest, str]] = []
    for ordinal, request in enumerate(ordered_requests, start=1):
        ledger_key = (request.retry_epoch_revision, request.entry_index)
        if ledger_key in seen:
            continue
        seen.add(ledger_key)
        current_requests = current_joint if len(request.context_entry_indexes) == 2 else current_regular
        if request not in current_requests:
            continue
        prepared.append((ordinal, request, _crossline_retry_helper_cache_key(request, capability)))

    if not prepared:
        return ()

    worker_count = max(1, min(int(worker_count or 1), 4))
    if worker_count == 1 or len(prepared) <= 1:
        outcomes: list[CrossLineRetryOutcome] = []
        for ordinal, request, cache_key in prepared:
            cache_hit = helper_cache is not None and cache_key in helper_cache
            if runtime_stats is not None:
                runtime_stats["cache_hits" if cache_hit else "cache_misses"] = (
                    runtime_stats.get("cache_hits" if cache_hit else "cache_misses", 0) + 1
                )
                if not cache_hit:
                    runtime_stats["helper_invocations"] = runtime_stats.get("helper_invocations", 0) + 1
            progress_log(
                f"crossline retry pass {pass_label}: {ordinal}/{total} "
                f"entry={request.entry_index + 1} "
                f"window={request.window_start:.2f}-{request.window_end:.2f}s "
                f"cache={'hit' if cache_hit else 'miss'}"
            )
            if helper_cache is None and timeout_seconds == 180.0:
                outcome = invoke_crossline_retry(request, capability, entries)
            else:
                outcome = invoke_crossline_retry(
                    request, capability, entries,
                    helper_cache=helper_cache, timeout_seconds=timeout_seconds,
                )
            if runtime_stats is not None and outcome.reason == "helper-timeout":
                runtime_stats["timeouts"] = runtime_stats.get("timeouts", 0) + 1
            progress_log(
                f"crossline retry pass {pass_label}: entry={request.entry_index + 1} "
                f"status={outcome.status} reason={outcome.reason}"
            )
            outcomes.append(outcome)
        return tuple(outcomes)

    # Recreate the exact serial cache semantics before dispatch: for every helper
    # input already cached, all requests are hits; otherwise the first request is
    # the one real invocation and later requests sharing that helper input are hits.
    grouped: dict[str, list[tuple[int, CrossLineRetryRequest]]] = {}
    for ordinal, request, cache_key in prepared:
        grouped.setdefault(cache_key, []).append((ordinal, request))

    outcomes_by_ordinal: dict[int, CrossLineRetryOutcome] = {}
    work: list[tuple[str, int, CrossLineRetryRequest]] = []
    for cache_key, group in grouped.items():
        pre_cached = helper_cache is not None and cache_key in helper_cache
        if runtime_stats is not None:
            if pre_cached:
                runtime_stats["cache_hits"] = runtime_stats.get("cache_hits", 0) + len(group)
            else:
                runtime_stats["cache_misses"] = runtime_stats.get("cache_misses", 0) + 1
                runtime_stats["cache_hits"] = runtime_stats.get("cache_hits", 0) + max(0, len(group) - 1)
                runtime_stats["helper_invocations"] = runtime_stats.get("helper_invocations", 0) + 1
        if pre_cached:
            for ordinal, request in group:
                progress_log(
                    f"crossline retry pass {pass_label}: cached {ordinal}/{total} "
                    f"entry={request.entry_index + 1}"
                )
                outcomes_by_ordinal[ordinal] = invoke_crossline_retry(
                    request, capability, entries,
                    helper_cache=helper_cache, timeout_seconds=timeout_seconds,
                )
        else:
            first_ordinal, first_request = group[0]
            work.append((cache_key, first_ordinal, first_request))

    progress_log(
        f"crossline retry pass {pass_label}: parallel_workers={worker_count} "
        f"unique_exec={len(work)}"
    )

    def _run_retry_parallel(
        item: tuple[str, int, CrossLineRetryRequest],
    ) -> tuple[str, int, CrossLineRetryOutcome, dict[str, tuple[str, object]]]:
        cache_key, ordinal, request = item
        private_cache: dict[str, tuple[str, object]] = {}
        progress_log(
            f"crossline retry pass {pass_label}: start {ordinal}/{total} "
            f"entry={request.entry_index + 1} "
            f"window={request.window_start:.2f}-{request.window_end:.2f}s"
        )
        outcome = invoke_crossline_retry(
            request, capability, entries,
            helper_cache=private_cache, timeout_seconds=timeout_seconds,
        )
        return cache_key, ordinal, outcome, private_cache

    if work:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(worker_count, len(work)),
            thread_name_prefix="lrc-ctc-retry",
        ) as pool:
            future_map = {pool.submit(_run_retry_parallel, item): item for item in work}
            for future in concurrent.futures.as_completed(future_map):
                cache_key, ordinal, outcome, private_cache = future.result()
                if helper_cache is not None:
                    helper_cache.update(private_cache)
                outcomes_by_ordinal[ordinal] = outcome
                progress_log(
                    f"crossline retry pass {pass_label}: done {ordinal}/{total} "
                    f"entry={outcome.entry_index + 1} "
                    f"status={outcome.status} reason={outcome.reason}"
                )
                # Same helper payload may map to more than one immutable request.
                # Parse those additional requests only after the representative
                # payload is in the shared cache; no second model invocation occurs.
                for grouped_ordinal, grouped_request in grouped[cache_key][1:]:
                    outcomes_by_ordinal[grouped_ordinal] = invoke_crossline_retry(
                        grouped_request, capability, entries,
                        helper_cache=helper_cache, timeout_seconds=timeout_seconds,
                    )

    ordered_outcomes = tuple(
        outcomes_by_ordinal[ordinal]
        for ordinal, _request, _cache_key in prepared
        if ordinal in outcomes_by_ordinal
    )
    if runtime_stats is not None:
        runtime_stats["timeouts"] = runtime_stats.get("timeouts", 0) + sum(
            item.reason == "helper-timeout" for item in ordered_outcomes
        )
    return ordered_outcomes


def _retry_diagnostic_payload(item: object) -> dict[str, object]:
    if isinstance(item, CrossLineRetryRequest):
        return {
            "entry": item.entry_index + 1,
            "ordinal": item.ordinal,
            "retry_epoch_revision": item.retry_epoch_revision,
            "request_revision": item.request_revision,
            "capability_revision": item.capability_revision,
            "alignment_audio_revision": item.alignment_audio_revision,
            "current": item.current.__dict__,
            "previous": item.previous.__dict__,
            "next_bound": item.next_bound.__dict__ if item.next_bound else None,
            "ownership_relation_revision": item.ownership_relation_revision,
            "previous_tail_evidence_revision": item.previous_tail_evidence_revision,
            "previous_tail": {
                "start": item.previous_tail_start,
                "end": item.previous_tail_end,
                "uncertainty": item.previous_tail_uncertainty,
                "validity": item.previous_tail_validity,
            },
            "audio_end_seconds": item.audio_end_seconds,
            "audio_duration_revision": item.audio_duration_revision,
            "context_entry_indexes": list(item.context_entry_indexes),
            "context_candidate_ids": list(item.context_candidate_ids),
            "transcript_digests": list(item.transcript_digests),
            "window": [item.window_start, item.window_end],
            "window_definition_revision": item.window_definition_revision,
            "transcript_digest": item.transcript_digest,
            "row_mapping": [list(pair) for pair in item.row_mapping],
            "row_mapping_revision": item.row_mapping_revision,
        }
    if isinstance(item, CrossLineRetryOutcome):
        return {
            "entry": item.entry_index + 1,
            "retry_epoch_revision": item.retry_epoch_revision,
            "request_revision": item.request_revision,
            "status": item.status,
            "reason": item.reason,
            "candidate_id": item.candidate.candidate_id if item.candidate else None,
            "output_revision": item.output_revision,
            "outcome_revision": item.outcome_revision,
            "previous_tail_evidence": _independent_tail_evidence_payload(
                item.previous_tail_evidence
            ),
        }
    raise TypeError("Unsupported retry diagnostic item")


def _retry_capability_diagnostic_payload(
    capability: BoundedRetryCapability | None,
) -> dict[str, object] | None:
    if capability is None:
        return None
    return {
        "kind": capability.kind,
        "backend_lineage": capability.backend_lineage,
        "helper_path": capability.helper_path,
        "helper_sha256": capability.helper_sha256,
        "helper_protocol_revision": capability.helper_protocol_revision,
        "alignment_audio_path": capability.alignment_audio_path,
        "alignment_audio_sha256": capability.alignment_audio_sha256,
        "alignment_audio_source": capability.alignment_audio_source,
        "sample_rate": capability.sample_rate,
        "model_identity": capability.model_identity,
        "device": capability.device,
        "capability_revision": capability.capability_revision,
    }


def collect_central_timing_candidates(
    audio_path: Path,
    entries: list[LyricEntry],
    scratch_timestamps: list[float],
    report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
    independent_evidence_audio_path: Path | None = None,
    retry_helper_cache: dict[str, tuple[str, object]] | None = None,
    vocal_onset_features: AudioFeatures | None = None,
) -> tuple[tuple[tuple[TimingCandidate, ...], ...], dict[str, object]]:
    """Collect immutable candidates without mutating final timing state.

    Inactive legacy boundary helpers are not called or imported here.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        raise LrcError("Central timing finalizer requires one assignment per lyric entry")
    if len(scratch_timestamps) != len(entries):
        raise LrcError("Central timing scratch clock length does not match lyric entries")
    # Runtime-only immutable lyric objects are retained so content identity can
    # be validated independently from timing provenance during central summary
    # and commit.  This key is explicitly excluded from serialized reports.
    report[_RUNTIME_LYRIC_ENTRIES_KEY] = tuple(entries)
    assert_lyric_identity_contract(entries, report, initialize=True)

    # The legacy prefix guard remains available for historical diagnostics, but
    # it is not executed by production candidate generation and contributes no
    # candidate, trusted bit, recovery, or flag clearing.
    evidence_audio_path = independent_evidence_audio_path or audio_path
    fusion_audio_revision: str | None = None
    if vocal_onset_features is not None:
        try:
            fusion_audio_revision = _path_sha256(evidence_audio_path)
        except (OSError, LrcError):
            fusion_audio_revision = None
    candidate_sets: list[tuple[TimingCandidate, ...]] = []
    for index, (entry, raw_assignment) in enumerate(zip(entries, assignments, strict=True)):
        if not isinstance(raw_assignment, dict):
            raise LrcError(f"Central timing finalizer received invalid assignment {index + 1}")
        # Re-bind the immutable backend baseline instead of trusting a scratch
        # timestamp that a pre-central heuristic may have shifted.  Legacy
        # scoring preserves the backend-selected path as a candidate row.
        baseline_time: float | None = None
        legacy_rows = raw_assignment.get("candidates")
        if isinstance(legacy_rows, list):
            for row in legacy_rows:
                if not isinstance(row, dict) or row.get("source") != "selected_path":
                    continue
                value = row.get("time")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    baseline_time = float(value)
                    break
        if baseline_time is None:
            raw_value = raw_assignment.get("timestamp", scratch_timestamps[index])
            baseline_time = float(raw_value if isinstance(raw_value, (int, float)) else scratch_timestamps[index])
        current_time = baseline_time
        spans = _timing_spans_from_value(raw_assignment.get("ctc_token_spans"))
        backend = str(report.get("backend") or "unknown")
        manual = bool(raw_assignment.get("manual_anchor_hint"))
        if manual:
            manual_value = raw_assignment.get("timestamp", scratch_timestamps[index])
            if isinstance(manual_value, (int, float)) and math.isfinite(float(manual_value)):
                current_time = float(manual_value)
            current_source = "manual-anchor"
        elif backend == "ctc":
            current_source = "ctc-current"
        elif backend == "whisperx":
            current_source = "whisperx-word"
        elif backend in {"lyrics", "checked_lrc"}:
            current_source = "checked-lrc" if backend == "checked_lrc" else "lyrics-source"
        else:
            current_source = f"{backend}-current"
        score = _candidate_confidence(raw_assignment)
        explicit_source = current_source in {"manual-anchor", "checked-lrc", "lyrics-source"}
        ctc_identity = bool(spans)
        aligned_non_ctc = backend == "whisperx" and score >= TRUSTED_ALIGNMENT_SCORE
        current_source_artifact = {
            "backend": backend,
            "timing_repair_source": raw_assignment.get("timing_repair_source"),
            "ctc_local_window": raw_assignment.get("ctc_local_window"),
            "spans": raw_assignment.get("ctc_token_spans"),
            "timestamp": current_time,
        }
        current_candidate = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entry),
            source=current_source,
            raw_time=current_time,
            spans=spans,
            confidence=score,
            identity_support="supported" if ctc_identity or aligned_non_ctc or explicit_source else "unavailable",
            sequence_support="supported",
            acoustic_support="unavailable",
            direct_onset_support=(
                "supported" if explicit_source else "unavailable"
            ),
            direct_onset_time=current_time if explicit_source else None,
            direct_onset_source_artifact=(
                current_source_artifact if explicit_source else None
            ),
            direct_onset_evidence_producer=(
                "external-reference" if explicit_source else None
            ),
            direct_onset_evidence_kind=(
                "external-reference" if explicit_source else "generic-onset"
            ),
            direct_onset_evidence_independent=explicit_source,
            current=True,
            right_edge_pileup=bool(raw_assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(raw_assignment.get("ctc_right_edge_truncation")),
            source_artifact=current_source_artifact,
        )
        candidates: list[TimingCandidate] = [current_candidate]

        raw_identity = runtime_raw_content_identity_for_entry(report, entries, index)
        consensus = runtime_raw_ctc_consensus_for_entry(
            report, entries, index, current_time
        )
        if consensus is not None and spans:
            raw_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                entries,
                index,
                producer="raw-asr-consensus",
                occurrence_binding=consensus.get("occurrence_binding"),
                upstream_evidence_revision=consensus.get("evidence_revision"),
            )
            raw_consensus_time = float(consensus["raw_time"])
            raw_consensus_score = float(consensus["raw_score"])
            consensus_artifact = {
                "producer": "raw-asr-consensus",
                "evidence_revision": consensus["evidence_revision"],
                "raw_time": raw_consensus_time,
                "ctc_time": current_time,
                "raw_score": raw_consensus_score,
                "raw_content_identity_revision": (
                    raw_identity.get("evidence_revision") if isinstance(raw_identity, dict) else None
                ),
                "runtime_only": True,
            }
            candidates.append(
                make_timing_candidate(
                    entry_index=index,
                    entry_text=entry_sung_text(entry),
                    source="ctc-raw-independent-consensus",
                    raw_time=current_time,
                    spans=spans,
                    confidence=min(score, raw_consensus_score),
                    identity_support="supported",
                    sequence_support="supported",
                    acoustic_support="unavailable",
                    direct_onset_support="supported",
                    direct_onset_time=raw_consensus_time,
                    direct_onset_source_artifact=consensus_artifact,
                    direct_onset_evidence_producer="raw-asr-consensus",
                    direct_onset_evidence_kind="cross-backend-onset",
                    direct_onset_evidence_independent=True,
                    occurrence_evidence=raw_occurrence_evidence,
                    independent_content_identity=raw_occurrence_evidence is not None,
                    current=False,
                    right_edge_pileup=bool(raw_assignment.get("ctc_right_edge_pileup")),
                    window_truncated=bool(raw_assignment.get("ctc_right_edge_truncation")),
                    source_artifact=consensus_artifact,
                )
            )

        whisperx_consensus = runtime_whisperx_ctc_consensus_for_entry(
            report, entries, index, current_time
        )
        if whisperx_consensus is not None and spans:
            whisperx_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                entries,
                index,
                producer="whisperx-consensus",
                occurrence_binding=whisperx_consensus.get("occurrence_binding"),
                upstream_evidence_revision=whisperx_consensus.get("evidence_revision"),
            )
            whisperx_time = float(whisperx_consensus["whisperx_forced_time"])
            whisperx_score = float(whisperx_consensus["score"])
            whisperx_artifact = {
                "producer": "whisperx-consensus",
                "evidence_revision": whisperx_consensus["evidence_revision"],
                "whisperx_forced_time": whisperx_time,
                "ctc_time": current_time,
                "score": whisperx_score,
                "runtime_only": True,
            }
            candidates.append(
                make_timing_candidate(
                    entry_index=index,
                    entry_text=entry_sung_text(entry),
                    source="ctc-whisperx-independent-consensus",
                    raw_time=current_time,
                    spans=spans,
                    confidence=min(score, whisperx_score),
                    identity_support="supported",
                    sequence_support="supported",
                    acoustic_support="unavailable",
                    direct_onset_support="supported",
                    direct_onset_time=whisperx_time,
                    direct_onset_source_artifact=whisperx_artifact,
                    direct_onset_evidence_producer="whisperx-consensus",
                    direct_onset_evidence_kind="cross-backend-onset",
                    direct_onset_evidence_independent=True,
                    occurrence_evidence=whisperx_occurrence_evidence,
                    independent_content_identity=whisperx_occurrence_evidence is not None,
                    current=False,
                    right_edge_pileup=bool(raw_assignment.get("ctc_right_edge_pileup")),
                    window_truncated=bool(raw_assignment.get("ctc_right_edge_truncation")),
                    source_artifact=whisperx_artifact,
                )
            )


        # Independent row-level backend identity can be fused with the isolated
        # vocal detector without requiring CTC to agree first.  This recovers
        # evidence that was previously thrown away when a backend failed only
        # at the whole-song aggregate level.  The runtime identity timestamp
        # must itself be close to the independently observed vocal onset.
        if isinstance(raw_identity, dict) and vocal_onset_features is not None and fusion_audio_revision:
            try:
                raw_identity_time = float(raw_identity.get("raw_time"))
                raw_identity_score = float(raw_identity.get("raw_score"))
            except (TypeError, ValueError):
                raw_identity_time = math.nan
                raw_identity_score = 0.0
            raw_peak = _strong_vocal_onset_near(
                vocal_onset_features, raw_identity_time, require_absolute_vocal=True
            )
            if raw_peak is not None and raw_identity_score >= TRUSTED_ALIGNMENT_SCORE:
                raw_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="raw-asr-vocal-fusion",
                    occurrence_binding=raw_identity.get("occurrence_binding"),
                    upstream_evidence_revision=raw_identity.get("evidence_revision"),
                )
                peak_time, peak_strength, peak_prominence = raw_peak
                artifact = {
                    "producer": "raw-asr-vocal-fusion",
                    "raw_content_identity_revision": raw_identity.get("evidence_revision"),
                    "raw_time": round(raw_identity_time, 6),
                    "raw_score": round(raw_identity_score, 6),
                    "vocal_onset_time": round(peak_time, 6),
                    "vocal_onset_strength": round(peak_strength, 8),
                    "vocal_onset_prominence": round(peak_prominence, 8),
                    "audio_sha256": fusion_audio_revision,
                    "feature_revision": "vocal-consonant-onset-v1",
                    "runtime_only": True,
                }
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source="raw-vocal-independent-fusion",
                        raw_time=peak_time,
                        spans=(),
                        confidence=min(raw_identity_score, peak_strength),
                        identity_support=(
                            "supported" if raw_occurrence_evidence is not None else "unavailable"
                        ),
                        sequence_support=(
                            "supported" if raw_occurrence_evidence is not None else "unavailable"
                        ),
                        acoustic_support="supported",
                        direct_onset_support="supported",
                        acoustic_strength=peak_strength,
                        acoustic_onset_time=peak_time,
                        acoustic_source_artifact=artifact,
                        acoustic_evidence_producer="raw-asr-vocal-fusion",
                        acoustic_evidence_kind="vocal-onset",
                        acoustic_evidence_independent=True,
                        direct_onset_time=peak_time,
                        direct_onset_source_artifact=artifact,
                        direct_onset_evidence_producer="raw-asr-vocal-fusion",
                        direct_onset_evidence_kind="vocal-onset",
                        direct_onset_evidence_independent=False,
                        occurrence_evidence=raw_occurrence_evidence,
                        independent_content_identity=raw_occurrence_evidence is not None,
                        current=False,
                        source_artifact=artifact,
                    )
                )

        whisperx_row = runtime_whisperx_row_evidence_for_entry(report, entries, index)
        if isinstance(whisperx_row, dict) and vocal_onset_features is not None and fusion_audio_revision:
            try:
                whisperx_forced_time = float(whisperx_row.get("first_char_time"))
                whisperx_row_score = float(whisperx_row.get("score"))
            except (TypeError, ValueError):
                whisperx_forced_time = math.nan
                whisperx_row_score = 0.0
            whisperx_peak = _strong_vocal_onset_near(vocal_onset_features, whisperx_forced_time)
            if whisperx_peak is not None and whisperx_row_score >= TRUSTED_ALIGNMENT_SCORE:
                whisperx_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="whisperx-vocal-fusion",
                    occurrence_binding=whisperx_row.get("occurrence_binding"),
                    upstream_evidence_revision=whisperx_row.get("evidence_revision"),
                )
                peak_time, peak_strength, peak_prominence = whisperx_peak
                artifact = {
                    "producer": "whisperx-vocal-fusion",
                    "whisperx_row_revision": whisperx_row.get("evidence_revision"),
                    "whisperx_forced_time": round(whisperx_forced_time, 6),
                    "score": round(whisperx_row_score, 6),
                    "vocal_onset_time": round(peak_time, 6),
                    "vocal_onset_strength": round(peak_strength, 8),
                    "vocal_onset_prominence": round(peak_prominence, 8),
                    "audio_sha256": fusion_audio_revision,
                    "feature_revision": "vocal-consonant-onset-v1",
                    "runtime_only": True,
                }
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source="whisperx-vocal-independent-fusion",
                        raw_time=whisperx_forced_time,
                        spans=(),
                        confidence=min(whisperx_row_score, peak_strength),
                        identity_support="supported",
                        sequence_support="supported",
                        acoustic_support="supported",
                        direct_onset_support="supported",
                        acoustic_strength=peak_strength,
                        acoustic_onset_time=peak_time,
                        acoustic_source_artifact=artifact,
                        acoustic_evidence_producer="whisperx-vocal-fusion",
                        acoustic_evidence_kind="vocal-onset",
                        acoustic_evidence_independent=True,
                        direct_onset_time=peak_time,
                        direct_onset_source_artifact=artifact,
                        direct_onset_evidence_producer="whisperx-vocal-fusion",
                        direct_onset_evidence_kind="cross-backend-onset",
                        direct_onset_evidence_independent=True,
                        occurrence_evidence=whisperx_occurrence_evidence,
                        independent_content_identity=whisperx_occurrence_evidence is not None,
                        current=False,
                        source_artifact=artifact,
                    )
                )

        local_whisper_identity = runtime_local_whisper_evidence_for_entry(
            report, entries, index
        )
        local_whisper_vocal = _local_whisper_vocal_fusion_candidate(
            entry_index=index,
            entry=entry,
            local_evidence=local_whisper_identity,
            vocal_onset_features=vocal_onset_features,
            audio_revision=fusion_audio_revision,
            occurrence_evidence=(
                candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="local-whisper-vocal-fusion",
                    occurrence_binding=local_whisper_identity.get("occurrence_binding"),
                    upstream_evidence_revision=local_whisper_identity.get("evidence_revision"),
                )
                if isinstance(local_whisper_identity, dict) else None
            ),
        )
        if local_whisper_vocal is not None:
            candidates.append(local_whisper_vocal)

        hypothesis_rows = raw_assignment.get("alignment_hypotheses")
        if isinstance(hypothesis_rows, list):
            for hypothesis in hypothesis_rows:
                if not isinstance(hypothesis, dict):
                    continue
                raw_time = hypothesis.get("raw_candidate_time")
                if not isinstance(raw_time, (int, float)) or not math.isfinite(float(raw_time)):
                    continue
                source = str(hypothesis.get("source") or "precentral-hypothesis")
                if source == "ctc-opening-local-retry-identity":
                    # Identity-only evidence is consumed by occurrence-level
                    # recurrence producers; it must never become a timing
                    # candidate merely because it contains helper spans.
                    continue
                if source == CTC_WIDE_REGION_FALLBACK_SOURCE:
                    # Gated against the finished candidate set below.
                    continue
                independent_identity_bound = False
                independent_acoustic_bound = False
                independent_direct_bound = False
                acoustic_producer: str | None = None
                direct_producer: str | None = None
                composite_identity_revision: str | None = None
                opening_vocal_fusion_bound = False
                raw_vocal_occurrence_bound = False
                acoustic_kind = "generic-acoustic"
                direct_kind = "generic-onset"
                if source == "global-acoustic-recurrence":
                    independent_identity_bound = recurrence_hypothesis_provenance_is_valid(
                        hypothesis, entry, evidence_audio_path
                    )
                    if not independent_identity_bound:
                        continue
                    independent_acoustic_bound = True
                    independent_direct_bound = True
                    acoustic_producer = direct_producer = "global-recurrence-search"
                    acoustic_kind = direct_kind = "recurrence-onset"
                elif source == "mutual-acoustic-recurrence":
                    independent_identity_bound = mutual_recurrence_hypothesis_provenance_is_valid(
                        hypothesis,
                        entry,
                        evidence_audio_path,
                        assignments,
                        entries,
                        index,
                    )
                    if not independent_identity_bound:
                        continue
                    independent_acoustic_bound = True
                    independent_direct_bound = True
                    acoustic_producer = direct_producer = "mutual-recurrence-search"
                    acoustic_kind = direct_kind = "recurrence-onset"
                elif source == "ctc-phonetic-onset-fusion":
                    if not phonetic_onset_fusion_provenance_is_valid(
                        hypothesis, raw_assignment, evidence_audio_path
                    ):
                        continue
                    # Stage-4 R1-2 resolution: keep the validated hypothesis as
                    # diagnostics, but remove this formerly-dead producer from
                    # Central timing authority so cleanup cannot move timestamps.
                    continue
                elif source == "ctc-vocal-onset-confirmed":
                    if not vocal_onset_confirmation_provenance_is_valid(
                        hypothesis, raw_assignment, evidence_audio_path,
                        recomputed_features=vocal_onset_features,
                        evidence_audio_sha256=fusion_audio_revision,
                    ):
                        continue
                    independent_acoustic_bound = True
                    independent_direct_bound = True
                    acoustic_producer = "consonant-onset-detector"
                    acoustic_kind = "vocal-onset"
                    direct_producer = "consonant-onset-detector"
                    direct_kind = "vocal-onset"
                    opening_vocal_fusion_bound = ctc_opening_vocal_fusion_is_valid(
                        hypothesis, raw_assignment, evidence_audio_path,
                        vocal_provenance_validated=True,
                        recomputed_features=vocal_onset_features,
                        evidence_audio_sha256=fusion_audio_revision,
                    )
                    if isinstance(raw_identity, dict):
                        try:
                            raw_identity_time = float(raw_identity.get("raw_time"))
                            raw_identity_score = float(raw_identity.get("raw_score"))
                            hypothesis_onset = float(hypothesis.get("direct_onset_time"))
                        except (TypeError, ValueError):
                            raw_identity_time = math.nan
                            raw_identity_score = 0.0
                            hypothesis_onset = math.nan
                        raw_vocal_occurrence_bound = bool(
                            math.isfinite(raw_identity_time)
                            and math.isfinite(hypothesis_onset)
                            and raw_identity_score >= RAW_CTC_CONSENSUS_MIN_SCORE
                            and abs(raw_identity_time - hypothesis_onset)
                            <= RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
                        )
                        if raw_vocal_occurrence_bound:
                            composite_identity_revision = str(
                                raw_identity.get("evidence_revision") or ""
                            )
                elif source == "ctc-vocal-activity-reentry":
                    if not vocal_activity_reentry_provenance_is_valid(
                        hypothesis, raw_assignment, evidence_audio_path
                    ):
                        continue
                    independent_direct_bound = True
                    direct_producer = "vocal-activity-detector"
                    direct_kind = "vocal-activity-reentry"
                elif source == "trusted-tail-release-onset":
                    if not trusted_tail_release_provenance_is_valid(
                        hypothesis,
                        evidence_audio_path,
                        assignments,
                        index,
                    ):
                        continue
                    # trusted-tail-release-diagnostic-only: a post-tail energy
                    # cluster is not phonetic proof of the next lyric's first
                    # mora. Preserve the serialized hypothesis for diagnostics,
                    # but do not manufacture an always-invalid Central candidate.
                    continue
                spans_value = _timing_spans_from_value(hypothesis.get("spans"))
                acoustic_strength = hypothesis.get("acoustic_strength")
                acoustic_onset_time = hypothesis.get("acoustic_onset_time")
                direct_onset_time = hypothesis.get("direct_onset_time")
                source_artifact = {
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "stage": hypothesis.get("stage"),
                    "parent": hypothesis.get("parent"),
                    "window": hypothesis.get("window"),
                    "mode": hypothesis.get("mode"),
                    "raw_content_identity_revision": composite_identity_revision,
                    "authoritative": False,
                }
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source=source,
                        raw_time=float(raw_time),
                        spans=spans_value,
                        confidence=float(hypothesis.get("confidence", 0.0) or 0.0),
                        identity_support=str(hypothesis.get("identity_support") or "unavailable"),  # type: ignore[arg-type]
                        sequence_support=str(hypothesis.get("sequence_support") or "unavailable"),  # type: ignore[arg-type]
                        acoustic_support=str(hypothesis.get("acoustic_support") or "unavailable"),  # type: ignore[arg-type]
                        direct_onset_support=(
                            "supported"
                            if independent_direct_bound and isinstance(direct_onset_time, (int, float))
                            else "unavailable"
                        ),
                        acoustic_strength=(float(acoustic_strength) if isinstance(acoustic_strength, (int, float)) else None),
                        acoustic_onset_time=(float(acoustic_onset_time) if isinstance(acoustic_onset_time, (int, float)) else None),
                        acoustic_source_artifact=source_artifact,
                        acoustic_evidence_producer=acoustic_producer,
                        acoustic_evidence_kind=acoustic_kind,
                        acoustic_evidence_independent=independent_acoustic_bound,
                        direct_onset_time=(
                            float(direct_onset_time)
                            if independent_direct_bound and isinstance(direct_onset_time, (int, float))
                            else None
                        ),
                        direct_onset_source_artifact=(
                            source_artifact
                            if independent_direct_bound and isinstance(direct_onset_time, (int, float))
                            else None
                        ),
                        direct_onset_evidence_producer=direct_producer,
                        direct_onset_evidence_kind=direct_kind,
                        direct_onset_evidence_independent=independent_direct_bound,
                        independent_content_identity=independent_identity_bound,
                        current=False,
                        right_edge_pileup=bool(hypothesis.get("right_edge_pileup")),
                        window_truncated=bool(hypothesis.get("window_truncated")),
                        source_artifact=source_artifact,
                    )
                )

                if source == "ctc-vocal-onset-confirmed" and opening_vocal_fusion_bound:
                    fusion_artifact = {
                        **source_artifact,
                        "producer": "ctc-opening-vocal-fusion",
                        "ctc_geometry_revision": _timing_content_digest({
                            "spans": hypothesis.get("spans"),
                            "assignment_time": raw_assignment.get("timestamp"),
                        }),
                    }
                    candidates.append(
                        make_timing_candidate(
                            entry_index=index,
                            entry_text=entry_sung_text(entry),
                            source="ctc-opening-vocal-fusion",
                            raw_time=float(raw_time),
                            spans=spans_value,
                            confidence=float(hypothesis.get("confidence", 0.0) or 0.0),
                            identity_support="supported",
                            sequence_support="supported",
                            acoustic_support="supported",
                            direct_onset_support="supported",
                            acoustic_strength=(float(acoustic_strength) if isinstance(acoustic_strength, (int, float)) else None),
                            acoustic_onset_time=(float(acoustic_onset_time) if isinstance(acoustic_onset_time, (int, float)) else None),
                            acoustic_source_artifact=fusion_artifact,
                            acoustic_evidence_producer="consonant-onset-detector",
                            acoustic_evidence_kind="vocal-onset",
                            acoustic_evidence_independent=True,
                            direct_onset_time=(float(direct_onset_time) if isinstance(direct_onset_time, (int, float)) else None),
                            direct_onset_source_artifact=fusion_artifact,
                            # CTC opening geometry belongs to the candidate itself.
                            # Combining it with an independent vocal peak improves
                            # current-line opening support, but does not make that
                            # peak independently lyric-specific.  Keep this generic
                            # so it can never cross UNKNOWN/VIOLATION ownership.
                            direct_onset_evidence_producer="consonant-onset-detector",
                            direct_onset_evidence_kind="vocal-onset",
                            direct_onset_evidence_independent=True,
                            independent_content_identity=False,
                            current=False,
                            right_edge_pileup=bool(hypothesis.get("right_edge_pileup")),
                            window_truncated=bool(hypothesis.get("window_truncated")),
                            source_artifact=fusion_artifact,
                        )
                    )
                if source == "ctc-vocal-onset-confirmed" and raw_vocal_occurrence_bound:
                    raw_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                        entries,
                        index,
                        producer="raw-asr-vocal-fusion",
                        occurrence_binding=raw_identity.get("occurrence_binding"),
                        upstream_evidence_revision=raw_identity.get("evidence_revision"),
                    )
                    raw_fusion_artifact = {
                        **source_artifact,
                        "producer": "raw-asr-vocal-fusion",
                        "raw_content_identity_revision": composite_identity_revision,
                        "raw_time": round(float(raw_identity.get("raw_time")), 6),
                        "raw_score": round(float(raw_identity.get("raw_score")), 6),
                    }
                    candidates.append(
                        make_timing_candidate(
                            entry_index=index,
                            entry_text=entry_sung_text(entry),
                            source="ctc-raw-vocal-independent-fusion",
                            raw_time=float(raw_time),
                            spans=spans_value,
                            confidence=min(
                                float(hypothesis.get("confidence", 0.0) or 0.0),
                                float(raw_identity.get("raw_score", 0.0) or 0.0),
                            ),
                            identity_support="supported",
                            sequence_support="supported",
                            acoustic_support="supported",
                            direct_onset_support="supported",
                            acoustic_strength=(float(acoustic_strength) if isinstance(acoustic_strength, (int, float)) else None),
                            acoustic_onset_time=(float(acoustic_onset_time) if isinstance(acoustic_onset_time, (int, float)) else None),
                            acoustic_source_artifact=raw_fusion_artifact,
                            acoustic_evidence_producer="consonant-onset-detector",
                            acoustic_evidence_kind="vocal-onset",
                            acoustic_evidence_independent=True,
                            direct_onset_time=(float(direct_onset_time) if isinstance(direct_onset_time, (int, float)) else None),
                            direct_onset_source_artifact=raw_fusion_artifact,
                            direct_onset_evidence_producer="raw-asr-vocal-fusion",
                            direct_onset_evidence_kind="vocal-onset",
                            direct_onset_evidence_independent=False,
                            occurrence_evidence=raw_occurrence_evidence,
                            independent_content_identity=raw_occurrence_evidence is not None,
                            current=False,
                            right_edge_pileup=bool(hypothesis.get("right_edge_pileup")),
                            window_truncated=bool(hypothesis.get("window_truncated")),
                            source_artifact=raw_fusion_artifact,
                        )
                    )

        first_token_candidates = raw_assignment.get("ctc_first_token_candidates")
        if isinstance(first_token_candidates, list):
            for raw_candidate in first_token_candidates:
                if not isinstance(raw_candidate, dict):
                    continue
                try:
                    raw_time = float(raw_candidate.get("time"))
                    raw_score = float(raw_candidate.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                # A first-token posterior is identity evidence, not by itself
                # a complete onset proof.  Its one-token span therefore enters
                # normalization and ownership gating but cannot become valid
                # until another producer contributes coherent prefix/acoustic
                # support for the same written time.
                posterior_span = TimingTokenSpan(
                    canonicalize_lrc_time(raw_time).seconds,
                    canonicalize_lrc_time(raw_time).seconds + HOP_SECONDS,
                    raw_score,
                    "",
                )
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source="ctc-first-token-posterior",
                        raw_time=raw_time,
                        spans=(posterior_span,),
                        confidence=raw_score,
                        identity_support="supported",
                        sequence_support="supported",
                        source_artifact=raw_candidate,
                    )
                )

        legacy_candidates = raw_assignment.get("candidates")
        if isinstance(legacy_candidates, list):
            for legacy in legacy_candidates:
                if not _active_legacy_candidate(legacy):
                    continue
                raw_time = legacy.get("time")
                if not isinstance(raw_time, (int, float)):
                    continue
                source = str(legacy.get("source") or "legacy-candidate")
                direct_onset = False
                identity = source in {"raw_asr", "whisperx_forced_first", "whisperx_word"}
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source=source.replace("_", "-"),
                        raw_time=float(raw_time),
                        confidence=float(legacy.get("score", 0.0) or 0.0),
                        identity_support="supported" if identity else "unavailable",
                        sequence_support="supported",
                        acoustic_support="unavailable",
                        direct_onset_support="supported" if direct_onset else "unavailable",
                        direct_onset_time=float(raw_time) if direct_onset else None,
                        direct_onset_source_artifact=legacy if direct_onset else None,
                        source_artifact=legacy,
                    )
                )

        if (report.get("segment_skeleton_status") or {}).get("status") == "SEGMENT_CURRENT":
            from segment_peer_outlier import exclude_uncorroborated_whisperx_fusion
            candidates, excluded = exclude_uncorroborated_whisperx_fusion(candidates)
            if excluded:
                raw_assignment["segment_peer_fusion_exclusions"] = [
                    {"source":x.source, "time":x.written_time.seconds,
                     "reason":"uncorroborated-distant-fused-onset"}
                    for x in excluded
                ]
        unique = {candidate.candidate_id: candidate for candidate in candidates}
        candidate_sets.append(tuple(unique[key] for key in sorted(unique)))

    ordinary_candidate_sets = _augment_local_ctc_candidates_with_runtime_backend_consensus(
        tuple(candidate_sets), report, entries
    )
    independent_tail_evidence = runtime_independent_tail_evidence(report, len(entries))
    independent_tail_payload = independent_tail_evidence_payload(independent_tail_evidence)
    fusion_source_counts = collections.Counter(
        candidate.source
        for candidates in ordinary_candidate_sets
        for candidate in candidates
        if candidate.source in {
            "raw-vocal-independent-fusion",
            "whisperx-vocal-independent-fusion",
            "ctc-opening-vocal-fusion",
            "ctc-raw-vocal-independent-fusion",
            "ctc-local-raw-independent-consensus",
            "ctc-local-whisperx-independent-consensus",
        }
    )
    hypothesis_count = sum(
        len(item.get("alignment_hypotheses", []))
        for item in assignments
        if isinstance(item, dict) and isinstance(item.get("alignment_hypotheses"), list)
    )
    diagnostics = {
        "ctc_prefix_probes": [],
        "ctc_prefix_probe_count": 0,
        "legacy_final_guard_changes_discarded": [],
        "legacy_final_guard_change_count": 0,
        "independent_tail_evidence": independent_tail_payload,
        "runtime_evidence_fusion_candidate_counts": dict(sorted(fusion_source_counts.items())),
        "alignment_hypothesis_layer": {
            "status": "active",
            "authoritative_precentral_mutation": False,
            "hypothesis_count": hypothesis_count,
            "central_commit_only": True,
        },
        **inactive_legacy_timing_diagnostics(),
    }
    capability = build_bounded_retry_capability(report, audio_path, args)
    tail_evidence_context = _tail_evidence_context_from_report(
        report, entries=entries, capability=capability
    )
    if tail_evidence_context is not None:
        report[_RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY] = tail_evidence_context
    independent_tail_evidence = runtime_independent_tail_evidence(
        report, len(entries), tail_evidence_context
    )
    independent_tail_payload = independent_tail_evidence_payload(independent_tail_evidence)
    if retry_helper_cache is None:
        retry_helper_cache = {}
    try:
        retry_timeout_seconds = float(getattr(args, "crossline_retry_timeout", 180.0))
    except (TypeError, ValueError):
        retry_timeout_seconds = 180.0
    if not math.isfinite(retry_timeout_seconds) or retry_timeout_seconds <= 0.0:
        retry_timeout_seconds = 180.0
    retry_timing_source_policy = str(
        getattr(args, "timing_source_policy", "default") or "default"
    )
    retry_runtime_stats: dict[str, int] = {
        "cache_hits": 0,
        "cache_misses": 0,
        "helper_invocations": 0,
        "timeouts": 0,
    }
    # Bounded fixed-point retry: resolving one boundary can make the next line's
    # previous-tail ownership provable. Re-plan from the newly evaluated state,
    # but cap passes so recovery can never become an unbounded heuristic loop.
    candidate_sets = ordinary_candidate_sets
    retry_passes: list[dict[str, object]] = []
    all_requests: list[CrossLineRetryRequest] = []
    all_outcomes: list[CrossLineRetryOutcome] = []
    processed_retry_semantics: set[str] = set()
    max_retry_passes = 4
    last_epoch_revision: str | None = None
    tail_evidence_changes = 0
    for pass_index in range(1, max_retry_passes + 1):
        before_candidate_revision, before_tail_revision = _retry_semantic_state_revision(
            candidate_sets,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        epoch = freeze_base_retry_epoch(
            str(report.get("backend") or "unknown"),
            report,
            candidate_sets,
            diagnostics,
            capability,
        )
        last_epoch_revision = epoch.retry_epoch_revision
        requests = plan_crossline_retry_requests(
            epoch,
            capability,
            entries,
            candidate_sets,
            duration,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        if capability is None:
            novel_requests: tuple[CrossLineRetryRequest, ...] = ()
            novel_semantics: tuple[str, ...] = ()
        else:
            novel_requests, novel_semantics = _novel_crossline_retry_requests(
                requests, capability, processed_retry_semantics
            )
        progress_log(
            f"crossline retry pass {pass_index}: requests={len(requests)} "
            f"new_semantic={len(novel_requests)} cache_entries={len(retry_helper_cache)}"
        )
        if not requests:
            retry_passes.append({
                "pass": pass_index,
                "retry_epoch_revision": epoch.retry_epoch_revision,
                "request_count": 0,
                "executed_request_count": 0,
                "new_semantic_request_count": 0,
                "outcome_count": 0,
                "new_candidate_count": 0,
                "semantic_fixed_point": True,
            })
            break
        if not novel_requests:
            progress_log(
                f"crossline retry pass {pass_index}: semantic fixed point; "
                "all planned helper inputs were already evaluated"
            )
            retry_passes.append({
                "pass": pass_index,
                "retry_epoch_revision": epoch.retry_epoch_revision,
                "request_count": len(requests),
                "executed_request_count": 0,
                "new_semantic_request_count": 0,
                "outcome_count": 0,
                "new_candidate_count": 0,
                "semantic_fixed_point": True,
            })
            break
        outcomes = execute_crossline_retry_epoch(
            epoch, capability, entries, candidate_sets, duration, novel_requests,
            helper_cache=retry_helper_cache,
            timeout_seconds=retry_timeout_seconds,
            pass_index=pass_index,
            runtime_stats=retry_runtime_stats,
            independent_tail_evidence=independent_tail_evidence,
            tail_evidence_context=tail_evidence_context,
            timing_source_policy=retry_timing_source_policy,
            worker_count=DEFAULT_EVIDENCE_WORKERS,
        )
        processed_retry_semantics.update(novel_semantics)
        tail_evidence_changes = sum(
            int(merge_runtime_independent_tail_evidence(
                report, outcome.previous_tail_evidence
            ))
            for outcome in outcomes
            if outcome.previous_tail_evidence is not None
        )
        if tail_evidence_changes:
            independent_tail_evidence = runtime_independent_tail_evidence(
                report, len(entries), tail_evidence_context
            )
        augmented = [list(items) for items in candidate_sets]
        existing_ids = {item.candidate_id for items in candidate_sets for item in items}
        new_candidate_count = 0
        for outcome in outcomes:
            record_crossline_retry_identity_hypothesis(report, outcome)
            if (
                outcome.status == "candidate"
                and outcome.candidate is not None
                and outcome.candidate.candidate_id not in existing_ids
            ):
                augmented[outcome.entry_index].append(outcome.candidate)
                existing_ids.add(outcome.candidate.candidate_id)
                new_candidate_count += 1
        candidate_sets = tuple(
            tuple({item.candidate_id: item for item in items}[key] for key in sorted(
                {item.candidate_id: item for item in items}
            ))
            for items in augmented
        )
        candidate_sets = _augment_local_ctc_candidates_with_runtime_backend_consensus(
            candidate_sets, report, entries
        )
        after_candidate_revision, after_tail_revision = _retry_semantic_state_revision(
            candidate_sets,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        semantic_state_changed = bool(
            before_candidate_revision != after_candidate_revision
            or before_tail_revision != after_tail_revision
        )
        all_requests.extend(novel_requests)
        all_outcomes.extend(outcomes)
        retry_passes.append({
            "pass": pass_index,
            "retry_epoch_revision": epoch.retry_epoch_revision,
            "request_count": len(requests),
            "executed_request_count": len(novel_requests),
            "new_semantic_request_count": len(novel_requests),
            "outcome_count": len(outcomes),
            "new_candidate_count": new_candidate_count,
            "previous_tail_evidence_change_count": tail_evidence_changes,
            "candidate_semantic_revision": after_candidate_revision,
            "authoritative_tail_semantic_revision": after_tail_revision,
            "semantic_fixed_point": not semantic_state_changed,
        })
        if new_candidate_count == 0 and not semantic_state_changed:
            break

    # Architectural recovery: after current-line retry reaches its bounded
    # fixed point, obtain a separate joint previous+current boundary observation
    # for every still-unresolved boundary.  The observation cannot be selected
    # directly; it must converge with an already-existing current candidate.
    joint_boundary_requests: tuple[CrossLineRetryRequest, ...] = ()
    joint_boundary_outcomes: tuple[CrossLineRetryOutcome, ...] = ()
    joint_boundary_added = 0
    if capability is not None:
        joint_epoch = freeze_base_retry_epoch(
            str(report.get("backend") or "unknown"),
            report,
            candidate_sets,
            diagnostics,
            capability,
        )
        joint_boundary_requests = plan_joint_boundary_retry_requests(
            joint_epoch, capability, entries, candidate_sets, duration,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        progress_log(
            f"joint boundary retry: requests={len(joint_boundary_requests)} "
            f"cache_entries={len(retry_helper_cache)}"
        )
        if joint_boundary_requests:
            joint_boundary_outcomes = execute_crossline_retry_epoch(
                joint_epoch, capability, entries, candidate_sets, duration,
                joint_boundary_requests,
                helper_cache=retry_helper_cache,
                timeout_seconds=retry_timeout_seconds,
                pass_index="joint",
                runtime_stats=retry_runtime_stats,
                independent_tail_evidence=independent_tail_evidence,
                tail_evidence_context=tail_evidence_context,
                timing_source_policy=retry_timing_source_policy,
                worker_count=DEFAULT_EVIDENCE_WORKERS,
            )
            joint_tail_evidence_changes = sum(
                int(merge_runtime_independent_tail_evidence(
                    report, outcome.previous_tail_evidence
                ))
                for outcome in joint_boundary_outcomes
                if outcome.previous_tail_evidence is not None
            )
            if joint_tail_evidence_changes:
                independent_tail_evidence = runtime_independent_tail_evidence(
                    report, len(entries), tail_evidence_context
                )
            before_ids = {
                item.candidate_id for items in candidate_sets for item in items
            }
            candidate_sets = _augment_candidates_with_joint_boundary_consensus(
                candidate_sets, joint_boundary_outcomes, entries
            )
            after_ids = {
                item.candidate_id for items in candidate_sets for item in items
            }
            joint_boundary_added = len(after_ids - before_ids)
            if joint_boundary_added:
                candidate_sets = _augment_local_ctc_candidates_with_runtime_backend_consensus(
                    candidate_sets, report, entries
                )
            progress_log(
                f"joint boundary retry: observations={len(joint_boundary_outcomes)} "
                f"new_consensus_candidates={joint_boundary_added}"
            )

    # The bounded CTC retry/joint phase is complete for this collection epoch.
    # Park resident MMS_FA workers on CPU before local whisper.cpp/WhisperX work.
    _park_persistent_helper_mode("ctc", reason="bounded-ctc-evidence-complete")

    # v14.6: the CTC graph is now rich enough that the remaining ceiling is
    # usually lack of an *independent* lyric-specific opening/occurrence witness.
    # Run whisper.cpp + WhisperX only for those unresolved/content-untrusted/review
    # rows.  The local observation is never selectable by itself; central may use
    # it only by fusing its onset/identity into an already-existing non-Whisper
    # candidate within the unchanged 0.18 s cross-backend tolerance.
    local_whisper_capability = build_local_whisper_capability(evidence_audio_path, args)
    local_whisper_requests: tuple[LocalWhisperRequest, ...] = ()
    local_whisper_outcomes: tuple[LocalWhisperOutcome, ...] = ()
    local_whisper_added = 0
    local_whisper_stats: dict[str, int] = {
        "cache_hits": 0, "cache_misses": 0, "helper_invocations": 0,
    }
    if local_whisper_capability is not None:
        local_whisper_requests = plan_local_whisper_requests(
            report, local_whisper_capability, entries, candidate_sets, duration,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        # Fixed-point recollection may change upstream candidate IDs and
        # generate hundreds of distinct optional Whisper context windows.
        # Preserve all real cache hits and cover the song with at most 32 new
        # incremental probes per recollection. The first collection remains
        # fully faithful. Deferred probes confer NO timing or identity proof.
        from local_whisper_incremental_budget import budget_incremental_whisper_requests
        local_whisper_requests, deferred_new_probes = budget_incremental_whisper_requests(
            local_whisper_requests,
            retry_helper_cache,
            lambda request: _local_whisper_cache_key(request, local_whisper_capability),
            maximum_new=32,
        )
        local_whisper_stats["budget_deferred"] = deferred_new_probes
        progress_log(
            f"local Whisper evidence: requests={len(local_whisper_requests)} "
            f"cache_entries={len(retry_helper_cache)} "
            f"deferred_untrusted_new={deferred_new_probes}"
        )
        local_whisper_outcomes = run_local_whisper_requests(
            local_whisper_requests, local_whisper_capability, entries, args,
            cache=retry_helper_cache, stats=local_whisper_stats,
        )
        bind_runtime_local_whisper_evidence(
            report, entries, local_whisper_capability,
            local_whisper_requests, local_whisper_outcomes,
        )
        before_ids = {
            item.candidate_id for items in candidate_sets for item in items
        }
        candidate_sets = _augment_candidates_with_local_whisper_vocal_consensus(
            candidate_sets, report, entries, vocal_onset_features, fusion_audio_revision
        )
        candidate_sets = _augment_candidates_with_local_whisper_consensus(
            candidate_sets, report, entries
        )
        after_ids = {
            item.candidate_id for items in candidate_sets for item in items
        }
        local_whisper_added = len(after_ids - before_ids)
        progress_log(
            f"local Whisper evidence: trusted_observations="
            f"{sum(item.status == 'evidence' for item in local_whisper_outcomes)} "
            f"new_consensus_candidates={local_whisper_added}"
        )

    # v14.7: same-model global/local Whisper contradictions are not independent
    # votes.  A weaker whole-song occurrence may be excluded only when the
    # bounded observation is sequence-bracketed and independently corroborated
    # by CTC at the competing onset.
    candidate_sets, hypothesis_falsifications = (
        _apply_context_conditioned_whisper_hypothesis_arbitration(
            candidate_sets, report, entries
        )
    )
    if hypothesis_falsifications:
        progress_log(
            "hypothesis arbitration: falsified_global_whisper_candidates="
            f"{sum(len(item.get('falsified_candidate_ids', [])) for item in hypothesis_falsifications)} "
            f"entries={','.join(str(item.get('entry')) for item in hypothesis_falsifications)}"
        )

    candidate_sets, wide_region_rows = _attach_ctc_wide_region_candidates(
        candidate_sets, report, entries, hypothesis_falsifications
    )

    diagnostics = {
        **diagnostics,
        "ctc_wide_region_fallback": {
            "source": CTC_WIDE_REGION_FALLBACK_SOURCE,
            "region_radius": CTC_WIDE_REGION_RADIUS,
            "rows": wide_region_rows,
            "central_commit_only": True,
        },
        "independent_tail_evidence": independent_tail_evidence_payload(
            runtime_independent_tail_evidence(report, len(entries))
        ),
        "independent_tail_authority": _tail_authority_diagnostics(
            candidate_sets,
            runtime_independent_tail_evidence(report, len(entries)),
            tail_evidence_context,
            retry_timing_source_policy,
        ),
        "crossline_retry": {
            "retry_epoch_revision": last_epoch_revision,
            "capability_revision": capability.capability_revision if capability else None,
            "capability": _retry_capability_diagnostic_payload(capability),
            "requests": [_retry_diagnostic_payload(item) for item in all_requests],
            "outcomes": [_retry_diagnostic_payload(item) for item in all_outcomes],
            "invocation_count": len(all_outcomes),
            "previous_tail_evidence_count": sum(
                item.previous_tail_evidence is not None for item in all_outcomes
            ),
            "helper_invocation_count": retry_runtime_stats["helper_invocations"],
            "helper_cache_hit_count": retry_runtime_stats["cache_hits"],
            "helper_cache_miss_count": retry_runtime_stats["cache_misses"],
            "helper_cache_entry_count": len(retry_helper_cache),
            "helper_timeout_count": retry_runtime_stats["timeouts"],
            "helper_timeout_seconds": retry_timeout_seconds,
            "processed_semantic_request_count": len(processed_retry_semantics),
            "semantic_fixed_point": bool(retry_passes and retry_passes[-1].get("semantic_fixed_point")),
            "recursive_planning": True,
            "max_passes": max_retry_passes,
            "passes": retry_passes,
        },
        "joint_boundary_retry": {
            "capability_revision": capability.capability_revision if capability else None,
            "requests": [_retry_diagnostic_payload(item) for item in joint_boundary_requests],
            "outcomes": [_retry_diagnostic_payload(item) for item in joint_boundary_outcomes],
            "request_count": len(joint_boundary_requests),
            "outcome_count": len(joint_boundary_outcomes),
            "previous_tail_evidence_count": sum(
                item.previous_tail_evidence is not None
                for item in joint_boundary_outcomes
            ),
            "strong_observation_count": sum(
                1
                for item in joint_boundary_outcomes
                if (
                    item.status == "candidate"
                    and item.candidate is not None
                    and item.candidate.source == "ctc-joint-boundary-observation"
                )
            ),
            "new_consensus_candidate_count": joint_boundary_added,
            "previous_tail_evidence_change_count": (
                joint_tail_evidence_changes
                if joint_boundary_requests
                else 0
            ),
            "central_commit_only": True,
            "observation_selectable_directly": False,
        },
        "local_whisper_evidence": {
            "available": local_whisper_capability is not None,
            "capability_revision": (
                local_whisper_capability.capability_revision
                if local_whisper_capability is not None else None
            ),
            "protocol_revision": (
                local_whisper_capability.protocol_revision
                if local_whisper_capability is not None else None
            ),
            "request_count": len(local_whisper_requests),
            "outcome_count": len(local_whisper_outcomes),
            "trusted_observation_count": sum(
                item.status == "evidence" for item in local_whisper_outcomes
            ),
            "invalid_count": sum(
                item.status == "invalid" for item in local_whisper_outcomes
            ),
            "error_count": sum(
                item.status == "error" for item in local_whisper_outcomes
            ),
            "new_consensus_candidate_count": local_whisper_added,
            "cache_hit_count": local_whisper_stats["cache_hits"],
            "cache_miss_count": local_whisper_stats["cache_misses"],
            "helper_invocation_count": local_whisper_stats["helper_invocations"],
            "entries": [
                {
                    "entry": item.entry_index + 1,
                    "status": item.status,
                    "reason": item.reason,
                    "onset_time": (
                        round(item.onset_time, 6) if item.onset_time is not None else None
                    ),
                    "score": (round(item.score, 6) if item.score is not None else None),
                    "evidence_revision": item.evidence_revision,
                    "occurrence_binding_mode": (
                        item.occurrence_binding.get("mode")
                        if isinstance(item.occurrence_binding, dict) else None
                    ),
                }
                for item in local_whisper_outcomes
            ],
            "central_commit_only": True,
            "observation_selectable_directly": False,
            "whispercpp_and_whisperx_count_as_one_lineage": True,
            "global_raw_and_local_whisper_count_as_one_family": True,
        },
        "hypothesis_arbitration": {
            "mode": "context-conditioned-falsification",
            "falsification_count": sum(
                len(item.get("falsified_candidate_ids", []))
                for item in hypothesis_falsifications
            ),
            "entry_count": len(hypothesis_falsifications),
            "entries": list(hypothesis_falsifications),
            "majority_vote_allowed": False,
            "central_commit_only": True,
        },
    }
    return candidate_sets, diagnostics
