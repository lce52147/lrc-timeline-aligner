"""Extracted from local_whisper_retry.py (split2): crossline and joint-boundary retry planning, base retry epoch and generation candidate helpers."""

from __future__ import annotations

import math
from typing import Literal
from report_trust import freeze_generation_diagnostics, timing_report_evaluation_revision
from timing_core import BaseRetryEpoch, BoundedRetryCapability, CrossLineRetryRequest, RetryCandidateReference, RuntimeTailEvidence, TailEvidenceContext, TimingCandidate, analyze_candidate_regions, build_final_timing_state, candidate_has_bound_direct_onset_evidence, derive_reliable_tail, generation_diagnostics_semantic_revision, resolve_reliable_tail
from lyrics_io import LyricEntry, entry_sung_text
from audio_features import HOP_SECONDS
from runtime_base import _timing_content_digest
from runtime_evidence_binding import _candidate_set_semantic_revision, bind_runtime_ctc_assignment_tail_fallback, bind_runtime_whisperx_ctc_consensus, bind_runtime_whisperx_row_evidence


CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS = 0.75


CROSSLINE_RETRY_MAX_WINDOW_EXPANSIONS = 3


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
