"""Extracted from auto_lrc.py (split phase 2): runtime evidence binding: bind runtime whisperx row evidence, CTC assignment tail fallback and independent tail evidence to timing candidates."""

from __future__ import annotations

import collections
import hashlib
import math
from dataclasses import replace
from typing import Literal, Sequence
from ctc_refinement import CTC_WIDE_REGION_PEER_RADIUS, CTC_WIDE_REGION_RADIUS, RAW_CTC_CONSENSUS_MIN_SCORE, SEGMENT_REALIGN_SOURCE
from report_trust import _RUNTIME_RAW_CONTENT_IDENTITY_KEY, _RUNTIME_RAW_CTC_CONSENSUS_KEY, _RUNTIME_WHISPERX_CTC_CONSENSUS_KEY, _RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY, _RUNTIME_WHISPERX_ROW_EVIDENCE_KEY, _suspicious_row_requires_review, runtime_local_whisper_evidence_for_entry, runtime_raw_content_identity_for_entry, runtime_whisperx_row_evidence_for_entry
from timing_core import BoundedRetryCapability, CTC_WIDE_REGION_FALLBACK_SOURCE, CrossLineRetryOutcome, IndependentTailEvidence, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, RuntimeTailEvidence, TRUSTED_ALIGNMENT_SCORE, TailEvidenceContext, TailEvidenceProvenance, TailEvidenceScope, TailObservationSet, TimingCandidate, WhisperXRuntimeProvenance, _RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY, _RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY, _RUNTIME_WHISPERX_TAIL_SCOPES_KEY, _authoritative_tail_observations, _candidate_confidence, _candidate_support_families, _entry_occurrence_binding, _finite_unit_confidence, _make_tail_provenance, _normalize_tail_observations, _runtime_occurrence_binding_is_valid, _runtime_tail_observation_map, _tail_backend_family, _tail_contexts_are_valid, _tail_helper_revision, _tail_provenance_is_valid, _tail_provenance_matches_context, _tail_provenance_payload, _whisperx_runtime_provenance_is_valid, build_final_timing_state, candidate_occurrence_evidence_from_binding, derive_reliable_tail, make_timing_candidate
from lyrics_io import LyricEntry, assert_lyric_identity_contract, entry_sung_text, normalize_match_text
from audio_features import HOP_SECONDS
from runtime_base import _timing_content_digest, canonicalize_lrc_time
from runtime_base import TimingTokenSpan, _timing_spans_from_value


CTC_WIDE_REGION_PEER_AGREEMENT_SECONDS = 0.3


CTC_WIDE_REGION_SUPPORT_SECONDS = 0.1


CTC_WIDE_REGION_VETO_SECONDS = 0.5


def bind_runtime_raw_content_identity(
    report: dict[str, object],
    entries: list[LyricEntry],
    raw_report: dict[str, object],
) -> None:
    """Bind independent raw-ASR lyric identity without granting timing authority."""
    raw_assignments = raw_report.get("assignments")
    if not isinstance(raw_assignments, list) or len(raw_assignments) != len(entries):
        return
    drift_entries = {
        int(entry)
        for run in report.get("raw_asr_systematic_drift_runs", [])
        if isinstance(run, dict)
        for entry in run.get("entries", [])
        if isinstance(entry, int)
    }
    segment_owners = collections.Counter(
        row.get("segment") for row in raw_assignments
        if isinstance(row, dict) and isinstance(row.get("segment"), int)
    )
    skipped_shared_segment: list[int] = []
    runtime: dict[int, dict[str, object]] = {}
    for index, raw_assignment in enumerate(raw_assignments):
        if (
            isinstance(raw_assignment, dict)
            and isinstance(raw_assignment.get("segment"), int)
            and segment_owners[raw_assignment["segment"]] > 1
        ):
            skipped_shared_segment.append(index + 1)
            continue
        if not isinstance(raw_assignment, dict):
            continue
        occurrence_binding = _entry_occurrence_binding(entries, raw_assignments, index)
        if occurrence_binding is None:
            continue
        try:
            raw_time = float(raw_assignment.get("timestamp"))
            raw_score = float(raw_assignment.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        segment = raw_assignment.get("segment")
        if (
            raw_score < TRUSTED_ALIGNMENT_SCORE
            or bool(raw_assignment.get("borrowed"))
            or segment is None
            or index + 1 in drift_entries
        ):
            continue
        lyric_sha = hashlib.sha256(
            normalize_match_text(entry_sung_text(entries[index])).encode("utf-8")
        ).hexdigest()
        payload = {
            "entry_index": index,
            "lyric_sha256": lyric_sha,
            "raw_time": round(raw_time, 6),
            "raw_score": round(raw_score, 6),
            "segment": int(segment) if isinstance(segment, int) else str(segment),
            "occurrence_binding": occurrence_binding,
            "producer": "raw-asr-content-identity",
        }
        runtime[index] = {
            **payload,
            "evidence_revision": _timing_content_digest(payload),
        }
    report[_RUNTIME_RAW_CONTENT_IDENTITY_KEY] = runtime
    report["raw_asr_content_identity"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "minimum_score": TRUSTED_ALIGNMENT_SCORE,
        "candidate_count": len(runtime),
        "entries": [index + 1 for index in sorted(runtime)],
        "skipped_shared_asr_segment_entries": skipped_shared_segment,
    }


def bind_runtime_whisperx_row_evidence(
    report: dict[str, object],
    entries: list[LyricEntry],
    evidence_candidates: list[tuple[list[float], dict[str, object], str]],
) -> None:
    """Preserve trusted per-line WhisperX identity/boundary geometry at runtime.

    Whole-backend rejection is an aggregate decision.  A row with a real,
    non-borrowed ASR match can still contribute independent identity and forced
    alignment geometry.  Authority remains runtime-only and row review defects
    are excluded before binding.
    """
    best: dict[int, dict[str, object]] = {}
    tail_candidates: dict[int, list[IndependentTailEvidence]] = {}
    scope_records: dict[tuple[int, str], list[TailEvidenceScope]] = {}
    for _candidate_times, candidate_report, _error in evidence_candidates:
        assignments = candidate_report.get("assignments")
        forced_rows = candidate_report.get("whisperx_forced_rows")
        runtime_provenance = candidate_report.get(
            _RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY
        )
        if not isinstance(assignments, list) or not isinstance(forced_rows, list):
            continue
        if len(assignments) != len(entries) or len(forced_rows) != len(entries):
            continue
        actionable_entries = {
            int(row.get("entry"))
            for row in candidate_report.get("suspicious_alignments", [])
            if isinstance(row, dict)
            and isinstance(row.get("entry"), int)
            and _suspicious_row_requires_review(row)
        }
        suppress_nst = bool(candidate_report.get("whisper_suppress_nst"))
        for index, (assignment, forced) in enumerate(zip(assignments, forced_rows)):
            if (
                not isinstance(assignment, dict)
                or not isinstance(forced, dict)
                or index + 1 in actionable_entries
            ):
                continue
            occurrence_binding = _entry_occurrence_binding(entries, assignments, index)
            if occurrence_binding is None:
                continue
            try:
                score = float(assignment.get("score", 0.0) or 0.0)
                segment_start = float(forced.get("segment_start"))
                segment_end = float(forced.get("segment_end"))
                first_time = float(forced.get("first_char_time"))
                last_time = float(forced.get("last_char_time"))
                char_count = int(forced.get("char_count", 0) or 0)
            except (TypeError, ValueError):
                continue
            numeric = (score, segment_start, segment_end, first_time, last_time)
            if (
                not all(math.isfinite(value) for value in numeric)
                or score < TRUSTED_ALIGNMENT_SCORE
                or bool(assignment.get("borrowed"))
                or assignment.get("segment") is None
                or char_count <= 0
                or segment_end < segment_start
                or not (segment_start - HOP_SECONDS <= first_time <= segment_end + HOP_SECONDS)
                or not (first_time - HOP_SECONDS <= last_time <= segment_end + HOP_SECONDS)
            ):
                continue
            lyric_sha = hashlib.sha256(
                normalize_match_text(entry_sung_text(entries[index])).encode("utf-8")
            ).hexdigest()
            payload = {
                "entry_index": index,
                "lyric_sha256": lyric_sha,
                "score": round(score, 6),
                "segment": assignment.get("segment"),
                "segment_start": round(segment_start, 6),
                "segment_end": round(segment_end, 6),
                "first_char_time": round(first_time, 6),
                "last_char_time": round(last_time, 6),
                "char_count": char_count,
                "suppress_nst": suppress_nst,
                "occurrence_binding": occurrence_binding,
                "producer": "whisperx-row-evidence",
            }
            revision = _timing_content_digest(payload)
            row = {**payload, "evidence_revision": revision}
            current = best.get(index)
            if current is None or (score, char_count, -segment_end) > (
                float(current.get("score", 0.0) or 0.0),
                int(current.get("char_count", 0) or 0),
                -float(current.get("segment_end", 0.0) or 0.0),
            ):
                best[index] = row
            uncertainty_seconds = max(0.0, segment_end - last_time)
            provenance: TailEvidenceProvenance | None = None
            scope: TailEvidenceScope | None = None
            if _whisperx_runtime_provenance_is_valid(runtime_provenance):
                assert isinstance(runtime_provenance, WhisperXRuntimeProvenance)
                model_identity = (
                    f"{runtime_provenance.align_model_id}@"
                    f"{runtime_provenance.align_model_revision}#"
                    f"{runtime_provenance.align_model_blob_sha256}"
                )
                helper_revision = _timing_content_digest({
                    "helper_sha256": runtime_provenance.helper_sha256,
                    "helper_protocol_revision": runtime_provenance.helper_protocol_revision,
                })
                transcript_revision = _timing_content_digest({
                    "raw_transcript_revision": runtime_provenance.raw_transcript_revision,
                    "entry_index": index,
                    "lyric_sha256": lyric_sha,
                    "occurrence_binding": occurrence_binding,
                })
                provenance = _make_tail_provenance(
                    entry_index=index,
                    tail_start=first_time,
                    tail_end=last_time,
                    uncertainty_seconds=uncertainty_seconds,
                    confidence=score,
                    producer="whisperx-forced-row",
                    backend_lineage=runtime_provenance.backend_lineage,
                    model_identity=model_identity,
                    helper_revision=helper_revision,
                    capability_revision=runtime_provenance.capability_revision,
                    audio_revision=runtime_provenance.alignment_wav_sha256,
                    transcript_revision=transcript_revision,
                    row_revision=revision,
                )
                if provenance is not None:
                    scope = TailEvidenceScope(
                        entry_index=index,
                        producer="whisperx-forced-row",
                        backend_lineage=runtime_provenance.backend_lineage,
                        model_identity=model_identity,
                        helper_revision=helper_revision,
                        capability_revision=runtime_provenance.capability_revision,
                        audio_revision=runtime_provenance.alignment_wav_sha256,
                        transcript_revisions=(transcript_revision,),
                        row_revisions=(revision,),
                    )
            source_revision = provenance.observation_revision if provenance is not None else revision
            if scope is not None:
                scope_records.setdefault((index, scope.producer), []).append(scope)
            tail_candidates.setdefault(index, []).append(
                IndependentTailEvidence(
                    entry_index=index,
                    tail_start=first_time,
                    tail_end=last_time,
                    uncertainty_seconds=uncertainty_seconds,
                    confidence=score,
                    producer="whisperx-forced-row",
                    source_revision=source_revision,
                    provenance=provenance,
                )
            )

    tails = _runtime_tail_observation_map(report)
    for index, candidates in tail_candidates.items():
        tails.setdefault(index, []).extend(candidates)

    report[_RUNTIME_WHISPERX_ROW_EVIDENCE_KEY] = best
    report[_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = _freeze_runtime_tail_observation_map(tails)
    scopes: dict[tuple[int, str], TailEvidenceScope] = {}
    for key, values in scope_records.items():
        first = values[0]
        if any(
            (
                item.backend_lineage != first.backend_lineage
                or item.model_identity != first.model_identity
                or item.helper_revision != first.helper_revision
                or item.audio_revision != first.audio_revision
            )
            for item in values[1:]
        ):
            continue
        scopes[key] = replace(first, **{
            "transcript_revisions": tuple(sorted({
                revision
                for item in values
                for revision in item.transcript_revisions
            })),
            "row_revisions": tuple(sorted({
                revision
                for item in values
                for revision in item.row_revisions
            })),
            "capability_revisions": tuple(sorted({
                item.capability_revision
                for item in values[1:]
            })),
        })
    report[_RUNTIME_WHISPERX_TAIL_SCOPES_KEY] = tuple(
        scopes[key]
        for key in sorted(scopes)
    )
    report["whisperx_row_evidence"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "minimum_score": TRUSTED_ALIGNMENT_SCORE,
        "candidate_count": len(best),
        "entries": [index + 1 for index in sorted(best)],
        "tail_evidence_count": len(tails),
    }


def bind_runtime_ctc_assignment_tail_fallback(
    report: dict[str, object],
    entries: list[LyricEntry],
) -> None:
    """Restore predecessor-tail geometry that would otherwise be lost after fusion.

    A selected Raw/WhisperX/onset-fusion candidate can legitimately carry no CTC
    spans even though the authoritative CTC assignment for the same immutable lyric
    row still contains usable terminal geometry.  Losing that geometry turns the
    next boundary into UNKNOWN for an implementation reason rather than an evidence
    reason.

    This is intentionally a *fallback*, not another independent backend vote:
    existing WhisperX row-tail evidence always wins.  The CTC row is admitted only
    when no distinct-backend runtime tail exists for that predecessor, and only when
    the normal ``derive_reliable_tail`` rules already classify its terminal run as
    valid.  No timestamp or tail limit is fabricated.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return
    assert_lyric_identity_contract(entries, report, initialize=True)
    tails = _runtime_tail_observation_map(report)
    generated: list[int] = []
    for index, (entry, assignment) in enumerate(zip(entries, assignments)):
        if index in tails or not isinstance(assignment, dict):
            continue
        spans = _timing_spans_from_value(assignment.get("ctc_token_spans"))
        if len(spans) < 2:
            continue
        raw_time = assignment.get("timestamp")
        if not isinstance(raw_time, (int, float)) or not math.isfinite(float(raw_time)):
            continue
        lyric_sha = hashlib.sha256(
            normalize_match_text(entry_sung_text(entry)).encode("utf-8")
        ).hexdigest()
        span_payload = [
            [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
            for span in spans
        ]
        artifact = {
            "producer": "ctc-assignment-terminal-row",
            "entry_index": index,
            "lyric_sha256": lyric_sha,
            "timing_repair_source": assignment.get("timing_repair_source"),
            "ctc_audio_source": report.get("ctc_audio_source"),
            "ctc_audio_sha256": report.get("ctc_audio_sha256") or report.get("vocal_stem_sha256"),
            "spans_revision": _timing_content_digest(span_payload),
        }
        probe = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entry),
            source="ctc-assignment-tail-fallback",
            raw_time=float(raw_time),
            spans=spans,
            confidence=_candidate_confidence(assignment),
            identity_support="supported",
            sequence_support="supported",
            current=False,
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(
                assignment.get("ctc_right_edge_truncation")
                or assignment.get("ctc_boundary_realign_required")
            ),
            source_artifact=artifact,
        )
        tail = derive_reliable_tail(probe)
        if (
            tail.validity != "valid"
            or tail.reliable_tail_start is None
            or tail.reliable_tail_end is None
        ):
            continue
        payload = {
            **artifact,
            "tail_start": round(tail.reliable_tail_start, 6),
            "tail_end": round(tail.reliable_tail_end, 6),
            "uncertainty_seconds": round(float(tail.uncertainty_seconds or 0.0), 6),
            "confidence": round(float(tail.confidence or 0.0), 6),
            "tail_evidence_revision": tail.tail_evidence_revision,
        }
        entry_index = index
        tail_start = float(tail.reliable_tail_start)
        tail_end = float(tail.reliable_tail_end)
        uncertainty_seconds = max(0.0, float(tail.uncertainty_seconds or 0.0))
        confidence = _finite_unit_confidence(tail.confidence or 0.0)
        capability = report.get(_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY)
        if isinstance(capability, BoundedRetryCapability):
            backend_lineage = capability.backend_lineage
            model_identity = capability.model_identity
            helper_revision = _tail_helper_revision(
                capability.helper_sha256,
                capability.helper_protocol_revision,
            )
            capability_revision = capability.capability_revision
            audio_revision = capability.alignment_audio_sha256
        else:
            backend_lineage = str(report.get("backend_lineage") or "")
            model_identity = str(report.get("model_identity") or "")
            helper_revision = str(report.get("helper_revision") or "")
            capability_revision = str(report.get("capability_revision") or "")
            audio_revision = str(
                report.get("ctc_audio_sha256") or report.get("vocal_stem_sha256") or ""
            )
        provenance = _make_tail_provenance(
            entry_index=entry_index,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty_seconds,
            confidence=confidence,
            producer="ctc-assignment-terminal-row",
            backend_lineage=backend_lineage,
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=capability_revision,
            audio_revision=audio_revision,
            transcript_revision=lyric_sha,
            row_revision=_timing_content_digest(span_payload),
        )
        source_revision = (
            provenance.observation_revision
            if provenance is not None else _timing_content_digest(payload)
        )
        observation = IndependentTailEvidence(
            entry_index=entry_index,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty_seconds,
            confidence=confidence,
            producer="ctc-assignment-terminal-row",
            source_revision=source_revision,
            provenance=provenance,
        )
        tails.setdefault(index, []).append(observation)
        generated.append(index + 1)
    report[_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = _freeze_runtime_tail_observation_map(tails)
    report["ctc_assignment_tail_fallback"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "role": "fallback-only-when-distinct-backend-tail-missing",
        "candidate_count": len(generated),
        "entries": generated,
    }


def _augment_local_ctc_candidates_with_runtime_backend_consensus(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Extend existing cross-backend onset proof to local/retry CTC candidates.

    v14.2 could bind Raw/WhisperX identity only to the baseline ``ctc-current``
    time.  A bounded local CTC retry may produce a better lyric-specific opening,
    yet remain unable to use the same independent backend observation.  This
    helper closes only that provenance gap.  It reuses the existing 0.18 s
    consensus tolerance and keeps the CTC candidate's existing timestamp; no
    averaging or synthetic boundary is introduced.
    """
    # Structural S1 is also local CTC. Independent Raw/WhisperX identities
    # may corroborate the SAME bounded candidate only if they already
    # satisfy the normal 0.18 s consensus and revision-bound occurrence.
    # Bare S1 CTC / vocal presence is never upgraded on its own.
    local_sources = {"ctc-local-window", "ctc-opening-local-retry", "segment-constrained-ctc", SEGMENT_REALIGN_SOURCE}
    augmented: list[tuple[TimingCandidate, ...]] = []
    for index, items in enumerate(candidate_sets):
        rows = list(items)
        raw_identity = runtime_raw_content_identity_for_entry(report, entries, index)
        whisperx_row = runtime_whisperx_row_evidence_for_entry(report, entries, index)
        for parent in items:
            if (
                parent.source not in local_sources
                or not parent.spans
                or parent.right_edge_pileup
                or parent.window_truncated
            ):
                continue
            backend_rows: list[tuple[str, float, float, str, str]] = []
            if isinstance(raw_identity, dict):
                try:
                    raw_time = float(raw_identity.get("raw_time"))
                    raw_score = float(raw_identity.get("raw_score"))
                except (TypeError, ValueError):
                    raw_time = math.nan
                    raw_score = 0.0
                if (
                    math.isfinite(raw_time)
                    and raw_score >= RAW_CTC_CONSENSUS_MIN_SCORE
                    and abs(raw_time - parent.written_time.seconds)
                    <= RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
                ):
                    backend_rows.append((
                        "raw", raw_time, raw_score,
                        str(raw_identity.get("evidence_revision") or ""),
                        "raw-asr-consensus",
                    ))
            if isinstance(whisperx_row, dict):
                try:
                    forced_time = float(whisperx_row.get("first_char_time"))
                    whisperx_score = float(whisperx_row.get("score"))
                except (TypeError, ValueError):
                    forced_time = math.nan
                    whisperx_score = 0.0
                if (
                    math.isfinite(forced_time)
                    and whisperx_score >= TRUSTED_ALIGNMENT_SCORE
                    and abs(forced_time - parent.written_time.seconds)
                    <= RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
                ):
                    backend_rows.append((
                        "whisperx", forced_time, whisperx_score,
                        str(whisperx_row.get("evidence_revision") or ""),
                        "whisperx-consensus",
                    ))
            for label, independent_time, independent_score, evidence_revision, producer in backend_rows:
                identity_evidence = (
                    raw_identity if label == "raw" else whisperx_row
                )
                occurrence_evidence = candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer=producer,
                    occurrence_binding=(
                        identity_evidence.get("occurrence_binding")
                        if isinstance(identity_evidence, dict) else None
                    ),
                    upstream_evidence_revision=(
                        identity_evidence.get("evidence_revision")
                        if isinstance(identity_evidence, dict) else None
                    ),
                )
                artifact = {
                    "producer": f"ctc-local-{label}-consensus",
                    "parent_candidate_id": parent.candidate_id,
                    "parent_generation_revision": parent.generation_revision,
                    "parent_time": round(parent.written_time.seconds, 6),
                    "independent_time": round(independent_time, 6),
                    "independent_score": round(independent_score, 6),
                    "independent_evidence_revision": evidence_revision,
                    "consensus_tolerance_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
                    "runtime_only": True,
                }
                rows.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entries[index]),
                        source=f"ctc-local-{label}-independent-consensus",
                        raw_time=parent.written_time.seconds,
                        spans=parent.spans,
                        confidence=min(parent.confidence, independent_score),
                        identity_support="supported",
                        sequence_support="supported",
                        acoustic_support="unavailable",
                        direct_onset_support="supported",
                        direct_onset_time=independent_time,
                        direct_onset_source_artifact=artifact,
                        direct_onset_evidence_producer=producer,
                        direct_onset_evidence_kind="cross-backend-onset",
                        direct_onset_evidence_independent=True,
                        occurrence_evidence=occurrence_evidence,
                        independent_content_identity=occurrence_evidence is not None,
                        current=False,
                        source_artifact=artifact,
                    )
                )
        unique = {candidate.candidate_id: candidate for candidate in rows}
        augmented.append(tuple(unique[key] for key in sorted(unique)))
    return tuple(augmented)


def _joint_consensus_spans(
    parent: TimingCandidate,
    observation: TimingCandidate,
) -> tuple[TimingTokenSpan, ...]:
    """Token spans for a joint-consensus candidate written at the parent time.

    A span-less parent may borrow the joint observation's spans (this gives the
    row a reliable tail for its successor, case34), but only when those spans
    describe the same written time: the observation's first token must start
    within one hop of the parent time, exactly the tolerance that
    ``normalize_timing_evidence`` enforces ("candidate-span-revision-mismatch").
    Borrowing spans that start elsewhere made every such joint candidate
    invalid (43 joint candidates across the 39-song corpus, e.g. ふり #55).
    """
    if parent.spans:
        return parent.spans
    borrowed = observation.spans
    if borrowed and abs(borrowed[0].start - parent.written_time.seconds) <= HOP_SECONDS:
        return borrowed
    return parent.spans


def _augment_candidates_with_joint_boundary_consensus(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    pair_outcomes: tuple[CrossLineRetryOutcome, ...],
    entries: list[LyricEntry],
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Bind joint previous+current CTC boundary observations to separate candidates.

    The pair observation is deliberately not selectable by itself.  It may only
    certify a separately generated current-line candidate when both boundaries
    converge within the already-established 0.18 s consensus tolerance.

    This is stronger than generic acoustic activity: the observation explicitly
    aligns the immutable previous lyric and current lyric in one CTC sequence,
    and the selected timestamp still comes from an independent candidate
    generation path rather than from the pair observation.
    """
    by_entry: dict[int, list[TimingCandidate]] = collections.defaultdict(list)
    for outcome in pair_outcomes:
        candidate = outcome.candidate
        if (
            outcome.status != "candidate"
            or candidate is None
            or candidate.source != "ctc-joint-boundary-observation"
            or candidate.window_truncated
            or candidate.right_edge_pileup
            or not candidate.spans
        ):
            continue
        by_entry[outcome.entry_index].append(candidate)

    augmented: list[tuple[TimingCandidate, ...]] = []
    for index, items in enumerate(candidate_sets):
        rows = list(items)
        observations = by_entry.get(index, [])
        if observations:
            for parent in items:
                if (
                    parent.source.startswith("ctc-joint-boundary")
                    or parent.window_truncated
                    or parent.right_edge_pileup
                    or parent.identity_support != "supported"
                    or parent.sequence_support != "supported"
                ):
                    continue
                for observation in observations:
                    if (
                        abs(
                            observation.written_time.seconds
                            - parent.written_time.seconds
                        )
                        > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
                    ):
                        continue
                    artifact = {
                        "producer": "joint-ctc-boundary-consensus",
                        "parent_candidate_id": parent.candidate_id,
                        "parent_generation_revision": parent.generation_revision,
                        "parent_time": round(parent.written_time.seconds, 6),
                        "pair_candidate_id": observation.candidate_id,
                        "pair_generation_revision": observation.generation_revision,
                        "pair_time": round(observation.written_time.seconds, 6),
                        "pair_span_digest": _timing_content_digest([
                            [
                                round(span.start, 6),
                                round(span.end, 6),
                                round(span.score, 8),
                                span.token,
                            ]
                            for span in observation.spans
                        ]),
                        "consensus_tolerance_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
                        "runtime_only": True,
                    }
                    rows.append(
                        make_timing_candidate(
                            entry_index=index,
                            entry_text=entry_sung_text(entries[index]),
                            source="ctc-joint-boundary-consensus",
                            raw_time=parent.written_time.seconds,
                            spans=_joint_consensus_spans(parent, observation),
                            confidence=min(parent.confidence, observation.confidence),
                            identity_support="supported",
                            sequence_support="supported",
                            acoustic_support=parent.acoustic_support,
                            direct_onset_support="supported",
                            acoustic_strength=parent.acoustic_strength,
                            acoustic_onset_time=(
                                parent.acoustic_evidence.onset_time
                                if parent.acoustic_evidence is not None else None
                            ),
                            acoustic_source_artifact=(
                                {
                                    "parent_acoustic_revision":
                                        parent.acoustic_evidence.source_revision
                                }
                                if parent.acoustic_evidence is not None else None
                            ),
                            acoustic_evidence_producer=(
                                parent.acoustic_evidence.producer
                                if parent.acoustic_evidence is not None else None
                            ),
                            acoustic_evidence_kind=(
                                parent.acoustic_evidence.kind
                                if parent.acoustic_evidence is not None
                                else "generic-acoustic"
                            ),
                            acoustic_evidence_independent=(
                                parent.acoustic_evidence.independent
                                if parent.acoustic_evidence is not None else False
                            ),
                            direct_onset_time=observation.written_time.seconds,
                            direct_onset_source_artifact=artifact,
                            direct_onset_evidence_producer="joint-ctc-boundary-consensus",
                            direct_onset_evidence_kind="joint-lyric-boundary",
                            direct_onset_evidence_independent=True,
                            occurrence_evidence=parent.occurrence_evidence,
                            independent_content_identity=(
                                parent.independent_content_identity
                                and parent.occurrence_evidence is not None
                            ),
                            current=False,
                            prefix_scope=parent.prefix_scope,
                            source_artifact=artifact,
                        )
                    )
        unique = {candidate.candidate_id: candidate for candidate in rows}
        augmented.append(tuple(unique[key] for key in sorted(unique)))
    return tuple(augmented)


def _freeze_runtime_tail_observation_map(
    observations: dict[int, list[IndependentTailEvidence]],
) -> dict[int, TailObservationSet]:
    frozen: dict[int, TailObservationSet] = {}
    for index, values in observations.items():
        unique: dict[str, IndependentTailEvidence] = {}
        for item in values:
            revision = (
                item.provenance.observation_revision
                if item.provenance is not None
                else item.source_revision
            )
            unique.setdefault(revision, item)
        if unique:
            frozen[index] = tuple(
                unique[key]
                for key in sorted(unique)
            )
    return frozen


def merge_runtime_independent_tail_evidence(
    report: dict[str, object],
    evidence: IndependentTailEvidence | None,
) -> bool:
    """Append one immutable observation without cross-lineage replacement."""
    if evidence is None or evidence.entry_index < 0:
        return False
    merged = _runtime_tail_observation_map(report)
    before = _freeze_runtime_tail_observation_map(merged)
    merged.setdefault(evidence.entry_index, []).append(evidence)
    after = _freeze_runtime_tail_observation_map(merged)
    report[_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = after
    return before != after


def _independent_tail_evidence_payload(
    evidence: IndependentTailEvidence | None,
) -> dict[str, object] | None:
    if evidence is None:
        return None
    payload: dict[str, object] = {
        "entry_index": evidence.entry_index,
        "tail_start": round(evidence.tail_start, 6),
        "tail_end": round(evidence.tail_end, 6),
        "uncertainty_seconds": round(evidence.uncertainty_seconds, 6),
        "confidence": round(evidence.confidence, 6),
        "producer": evidence.producer,
        "source_revision": evidence.source_revision,
    }
    if evidence.provenance is None:
        payload["provenance_status"] = "missing"
        payload["authority_level"] = "DIAGNOSTIC_ONLY"
        payload["rejection_reason"] = "missing-tail-provenance"
    elif _tail_provenance_is_valid(evidence):
        payload["provenance_status"] = "valid"
        payload["authority_level"] = "PENDING_PREDECESSOR_LINEAGE_CHECK"
        payload["provenance"] = {
            **_tail_provenance_payload(evidence.provenance),
            "observation_revision": evidence.provenance.observation_revision,
        }
    else:
        payload["provenance_status"] = "invalid"
        payload["authority_level"] = "DIAGNOSTIC_ONLY"
        payload["rejection_reason"] = "invalid-tail-provenance"
    return payload


def independent_tail_evidence_payload(
    evidence: RuntimeTailEvidence | tuple[IndependentTailEvidence | None, ...],
) -> list[object]:
    payload: list[object] = []
    for item in evidence:
        observations = _normalize_tail_observations(item)  # type: ignore[arg-type]
        if not observations:
            payload.append(None)
        elif len(observations) == 1:
            payload.append(_independent_tail_evidence_payload(observations[0]))
        else:
            payload.append([
                _independent_tail_evidence_payload(observation)
                for observation in observations
            ])
    return payload


def _candidate_set_semantic_revision(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> str:
    return _timing_content_digest([
        sorted(candidate.candidate_id for candidate in candidates)
        for candidates in candidate_sets
    ])


def _authoritative_tail_semantic_revision(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    independent_tail_evidence: RuntimeTailEvidence,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> str:
    if len(candidate_sets) != len(independent_tail_evidence):
        return _timing_content_digest({"invalid_length": True})
    try:
        state = build_final_timing_state(
            candidate_sets,
            independent_tail_evidence=independent_tail_evidence,
            tail_evidence_context=tail_evidence_context,
            timing_source_policy=timing_source_policy,
        )
    except (TypeError, ValueError):
        return _timing_content_digest({"invalid_state": True})
    return _timing_content_digest([
        {
            "entry_index": decision.entry_index,
            "ownership_status": decision.evidence.ownership.status,
            "tail_revision": decision.evidence.referenced_tail_evidence_revision,
            "tail_validity": decision.evidence.ownership.previous_tail_validity,
            "tail_end": decision.evidence.ownership.reliable_tail_end,
            "tail_uncertainty": decision.evidence.ownership.uncertainty_seconds,
        }
        for decision in state.decisions
        if decision.entry_index > 0
    ])


def _retry_semantic_state_revision(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    independent_tail_evidence: RuntimeTailEvidence,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> tuple[str, str]:
    return (
        _candidate_set_semantic_revision(candidate_sets),
        _authoritative_tail_semantic_revision(
            candidate_sets,
            independent_tail_evidence,
            tail_evidence_context,
            timing_source_policy,
        ),
    )


def _tail_authority_diagnostics(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    independent_tail_evidence: RuntimeTailEvidence,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> list[dict[str, object]]:
    """Explain Contract-A acceptance/rejection without adding trust logic."""
    if len(candidate_sets) != len(independent_tail_evidence):
        return []
    try:
        state = build_final_timing_state(
            candidate_sets,
            independent_tail_evidence=independent_tail_evidence,
            tail_evidence_context=tail_evidence_context,
            timing_source_policy=timing_source_policy,
        )
    except (TypeError, ValueError):
        return []
    rows: list[dict[str, object]] = []
    for index, observations in enumerate(independent_tail_evidence):
        if not observations or index >= len(state.decisions):
            continue
        predecessor = state.decisions[index].candidate
        families = sorted(_candidate_support_families(predecessor))
        authoritative = _authoritative_tail_observations(
            predecessor, observations, tail_evidence_context
        )
        selected = authoritative[0] if authoritative else None
        intrinsic = derive_reliable_tail(predecessor)
        intrinsic_upper = (
            intrinsic.reliable_tail_end + (intrinsic.uncertainty_seconds or 0.0)
            if intrinsic.validity == "valid" and intrinsic.reliable_tail_end is not None
            else None
        )
        for observation in observations:
            provenance = observation.provenance
            family = _tail_backend_family(provenance.backend_lineage if provenance else None)
            if not _tail_provenance_is_valid(observation):
                authority, reason = "DIAGNOSTIC_ONLY", "invalid-or-missing-provenance"
            elif not _tail_contexts_are_valid(tail_evidence_context):
                authority, reason = "DIAGNOSTIC_ONLY", "missing-tail-evidence-context"
            elif not _tail_provenance_matches_context(observation, tail_evidence_context):
                authority, reason = "DIAGNOSTIC_ONLY", "stale-tail-evidence-context"
            elif not families:
                authority, reason = "REJECTED", "unknown-predecessor-support-family"
            elif family in families:
                authority, reason = "REJECTED", "same-lineage-overlap"
            elif selected is not observation:
                authority, reason = "VALID_NONSELECTED", "another-disjoint-bound-dominates"
            elif intrinsic_upper is not None and intrinsic_upper >= (
                observation.tail_end + observation.uncertainty_seconds
            ):
                authority, reason = "VALID_NONSELECTED", "intrinsic-bound-dominates"
            else:
                authority, reason = "ACCEPTED", "disjoint-lineage-authoritative-bound"
            rows.append({
                "entry": index + 1,
                "producer": observation.producer,
                "backend_lineage": provenance.backend_lineage if provenance else None,
                "candidate_source": predecessor.source,
                "candidate_support_families": families,
                "authority": authority,
                "reason": reason,
                "observation_revision": observation.source_revision,
                "tail_start": round(observation.tail_start, 6),
                "tail_end": round(observation.tail_end, 6),
                "uncertainty_seconds": round(observation.uncertainty_seconds, 6),
            })
    return rows


def bind_runtime_whisperx_ctc_consensus(
    report: dict[str, object],
    entries: list[LyricEntry],
    ctc_timestamps: list[float],
    evidence_candidates: list[tuple[list[float], dict[str, object], str]],
) -> None:
    """Preserve line-valid WhisperX evidence even when the backend fails globally."""
    best: dict[int, dict[str, object]] = {}
    for _candidate_times, candidate_report, _error in evidence_candidates:
        assignments = candidate_report.get("assignments")
        forced_times = candidate_report.get("whisperx_forced_first_times")
        if not isinstance(assignments, list) or not isinstance(forced_times, list):
            continue
        if len(assignments) != len(entries) or len(forced_times) != len(entries):
            continue
        suppress_nst = bool(candidate_report.get("whisper_suppress_nst"))
        for index, (assignment, forced_value) in enumerate(zip(assignments, forced_times)):
            if not isinstance(assignment, dict):
                continue
            occurrence_binding = _entry_occurrence_binding(entries, assignments, index)
            if occurrence_binding is None:
                continue
            try:
                score = float(assignment.get("score", 0.0) or 0.0)
                forced_time = float(forced_value)
                ctc_time = float(ctc_timestamps[index])
            except (TypeError, ValueError, IndexError):
                continue
            if (
                score < TRUSTED_ALIGNMENT_SCORE
                or bool(assignment.get("borrowed"))
                or assignment.get("segment") is None
                or not math.isfinite(forced_time)
                or abs(forced_time - ctc_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
            ):
                continue
            lyric_sha = hashlib.sha256(
                normalize_match_text(entry_sung_text(entries[index])).encode("utf-8")
            ).hexdigest()
            payload = {
                "entry_index": index,
                "lyric_sha256": lyric_sha,
                "whisperx_forced_time": round(forced_time, 6),
                "ctc_time": round(ctc_time, 6),
                "score": round(score, 6),
                "segment": assignment.get("segment"),
                "suppress_nst": suppress_nst,
                "occurrence_binding": occurrence_binding,
                "producer": "whisperx-consensus",
            }
            row = {**payload, "evidence_revision": _timing_content_digest(payload)}
            current = best.get(index)
            if current is None or (score, -abs(forced_time - ctc_time)) > (
                float(current.get("score", 0.0) or 0.0),
                -abs(float(current.get("whisperx_forced_time", 0.0)) - ctc_time),
            ):
                best[index] = row
    report[_RUNTIME_WHISPERX_CTC_CONSENSUS_KEY] = best
    report["whisperx_ctc_onset_consensus"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "minimum_score": TRUSTED_ALIGNMENT_SCORE,
        "max_delta_seconds": PREFIX_ACOUSTIC_ASSOCIATION_SECONDS,
        "candidate_count": len(best),
        "entries": [index + 1 for index in sorted(best)],
    }


def runtime_whisperx_ctc_consensus_for_entry(
    report: dict[str, object],
    entries: list[LyricEntry],
    entry_index: int,
    current_time: float,
) -> dict[str, object] | None:
    runtime = report.get(_RUNTIME_WHISPERX_CTC_CONSENSUS_KEY)
    if not isinstance(runtime, dict):
        return None
    evidence = runtime.get(entry_index)
    if not isinstance(evidence, dict):
        return None
    try:
        forced_time = float(evidence.get("whisperx_forced_time"))
        ctc_time = float(evidence.get("ctc_time"))
        score = float(evidence.get("score"))
    except (TypeError, ValueError):
        return None
    lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entries[entry_index])).encode("utf-8")
    ).hexdigest()
    payload = {
        "entry_index": entry_index,
        "lyric_sha256": lyric_sha,
        "whisperx_forced_time": round(forced_time, 6),
        "ctc_time": round(ctc_time, 6),
        "score": round(score, 6),
        "segment": evidence.get("segment"),
        "suppress_nst": bool(evidence.get("suppress_nst")),
        "occurrence_binding": evidence.get("occurrence_binding"),
        "producer": "whisperx-consensus",
    }
    if (
        evidence.get("producer") != "whisperx-consensus"
        or score < TRUSTED_ALIGNMENT_SCORE
        or evidence.get("lyric_sha256") != lyric_sha
        or not _runtime_occurrence_binding_is_valid(
            entries, entry_index, evidence.get("occurrence_binding")
        )
        or abs(forced_time - ctc_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        or canonicalize_lrc_time(ctc_time).centiseconds
        != canonicalize_lrc_time(current_time).centiseconds
        or evidence.get("evidence_revision") != _timing_content_digest(payload)
    ):
        return None
    return dict(evidence)


def runtime_raw_ctc_consensus_for_entry(
    report: dict[str, object],
    entries: list[LyricEntry],
    entry_index: int,
    current_time: float,
) -> dict[str, object] | None:
    """Return fresh in-process raw/CTC onset agreement for one entry.

    The authority is deliberately non-serializable and is re-bound to the
    current CTC timestamp.  This prevents stale report metadata from being used
    as independent evidence on another run or after the baseline moved.
    """
    if str(report.get("backend") or "") != "ctc":
        return None
    runtime = report.get(_RUNTIME_RAW_CTC_CONSENSUS_KEY)
    if not isinstance(runtime, dict):
        return None
    evidence = runtime.get(entry_index)
    if not isinstance(evidence, dict):
        return None
    try:
        raw_time = float(evidence.get("raw_time"))
        ctc_time = float(evidence.get("ctc_time"))
        raw_score = float(evidence.get("raw_score"))
        max_delta = float(evidence.get("max_delta_seconds"))
    except (TypeError, ValueError):
        return None
    if (
        evidence.get("producer") != "raw-asr-consensus"
        or raw_score < RAW_CTC_CONSENSUS_MIN_SCORE
        or abs(max_delta - RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS) > 1e-9
        or abs(raw_time - ctc_time) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
        or canonicalize_lrc_time(ctc_time).centiseconds
        != canonicalize_lrc_time(current_time).centiseconds
    ):
        return None
    payload = {
        "entry_index": entry_index,
        "raw_time": round(raw_time, 6),
        "ctc_time": round(ctc_time, 6),
        "raw_score": round(raw_score, 6),
        "max_delta_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
        "occurrence_binding": evidence.get("occurrence_binding"),
        "producer": "raw-asr-consensus",
    }
    if (
        not _runtime_occurrence_binding_is_valid(
            entries, entry_index, evidence.get("occurrence_binding")
        )
        or evidence.get("evidence_revision") != _timing_content_digest(payload)
    ):
        return None
    return dict(evidence)


def _attach_ctc_wide_region_candidates(
    candidate_sets: Sequence[tuple[TimingCandidate, ...]],
    report: dict[str, object],
    entries: list[LyricEntry],
    hypothesis_falsifications: Sequence[dict[str, object]],
) -> tuple[tuple[tuple[TimingCandidate, ...], ...], list[dict[str, object]]]:
    """Admit a row's radius-3 replay as its wide-region fallback candidate.

    Agreement: the radius-2 replay lands within 0.3 s, or another candidate
    that is not a first-token posterior lands within 0.1 s.  Veto: any
    independent onset for the row (acoustic recurrence, a trusted local
    Whisper onset or its arbitration) more than 0.5 s away.  The vocal-fusion
    onsets riding on candidates are not vetoes: on c17 #20 one sat 1.3 s off
    BK and blocked a correct wide time.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(candidate_sets):
        return tuple(candidate_sets), []
    independent_onsets: dict[int, list[float]] = collections.defaultdict(list)
    for key in ("mutual_acoustic_recurrence_hypotheses", "global_acoustic_recurrence_hypotheses"):
        rows = report.get(key)
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and isinstance(row.get("candidate"), (int, float)):
                independent_onsets[int(row.get("entry", 0)) - 1].append(float(row["candidate"]))
    for row in hypothesis_falsifications:
        if isinstance(row.get("local_onset"), (int, float)):
            independent_onsets[int(row.get("entry", 0)) - 1].append(float(row["local_onset"]))
    revised: list[tuple[TimingCandidate, ...]] = []
    diagnostics: list[dict[str, object]] = []
    for index, (candidates, assignment) in enumerate(zip(candidate_sets, assignments)):
        hypotheses = assignment.get("alignment_hypotheses") if isinstance(assignment, dict) else None
        wide: dict[int, dict[str, object]] = {}
        for hypothesis in hypotheses if isinstance(hypotheses, list) else []:
            if (
                isinstance(hypothesis, dict)
                and hypothesis.get("source") == CTC_WIDE_REGION_FALLBACK_SOURCE
                and isinstance(hypothesis.get("raw_candidate_time"), (int, float))
            ):
                for radius in (CTC_WIDE_REGION_RADIUS, CTC_WIDE_REGION_PEER_RADIUS):
                    if hypothesis.get("mode") == f"region-radius-{radius}":
                        wide[radius] = hypothesis
        primary = wide.get(CTC_WIDE_REGION_RADIUS)
        if primary is None:
            revised.append(tuple(candidates))
            continue
        seconds = canonicalize_lrc_time(float(primary["raw_candidate_time"])).seconds
        peer = wide.get(CTC_WIDE_REGION_PEER_RADIUS)
        peer_agrees = bool(
            peer is not None
            and abs(float(peer["raw_candidate_time"]) - seconds)
            <= CTC_WIDE_REGION_PEER_AGREEMENT_SECONDS
        )
        supported = any(
            candidate.source != "ctc-first-token-posterior"
            and abs(candidate.written_time.seconds - seconds) <= CTC_WIDE_REGION_SUPPORT_SECONDS
            for candidate in candidates
        )
        onsets = list(independent_onsets.get(index, []))
        local = runtime_local_whisper_evidence_for_entry(report, entries, index)
        if isinstance(local, dict) and isinstance(local.get("onset_time"), (int, float)):
            onsets.append(float(local["onset_time"]))
        vetoed = any(abs(onset - seconds) > CTC_WIDE_REGION_VETO_SECONDS for onset in onsets)
        admitted = (peer_agrees or supported) and not vetoed
        diagnostics.append({
            "entry": index + 1,
            "time": seconds,
            "peer_time": (
                round(float(peer["raw_candidate_time"]), 3) if peer is not None else None
            ),
            "peer_agrees": peer_agrees,
            "candidate_support": supported,
            "independent_onsets": sorted(round(onset, 3) for onset in onsets),
            "vetoed": vetoed,
            "admitted": admitted,
        })
        if not admitted:
            revised.append(tuple(candidates))
            continue
        spans = _timing_spans_from_value(primary.get("spans"))
        wide_candidate = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entries[index]),
            source=CTC_WIDE_REGION_FALLBACK_SOURCE,
            raw_time=float(primary["raw_candidate_time"]),
            spans=spans,
            confidence=float(primary.get("confidence", 0.0) or 0.0),
            sequence_support="supported",
            source_artifact=primary,
        )
        unique = {candidate.candidate_id: candidate for candidate in (*candidates, wide_candidate)}
        revised.append(tuple(unique[key] for key in sorted(unique)))
    return tuple(revised), diagnostics
