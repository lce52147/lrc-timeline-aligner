"""Extracted from timing_core.py (split2): candidate occurrence evidence, tail provenance and context validation, serialized-time helpers and runtime occurrence binding."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import replace
from multimodel_consensus import model_identity
from lyrics_io import LyricEntry, _RUNTIME_LYRIC_ENTRIES_KEY, entry_sung_text, has_ambiguous_lyric_prefix, lyric_identity_revision, normalize_match_text
from runtime_base import _timing_content_digest
from runtime_base import _timing_spans_from_value
from timing_evidence_types import (
    TRUSTED_ALIGNMENT_SCORE,
    _RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY,
    _RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY,
    _RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY,
    _RUNTIME_WHISPERX_TAIL_SCOPES_KEY,
    _WHISPERX_HELPER_PROTOCOL_REVISION,
    CandidateOccurrenceEvidence,
    TimingCandidate,
    TailEvidenceProvenance,
    TailEvidenceScope,
    TailEvidenceContext,
    WhisperXRuntimeProvenance,
    IndependentTailEvidence,
    TailObservationSet,
    RuntimeTailEvidence,
    BoundedRetryCapability,
    _CANDIDATE_OCCURRENCE_MODES,
    _CANDIDATE_OCCURRENCE_DIRECT_PRODUCERS,
    _CANDIDATE_OCCURRENCE_BINDING_KEYS,
)


def _candidate_occurrence_binding_keys_are_canonical(
    binding: object,
) -> bool:
    if not isinstance(binding, dict):
        return False
    mode = binding.get("mode")
    if not isinstance(mode, str):
        return False
    allowed = _CANDIDATE_OCCURRENCE_BINDING_KEYS.get(mode)
    return allowed is not None and set(binding).issubset(allowed)


def _candidate_occurrence_context_indexes(
    binding: dict[str, object],
    entry_index: int,
) -> tuple[int, ...] | None:
    """Return the canonical context offsets represented by one binding."""
    mode = binding.get("mode")
    if mode == "global-unique":
        return (entry_index,)
    key = (
        "context_entry_indexes"
        if mode == "sequence-bracket"
        else "anchor_entry_indexes"
        if mode == "sparse-sequence-bracket"
        else None
    )
    if key is None:
        return None
    indexes = binding.get(key)
    if not isinstance(indexes, list):
        return None
    try:
        typed = tuple(int(value) for value in indexes)
    except (TypeError, ValueError):
        return None
    return typed


def _canonical_occurrence_binding_json(binding: object) -> str | None:
    if not isinstance(binding, dict):
        return None
    try:
        return json.dumps(
            binding,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


def _candidate_occurrence_evidence_payload(
    evidence: CandidateOccurrenceEvidence | None,
) -> dict[str, object] | None:
    """Return the path-free semantic payload used by the evidence digest."""
    if evidence is None:
        return None
    try:
        binding = json.loads(evidence.binding_payload_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        binding = None
    if (
        not isinstance(binding, dict)
        or not _candidate_occurrence_binding_keys_are_canonical(binding)
    ):
        return None
    return {
        "entry_index": evidence.entry_index,
        "producer": evidence.producer,
        "binding": binding,
        "binding_revision": evidence.binding_revision,
        "lyric_identity_revision": evidence.lyric_identity_revision,
        "lyric_sha256": evidence.lyric_sha256,
        "occurrence_mode": evidence.occurrence_mode,
        "context_entry_indexes": list(evidence.context_entry_indexes),
        "unique": evidence.unique,
        "upstream_evidence_revision": evidence.upstream_evidence_revision,
    }


def _candidate_occurrence_evidence_projection(
    evidence: CandidateOccurrenceEvidence | None,
) -> dict[str, object] | None:
    """Return a JSON-safe projection without paths or raw transcript data."""
    if evidence is None:
        return None
    payload = _candidate_occurrence_evidence_payload(evidence)
    if payload is None:
        return None
    return {
        **payload,
        "binding_payload_json": evidence.binding_payload_json,
        "evidence_revision": evidence.evidence_revision,
    }


def candidate_occurrence_evidence_from_binding(
    entries: list[LyricEntry] | tuple[LyricEntry, ...],
    entry_index: int,
    *,
    producer: str,
    occurrence_binding: object,
    upstream_evidence_revision: object,
    lyric_identity_revision_value: str | None = None,
) -> CandidateOccurrenceEvidence | None:
    """Freeze one already validated runtime occurrence binding for a candidate."""
    if (
        not 0 <= entry_index < len(entries)
        or not isinstance(producer, str)
        or not producer.strip()
        or producer not in _CANDIDATE_OCCURRENCE_DIRECT_PRODUCERS
        or not isinstance(upstream_evidence_revision, str)
        or not upstream_evidence_revision.strip()
        or not isinstance(occurrence_binding, dict)
    ):
        return None
    if not _runtime_occurrence_binding_is_valid(
        list(entries), entry_index, occurrence_binding
    ):
        return None
    mode = occurrence_binding.get("mode")
    if mode not in _CANDIDATE_OCCURRENCE_MODES:
        return None
    context_indexes = _candidate_occurrence_context_indexes(
        occurrence_binding, entry_index
    )
    if (
        context_indexes is None
        or not context_indexes
        or entry_index not in context_indexes
        or tuple(sorted(set(context_indexes))) != context_indexes
    ):
        return None
    binding_json = _canonical_occurrence_binding_json(occurrence_binding)
    binding_revision = occurrence_binding.get("binding_revision")
    lyric_sha256 = occurrence_binding.get("lyric_sha256")
    if (
        binding_json is None
        or not isinstance(binding_revision, str)
        or not binding_revision.strip()
        or not isinstance(lyric_sha256, str)
        or not lyric_sha256.strip()
    ):
        return None
    target = normalize_match_text(entry_sung_text(entries[entry_index]))
    expected_lyric_sha256 = hashlib.sha256(target.encode("utf-8")).hexdigest()
    if lyric_sha256 != expected_lyric_sha256:
        return None
    lyric_revision = lyric_identity_revision_value or lyric_identity_revision(
        list(entries)
    )
    if not isinstance(lyric_revision, str) or not lyric_revision.strip():
        return None
    provisional = CandidateOccurrenceEvidence(
        entry_index=entry_index,
        producer=producer,
        evidence_revision="",
        binding_payload_json=binding_json,
        binding_revision=binding_revision,
        lyric_identity_revision=lyric_revision,
        lyric_sha256=lyric_sha256,
        occurrence_mode=str(mode),
        context_entry_indexes=context_indexes,
        unique=True,
        upstream_evidence_revision=upstream_evidence_revision,
    )
    payload = _candidate_occurrence_evidence_payload(provisional)
    if payload is None:
        return None
    return replace(
        provisional,
        evidence_revision=_timing_content_digest(payload),
    )


def candidate_occurrence_evidence_is_valid(
    candidate: TimingCandidate,
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Validate candidate occurrence evidence, including current lyric context."""
    evidence = candidate.occurrence_evidence
    if evidence is None:
        return False
    if (
        evidence.entry_index != candidate.entry_index
        or evidence.producer not in _CANDIDATE_OCCURRENCE_DIRECT_PRODUCERS
        or not evidence.evidence_revision
        or not evidence.binding_revision
        or not evidence.lyric_identity_revision
        or not evidence.lyric_sha256
        or evidence.occurrence_mode not in _CANDIDATE_OCCURRENCE_MODES
        or not evidence.context_entry_indexes
        or tuple(sorted(set(evidence.context_entry_indexes))) != evidence.context_entry_indexes
        or not evidence.unique
        or not evidence.upstream_evidence_revision
    ):
        return False
    direct = candidate.direct_onset_evidence
    if (
        _candidate_requires_occurrence_evidence(candidate)
        and (direct is None or direct.producer != evidence.producer)
    ):
        return False
    try:
        binding = json.loads(evidence.binding_payload_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(binding, dict):
        return False
    if _canonical_occurrence_binding_json(binding) != evidence.binding_payload_json:
        return False
    if binding.get("entry_index") != evidence.entry_index:
        return False
    mode = binding.get("mode")
    if mode != evidence.occurrence_mode:
        return False
    expected_context = _candidate_occurrence_context_indexes(
        binding, evidence.entry_index
    )
    if expected_context != evidence.context_entry_indexes:
        return False
    binding_payload = {
        key: value for key, value in binding.items() if key != "binding_revision"
    }
    if binding.get("binding_revision") != _timing_content_digest(binding_payload):
        return False
    payload = _candidate_occurrence_evidence_payload(evidence)
    if payload is None or evidence.evidence_revision != _timing_content_digest(payload):
        return False
    if entries is None:
        return True
    if (
        len(entries) == 0
        or not 0 <= candidate.entry_index < len(entries)
        or lyric_identity_revision(list(entries)) != evidence.lyric_identity_revision
    ):
        return False
    target = normalize_match_text(entry_sung_text(entries[candidate.entry_index]))
    if hashlib.sha256(target.encode("utf-8")).hexdigest() != evidence.lyric_sha256:
        return False
    return _runtime_occurrence_binding_is_valid(
        list(entries), candidate.entry_index, binding
    )


def _candidate_requires_occurrence_evidence(candidate: TimingCandidate) -> bool:
    direct = candidate.direct_onset_evidence
    return bool(
        direct is not None
        and direct.kind == "cross-backend-onset"
        and direct.producer in _CANDIDATE_OCCURRENCE_DIRECT_PRODUCERS
    )


def _tail_helper_revision(helper_sha256: str, helper_protocol_revision: str) -> str:
    """Digest helper identity without making its execution path semantic."""
    return _timing_content_digest({
        "helper_sha256": helper_sha256,
        "helper_protocol_revision": helper_protocol_revision,
    })


_TAIL_BACKEND_LINEAGE_FAMILIES = {
    "ctc": "ctc",
    "mms-ctc": "ctc",
    "whisper-family": "whisper-family",
    "whisperx": "whisper-family",
    "raw-asr": "whisper-family",
    "recurrence": "recurrence",
    "hubert-ja-known-lyric": "hubert-ja-known-lyric",
}


def _tail_backend_family(lineage: object) -> str | None:
    if not isinstance(lineage, str):
        return None
    return _TAIL_BACKEND_LINEAGE_FAMILIES.get(lineage.strip().lower())


def _tail_provenance_payload(provenance: TailEvidenceProvenance) -> dict[str, str]:
    return {
        "producer": provenance.producer,
        "backend_lineage": provenance.backend_lineage,
        "model_identity": provenance.model_identity,
        "helper_revision": provenance.helper_revision,
        "capability_revision": provenance.capability_revision,
        "audio_revision": provenance.audio_revision,
        "transcript_revision": provenance.transcript_revision,
        "row_revision": provenance.row_revision,
    }


def _tail_provenance_is_valid(
    evidence: IndependentTailEvidence,
    expected_context: dict[str, str] | None = None,
) -> bool:
    provenance = evidence.provenance
    if provenance is None:
        return False
    fields = _tail_provenance_payload(provenance)
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        return False
    if _tail_backend_family(provenance.backend_lineage) is None:
        return False
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (
            evidence.tail_start,
            evidence.tail_end,
            evidence.uncertainty_seconds,
            evidence.confidence,
        )
    ):
        return False
    if (
        evidence.entry_index < 0
        or evidence.tail_start > evidence.tail_end
        or evidence.uncertainty_seconds < 0.0
        or not 0.0 <= evidence.confidence <= 1.0
    ):
        return False
    if evidence.source_revision != provenance.observation_revision:
        return False
    expected_revision = _timing_content_digest({
        **fields,
        "entry_index": evidence.entry_index,
        "tail_start": round(evidence.tail_start, 6),
        "tail_end": round(evidence.tail_end, 6),
        "uncertainty_seconds": round(evidence.uncertainty_seconds, 6),
        "confidence": round(evidence.confidence, 8),
    })
    if provenance.observation_revision != expected_revision:
        return False
    if expected_context is not None:
        for key, expected in expected_context.items():
            if fields.get(key) != expected:
                return False
    return True


def _make_tail_provenance(
    *,
    entry_index: int,
    tail_start: float,
    tail_end: float,
    uncertainty_seconds: float,
    confidence: float,
    producer: str,
    backend_lineage: str,
    model_identity: str,
    helper_revision: str,
    capability_revision: str,
    audio_revision: str,
    transcript_revision: str,
    row_revision: str,
) -> TailEvidenceProvenance | None:
    payload = {
        "producer": producer,
        "backend_lineage": backend_lineage,
        "model_identity": model_identity,
        "helper_revision": helper_revision,
        "capability_revision": capability_revision,
        "audio_revision": audio_revision,
        "transcript_revision": transcript_revision,
        "row_revision": row_revision,
        "entry_index": entry_index,
        "tail_start": round(float(tail_start), 6),
        "tail_end": round(float(tail_end), 6),
        "uncertainty_seconds": round(float(uncertainty_seconds), 6),
        "confidence": round(float(confidence), 8),
    }
    numeric = (tail_start, tail_end, uncertainty_seconds, confidence)
    if (
        _tail_backend_family(backend_lineage) is None
        or any(
            not isinstance(value, str) or not value.strip()
            for key, value in payload.items()
            if key not in {"entry_index", "tail_start", "tail_end", "uncertainty_seconds", "confidence"}
        )
        or not all(math.isfinite(float(value)) for value in numeric)
        or tail_start > tail_end
        or uncertainty_seconds < 0.0
        or not 0.0 <= confidence <= 1.0
    ):
        return None
    return TailEvidenceProvenance(
        producer=producer,
        backend_lineage=backend_lineage,
        model_identity=model_identity,
        helper_revision=helper_revision,
        capability_revision=capability_revision,
        audio_revision=audio_revision,
        transcript_revision=transcript_revision,
        row_revision=row_revision,
        observation_revision=_timing_content_digest(payload),
    )


# Contexts are frozen and Central re-validates the same few objects thousands
# of times per song, so each object's verdict is remembered by identity (the
# entry keeps the object alive, so its id cannot be reused while cached).
_TAIL_CONTEXT_VALIDITY: dict[int, tuple[TailEvidenceContext, bool]] = {}


def _tail_context_is_valid(context: TailEvidenceContext | None) -> bool:
    if not isinstance(context, TailEvidenceContext):
        return False
    cached = _TAIL_CONTEXT_VALIDITY.get(id(context))
    if cached is not None and cached[0] is context:
        return cached[1]
    result = _tail_context_is_valid_uncached(context)
    if len(_TAIL_CONTEXT_VALIDITY) >= 4096:
        _TAIL_CONTEXT_VALIDITY.clear()
    _TAIL_CONTEXT_VALIDITY[id(context)] = (context, result)
    return result


def _tail_context_is_valid_uncached(context: TailEvidenceContext) -> bool:
    scalar_fields = (
        context.backend_lineage,
        context.model_identity,
        context.helper_revision,
        context.capability_revision,
        context.audio_revision,
    )
    scope_keys = tuple(
        (scope.entry_index, scope.producer)
        for scope in context.scopes
        if isinstance(scope, TailEvidenceScope)
    )
    return bool(
        _tail_backend_family(context.backend_lineage)
        and all(isinstance(value, str) and value.strip() for value in scalar_fields)
        and context.transcript_revisions
        and context.row_revisions
        and all(
            isinstance(value, str) and value.strip()
            for value in (*context.transcript_revisions, *context.row_revisions)
        )
        and tuple(sorted(set(context.transcript_revisions))) == context.transcript_revisions
        and tuple(sorted(set(context.row_revisions))) == context.row_revisions
        and all(
            isinstance(item, tuple)
            and len(item) == 5
            and all(isinstance(value, str) and value.strip() for value in item)
            for item in context.producer_contexts
        )
        and tuple(sorted(set(context.producer_contexts))) == context.producer_contexts
        and context.scopes
        and len(scope_keys) == len(context.scopes)
        and tuple(sorted(set(scope_keys))) == scope_keys
        and all(
            scope.entry_index >= 0
            and isinstance(scope.producer, str)
            and scope.producer.strip()
            and _tail_backend_family(scope.backend_lineage)
            and all(
                isinstance(value, str) and value.strip()
                for value in (
                    scope.backend_lineage,
                    scope.model_identity,
                    scope.helper_revision,
                    scope.capability_revision,
                    scope.audio_revision,
                )
            )
            and scope.transcript_revisions
            and scope.row_revisions
            and tuple(sorted(set(scope.transcript_revisions))) == scope.transcript_revisions
            and tuple(sorted(set(scope.row_revisions))) == scope.row_revisions
            and tuple(sorted(set(scope.capability_revisions))) == scope.capability_revisions
            and all(
                isinstance(value, str) and value.strip()
                for value in (
                    *scope.transcript_revisions,
                    *scope.row_revisions,
                    *scope.capability_revisions,
                )
            )
            for scope in context.scopes
        )
    )


def _tail_provenance_matches_context(
    evidence: IndependentTailEvidence,
    context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None,
) -> bool:
    if isinstance(context, tuple):
        return any(
            _tail_provenance_matches_context(evidence, item)
            for item in context
        )
    if not _tail_context_is_valid(context) or evidence.provenance is None:
        return False
    provenance = evidence.provenance
    scope = next(
        (
            item
            for item in context.scopes
            if item.entry_index == evidence.entry_index
            and item.producer == provenance.producer
        ),
        None,
    )
    if scope is None:
        return False
    execution_context = (
        provenance.backend_lineage,
        provenance.model_identity,
        provenance.helper_revision,
        provenance.audio_revision,
    )
    if (
        execution_context != (
            scope.backend_lineage,
            scope.model_identity,
            scope.helper_revision,
            scope.audio_revision,
        )
        or provenance.capability_revision not in (
            scope.capability_revision,
            *scope.capability_revisions,
        )
        or provenance.transcript_revision not in scope.transcript_revisions
        or provenance.row_revision not in scope.row_revisions
    ):
        return False
    return True


def _tail_contexts_are_valid(
    context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None,
) -> bool:
    """Validate one or more exact execution contexts without collapsing them."""

    if isinstance(context, tuple):
        return bool(context) and all(_tail_context_is_valid(item) for item in context)
    return _tail_context_is_valid(context)


def _whisperx_runtime_provenance_payload(
    provenance: WhisperXRuntimeProvenance,
) -> dict[str, str]:
    return {
        "backend_lineage": provenance.backend_lineage,
        "raw_executable_sha256": provenance.raw_executable_sha256,
        "raw_model_sha256": provenance.raw_model_sha256,
        "python_sha256": provenance.python_sha256,
        "helper_sha256": provenance.helper_sha256,
        "helper_protocol_revision": provenance.helper_protocol_revision,
        "whisperx_package_version": provenance.whisperx_package_version,
        "align_model_id": provenance.align_model_id,
        "align_model_revision": provenance.align_model_revision,
        "align_model_blob_sha256": provenance.align_model_blob_sha256,
        "source_audio_sha256": provenance.source_audio_sha256,
        "alignment_wav_sha256": provenance.alignment_wav_sha256,
        "device": provenance.device,
        "language": provenance.language,
        "raw_transcript_revision": provenance.raw_transcript_revision,
        "initial_aligned_output_revision": provenance.initial_aligned_output_revision,
        "forced_output_revision": provenance.forced_output_revision,
    }


def _whisperx_runtime_provenance_is_valid(
    provenance: WhisperXRuntimeProvenance | None,
) -> bool:
    if not isinstance(provenance, WhisperXRuntimeProvenance):
        return False
    fields = _whisperx_runtime_provenance_payload(provenance)
    if (
        provenance.backend_lineage != "whisper-family"
        or any(not isinstance(value, str) or not value.strip() for value in fields.values())
        or provenance.helper_protocol_revision != _WHISPERX_HELPER_PROTOCOL_REVISION
    ):
        return False
    return provenance.capability_revision == _timing_content_digest(fields)


def _tail_context_revision_values(
    report: dict[str, object],
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive current scoped transcript/row revisions without using evidence.

    The producer-specific revisions are deterministic projections of the
    current immutable lyric rows and CTC assignments.  Serialized retry
    diagnostics and sidecar observations are never used to establish context.
    """
    transcript_revisions: set[str] = set()
    row_revisions: set[str] = set()
    if isinstance(report.get("lyric_entry_revision"), str):
        transcript_revisions.add(str(report["lyric_entry_revision"]))
    if not entries:
        return (), ()

    normalized_texts: list[str] = []
    for index, entry in enumerate(entries):
        text = entry_sung_text(entry)
        normalized_texts.append(text)
        normalized = normalize_match_text(text)
        transcript_revisions.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
        transcript_revisions.add(_timing_content_digest(text))
        transcript_revisions.add(_timing_content_digest((text,)))
        assignments = report.get("assignments")
        if isinstance(assignments, list) and index < len(assignments):
            assignment = assignments[index]
            if isinstance(assignment, dict):
                spans = _timing_spans_from_value(assignment.get("ctc_token_spans"))
                if spans:
                    span_payload = [
                        [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
                        for span in spans
                    ]
                    row_revisions.add(_timing_content_digest(span_payload))

    # Joint/current-line producers bind one row or an adjacent pair.  Include
    # the exact request projections for every such current row pair.
    for index in range(len(normalized_texts)):
        one_texts = (normalized_texts[index],)
        transcript_revisions.add(_timing_content_digest(one_texts))
        row_revisions.add(_timing_content_digest(((index, 0),)))
        if index:
            pair_texts = (normalized_texts[index - 1], normalized_texts[index])
            transcript_revisions.add(_timing_content_digest(pair_texts))
            row_revisions.add(_timing_content_digest(((index - 1, 0), (index, 1))))

    return tuple(sorted(transcript_revisions)), tuple(sorted(row_revisions))


def _tail_context_scopes_from_report(
    report: dict[str, object],
    entries: list[LyricEntry] | tuple[LyricEntry, ...],
    capability: BoundedRetryCapability,
) -> tuple[TailEvidenceScope, ...]:
    """Build exact producer scopes from current rows/assignments only."""
    helper_revision = _tail_helper_revision(
        capability.helper_sha256,
        capability.helper_protocol_revision,
    )
    scopes: list[TailEvidenceScope] = []
    assignments = report.get("assignments")
    for index, entry in enumerate(entries):
        lyric_revision = hashlib.sha256(
            normalize_match_text(entry_sung_text(entry)).encode("utf-8")
        ).hexdigest()
        if not isinstance(assignments, list) or index >= len(assignments):
            continue
        assignment = assignments[index]
        if not isinstance(assignment, dict):
            continue
        spans = _timing_spans_from_value(assignment.get("ctc_token_spans"))
        if not spans:
            continue
        span_payload = [
            [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
            for span in spans
        ]
        scopes.append(TailEvidenceScope(
            entry_index=index,
            producer="ctc-assignment-terminal-row",
            backend_lineage=capability.backend_lineage,
            model_identity=capability.model_identity,
            helper_revision=helper_revision,
            capability_revision=capability.capability_revision,
            audio_revision=capability.alignment_audio_sha256,
            transcript_revisions=(lyric_revision,),
            row_revisions=(_timing_content_digest(span_payload),),
        ))

    texts = tuple(entry_sung_text(entry) for entry in entries)
    for index in range(1, len(texts)):
        pair_mapping = ((index - 1, 0), (index, 1))
        scopes.append(TailEvidenceScope(
            entry_index=index - 1,
            producer="joint-ctc-boundary-previous-tail",
            backend_lineage=capability.backend_lineage,
            model_identity=capability.model_identity,
            helper_revision=helper_revision,
            capability_revision=capability.capability_revision,
            audio_revision=capability.alignment_audio_sha256,
            transcript_revisions=(_timing_content_digest((texts[index - 1], texts[index])),),
            row_revisions=(_timing_content_digest(pair_mapping),),
        ))

    runtime_scopes = report.get(_RUNTIME_WHISPERX_TAIL_SCOPES_KEY)
    if isinstance(runtime_scopes, (tuple, list)):
        scopes.extend(
            item for item in runtime_scopes
            if isinstance(item, TailEvidenceScope)
        )
    unique = {
        (scope.entry_index, scope.producer): scope
        for scope in scopes
    }
    return tuple(
        unique[key]
        for key in sorted(unique)
    )


def _tail_evidence_context_from_report(
    report: dict[str, object],
    *,
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
    capability: BoundedRetryCapability | None = None,
) -> TailEvidenceContext | None:
    """Return only an in-process, fully specified active tail context."""
    explicit = report.get(_RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY)
    if isinstance(explicit, TailEvidenceContext):
        return explicit if _tail_context_is_valid(explicit) else None
    if capability is None:
        active = report.get(_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY)
        capability = active if isinstance(active, BoundedRetryCapability) else None
    if entries is None:
        runtime_entries = report.get(_RUNTIME_LYRIC_ENTRIES_KEY)
        if isinstance(runtime_entries, (list, tuple)):
            entries = tuple(item for item in runtime_entries if isinstance(item, LyricEntry))
    if capability is None or not entries:
        return None
    helper_revision = _tail_helper_revision(
        capability.helper_sha256,
        capability.helper_protocol_revision,
    )
    transcript_revisions, row_revisions = _tail_context_revision_values(report, entries)
    scopes = _tail_context_scopes_from_report(report, entries, capability)
    context = TailEvidenceContext(
        backend_lineage=capability.backend_lineage,
        model_identity=capability.model_identity,
        helper_revision=helper_revision,
        capability_revision=capability.capability_revision,
        audio_revision=capability.alignment_audio_sha256,
        transcript_revisions=transcript_revisions,
        row_revisions=row_revisions,
        scopes=scopes,
    )
    return context if _tail_context_is_valid(context) else None


_SERIALIZED_SECOND_SCALE = 1_000_000


def _serialized_microseconds(value: object) -> int:
    """Project a serialized second value onto the report's 6-decimal lattice."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("serialized time must be finite")
    return int(round(numeric * _SERIALIZED_SECOND_SCALE))


def _serialized_seconds_equal(left: object, right: object) -> bool:
    """Compare report/provenance times after the same 6-decimal quantization."""
    try:
        return _serialized_microseconds(left) == _serialized_microseconds(right)
    except (TypeError, ValueError, OverflowError):
        return False


def _serialized_delta_matches(
    end: object,
    start: object,
    delta: object,
    *,
    tolerance_microseconds: int = 1,
) -> bool:
    """Validate a derived delta without raw-float cancellation artifacts.

    Each operand is independently serialized to 6 decimals.  Subtracting two
    independently rounded endpoints can differ from an independently rounded
    stored delta by at most one microsecond.  This tolerance is serialization
    closure only; it is not an acoustic or alignment threshold.
    """
    if tolerance_microseconds < 0:
        return False
    try:
        end_us = _serialized_microseconds(end)
        start_us = _serialized_microseconds(start)
        delta_us = _serialized_microseconds(delta)
    except (TypeError, ValueError, OverflowError):
        return False
    return abs((end_us - start_us) - delta_us) <= tolerance_microseconds


def _finite_unit_confidence(value: object) -> float:
    """Clamp finite confidence to [0, 1]; malformed/non-finite evidence fails closed."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _candidate_confidence(item: dict[str, object]) -> float:
    for key in ("confidence", "score", "ctc_score"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return _finite_unit_confidence(value)
    return 0.0


def _entry_identity_is_unique(entries: list[LyricEntry], index: int) -> bool:
    normalized = normalize_match_text(entry_sung_text(entries[index]))
    if not normalized:
        return False
    occurrences = sum(
        1
        for entry in entries
        if normalize_match_text(entry_sung_text(entry)) == normalized
    )
    return occurrences == 1 and not has_ambiguous_lyric_prefix(entries, index)


def _minimal_unique_sequence_context(
    entries: list[LyricEntry],
    index: int,
    *,
    max_items: int = 9,
) -> tuple[int, int] | None:
    """Return the shortest immutable lyric window uniquely identifying one occurrence.

    Individual chorus lines often repeat. Requiring globally-unique single-line
    anchors can therefore force an unnecessarily large window or make a valid
    occurrence impossible to probe. A contiguous lyric sequence is sufficient
    identity evidence when that exact normalized sequence occurs once in the
    immutable lyric list. The target must be strictly inside the bracket.
    """
    if not 0 <= index < len(entries) or max_items < 3:
        return None
    lyrics = [
        normalize_match_text(entry_sung_text(entry))
        for entry in entries
    ]
    if not lyrics[index]:
        return None
    upper = min(max_items, len(entries))
    candidates: list[tuple[int, int, int, int, int]] = []
    for length in range(3, upper + 1):
        min_left = max(0, index - length + 1)
        max_left = min(index - 1, len(entries) - length)
        for left in range(min_left, max_left + 1):
            right = left + length - 1
            if not (left < index < right):
                continue
            sequence = tuple(lyrics[left : right + 1])
            if any(not item for item in sequence):
                continue
            occurrences = [
                start
                for start in range(0, len(lyrics) - length + 1)
                if tuple(lyrics[start : start + length]) == sequence
            ]
            if occurrences != [left]:
                continue
            left_distance = index - left
            right_distance = right - index
            candidates.append((length, max(left_distance, right_distance), abs(left_distance - right_distance), left, right))
        if candidates:
            break
    if not candidates:
        return None
    _, _, _, left, right = min(candidates)
    return left, right


def _sparse_unique_sequence_anchor_set(
    entries: list[LyricEntry],
    assignments: list[object],
    index: int,
    *,
    max_items: int = 9,
) -> tuple[int, int, tuple[int, ...]] | None:
    """Return the smallest trusted sparse fingerprint that uniquely binds a target.

    Only real trusted backend rows participate. Missing/weak context rows are
    ignored rather than interpolated. The selected target+anchor lyric offsets
    must identify exactly one target occurrence in the immutable lyric list and
    fit inside the existing bounded context size.
    """
    if not 0 <= index < len(entries) or len(assignments) != len(entries) or max_items < 3:
        return None
    if _identity_assignment_observation(assignments[index]) is None:
        return None
    search_left = max(0, index - max_items + 1)
    search_right = min(len(entries) - 1, index + max_items - 1)
    observed = [
        item
        for item in range(search_left, search_right + 1)
        if _identity_assignment_observation(assignments[item]) is not None
    ]
    left_observed = [item for item in observed if item < index]
    right_observed = [item for item in observed if item > index]
    # A strong sparse bracket must independently bound the target occurrence
    # from both sides.  One-sided fingerprints may be unique as lyric strings,
    # but they cannot prove a timing bracket around the selected occurrence.
    if not left_observed or not right_observed:
        return None

    lyrics = [normalize_match_text(entry_sung_text(entry)) for entry in entries]
    target = lyrics[index]
    if not target:
        return None
    target_occurrences = [item for item, text in enumerate(lyrics) if text == target]
    anchors = [item for item in observed if item != index]
    ranked: list[tuple[int, int, int, tuple[int, ...]]] = []
    for count in range(2, len(anchors) + 1):
        for subset in itertools.combinations(anchors, count):
            selected = tuple(sorted((index, *subset)))
            if selected[-1] - selected[0] + 1 > max_items:
                continue
            observations = [_identity_assignment_observation(assignments[item]) for item in selected]
            if any(item is None for item in observations):
                continue
            typed = [item for item in observations if item is not None]
            timestamps = [item[0] for item in typed]
            segments = [item[1] for item in typed]
            if (
                any(left >= right for left, right in zip(timestamps, timestamps[1:]))
                or any(left > right for left, right in zip(segments, segments[1:]))
            ):
                continue
            offsets = tuple(item - index for item in selected)
            expected = tuple(lyrics[item] for item in selected)
            matches: list[int] = []
            for occurrence in target_occurrences:
                candidate_indexes = [occurrence + offset for offset in offsets]
                if any(item < 0 or item >= len(lyrics) for item in candidate_indexes):
                    continue
                if tuple(lyrics[item] for item in candidate_indexes) == expected:
                    matches.append(occurrence)
            if matches != [index]:
                continue
            span = selected[-1] - selected[0]
            distance = sum(abs(item - index) for item in selected if item != index)
            two_sided = any(item < index for item in selected) and any(item > index for item in selected)
            if not two_sided:
                continue
            ranked.append((count, span, distance, selected))
        if ranked:
            break
    if not ranked:
        return None
    _, _, _, selected = min(ranked)
    return selected[0], selected[-1], selected


def _sparse_sequence_fingerprint_is_unique(
    entries: list[LyricEntry],
    index: int,
    anchor_indexes: list[int],
) -> bool:
    if index not in anchor_indexes or len(set(anchor_indexes) - {index}) < 2:
        return False
    lyrics = [normalize_match_text(entry_sung_text(entry)) for entry in entries]
    if not 0 <= index < len(lyrics) or not lyrics[index]:
        return False
    ordered = sorted(anchor_indexes)
    offsets = tuple(item - index for item in ordered)
    expected = tuple(lyrics[item] for item in ordered)
    matches: list[int] = []
    for occurrence, text in enumerate(lyrics):
        if text != lyrics[index]:
            continue
        candidate_indexes = [occurrence + offset for offset in offsets]
        if any(item < 0 or item >= len(lyrics) for item in candidate_indexes):
            continue
        if tuple(lyrics[item] for item in candidate_indexes) == expected:
            matches.append(occurrence)
    return matches == [index]


def _occurrence_binding_has_strong_context(binding: object) -> bool:
    if not isinstance(binding, dict):
        return False
    mode = binding.get("mode")
    if mode == "sequence-bracket":
        return True
    if mode != "sparse-sequence-bracket":
        return False
    entry_index = binding.get("entry_index")
    indexes = binding.get("anchor_entry_indexes")
    if type(entry_index) is not int or not isinstance(indexes, list):
        return False
    try:
        typed = [int(item) for item in indexes]
    except (TypeError, ValueError):
        return False
    return (
        entry_index in typed
        and any(item < entry_index for item in typed)
        and any(item > entry_index for item in typed)
    )


def _identity_assignment_observation(
    assignment: object,
) -> tuple[float, int, float] | None:
    """Return a real backend row suitable for occurrence binding."""
    if not isinstance(assignment, dict):
        return None
    try:
        timestamp = float(assignment.get("timestamp"))
        score = float(assignment.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    segment = assignment.get("segment")
    if (
        not math.isfinite(timestamp)
        or score < TRUSTED_ALIGNMENT_SCORE
        or bool(assignment.get("borrowed"))
        or not isinstance(segment, int)
    ):
        return None
    return timestamp, segment, score


def _entry_occurrence_binding(
    entries: list[LyricEntry],
    assignments: list[object],
    index: int,
    *,
    require_context: bool = False,
) -> dict[str, object] | None:
    """Prove which immutable lyric occurrence an independent backend identified.

    Prefer a fully trusted contiguous sequence bracket.  If one or more unrelated
    context rows are weak, fall back to the smallest trusted sparse fingerprint
    that still uniquely identifies the target occurrence.  No timestamp or lyric
    is inferred for missing rows.
    """
    if not (0 <= index < len(entries)) or len(assignments) != len(entries):
        return None
    current = _identity_assignment_observation(assignments[index])
    if current is None:
        return None
    current_time, current_segment, current_score = current
    target = normalize_match_text(entry_sung_text(entries[index]))
    if not target:
        return None

    if _entry_identity_is_unique(entries, index) and not require_context:
        payload = {
            "mode": "global-unique",
            "entry_index": index,
            "lyric_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "segment": current_segment,
            "timestamp": round(current_time, 6),
            "score": round(current_score, 6),
        }
        return {**payload, "binding_revision": _timing_content_digest(payload)}

    context = _minimal_unique_sequence_context(entries, index, max_items=9)
    if context is None:
        return None
    left_index, right_index = context
    context_indexes = list(range(left_index, right_index + 1))
    observations = [
        _identity_assignment_observation(assignments[context_index])
        for context_index in context_indexes
    ]
    if all(item is not None for item in observations):
        typed = [item for item in observations if item is not None]
        timestamps = [item[0] for item in typed]
        segments = [item[1] for item in typed]
        scores = [item[2] for item in typed]
        if (
            segments.count(current_segment) == 1
            and not any(left >= right for left, right in zip(timestamps, timestamps[1:]))
            and not any(left > right for left, right in zip(segments, segments[1:]))
        ):
            relative_index = index - left_index
            current_time, current_segment, current_score = typed[relative_index]
            context_payload = {
                "left_index": left_index,
                "target_index": index,
                "right_index": right_index,
                "lyrics": [
                    normalize_match_text(entry_sung_text(entry))
                    for entry in entries[left_index : right_index + 1]
                ],
            }
            payload = {
                "mode": "sequence-bracket",
                "entry_index": index,
                "lyric_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                "segment": current_segment,
                "timestamp": round(current_time, 6),
                "score": round(current_score, 6),
                "left_entry_index": left_index,
                "left_segment": segments[0],
                "left_timestamp": round(timestamps[0], 6),
                "left_score": round(scores[0], 6),
                "right_entry_index": right_index,
                "right_segment": segments[-1],
                "right_timestamp": round(timestamps[-1], 6),
                "right_score": round(scores[-1], 6),
                "context_entry_indexes": context_indexes,
                "context_segments": segments,
                "context_timestamps": [round(value, 6) for value in timestamps],
                "context_scores": [round(value, 6) for value in scores],
                "context_revision": _timing_content_digest(context_payload),
            }
            return {**payload, "binding_revision": _timing_content_digest(payload)}

    sparse = _sparse_unique_sequence_anchor_set(entries, assignments, index, max_items=9)
    if sparse is None:
        return None
    sparse_left, sparse_right, anchor_indexes = sparse
    sparse_observations = [
        _identity_assignment_observation(assignments[item])
        for item in anchor_indexes
    ]
    if any(item is None for item in sparse_observations):
        return None
    typed_sparse = [item for item in sparse_observations if item is not None]
    anchor_timestamps = [item[0] for item in typed_sparse]
    anchor_segments = [item[1] for item in typed_sparse]
    anchor_scores = [item[2] for item in typed_sparse]
    target_position = anchor_indexes.index(index)
    current_time, current_segment, current_score = typed_sparse[target_position]
    if anchor_segments.count(current_segment) > 1:
        return None
    fingerprint_payload = {
        "target_index": index,
        "context_left_index": sparse_left,
        "context_right_index": sparse_right,
        "anchor_entry_indexes": list(anchor_indexes),
        "anchor_lyrics": [
            normalize_match_text(entry_sung_text(entries[item]))
            for item in anchor_indexes
        ],
    }
    payload = {
        "mode": "sparse-sequence-bracket",
        "entry_index": index,
        "lyric_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        "segment": current_segment,
        "timestamp": round(current_time, 6),
        "score": round(current_score, 6),
        "context_left_index": sparse_left,
        "context_right_index": sparse_right,
        "anchor_entry_indexes": list(anchor_indexes),
        "anchor_segments": anchor_segments,
        "anchor_timestamps": [round(value, 6) for value in anchor_timestamps],
        "anchor_scores": [round(value, 6) for value in anchor_scores],
        "fingerprint_revision": _timing_content_digest(fingerprint_payload),
    }
    return {**payload, "binding_revision": _timing_content_digest(payload)}


def _runtime_occurrence_binding_is_valid(
    entries: list[LyricEntry],
    index: int,
    binding: object,
) -> bool:
    """Revalidate immutable lyric context and monotonic runtime binding data."""
    if not isinstance(binding, dict) or not (0 <= index < len(entries)):
        return False
    if not _candidate_occurrence_binding_keys_are_canonical(binding):
        return False
    target = normalize_match_text(entry_sung_text(entries[index]))
    if not target or binding.get("entry_index") != index:
        return False
    if binding.get("lyric_sha256") != hashlib.sha256(target.encode("utf-8")).hexdigest():
        return False
    payload = {key: value for key, value in binding.items() if key != "binding_revision"}
    if binding.get("binding_revision") != _timing_content_digest(payload):
        return False
    mode = binding.get("mode")
    if mode == "global-unique":
        return _entry_identity_is_unique(entries, index)

    if mode == "sequence-bracket":
        try:
            left_index = int(binding.get("left_entry_index"))
            right_index = int(binding.get("right_entry_index"))
            current_segment = int(binding.get("segment"))
            current_time = float(binding.get("timestamp"))
        except (TypeError, ValueError):
            return False
        expected_context = _minimal_unique_sequence_context(entries, index, max_items=9)
        if expected_context != (left_index, right_index):
            return False
        context_indexes = binding.get("context_entry_indexes")
        context_segments = binding.get("context_segments")
        context_timestamps = binding.get("context_timestamps")
        context_scores = binding.get("context_scores")
        expected_indexes = list(range(left_index, right_index + 1))
        if (
            context_indexes != expected_indexes
            or not isinstance(context_segments, list)
            or not isinstance(context_timestamps, list)
            or not isinstance(context_scores, list)
            or not (len(context_segments) == len(context_timestamps) == len(context_scores) == len(expected_indexes))
        ):
            return False
        try:
            segments = [int(value) for value in context_segments]
            timestamps = [float(value) for value in context_timestamps]
            scores = [float(value) for value in context_scores]
        except (TypeError, ValueError):
            return False
        if (
            any(not math.isfinite(value) for value in timestamps + scores)
            or any(score < TRUSTED_ALIGNMENT_SCORE for score in scores)
            or any(left > right for left, right in zip(segments, segments[1:]))
            or any(left >= right for left, right in zip(timestamps, timestamps[1:]))
        ):
            return False
        relative_index = index - left_index
        if segments.count(current_segment) > 1:
            return False
        if (
            segments[relative_index] != current_segment
            or abs(timestamps[relative_index] - current_time) > 1e-6
            or int(binding.get("left_segment")) != segments[0]
            or int(binding.get("right_segment")) != segments[-1]
            or abs(float(binding.get("left_timestamp")) - timestamps[0]) > 1e-6
            or abs(float(binding.get("right_timestamp")) - timestamps[-1]) > 1e-6
        ):
            return False
        context_payload = {
            "left_index": left_index,
            "target_index": index,
            "right_index": right_index,
            "lyrics": [
                normalize_match_text(entry_sung_text(entry))
                for entry in entries[left_index : right_index + 1]
            ],
        }
        return binding.get("context_revision") == _timing_content_digest(context_payload)

    if mode != "sparse-sequence-bracket":
        return False
    try:
        context_left = int(binding.get("context_left_index"))
        context_right = int(binding.get("context_right_index"))
        current_segment = int(binding.get("segment"))
        current_time = float(binding.get("timestamp"))
    except (TypeError, ValueError):
        return False
    anchor_indexes = binding.get("anchor_entry_indexes")
    anchor_segments = binding.get("anchor_segments")
    anchor_timestamps = binding.get("anchor_timestamps")
    anchor_scores = binding.get("anchor_scores")
    if (
        not isinstance(anchor_indexes, list)
        or not isinstance(anchor_segments, list)
        or not isinstance(anchor_timestamps, list)
        or not isinstance(anchor_scores, list)
        or not (len(anchor_indexes) == len(anchor_segments) == len(anchor_timestamps) == len(anchor_scores))
        or len(anchor_indexes) < 3
    ):
        return False
    try:
        indexes = [int(value) for value in anchor_indexes]
        segments = [int(value) for value in anchor_segments]
        timestamps = [float(value) for value in anchor_timestamps]
        scores = [float(value) for value in anchor_scores]
    except (TypeError, ValueError):
        return False
    if (
        indexes != sorted(indexes)
        or len(set(indexes)) != len(indexes)
        or index not in indexes
        or context_left != indexes[0]
        or context_right != indexes[-1]
        or context_right - context_left + 1 > 9
        or any(item < context_left or item > context_right for item in indexes)
        or len(set(indexes) - {index}) < 2
        or not any(item < index for item in indexes)
        or not any(item > index for item in indexes)
        or any(not math.isfinite(value) for value in timestamps + scores)
        or any(score < TRUSTED_ALIGNMENT_SCORE for score in scores)
        or any(left > right for left, right in zip(segments, segments[1:]))
        or any(left >= right for left, right in zip(timestamps, timestamps[1:]))
        or not _sparse_sequence_fingerprint_is_unique(entries, index, indexes)
    ):
        return False
    target_position = indexes.index(index)
    if segments.count(current_segment) > 1:
        return False
    if segments[target_position] != current_segment or abs(timestamps[target_position] - current_time) > 1e-6:
        return False
    fingerprint_payload = {
        "target_index": index,
        "context_left_index": context_left,
        "context_right_index": context_right,
        "anchor_entry_indexes": indexes,
        "anchor_lyrics": [
            normalize_match_text(entry_sung_text(entries[item]))
            for item in indexes
        ],
    }
    return binding.get("fingerprint_revision") == _timing_content_digest(fingerprint_payload)


def runtime_independent_tail_evidence(
    report: dict[str, object],
    count: int,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
) -> RuntimeTailEvidence:
    if tail_evidence_context is None:
        tail_evidence_context = _tail_evidence_context_from_report(report)
    runtime = _runtime_tail_observation_map(report)
    rows: list[TailObservationSet | None] = []
    for index in range(count):
        observations = tuple(
            item for item in runtime.get(index, [])
            if _runtime_tail_observation_is_current(
                report, item, tail_evidence_context
            )
        )
        rows.append(observations or None)
    return tuple(rows)


def _runtime_tail_observation_map(
    report: dict[str, object],
) -> dict[int, list[IndependentTailEvidence]]:
    runtime = report.get(_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY)
    if not isinstance(runtime, dict):
        return {}
    normalized: dict[int, list[IndependentTailEvidence]] = {}
    for raw_index, value in runtime.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if isinstance(value, IndependentTailEvidence):
            values = (value,)
        elif isinstance(value, (tuple, list)):
            values = tuple(item for item in value if isinstance(item, IndependentTailEvidence))
        else:
            values = ()
        if values:
            normalized.setdefault(index, []).extend(values)
    return normalized


def _runtime_tail_observation_is_current(
    report: dict[str, object],
    evidence: IndependentTailEvidence,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
) -> bool:
    if not _tail_provenance_is_valid(evidence):
        return False
    # Without a central context the observation may be inspected as a
    # diagnostic, but it must not be mistaken for authority.  The resolver
    # independently fails closed when this context is absent.
    if tail_evidence_context is None:
        return True
    if not _tail_provenance_matches_context(evidence, tail_evidence_context):
        return False
    provenance = evidence.provenance
    assert provenance is not None
    # A peer bundle validates each observation against its originating context.
    # There is no single report capability that can safely stand in for those
    # multiple executions, so the exact context match is the authority check.
    if isinstance(tail_evidence_context, tuple):
        return True
    if _tail_backend_family(provenance.backend_lineage) != "ctc":
        return True
    capability = report.get(_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY)
    if not isinstance(capability, BoundedRetryCapability):
        return False
    expected_context = {
        "backend_lineage": capability.backend_lineage,
        "model_identity": capability.model_identity,
        "helper_revision": _tail_helper_revision(
            capability.helper_sha256,
            capability.helper_protocol_revision,
        ),
        "capability_revision": capability.capability_revision,
        "audio_revision": capability.alignment_audio_sha256,
    }
    return _tail_provenance_is_valid(evidence, expected_context)
