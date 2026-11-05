"""Extracted from auto_lrc.py (split step 4): timing dataclasses, candidate/tail evidence, ownership, timing decision and final timing state."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import statistics
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Sequence
import arbiter as timing_arbiter
from multimodel_consensus import model_identity
from lyrics_io import LyricEntry, _RUNTIME_LYRIC_ENTRIES_KEY, entry_sung_text, has_ambiguous_lyric_prefix, lyric_identity_revision, normalize_match_text
from audio_features import HOP_SECONDS
from runtime_base import CanonicalWrittenTime, LrcError, _timing_content_digest, canonicalize_lrc_time, progress_log
from runtime_base import TimingTokenSpan, _timing_spans_from_value


CTC_SEVERE_INTERNAL_GAP_SECONDS = 3.0


PREFIX_ACOUSTIC_ASSOCIATION_SECONDS = 0.22


TRUSTED_ALIGNMENT_SCORE = 0.62


_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY = "_runtime_ctc_bounded_retry_capability"


_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY = "_runtime_independent_tail_evidence"


_RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY = "_runtime_tail_evidence_context"


_RUNTIME_WHISPERX_TAIL_SCOPES_KEY = "_runtime_whisperx_tail_scopes"


RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS = 0.18


_WHISPERX_HELPER_PROTOCOL_REVISION = "whisperx-refine-runtime-provenance-v1"


EvidenceState = Literal["supported", "contradicted", "unavailable"]


OwnershipState = Literal["clear", "violation", "unknown"]


TailValidity = Literal["valid", "invalid", "unknown"]


@dataclass(frozen=True)
class CandidateAcousticEvidence:
    onset_time: float
    strength: float
    distance_seconds: float
    source_revision: str
    producer: str
    kind: str
    independent: bool


@dataclass(frozen=True)
class CandidateOnsetEvidence:
    onset_time: float
    distance_seconds: float
    source_revision: str
    producer: str
    kind: str
    independent: bool


@dataclass(frozen=True)
class CandidateOccurrenceEvidence:
    """Immutable occurrence identity carried by independent direct evidence.

    The binding is retained as canonical JSON rather than as a mutable mapping.
    It contains only normalized lyric/context facts and revision digests; paths,
    transcripts, and execution objects remain outside the candidate contract.
    """

    entry_index: int
    producer: str
    evidence_revision: str
    binding_payload_json: str
    binding_revision: str
    lyric_identity_revision: str
    lyric_sha256: str
    occurrence_mode: str
    context_entry_indexes: tuple[int, ...]
    unique: bool
    upstream_evidence_revision: str


@dataclass(frozen=True)
class TimingCandidate:
    candidate_id: str
    entry_index: int
    source: str
    raw_time: float
    written_time: CanonicalWrittenTime
    generation_revision: str
    spans: tuple[TimingTokenSpan, ...]
    confidence: float
    identity_support: EvidenceState
    sequence_support: EvidenceState
    acoustic_support: EvidenceState
    direct_onset_support: EvidenceState
    acoustic_strength: float | None = None
    acoustic_evidence: CandidateAcousticEvidence | None = None
    direct_onset_evidence: CandidateOnsetEvidence | None = None
    occurrence_evidence: CandidateOccurrenceEvidence | None = None
    independent_content_identity: bool = False
    current: bool = False
    prefix_scope: bool = False
    right_edge_pileup: bool = False
    window_truncated: bool = False


@dataclass(frozen=True)
class CandidateRegionEvidence:
    candidate_id: str
    generation_revision: str
    computed_for_centiseconds: int
    region_revision: str
    parsed_gaps: tuple[float, ...]
    prefix_duration: float | None
    max_internal_gap: float | None
    median_internal_gap: float | None
    p90_internal_gap: float | None
    p95_internal_gap: float | None
    token_density: float | None
    longest_supported_run: int
    opening_supported_run: int
    later_supported_run: int
    reliable_support_count: int
    onset_proof_end: float | None
    severe_gap_indexes: tuple[int, ...]
    temporal_coherence: Literal["coherent", "ambiguous", "incoherent", "unavailable"]
    fracture_region: Literal["opening", "prefix", "middle", "tail", "unknown", "none"]


@dataclass(frozen=True)
class TailRegionEvidence:
    previous_candidate_id: str
    previous_generation_revision: str
    previous_region_revision: str
    tail_evidence_revision: str
    validity: TailValidity
    reliable_tail_start: float | None
    reliable_tail_end: float | None
    uncertainty_seconds: float | None
    confidence: float | None
    reason: str
    terminal_span_indexes: tuple[int, ...] = ()
    local_gap_references: tuple[float, ...] = ()
    # Runtime external-tail authority metadata.  Intrinsic predecessor tails
    # leave these fields unset; an external observation must carry all three
    # fields before it can participate in candidate ownership.
    authority_backend_lineage: str | None = None
    authority_producer: str | None = None
    authority_observation_revision: str | None = None


@dataclass(frozen=True)
class TailEvidenceProvenance:
    """Immutable provenance required before a tail observation has authority."""

    producer: str
    backend_lineage: str
    model_identity: str
    helper_revision: str
    capability_revision: str
    audio_revision: str
    transcript_revision: str
    row_revision: str
    observation_revision: str


@dataclass(frozen=True)
class TailEvidenceScope:
    """Exact authority scope for one entry and one tail producer."""

    entry_index: int
    producer: str
    backend_lineage: str
    model_identity: str
    helper_revision: str
    capability_revision: str
    audio_revision: str
    transcript_revisions: tuple[str, ...]
    row_revisions: tuple[str, ...]
    capability_revisions: tuple[str, ...] = ()


@dataclass(frozen=True)
class TailEvidenceContext:
    """Immutable execution context required for tail authority.

    The scalar revisions identify the active backend execution.  The scoped
    transcript/row revision sets are derived from the immutable current lyric
    rows and assignments (or supplied by an in-process caller that owns that
    execution).  They prevent a digest-valid observation from being reused
    after its lyric or row context changed.
    """

    backend_lineage: str
    model_identity: str
    helper_revision: str
    capability_revision: str
    audio_revision: str
    transcript_revisions: tuple[str, ...]
    row_revisions: tuple[str, ...]
    producer_contexts: tuple[tuple[str, str, str, str, str], ...] = ()
    scopes: tuple[TailEvidenceScope, ...] = ()


@dataclass(frozen=True)
class WhisperXRuntimeProvenance:
    """Runtime-only provenance for one WhisperX candidate execution."""

    backend_lineage: str
    raw_executable_sha256: str
    raw_model_sha256: str
    python_sha256: str
    helper_sha256: str
    helper_protocol_revision: str
    whisperx_package_version: str
    align_model_id: str
    align_model_revision: str
    align_model_blob_sha256: str
    source_audio_sha256: str
    alignment_wav_sha256: str
    device: str
    language: str
    raw_transcript_revision: str
    initial_aligned_output_revision: str
    forced_output_revision: str
    capability_revision: str


@dataclass(frozen=True)
class IndependentTailEvidence:
    """Runtime predecessor-tail observation with fail-closed provenance."""

    entry_index: int
    tail_start: float
    tail_end: float
    uncertainty_seconds: float
    confidence: float
    producer: str
    source_revision: str
    provenance: TailEvidenceProvenance | None = None


@dataclass(frozen=True)
class IndependentCurrentOnsetEvidence:
    """Runtime-only occurrence-bound HuBERT current-opening observation.

    The observation itself is never a TimingCandidate.  A final-associated
    subset may falsify an already-existing post-Central distant hypothesis.
    V14 may also use a validated non-final-associated observation as independent
    onset evidence for a private Central replay that enriches an already-existing
    CTC/recurrence candidate at its original timestamp; only the ordinary Central
    selector may commit that recovered timestamp.
    """

    entry_index: int
    onset_time: float
    opening_proof_end: float
    opening_region_revision: str
    identity_owner_candidate_id: str
    identity_owner_generation_revision: str
    producer: str
    backend_lineage: str
    model_identity: str
    helper_revision: str
    capability_revision: str
    audio_revision: str
    transcript_revision: str
    row_revision: str
    request_revision: str
    hypothesis_set_revision: str
    occurrence_binding_revision: str
    observation_revision: str


TailObservationSet = tuple[IndependentTailEvidence, ...]


RuntimeTailEvidence = tuple[TailObservationSet | None, ...]


@dataclass(frozen=True)
class OwnershipRelationEvidence:
    status: OwnershipState
    relation_revision: str
    current_candidate_id: str
    current_generation_revision: str
    current_region_revision: str
    current_centiseconds: int
    previous_candidate_id: str | None
    previous_generation_revision: str | None
    previous_region_revision: str | None
    previous_tail_evidence_revision: str | None
    previous_tail_validity: TailValidity | None
    reliable_tail_start: float | None
    reliable_tail_end: float | None
    uncertainty_seconds: float | None
    confidence: float | None
    distance_seconds: float | None
    reason: str
    previous_tail_authority_backend_lineage: str | None = None
    previous_tail_authority_producer: str | None = None
    previous_tail_authority_observation_revision: str | None = None


@dataclass(frozen=True)
class NormalizedTimingEvidence:
    candidate_id: str
    candidate_revision: str
    computed_for_centiseconds: int
    region_revision: str
    ownership_relation_revision: str
    referenced_tail_evidence_revision: str | None
    prefix_duration: float | None
    max_internal_gap: float | None
    median_internal_gap: float | None
    p90_internal_gap: float | None
    p95_internal_gap: float | None
    token_density: float | None
    longest_supported_run: int
    opening_supported_run: int
    later_supported_run: int
    reliable_support_count: int
    onset_proof_end: float | None
    island_count: int
    severe_gap_count: int
    first_severe_gap_index: int | None
    temporal_coherence: Literal["coherent", "ambiguous", "incoherent", "unavailable"]
    fracture_region: Literal["opening", "prefix", "middle", "tail", "unknown", "none"]
    ownership: OwnershipRelationEvidence
    tail_distance_seconds: float | None
    ownership_independent: bool
    structural_failures: tuple[str, ...]
    minimum_proof_satisfied: bool


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: TimingCandidate
    evidence: NormalizedTimingEvidence
    valid: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryProvenance:
    candidate_id: str
    candidate_revision: str
    evidence_revision: int
    canonical_written_centiseconds: int
    old_selection_revision: int
    new_selection_revision: int
    resolved_finding_ids: tuple[str, ...]
    passed_invariants: tuple[str, ...]
    resolution_mode: Literal["challenger-recovery"]
    evaluator_version: str


@dataclass(frozen=True)
class TimingDecision:
    entry_index: int
    selection_revision: int
    status: Literal["selected_valid", "provisional_unresolved"]
    selected_candidate_id: str | None
    provisional_candidate_id: str | None
    written_time: CanonicalWrittenTime
    timestamp_revision: int
    timing_evidence_revision: int
    evidence: NormalizedTimingEvidence
    candidate: TimingCandidate
    rejected_candidates: tuple[CandidateEvaluation, ...]
    recovery: RecoveryProvenance | None
    arbiter: timing_arbiter.GrossRescueDecision | None = None
    replaced_candidate: TimingCandidate | None = None


@dataclass(frozen=True)
class TimingInvariantAudit:
    entry_index: int
    timing_trusted: bool
    status: Literal["selected_valid", "provisional_unresolved"]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class FinalTimingState:
    decisions: tuple[TimingDecision, ...]
    written_times: tuple[CanonicalWrittenTime, ...]
    audits: tuple[TimingInvariantAudit, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.decisions) == len(self.written_times) == len(self.audits)
        ):
            raise LrcError("Final timing state cardinality mismatch")


@dataclass(frozen=True)
class TimingFinding:
    finding_id: str
    code: str
    raised_revision: int
    evidence_scope: str


@dataclass(frozen=True)
class EntryTimingFindings:
    historical: tuple[TimingFinding, ...]
    resolved: tuple[TimingFinding, ...]
    unresolved: tuple[TimingFinding, ...]


@dataclass(frozen=True)
class CentralTrustSummary:
    timing_entries: int
    assigned_entries: int
    content_trusted_entries: int
    timing_trusted_entries: int
    overall_trusted_entries: int
    unverified_count: int
    review_required_count: int
    assigned_percent: float
    content_trusted_percent: float
    timing_trusted_percent: float
    overall_trusted_percent: float
    unverified_percent: float
    review_required_percent: float
    review_required: bool


@dataclass(frozen=True)
class EvaluatedBackendTiming:
    backend: str
    input_revision: str
    report_projection_revision: str
    generation_diagnostics_revision: str
    generation_diagnostics_semantic_revision: str
    generation_diagnostics_json: str
    evaluation_revision: str
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...]
    state: FinalTimingState
    finding_states: tuple[EntryTimingFindings, ...]
    summary: CentralTrustSummary
    quality: float
    arbiter_mode: Literal["off", "gross-rescue", "reviewer-validity"] = "off"
    timing_source_policy: Literal["default", "ctc-first"] = "default"


@dataclass(frozen=True)
class PeerGraphPolicy:
    """Selection policy for a runtime-only cross-backend peer graph.

    The policy is deliberately separate from ``TimingCandidate.current``.
    ``current`` is part of candidate identity and revision provenance; peer
    mode only changes which selection branch is allowed to use that metadata.
    """

    mode: Literal["backend_local", "cross_backend_neutral"]
    revision: str


@dataclass(frozen=True)
class BackendContentIdentityObservation:
    """Immutable runtime content observation retained for peer simulation."""

    entry_index: int
    producer: str
    backend_lineage: str
    lyric_revision: str
    occurrence_revision: str
    evidence_revision: str


@dataclass(frozen=True)
class TailObservationBinding:
    """One tail observation bound to its originating execution context."""

    observation: IndependentTailEvidence
    context: TailEvidenceContext | None
    context_revision: str


@dataclass(frozen=True)
class EvaluatedBackendBundle:
    """Runtime-only immutable preview bundle.

    This object is never serialized as authority.  Reports retain only the
    public, non-authoritative projection emitted by the peer simulation.
    """

    evaluated: EvaluatedBackendTiming
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...]
    candidate_sets_revision: str
    report_projection_revision: str
    central_scaffold_json: str
    content_identity_observations: tuple[BackendContentIdentityObservation, ...]
    tail_bindings: tuple[TailObservationBinding, ...]
    capability_projection_json: str
    source_audio_revision: str | None
    alignment_audio_revision: str | None
    algorithm_source_revision: str
    algorithm_revision: str
    input_revision: str
    diagnostics_revision: str
    diagnostics_semantic_revision: str
    evaluation_revision: str
    bundle_revision: str


@dataclass(frozen=True)
class ObservationClusterSummary:
    """Non-authoritative, immutable summary of one bounded timing cluster.

    Cluster membership is diagnostic only.  The existing Central candidate
    evaluator remains the authority for selection and review; this summary
    prevents same-family wrappers from being counted as independent votes in
    the peer projection while preserving every distant hypothesis.
    """

    cluster_revision: str
    anchor_time: float
    member_candidate_ids: tuple[str, ...]
    member_generation_revisions: tuple[str, ...]
    support_families: tuple[str, ...]
    representative_candidate_id: str | None
    relation: Literal["baseline", "distant"]
    evidence_state: Literal["independent", "same-lineage", "mixed", "unavailable"]
    explicit_falsification: bool
    review_required: bool


@dataclass(frozen=True)
class BackendSelection:
    selected: EvaluatedBackendTiming
    candidates: tuple[EvaluatedBackendTiming, ...]
    selection_revision: str


@dataclass(frozen=True)
class BoundedRetryCapability:
    kind: Literal["mms-known-lyric-ctc"]
    backend_lineage: str
    helper_path: str
    helper_sha256: str
    helper_protocol_revision: str
    alignment_audio_path: str
    alignment_audio_sha256: str
    alignment_audio_source: str
    sample_rate: int
    model_identity: str
    device: str
    capability_revision: str


@dataclass(frozen=True)
class RetryCandidateReference:
    candidate_id: str
    generation_revision: str
    region_revision: str
    canonical_centiseconds: int


@dataclass(frozen=True)
class BaseRetryEpoch:
    backend: str
    report_projection_revision: str
    ordinary_candidate_sets_revision: str
    ordinary_diagnostics_revision: str
    ordinary_diagnostics_integrity_revision: str
    capability_revision: str | None
    schema_version: int
    retry_epoch_revision: str


@dataclass(frozen=True)
class PostcentralCtcTailProducerContext:
    """Runtime-only authority for targeted post-Central CTC tail observations."""

    capability: BoundedRetryCapability
    epoch: BaseRetryEpoch
    tail_context: TailEvidenceContext
    bundle_revision: str
    candidate_sets_revision: str
    helper_cache: dict[str, tuple[str, object]]


@dataclass(frozen=True)
class CrossLineRetryRequest:
    retry_epoch_revision: str
    entry_index: int
    ordinal: int
    capability_revision: str
    alignment_audio_revision: str
    current: RetryCandidateReference
    previous: RetryCandidateReference
    next_bound: RetryCandidateReference | None
    audio_end_seconds: float
    audio_duration_revision: str
    previous_tail_evidence_revision: str | None
    previous_tail_start: float | None
    previous_tail_end: float | None
    previous_tail_uncertainty: float | None
    previous_tail_validity: TailValidity
    ownership_relation_revision: str
    context_entry_indexes: tuple[int, ...]
    context_candidate_ids: tuple[str, ...]
    transcript_digests: tuple[str, ...]
    transcript_digest: str
    row_mapping: tuple[tuple[int, int], ...]
    row_mapping_revision: str
    window_start: float
    window_end: float
    window_definition_revision: str
    request_revision: str


@dataclass(frozen=True)
class CrossLineRetryOutcome:
    retry_epoch_revision: str
    request_revision: str
    entry_index: int
    status: Literal["candidate", "invalid", "error"]
    reason: str
    candidate: TimingCandidate | None
    output_revision: str | None
    outcome_revision: str
    previous_tail_evidence: IndependentTailEvidence | None = None


@dataclass(frozen=True)
class LocalWhisperCapability:
    kind: Literal["whispercpp-whisperx-local"]
    backend_lineage: str
    audio_path: str
    audio_sha256: str
    whisper_cli_path: str
    whisper_cli_sha256: str
    whisper_model_path: str
    whisper_model_sha256: str
    whisperx_python_path: str
    whisperx_helper_path: str
    whisperx_helper_sha256: str
    language: str
    device: str
    protocol_revision: str
    capability_revision: str


@dataclass(frozen=True)
class LocalWhisperRequest:
    entry_index: int
    capability_revision: str
    audio_revision: str
    context_entry_indexes: tuple[int, ...]
    context_candidate_ids: tuple[str, ...]
    transcript_digests: tuple[str, ...]
    transcript_digest: str
    window_start: float
    window_end: float
    window_definition_revision: str
    request_revision: str
    execution_revision: str = ""


@dataclass(frozen=True)
class LocalWhisperOutcome:
    entry_index: int
    request_revision: str
    status: Literal["evidence", "invalid", "error"]
    reason: str
    onset_time: float | None
    score: float | None
    evidence_revision: str | None
    occurrence_binding: dict[str, object] | None
    raw_transcript_revision: str | None
    aligned_transcript_revision: str | None
    outcome_revision: str


_CANDIDATE_OCCURRENCE_MODES = frozenset({
    "global-unique",
    "sequence-bracket",
    "sparse-sequence-bracket",
})


_CANDIDATE_OCCURRENCE_DIRECT_PRODUCERS = frozenset({
    "raw-asr-consensus",
    "raw-asr-vocal-fusion",
    "whisperx-consensus",
    "whisperx-vocal-fusion",
    "local-whisper-aligner",
    "local-whisper-vocal-fusion",
})


_CANDIDATE_OCCURRENCE_BINDING_KEYS = {
    "global-unique": frozenset({
        "mode",
        "entry_index",
        "lyric_sha256",
        "segment",
        "timestamp",
        "score",
        "binding_revision",
    }),
    "sequence-bracket": frozenset({
        "mode",
        "entry_index",
        "lyric_sha256",
        "segment",
        "timestamp",
        "score",
        "left_entry_index",
        "left_segment",
        "left_timestamp",
        "left_score",
        "right_entry_index",
        "right_segment",
        "right_timestamp",
        "right_score",
        "context_entry_indexes",
        "context_segments",
        "context_timestamps",
        "context_scores",
        "context_revision",
        "binding_revision",
    }),
    "sparse-sequence-bracket": frozenset({
        "mode",
        "entry_index",
        "lyric_sha256",
        "segment",
        "timestamp",
        "score",
        "context_left_index",
        "context_right_index",
        "anchor_entry_indexes",
        "anchor_segments",
        "anchor_timestamps",
        "anchor_scores",
        "fingerprint_revision",
        "binding_revision",
    }),
}


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


def make_timing_candidate(
    *,
    entry_index: int,
    entry_text: str,
    source: str,
    raw_time: float,
    spans: tuple[TimingTokenSpan, ...] = (),
    confidence: float = 0.0,
    identity_support: EvidenceState = "unavailable",
    sequence_support: EvidenceState = "unavailable",
    acoustic_support: EvidenceState = "unavailable",
    direct_onset_support: EvidenceState = "unavailable",
    acoustic_strength: float | None = None,
    acoustic_onset_time: float | None = None,
    acoustic_source_artifact: object = None,
    acoustic_evidence_producer: str | None = None,
    acoustic_evidence_kind: str = "generic-acoustic",
    acoustic_evidence_independent: bool = False,
    direct_onset_time: float | None = None,
    direct_onset_source_artifact: object = None,
    direct_onset_evidence_producer: str | None = None,
    direct_onset_evidence_kind: str = "generic-onset",
    direct_onset_evidence_independent: bool = False,
    occurrence_evidence: CandidateOccurrenceEvidence | None = None,
    independent_content_identity: bool = False,
    current: bool = False,
    prefix_scope: bool = False,
    right_edge_pileup: bool = False,
    window_truncated: bool = False,
    source_artifact: object = None,
) -> TimingCandidate:
    """Build an order-independent candidate identity from source content."""
    written_time = canonicalize_lrc_time(raw_time)
    span_payload = [
        [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
        for span in spans
    ]
    acoustic_evidence: CandidateAcousticEvidence | None = None
    if acoustic_onset_time is not None and acoustic_strength is not None:
        acoustic_revision = _timing_content_digest(acoustic_source_artifact)
        acoustic_evidence = CandidateAcousticEvidence(
            onset_time=float(acoustic_onset_time),
            strength=float(acoustic_strength),
            distance_seconds=float(acoustic_onset_time) - written_time.seconds,
            source_revision=acoustic_revision,
            producer=str(acoustic_evidence_producer or ""),
            kind=str(acoustic_evidence_kind or "generic-acoustic"),
            independent=bool(acoustic_evidence_independent and acoustic_evidence_producer),
        )
    direct_onset_evidence: CandidateOnsetEvidence | None = None
    if direct_onset_time is not None and direct_onset_source_artifact is not None:
        direct_onset_evidence = CandidateOnsetEvidence(
            onset_time=float(direct_onset_time),
            distance_seconds=float(direct_onset_time) - written_time.seconds,
            source_revision=_timing_content_digest(direct_onset_source_artifact),
            producer=str(direct_onset_evidence_producer or ""),
            kind=str(direct_onset_evidence_kind or "generic-onset"),
            independent=bool(direct_onset_evidence_independent and direct_onset_evidence_producer),
        )
    # Cross-backend direct evidence must carry its immutable occurrence binding.
    # Other direct producers (notably the existing recurrence producers) already
    # have their own provenance contract and must retain that contract here.  The
    # central validators still re-check occurrence evidence for every
    # cross-backend candidate, so a caller cannot restore the old opaque
    # ``source_artifact`` trust path by omitting the binding.
    cross_backend_identity_claim = bool(
        independent_content_identity
        and direct_onset_evidence is not None
        and direct_onset_evidence.kind == "cross-backend-onset"
    )
    independent_content_identity = bool(
        independent_content_identity
        and (occurrence_evidence is not None or not cross_backend_identity_claim)
    )
    normalized_confidence = _finite_unit_confidence(confidence)
    evidence_payload = {
        "confidence": round(normalized_confidence, 8),
        "identity_support": identity_support,
        "sequence_support": sequence_support,
        "acoustic_support": acoustic_support,
        "direct_onset_support": direct_onset_support,
        "direct_onset_evidence": (
            {
                "onset_time": round(direct_onset_evidence.onset_time, 6),
                "distance_seconds": round(direct_onset_evidence.distance_seconds, 6),
                "source_revision": direct_onset_evidence.source_revision,
                "producer": direct_onset_evidence.producer,
                "kind": direct_onset_evidence.kind,
                "independent": direct_onset_evidence.independent,
            }
            if direct_onset_evidence is not None
            else None
        ),
        "independent_content_identity": bool(independent_content_identity),
        "occurrence_evidence": _candidate_occurrence_evidence_payload(
            occurrence_evidence
        ),
        "acoustic_evidence": (
            {
                "onset_time": round(acoustic_evidence.onset_time, 6),
                "strength": round(acoustic_evidence.strength, 8),
                "distance_seconds": round(acoustic_evidence.distance_seconds, 6),
                "source_revision": acoustic_evidence.source_revision,
                "producer": acoustic_evidence.producer,
                "kind": acoustic_evidence.kind,
                "independent": acoustic_evidence.independent,
            }
            if acoustic_evidence is not None
            else None
        ),
        "current": bool(current),
        "prefix_scope": bool(prefix_scope),
        "right_edge_pileup": bool(right_edge_pileup),
        "window_truncated": bool(window_truncated),
    }
    generation_revision = _timing_content_digest(
        {
            "source": source,
            "artifact": source_artifact,
            "spans": span_payload,
            "evidence": evidence_payload,
        }
    )
    candidate_id = _timing_content_digest(
        {
            "schema": 1,
            "entry": entry_index,
            "entry_digest": _timing_content_digest(unicodedata.normalize("NFKC", entry_text)),
            "source": source,
            "generation_revision": generation_revision,
            "raw_time": round(float(raw_time), 6),
            "canonical_centiseconds": written_time.centiseconds,
            "spans": span_payload,
            "evidence": evidence_payload,
        }
    )
    return TimingCandidate(
        candidate_id=candidate_id,
        entry_index=entry_index,
        source=source,
        raw_time=float(raw_time),
        written_time=written_time,
        generation_revision=generation_revision,
        spans=spans,
        confidence=normalized_confidence,
        identity_support=identity_support,
        sequence_support=sequence_support,
        acoustic_support=acoustic_support,
        direct_onset_support=direct_onset_support,
        acoustic_strength=acoustic_strength,
        acoustic_evidence=acoustic_evidence,
        direct_onset_evidence=direct_onset_evidence,
        occurrence_evidence=occurrence_evidence,
        independent_content_identity=bool(independent_content_identity),
        current=current,
        prefix_scope=prefix_scope,
        right_edge_pileup=right_edge_pileup,
        window_truncated=window_truncated,
    )


def record_alignment_hypothesis(
    assignment: dict[str, object],
    *,
    source: str,
    stage: str,
    raw_time: float,
    confidence: float = 0.0,
    spans: object = None,
    identity_support: EvidenceState = "unavailable",
    sequence_support: EvidenceState = "unavailable",
    acoustic_support: EvidenceState = "unavailable",
    direct_onset_support: EvidenceState = "unavailable",
    acoustic_strength: float | None = None,
    acoustic_onset_time: float | None = None,
    acoustic_evidence_producer: str | None = None,
    acoustic_evidence_kind: str = "generic-acoustic",
    direct_onset_time: float | None = None,
    direct_onset_evidence_producer: str | None = None,
    direct_onset_evidence_kind: str = "generic-onset",
    parent_source: str | None = None,
    parent_time: float | None = None,
    window: dict[str, object] | None = None,
    mode: str | None = None,
    right_edge_pileup: bool = False,
    window_truncated: bool = False,
) -> dict[str, object]:
    """Append a JSON-safe immutable pre-central timing hypothesis.

    Candidate generators may use mutable scratch state internally, but they must
    publish alternatives through this ledger instead of replacing the backend's
    baseline assignment timestamp.  The central evaluator is the only authority
    allowed to select/commit a final timestamp.
    """
    hypotheses = assignment.get("alignment_hypotheses")
    if not isinstance(hypotheses, list):
        hypotheses = []
        assignment["alignment_hypotheses"] = hypotheses
    spans_payload = copy.deepcopy(spans) if isinstance(spans, list) else []
    payload: dict[str, object] = {
        "source": source,
        "stage": stage,
        "raw_candidate_time": round(float(raw_time), 6),
        "confidence": round(_finite_unit_confidence(confidence), 8),
        "spans": spans_payload,
        "identity_support": identity_support,
        "sequence_support": sequence_support,
        "acoustic_support": acoustic_support,
        "direct_onset_support": direct_onset_support,
        "acoustic_strength": (
            round(float(acoustic_strength), 8) if acoustic_strength is not None else None
        ),
        "acoustic_onset_time": (
            round(float(acoustic_onset_time), 6) if acoustic_onset_time is not None else None
        ),
        "direct_onset_time": (
            round(float(direct_onset_time), 6) if direct_onset_time is not None else None
        ),
        # These labels are descriptive only.  Central collection independently
        # validates whether a producer is allowed to establish timing trust.
        "acoustic_evidence_producer": acoustic_evidence_producer,
        "acoustic_evidence_kind": acoustic_evidence_kind,
        "direct_onset_evidence_producer": direct_onset_evidence_producer,
        "direct_onset_evidence_kind": direct_onset_evidence_kind,
        "parent": (
            {
                "source": parent_source,
                "raw_candidate_time": round(float(parent_time), 6),
            }
            if parent_source is not None and parent_time is not None
            else None
        ),
        "window": copy.deepcopy(window) if isinstance(window, dict) else None,
        "mode": mode,
        "right_edge_pileup": bool(right_edge_pileup),
        "window_truncated": bool(window_truncated),
        "authoritative": False,
    }
    payload["hypothesis_id"] = _timing_content_digest(payload)
    if not any(
        isinstance(item, dict) and item.get("hypothesis_id") == payload["hypothesis_id"]
        for item in hypotheses
    ):
        hypotheses.append(payload)
    return payload


def _percentile(values: tuple[float, ...], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _supported_run_metrics(
    spans: tuple[TimingTokenSpan, ...],
    gaps: tuple[float, ...],
) -> tuple[int, int, int]:
    """Return all/opening/later runs without joining disconnected islands."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, span in enumerate(spans):
        disconnected = (
            index > 0 and gaps[index - 1] >= CTC_SEVERE_INTERNAL_GAP_SECONDS
        )
        if span.score < 0.05 or disconnected:
            if run_start is not None:
                runs.append((run_start, index))
                run_start = None
            if span.score >= 0.05:
                run_start = index
        elif run_start is None:
            run_start = index
    if run_start is not None:
        runs.append((run_start, len(spans)))
    longest = max((end - start for start, end in runs), default=0)
    opening = next((end - start for start, end in runs if start == 0), 0)
    later = max((end - start for start, end in runs if start > 0), default=0)
    return longest, opening, later


STRICT_DIRECT_ONSET_KINDS = frozenset({
    "phonetic-onset",
    "vocal-onset",
    "vocal-activity-reentry",
    "recurrence-onset",
    "external-reference",
    "cross-backend-onset",
    "joint-lyric-boundary",
    "known-lyric-onset",
})


STRICT_DIRECT_ONSET_PRODUCERS: dict[str, frozenset[str]] = {
    "phonetic-onset": frozenset({"phonetic-onset-fusion"}),
    "vocal-onset": frozenset({"consonant-onset-detector"}),
    "vocal-activity-reentry": frozenset({"vocal-activity-detector"}),
    "recurrence-onset": frozenset({"global-recurrence-search", "mutual-recurrence-search"}),
    "external-reference": frozenset({"external-reference"}),
    "cross-backend-onset": frozenset({
        "raw-asr-consensus",
        "raw-asr-vocal-fusion",
        "whisperx-consensus",
        "whisperx-vocal-fusion",
        "local-whisper-aligner",
        "local-whisper-vocal-fusion",
    }),
    "joint-lyric-boundary": frozenset({"joint-ctc-boundary-consensus"}),
    "known-lyric-onset": frozenset({"hubert-ja-known-lyric-current-onset"}),
}


BOUNDARY_GRADE_DIRECT_ONSET_KINDS = frozenset({
    "phonetic-onset",
    "vocal-activity-reentry",
    "recurrence-onset",
    "external-reference",
    "cross-backend-onset",
    "joint-lyric-boundary",
    "known-lyric-onset",
})


def candidate_has_bound_acoustic_evidence(candidate: TimingCandidate) -> bool:
    evidence = candidate.acoustic_evidence
    return bool(
        candidate.acoustic_support == "supported"
        and evidence is not None
        and evidence.source_revision
        and evidence.independent
        and bool(evidence.producer)
        and abs(evidence.distance_seconds) <= PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
    )


def candidate_has_bound_direct_onset_evidence(
    candidate: TimingCandidate,
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Return true for revision-bound onset evidence independent of the candidate.

    This establishes that an onset observation is genuine.  It does *not* imply
    that every onset kind is lyric-specific enough to defeat an UNKNOWN or
    VIOLATION previous-tail relation; use
    :func:`candidate_has_boundary_grade_direct_onset_evidence` for that.
    """
    evidence = candidate.direct_onset_evidence
    if _candidate_requires_occurrence_evidence(candidate) and not candidate_occurrence_evidence_is_valid(
        candidate, entries
    ):
        return False
    return bool(
        candidate.direct_onset_support == "supported"
        and evidence is not None
        and evidence.source_revision
        and evidence.independent
        and bool(evidence.producer)
        and evidence.kind in STRICT_DIRECT_ONSET_KINDS
        and evidence.producer in STRICT_DIRECT_ONSET_PRODUCERS.get(evidence.kind, frozenset())
        and (
            evidence.kind != "external-reference"
            or candidate.source in {"manual-anchor", "checked-lrc", "lyrics-source"}
        )
        and abs(evidence.distance_seconds) <= PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
    )


def candidate_has_boundary_grade_direct_onset_evidence(
    candidate: TimingCandidate,
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Whether independent onset evidence may cross a previous-tail conflict.

    A generic isolated-vocal energy peak is independent acoustic evidence, but
    it is not independently phonetic/lyric-specific.  It can support an opening
    when ownership is already CLEAR, never by itself override UNKNOWN/VIOLATION.
    """
    evidence = candidate.direct_onset_evidence
    return bool(
        candidate_has_bound_direct_onset_evidence(candidate, entries)
        and evidence is not None
        and evidence.kind in BOUNDARY_GRADE_DIRECT_ONSET_KINDS
    )


def candidate_has_independent_backend_identity(
    candidate: TimingCandidate,
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Whether an independent ASR/alignment backend proves this lyric row's identity."""
    evidence = candidate.direct_onset_evidence
    return bool(
        candidate.independent_content_identity
        and candidate.identity_support == "supported"
        and candidate.sequence_support == "supported"
        and evidence is not None
        and evidence.kind == "cross-backend-onset"
        and evidence.producer in {
            "raw-asr-consensus",
            "raw-asr-vocal-fusion",
            "whisperx-consensus",
            "whisperx-vocal-fusion",
            "local-whisper-aligner",
            "local-whisper-vocal-fusion",
        }
        and candidate_has_bound_direct_onset_evidence(candidate, entries)
    )


_ROMAJI_VOWELS = frozenset("aeiou")


def _opening_mora_proof_end(
    spans: tuple[TimingTokenSpan, ...],
    parsed_gaps: tuple[float, ...],
) -> float | None:
    """Return the end of a locally-supported first mora.

    Sentence timing is defined by the start of the current lyric, not by the
    release of the previous lyric.  CTC often assigns weak posteriors to short
    Japanese onset consonants (/k h s t .../) while strongly supporting the
    following vowel.  Treat a compact C(+C)+V onset as phonetic opening proof
    when the mora as a whole is supported.

    This intentionally does *not* accept an isolated high posterior with no
    token identity or a delayed vowel: it is a local phonetic-sequence test,
    not a generic score threshold.
    """
    if not spans:
        return None

    # Vowel-initial mora.  Require immediate continuation so an isolated
    # token before a multi-second fracture cannot certify the whole opening.
    first_token = (spans[0].token or "").lower()
    if (
        first_token in _ROMAJI_VOWELS
        and spans[0].score >= 0.12
        and len(spans) >= 2
        and parsed_gaps
        and parsed_gaps[0] < 0.18
        and spans[1].score >= 0.01
    ):
        return spans[0].end

    # Japanese Hepburn onsets are normally C+V or CC+V (e.g. shi/chi/tsu).
    # Search only the first three romanized characters and require a compact
    # onset; this prevents later strong vowels from back-certifying a missing
    # sentence start.
    limit = min(3, len(spans))
    vowel_index = next(
        (
            index
            for index in range(limit)
            if (spans[index].token or "").lower() in _ROMAJI_VOWELS
        ),
        None,
    )
    if vowel_index is None or vowel_index == 0:
        return None
    vowel = spans[vowel_index]
    consonants = spans[:vowel_index]
    if vowel.score < 0.12:
        return None
    if any((span.token or "").lower() in _ROMAJI_VOWELS for span in consonants):
        return None
    if max((span.score for span in consonants), default=0.0) < 0.01:
        return None
    if vowel.start - spans[0].start > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS:
        return None
    for gap_index in range(vowel_index):
        if parsed_gaps[gap_index] >= 0.18:
            return None
    # The first mora must connect to at least one following token of the same
    # lyric sequence.  This is what distinguishes a sentence opening from a
    # lone accidental posterior island.
    if vowel_index + 1 >= len(spans):
        return None
    if parsed_gaps[vowel_index] >= 0.18 or spans[vowel_index + 1].score < 0.01:
        return None
    return vowel.end


# Candidates are frozen and their region analysis depends on nothing else, yet
# Central asks for the same candidate's regions tens of thousands of times per
# song; each object's result is remembered by identity (the entry keeps the
# object alive, so its id cannot be reused while cached).
_CANDIDATE_REGIONS: dict[int, tuple[TimingCandidate, CandidateRegionEvidence]] = {}


def analyze_candidate_regions(candidate: TimingCandidate) -> CandidateRegionEvidence:
    cached = _CANDIDATE_REGIONS.get(id(candidate))
    if cached is not None and cached[0] is candidate:
        return cached[1]
    result = _analyze_candidate_regions_uncached(candidate)
    if len(_CANDIDATE_REGIONS) >= 65536:
        _CANDIDATE_REGIONS.clear()
    _CANDIDATE_REGIONS[id(candidate)] = (candidate, result)
    return result


def _analyze_candidate_regions_uncached(candidate: TimingCandidate) -> CandidateRegionEvidence:
    spans = candidate.spans
    parsed_gaps = tuple(max(0.0, right.start - left.end) for left, right in zip(spans, spans[1:]))
    positive_gaps = tuple(gap for gap in parsed_gaps if gap > 0.0)
    median_gap = float(statistics.median(positive_gaps)) if positive_gaps else None
    max_gap = max(parsed_gaps) if parsed_gaps else None
    duration = max(0.0, spans[-1].end - spans[0].start) if spans else None
    density = len(spans) / max(duration, HOP_SECONDS) if duration is not None else None
    longest_run, opening_run, later_run = _supported_run_metrics(spans, parsed_gaps)
    reliable_count = sum(span.score >= 0.08 for span in spans)
    bound_acoustic = candidate_has_bound_acoustic_evidence(candidate)
    bound_direct = candidate_has_bound_direct_onset_evidence(candidate)
    proof_end: float | None = _opening_mora_proof_end(spans, parsed_gaps)
    if proof_end is None and spans and spans[0].score >= 0.05 and opening_run >= 2:
        first_reliable = next(
            (span for span in spans[:opening_run] if span.score >= 0.08), None
        )
        if first_reliable is not None:
            proof_end = first_reliable.end
    elif (
        spans
        and spans[0].score >= 0.08
        and bound_direct
        and reliable_count >= 8
        and (max_gap is None or max_gap < CTC_SEVERE_INTERNAL_GAP_SECONDS)
        and duration is not None
        and duration >= 3.0
    ):
        # A bound independent vocal onset plus a reliable first CTC token can
        # establish the opening even when the second token is weak.  This does
        # not rescue fractured/short paths and therefore cannot validate the
        # collapsed-tail failure mode seen in late-song false positives.
        proof_end = spans[0].end
    elif (
        len(spans) >= 2
        and spans[0].score < 0.05
        and spans[1].score >= 0.08
        and (bound_acoustic or bound_direct)
        and spans[1].start - candidate.written_time.seconds
        <= PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        and parsed_gaps[0] < CTC_SEVERE_INTERNAL_GAP_SECONDS
    ):
        proof_end = spans[1].end

    severe_gap_indexes = tuple(
        index
        for index, gap in enumerate(parsed_gaps)
        if gap >= CTC_SEVERE_INTERNAL_GAP_SECONDS
    )
    severe_gap_index: int | None = (
        severe_gap_indexes[0] if severe_gap_indexes else None
    )

    fracture_region: Literal["opening", "prefix", "middle", "tail", "unknown", "none"] = "none"
    if severe_gap_index is not None:
        left = spans[severe_gap_index]
        if proof_end is None:
            fracture_region = "unknown"
        elif left.end <= candidate.written_time.seconds:
            fracture_region = "opening"
        elif left.end < proof_end:
            fracture_region = "prefix"
        else:
            # A disconnected terminal island is not evidence that the onset itself
            # fractured.  Classify the path break after a proven opening as middle;
            # derive_reliable_tail() independently decides whether the terminal
            # island can own a cross-line boundary.  This keeps whole-row retries
            # from reintroducing the old "prefix fracture invalidates onset" bug.
            fracture_region = "middle"

    temporal: Literal["coherent", "ambiguous", "incoherent", "unavailable"]
    if not spans:
        temporal = "coherent" if bound_direct else "unavailable"
    elif candidate.prefix_scope and severe_gap_indexes:
        temporal = "incoherent"
    elif severe_gap_index is not None and fracture_region in {"opening", "prefix", "unknown"}:
        temporal = "incoherent"
    elif proof_end is None:
        temporal = "ambiguous"
    else:
        temporal = "coherent"

    region_payload = {
        "candidate_id": candidate.candidate_id,
        "generation_revision": candidate.generation_revision,
        "computed_for_centiseconds": candidate.written_time.centiseconds,
        "gaps": parsed_gaps,
        "duration": duration,
        "opening_supported_run": opening_run,
        "later_supported_run": later_run,
        "onset_proof_end": proof_end,
        "severe_gap_indexes": severe_gap_indexes,
        "temporal_coherence": temporal,
        "fracture_region": fracture_region,
        "right_edge_pileup": candidate.right_edge_pileup,
        "window_truncated": candidate.window_truncated,
    }
    return CandidateRegionEvidence(
        candidate_id=candidate.candidate_id,
        generation_revision=candidate.generation_revision,
        computed_for_centiseconds=candidate.written_time.centiseconds,
        region_revision=_timing_content_digest(region_payload),
        parsed_gaps=parsed_gaps,
        prefix_duration=round(duration, 6) if duration is not None else None,
        max_internal_gap=round(max_gap, 6) if max_gap is not None else None,
        median_internal_gap=round(median_gap, 6) if median_gap is not None else None,
        p90_internal_gap=round(_percentile(parsed_gaps, 0.90), 6) if parsed_gaps else None,
        p95_internal_gap=round(_percentile(parsed_gaps, 0.95), 6) if parsed_gaps else None,
        token_density=round(density, 6) if density is not None else None,
        longest_supported_run=longest_run,
        opening_supported_run=opening_run,
        later_supported_run=later_run,
        reliable_support_count=reliable_count,
        onset_proof_end=round(proof_end, 6) if proof_end is not None else None,
        severe_gap_indexes=severe_gap_indexes,
        temporal_coherence=temporal,
        fracture_region=fracture_region,
    )


def _tail_region_result(
    candidate: TimingCandidate,
    regions: CandidateRegionEvidence,
    validity: TailValidity,
    reason: str,
    start: float | None = None,
    end: float | None = None,
    uncertainty: float | None = None,
    confidence: float | None = None,
    terminal_span_indexes: tuple[int, ...] = (),
    local_gap_references: tuple[float, ...] = (),
) -> TailRegionEvidence:
    revision = _timing_content_digest({
        "previous_candidate_id": candidate.candidate_id,
        "previous_generation_revision": candidate.generation_revision,
        "previous_region_revision": regions.region_revision,
        "validity": validity,
        "start": start,
        "end": end,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "reason": reason,
        "terminal_span_indexes": terminal_span_indexes,
        "local_gap_references": local_gap_references,
    })
    return TailRegionEvidence(
        candidate.candidate_id,
        candidate.generation_revision,
        regions.region_revision,
        revision,
        validity,
        start,
        end,
        uncertainty,
        confidence,
        reason,
        terminal_span_indexes,
        local_gap_references,
    )


def derive_reliable_tail(
    candidate: TimingCandidate,
    regions: CandidateRegionEvidence | None = None,
) -> TailRegionEvidence:
    regions = regions or analyze_candidate_regions(candidate)
    if (
        regions.candidate_id != candidate.candidate_id
        or regions.generation_revision != candidate.generation_revision
    ):
        return _tail_region_result(candidate, regions, "invalid", "tail-region-revision-mismatch")
    if candidate.right_edge_pileup or candidate.window_truncated:
        return _tail_region_result(candidate, regions, "invalid", "tail-window-invalid")
    if regions.fracture_region in {"tail", "unknown"}:
        return _tail_region_result(candidate, regions, "invalid", "tail-fracture-invalid")
    if candidate.identity_support != "supported" or candidate.sequence_support != "supported":
        return _tail_region_result(candidate, regions, "unknown", "tail-identity-or-sequence-unavailable")
    spans = candidate.spans
    attached_weak_terminal = False
    detached_low_reliability_terminal = False
    if (
        candidate.source == "ctc-local-window"
        and len(spans) >= 3
        and regions.onset_proof_end is not None
    ):
        terminal_gap = max(0.0, spans[-1].start - spans[-2].end)
        reliable_prefix_count = sum(span.score >= 0.08 for span in spans[:-1])
        if reliable_prefix_count >= 2 and spans[-2].score >= 0.08:
            if (
                spans[-1].score < 0.08
                and terminal_gap >= CTC_SEVERE_INTERNAL_GAP_SECONDS
            ):
                spans = spans[:-1]
                detached_low_reliability_terminal = True
            elif (
                spans[-1].score < 0.05
                and not regions.severe_gap_indexes
                and regions.p95_internal_gap is not None
                and terminal_gap <= max(HOP_SECONDS, float(regions.p95_internal_gap))
            ):
                attached_weak_terminal = True
    if len(spans) < 2 or (spans[-1].score < 0.05 and not attached_weak_terminal):
        return _tail_region_result(candidate, regions, "unknown", "no-terminal-supported-run")
    gaps = tuple(max(0.0, right.start - left.end) for left, right in zip(spans, spans[1:]))
    separating_gap_index = len(spans) - 2
    reference_start = separating_gap_index - 1
    while reference_start >= 0:
        if (
            gaps[reference_start] >= CTC_SEVERE_INTERNAL_GAP_SECONDS
            or spans[reference_start].score < 0.05
            or spans[reference_start + 1].score < 0.05
        ):
            break
        reference_start -= 1
    if reference_start >= 0 and gaps[reference_start] >= CTC_SEVERE_INTERNAL_GAP_SECONDS:
        return _tail_region_result(candidate, regions, "invalid", "tail-fracture-invalid")
    local_gap_references = tuple(
        gaps[index]
        for index in range(reference_start + 1, separating_gap_index)
        if gaps[index] > 0.0
    )
    if not local_gap_references:
        # A weak token immediately before an otherwise strong terminal pair can
        # cut the local reference walk even when the candidate has ample robust
        # cadence evidence elsewhere.  Fall back only to gaps whose two adjacent
        # tokens are both independently reliable, and only when the terminal pair
        # is itself reliable.  This reuses the existing reliability/cadence rules
        # rather than inventing a song-specific tail threshold.
        global_gap_references = tuple(
            gap
            for gap_index, gap in enumerate(gaps[:-1])
            if gap > 0.0
            and gap < CTC_SEVERE_INTERNAL_GAP_SECONDS
            and spans[gap_index].score >= 0.08
            and spans[gap_index + 1].score >= 0.08
        )
        if (
            spans[-2].score >= 0.08
            and spans[-1].score >= 0.08
            and global_gap_references
        ):
            local_gap_references = global_gap_references
        else:
            return _tail_region_result(
                candidate, regions, "unknown", "terminal-local-gap-reference-unavailable"
            )
    median_gap = float(statistics.median(local_gap_references))
    contiguous_limit = max(HOP_SECONDS, median_gap * 3.0)
    if gaps[separating_gap_index] >= CTC_SEVERE_INTERNAL_GAP_SECONDS:
        return _tail_region_result(candidate, regions, "invalid", "tail-fracture-invalid")
    if gaps[separating_gap_index] > contiguous_limit:
        return _tail_region_result(
            candidate,
            regions,
            "unknown",
            "detached-terminal-ghost",
            local_gap_references=local_gap_references,
        )
    start_index = len(spans) - 1
    reliable_count = 1 if spans[-1].score >= 0.08 else 0
    for index in range(len(spans) - 2, -1, -1):
        gap = max(0.0, spans[index + 1].start - spans[index].end)
        if gap > contiguous_limit or spans[index].score < 0.05:
            break
        start_index = index
        reliable_count += int(spans[index].score >= 0.08)
    tail = spans[start_index:]
    if len(tail) < 2 or reliable_count < 1:
        return _tail_region_result(candidate, regions, "unknown", "terminal-run-not-reliable")
    if any(index == start_index - 1 for index in regions.severe_gap_indexes):
        return _tail_region_result(candidate, regions, "invalid", "tail-fracture-invalid")
    tail_uncertainty = contiguous_limit
    tail_reason = "reliable-terminal-run"
    if attached_weak_terminal:
        tail_uncertainty = max(HOP_SECONDS, spans[-1].end - spans[-2].end)
        tail_reason = "local-window-attached-weak-terminal"
    elif detached_low_reliability_terminal:
        tail_reason = "local-window-detached-low-reliability-terminal-ignored"
    return _tail_region_result(
        candidate,
        regions,
        "valid",
        tail_reason,
        tail[0].start,
        tail[-1].end,
        tail_uncertainty,
        statistics.mean(span.score for span in tail),
        tuple(range(start_index, len(spans))),
        local_gap_references,
    )


def _bind_independent_tail_to_candidate(
    candidate: TimingCandidate,
    evidence: IndependentTailEvidence,
) -> TailRegionEvidence:
    """Re-bind runtime tail evidence to the selected predecessor candidate.

    The external row proves a lyric-tail upper bound, not the predecessor's
    start timestamp.  Binding the evidence to the selected predecessor keeps
    the ownership relation revision-safe without forcing that external backend
    to become the selected timing candidate.
    """
    regions = analyze_candidate_regions(candidate)
    provenance = evidence.provenance
    authority_backend_lineage = (
        provenance.backend_lineage if provenance is not None else None
    )
    authority_producer = provenance.producer if provenance is not None else None
    authority_observation_revision = (
        provenance.observation_revision if provenance is not None else None
    )
    authority_metadata_valid = bool(
        isinstance(authority_backend_lineage, str)
        and authority_backend_lineage.strip()
        and _tail_backend_family(authority_backend_lineage) is not None
        and isinstance(authority_producer, str)
        and authority_producer.strip()
        and isinstance(authority_observation_revision, str)
        and authority_observation_revision.strip()
    )
    payload = {
        "previous_candidate_id": candidate.candidate_id,
        "previous_generation_revision": candidate.generation_revision,
        "previous_region_revision": regions.region_revision,
        "external_entry_index": evidence.entry_index,
        "external_tail_start": round(evidence.tail_start, 6),
        "external_tail_end": round(evidence.tail_end, 6),
        "external_uncertainty_seconds": round(evidence.uncertainty_seconds, 6),
        "external_confidence": round(evidence.confidence, 6),
        "external_producer": evidence.producer,
        "external_source_revision": evidence.source_revision,
        "external_provenance": (
            _tail_provenance_payload(evidence.provenance)
            if evidence.provenance is not None else None
        ),
    }
    return TailRegionEvidence(
        previous_candidate_id=candidate.candidate_id,
        previous_generation_revision=candidate.generation_revision,
        previous_region_revision=regions.region_revision,
        tail_evidence_revision=_timing_content_digest(payload),
        validity="valid" if authority_metadata_valid else "unknown",
        reliable_tail_start=float(evidence.tail_start) if authority_metadata_valid else None,
        reliable_tail_end=float(evidence.tail_end) if authority_metadata_valid else None,
        uncertainty_seconds=(
            max(0.0, float(evidence.uncertainty_seconds))
            if authority_metadata_valid else None
        ),
        confidence=(
            _finite_unit_confidence(evidence.confidence)
            if authority_metadata_valid else None
        ),
        reason=(
            f"independent-{evidence.producer}-tail"
            if authority_metadata_valid
            else "independent-tail-authority-metadata-missing"
        ),
        terminal_span_indexes=(),
        local_gap_references=(),
        authority_backend_lineage=authority_backend_lineage if authority_metadata_valid else None,
        authority_producer=authority_producer if authority_metadata_valid else None,
        authority_observation_revision=(
            authority_observation_revision if authority_metadata_valid else None
        ),
    )


def _normalize_tail_observations(
    independent: IndependentTailEvidence | TailObservationSet | None,
) -> TailObservationSet:
    if independent is None:
        return ()
    if isinstance(independent, IndependentTailEvidence):
        return (independent,)
    if not isinstance(independent, tuple):
        return ()
    return tuple(
        item for item in independent
        if isinstance(item, IndependentTailEvidence)
    )


def _authoritative_tail_observations(
    candidate: TimingCandidate,
    observations: TailObservationSet,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
) -> TailObservationSet:
    if not _tail_contexts_are_valid(tail_evidence_context):
        return ()
    candidate_families = _candidate_support_families(candidate)
    if not candidate_families:
        return ()
    selected: list[IndependentTailEvidence] = []
    for observation in observations:
        if observation.entry_index != candidate.entry_index:
            continue
        if (
            not _tail_provenance_is_valid(observation)
            or not _tail_provenance_matches_context(observation, tail_evidence_context)
        ):
            continue
        evidence_family = _tail_backend_family(
            observation.provenance.backend_lineage if observation.provenance else None
        )
        if evidence_family is None or evidence_family in candidate_families:
            continue
        selected.append(observation)
    return tuple(sorted(
        selected,
        key=lambda item: (
            -(item.tail_end + max(0.0, item.uncertainty_seconds)),
            -item.tail_end,
            item.provenance.observation_revision if item.provenance else item.source_revision,
        ),
    ))


def resolve_reliable_tail(
    candidate: TimingCandidate,
    independent: IndependentTailEvidence | TailObservationSet | None = None,
    *,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
) -> TailRegionEvidence:
    """Return the most conservative valid predecessor-tail proof available.

    Intrinsic CTC tail geometry remains the default.  Only a provenance-valid
    observation whose backend lineage is disjoint from the selected predecessor
    candidate may replace UNKNOWN/invalid intrinsic evidence or tighten an
    already-valid release bound.  When both are valid the later
    ``tail_end + uncertainty`` wins, so fusion can never make overlap easier to
    clear than either source.  Same-lineage observations remain diagnostic.
    A missing or stale execution context makes every external observation
    diagnostic-only; intrinsic geometry is still evaluated normally.
    """
    intrinsic = derive_reliable_tail(candidate)
    observations = _authoritative_tail_observations(
        candidate,
        _normalize_tail_observations(independent),
        tail_evidence_context,
    )
    if not observations:
        return intrinsic
    external = _bind_independent_tail_to_candidate(candidate, observations[0])
    external_upper = external.reliable_tail_end + (external.uncertainty_seconds or 0.0)
    if intrinsic.validity != "valid" or intrinsic.reliable_tail_end is None:
        # A disjoint backend may resolve an unknown predecessor tail, but it
        # cannot erase reliable support that the selected predecessor itself
        # still carries beyond that backend's claimed release boundary.
        if any(
            span.score >= 0.08 and span.start > external_upper
            for span in candidate.spans
        ):
            return intrinsic
        return external
    intrinsic_upper = intrinsic.reliable_tail_end + (intrinsic.uncertainty_seconds or 0.0)
    if intrinsic_upper >= external_upper:
        return intrinsic
    return external


def classify_candidate_ownership(
    candidate: TimingCandidate,
    current_regions: CandidateRegionEvidence,
    previous_tail: TailRegionEvidence | None,
) -> OwnershipRelationEvidence:
    status: OwnershipState
    reason: str
    distance: float | None = None
    lineage_relation = _external_tail_current_lineage_relation(
        candidate, previous_tail
    )
    if previous_tail is None:
        status, reason = "clear", "first-entry"
    elif previous_tail.validity != "valid" or previous_tail.reliable_tail_end is None:
        status, reason = "unknown", previous_tail.reason
    elif lineage_relation == "overlap":
        status, reason = "unknown", "external-tail-current-lineage-overlap"
    elif lineage_relation == "unknown":
        status, reason = "unknown", "external-tail-current-lineage-unresolved"
    else:
        distance = candidate.written_time.seconds - previous_tail.reliable_tail_end
        if candidate.written_time.seconds <= previous_tail.reliable_tail_end:
            status, reason = "violation", "candidate-invades-reliable-tail"
        elif candidate.written_time.seconds <= previous_tail.reliable_tail_end + (
            previous_tail.uncertainty_seconds or HOP_SECONDS
        ):
            status, reason = "unknown", "candidate-inside-tail-release-uncertainty"
        else:
            status, reason = "clear", "candidate-after-valid-tail-release"
    payload = {
        "status": status,
        "reason": reason,
        "current_candidate_id": candidate.candidate_id,
        "current_generation_revision": candidate.generation_revision,
        "current_region_revision": current_regions.region_revision,
        "current_centiseconds": candidate.written_time.centiseconds,
        "previous_candidate_id": previous_tail.previous_candidate_id if previous_tail else None,
        "previous_generation_revision": previous_tail.previous_generation_revision if previous_tail else None,
        "previous_region_revision": previous_tail.previous_region_revision if previous_tail else None,
        "previous_tail_evidence_revision": previous_tail.tail_evidence_revision if previous_tail else None,
        "previous_tail_validity": previous_tail.validity if previous_tail else None,
        "previous_tail_authority_backend_lineage": (
            previous_tail.authority_backend_lineage if previous_tail else None
        ),
        "previous_tail_authority_producer": (
            previous_tail.authority_producer if previous_tail else None
        ),
        "previous_tail_authority_observation_revision": (
            previous_tail.authority_observation_revision if previous_tail else None
        ),
        "external_tail_current_lineage_relation": lineage_relation,
        "distance": distance,
    }
    return OwnershipRelationEvidence(
        status=status,
        relation_revision=_timing_content_digest(payload),
        current_candidate_id=candidate.candidate_id,
        current_generation_revision=candidate.generation_revision,
        current_region_revision=current_regions.region_revision,
        current_centiseconds=candidate.written_time.centiseconds,
        previous_candidate_id=previous_tail.previous_candidate_id if previous_tail else None,
        previous_generation_revision=previous_tail.previous_generation_revision if previous_tail else None,
        previous_region_revision=previous_tail.previous_region_revision if previous_tail else None,
        previous_tail_evidence_revision=previous_tail.tail_evidence_revision if previous_tail else None,
        previous_tail_validity=previous_tail.validity if previous_tail else None,
        reliable_tail_start=previous_tail.reliable_tail_start if previous_tail else None,
        reliable_tail_end=previous_tail.reliable_tail_end if previous_tail else None,
        uncertainty_seconds=previous_tail.uncertainty_seconds if previous_tail else None,
        confidence=previous_tail.confidence if previous_tail else None,
        distance_seconds=round(distance, 6) if distance is not None else None,
        reason=reason,
        previous_tail_authority_backend_lineage=(
            previous_tail.authority_backend_lineage if previous_tail else None
        ),
        previous_tail_authority_producer=(
            previous_tail.authority_producer if previous_tail else None
        ),
        previous_tail_authority_observation_revision=(
            previous_tail.authority_observation_revision if previous_tail else None
        ),
    )


def normalize_timing_candidate(
    candidate: TimingCandidate,
    previous_candidate: TimingCandidate | None = None,
    *,
    current_regions: CandidateRegionEvidence | None = None,
    previous_tail: TailRegionEvidence | None = None,
    ownership_relation: OwnershipRelationEvidence | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> NormalizedTimingEvidence:
    spans = candidate.spans
    regions = current_regions or analyze_candidate_regions(candidate)
    if previous_candidate is not None and previous_tail is None:
        previous_tail = derive_reliable_tail(previous_candidate)
    ownership = ownership_relation or classify_candidate_ownership(
        candidate, regions, previous_tail
    )
    bound_acoustic = candidate_has_bound_acoustic_evidence(candidate)
    occurrence_valid = bool(
        not _candidate_requires_occurrence_evidence(candidate)
        or candidate_occurrence_evidence_is_valid(candidate, occurrence_entries)
    )
    bound_direct = candidate_has_bound_direct_onset_evidence(
        candidate, occurrence_entries
    )
    identity_ok = candidate.identity_support == "supported"
    sequence_ok = candidate.sequence_support == "supported"
    # Independent onset evidence proves a boundary, not an arbitrarily collapsed
    # CTC path.  When token spans exist, require local continuation support unless
    # the candidate carries independent full-line content identity (recurrence).
    direct_onset_path_shape_ok = bool(
        not bound_direct
        or not spans
        or candidate.independent_content_identity
        or regions.reliable_support_count >= 3
    )
    direct_onset_path_support = bool(bound_direct and direct_onset_path_shape_ok)
    onset_ok = bool(
        direct_onset_path_shape_ok
        and (regions.onset_proof_end is not None or direct_onset_path_support)
    )
    independent_backend_identity = candidate_has_independent_backend_identity(
        candidate, occurrence_entries
    )
    coherence_ok = bool(
        regions.temporal_coherence == "coherent"
        or (
            independent_backend_identity
            and regions.temporal_coherence == "ambiguous"
        )
    )
    opening_identity_proven = bool(
        identity_ok
        and sequence_ok
        and onset_ok
        and coherence_ok
    )
    # A candidate may use its own spans to establish intrinsic identity,
    # sequence, and opening geometry, but those same spans are not independent
    # evidence against a previous-tail UNKNOWN/VIOLATION relation. Crossing a
    # tail conflict requires separately produced, revision-bound direct-onset
    # evidence.
    boundary_grade_opening = _boundary_grade_opening_can_cross_tail(
        candidate,
        previous_tail=previous_tail,
        occurrence_entries=occurrence_entries,
    )
    independent_ownership = bool(
        boundary_grade_opening and ownership.status != "clear"
    )

    failures: list[str] = []
    if not occurrence_valid:
        failures.append("occurrence-evidence-invalid")
    if candidate.right_edge_pileup:
        failures.append("right-edge-pileup")
    if candidate.window_truncated:
        failures.append("window-truncated")
    if (
        previous_candidate is not None
        and candidate.written_time.seconds
        <= previous_candidate.written_time.seconds + HOP_SECONDS
    ):
        failures.append("nonmonotonic-line-start")
    if ownership.status == "violation" and not boundary_grade_opening:
        failures.append("previous-tail-overlap")
    if ownership.status == "unknown" and not boundary_grade_opening:
        failures.append("previous-tail-ownership-unknown")
    if regions.temporal_coherence == "incoherent":
        failures.append("temporal-incoherence")
    if spans and abs(spans[0].start - candidate.written_time.seconds) > HOP_SECONDS:
        failures.append("candidate-span-revision-mismatch")
    relation_revision_ok = bool(
        ownership.current_candidate_id == candidate.candidate_id
        and ownership.current_generation_revision == candidate.generation_revision
        and ownership.current_region_revision == regions.region_revision
        and ownership.current_centiseconds == candidate.written_time.centiseconds
        and (
            previous_tail is None
            or (
                ownership.previous_candidate_id == previous_tail.previous_candidate_id
                and ownership.previous_generation_revision == previous_tail.previous_generation_revision
                and ownership.previous_region_revision == previous_tail.previous_region_revision
                and ownership.previous_tail_evidence_revision == previous_tail.tail_evidence_revision
            )
        )
    )
    if not relation_revision_ok:
        failures.append("ownership-revision-mismatch")

    ownership_ok = ownership.status == "clear" or boundary_grade_opening
    if candidate.source == "acoustic-only":
        identity_ok = False
    if candidate.source == "duplicate-projection":
        onset_ok = False
    minimum_proof = bool(
        identity_ok
        and sequence_ok
        and occurrence_valid
        and onset_ok
        and ownership_ok
        and coherence_ok
        and relation_revision_ok
    )

    revision = _timing_content_digest(
        {
            "candidate_id": candidate.candidate_id,
            "computed_for_centiseconds": candidate.written_time.centiseconds,
            "generation_revision": candidate.generation_revision,
            "region_revision": regions.region_revision,
            "ownership_relation_revision": ownership.relation_revision,
            "referenced_tail_evidence_revision": ownership.previous_tail_evidence_revision,
            "bound_acoustic": bound_acoustic,
            "bound_direct": bound_direct,
            "minimum_proof": minimum_proof,
            "failures": failures,
        }
    )
    return NormalizedTimingEvidence(
        candidate_id=candidate.candidate_id,
        candidate_revision=revision,
        computed_for_centiseconds=candidate.written_time.centiseconds,
        region_revision=regions.region_revision,
        ownership_relation_revision=ownership.relation_revision,
        referenced_tail_evidence_revision=ownership.previous_tail_evidence_revision,
        prefix_duration=regions.prefix_duration,
        max_internal_gap=regions.max_internal_gap,
        median_internal_gap=regions.median_internal_gap,
        p90_internal_gap=regions.p90_internal_gap,
        p95_internal_gap=regions.p95_internal_gap,
        token_density=regions.token_density,
        longest_supported_run=regions.longest_supported_run,
        opening_supported_run=regions.opening_supported_run,
        later_supported_run=regions.later_supported_run,
        reliable_support_count=regions.reliable_support_count,
        onset_proof_end=regions.onset_proof_end,
        island_count=(len(regions.severe_gap_indexes) + 1 if spans else 0),
        severe_gap_count=len(regions.severe_gap_indexes),
        first_severe_gap_index=regions.severe_gap_indexes[0] if regions.severe_gap_indexes else None,
        temporal_coherence=regions.temporal_coherence,
        fracture_region=regions.fracture_region,
        ownership=ownership,
        tail_distance_seconds=ownership.distance_seconds,
        ownership_independent=independent_ownership,
        structural_failures=tuple(failures),
        minimum_proof_satisfied=minimum_proof,
    )


def evaluate_timing_candidate(
    candidate: TimingCandidate,
    previous_candidate: TimingCandidate | None = None,
    *,
    previous_tail: TailRegionEvidence | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> CandidateEvaluation:
    evidence = normalize_timing_candidate(
        candidate,
        previous_candidate,
        previous_tail=previous_tail,
        occurrence_entries=occurrence_entries,
    )
    reasons = list(evidence.structural_failures)
    if not evidence.minimum_proof_satisfied:
        reasons.append("insufficient-minimum-proof")
    reasons = list(dict.fromkeys(reasons))
    return CandidateEvaluation(candidate, evidence, not reasons, tuple(reasons))


def _candidate_vector(evaluation: CandidateEvaluation) -> tuple[int, int, int, int, float]:
    ownership_rank = {"violation": 0, "unknown": 1, "clear": 2}
    temporal_rank = {"incoherent": 0, "unavailable": 1, "ambiguous": 2, "coherent": 3}
    candidate = evaluation.candidate
    opening_rank = int(
        evaluation.evidence.onset_proof_end is not None
        or candidate_has_bound_direct_onset_evidence(candidate)
    )
    return (
        opening_rank,
        int(candidate.identity_support == "supported" and candidate.sequence_support == "supported"),
        temporal_rank[evaluation.evidence.temporal_coherence],
        ownership_rank[evaluation.evidence.ownership.status],
        candidate.confidence,
    )


def candidate_dominates(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    if not left.valid or not right.valid:
        return left.valid and not right.valid
    left_vector = _candidate_vector(left)
    right_vector = _candidate_vector(right)
    dominates = all(a >= b for a, b in zip(left_vector, right_vector)) and any(
        a > b for a, b in zip(left_vector, right_vector)
    )
    if not dominates:
        return False
    # Producer-local confidence values are useful within one lineage, but are
    # not calibrated strongly enough to let a distant candidate from one
    # backend erase an otherwise structurally-equivalent candidate from a
    # disjoint backend.  Preserve confidence ranking everywhere else and let
    # the existing bounded-consensus owner arbitrate this cross-lineage case.
    left_has_extra_authority = (
        candidate_has_boundary_grade_direct_onset_evidence(left.candidate)
        and not candidate_has_boundary_grade_direct_onset_evidence(right.candidate)
    ) or (
        left.candidate.independent_content_identity
        and not right.candidate.independent_content_identity
    )
    if (
        left_vector[:-1] == right_vector[:-1]
        and left_vector[-1] > right_vector[-1]
        and not left_has_extra_authority
    ):
        left_families = _lineage_families_from_labels(
            left.candidate.source,
            source=left.candidate.source,
        )
        right_families = _lineage_families_from_labels(
            right.candidate.source,
            source=right.candidate.source,
        )
        distant = abs(
            left.candidate.written_time.seconds - right.candidate.written_time.seconds
        ) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
        if left_families and right_families and left_families.isdisjoint(right_families) and distant:
            return False
    return True


def _semantically_equivalent_valid_winner(
    evaluations: tuple[CandidateEvaluation, ...],
) -> CandidateEvaluation | None:
    """Select one provenance record when all valid winners mean the same LRC time.

    Multiple independent producers can legitimately prove the same canonical
    boundary.  Treating those proofs as mutually exclusive creates the
    pathological state where *more* evidence causes a provisional fallback.
    This helper never chooses between different timestamps: it only collapses
    an already-equivalent centisecond class and keeps the strongest provenance
    deterministically.
    """
    if not evaluations or any(not item.valid for item in evaluations):
        return None
    centiseconds = {item.candidate.written_time.centiseconds for item in evaluations}
    if len(centiseconds) != 1:
        return None

    def rank(item: CandidateEvaluation) -> tuple[int, int, int, int, int, int, float, str, str]:
        candidate = item.candidate
        vector = _candidate_vector(item)
        return (
            int(candidate_has_boundary_grade_direct_onset_evidence(candidate)),
            int(candidate.independent_content_identity),
            vector[0], vector[1], vector[2], vector[3], vector[4],
            candidate.source, candidate.candidate_id,
        )

    # Numeric evidence ranks are descending.  Lexical fields are only stable
    # provenance tie-breakers after timestamp/evidence equivalence is proven.
    best_rank = max(rank(item)[:-2] for item in evaluations)
    strongest = [item for item in evaluations if rank(item)[:-2] == best_rank]
    return min(strongest, key=lambda item: (item.candidate.source, item.candidate.candidate_id))


def _lineage_families_from_labels(
    *labels: str,
    source: str | None = None,
) -> frozenset[str]:
    """Map existing producer/source labels to coarse backend lineages."""
    normalized = " ".join(
        label.lower()
        for label in labels
        if isinstance(label, str) and label
    )
    families: set[str] = set()
    if (
        "local-whisper" in normalized
        or "whisperx" in normalized
        or "whispercpp" in normalized
        or "raw-asr" in normalized
        or (isinstance(source, str) and source.lower().startswith("raw-"))
    ):
        families.add("whisper-family")
    if "ctc" in normalized or "mms" in normalized:
        families.add("ctc")
    if "recurrence" in normalized:
        families.add("recurrence")
    if "hubert" in normalized:
        families.add("hubert-ja-known-lyric")
    return frozenset(families)


def _candidate_support_families(candidate: TimingCandidate) -> frozenset[str]:
    """Return coarse *independent* evidence lineages for consensus.

    Global raw ASR, bounded local Whisper, and WhisperX-assisted observations all
    ultimately depend on the same Whisper recognizer stack for lyric identity.
    Treating them as separate votes would manufacture independence.  They therefore
    collapse to one ``whisper-family`` lineage.  CTC and recurrence remain separate
    lineages because their evidence is produced independently.
    """
    return _lineage_families_from_labels(
        candidate.source,
        candidate.direct_onset_evidence.producer if candidate.direct_onset_evidence else "",
        candidate.acoustic_evidence.producer if candidate.acoustic_evidence else "",
        source=candidate.source,
    )


def _single_direct_onset_family(candidate: TimingCandidate) -> str | None:
    """Return a known family for the candidate's boundary-grade direct producer."""
    if not candidate_has_boundary_grade_direct_onset_evidence(candidate):
        return None
    evidence = candidate.direct_onset_evidence
    if evidence is None:
        return None
    families = _lineage_families_from_labels(evidence.producer)
    return next(iter(families)) if len(families) == 1 else None


def _current_tail_lineage_relation(
    candidate: TimingCandidate,
    authority_backend_lineage: object,
    authority_producer: object,
    authority_observation_revision: object,
) -> Literal["disjoint", "direct-disjoint", "overlap", "unknown"]:
    """Classify external-tail dependence for one current candidate.

    Broad candidate support is the default dependency boundary.  A candidate's
    existing boundary-grade direct-onset producer may establish a disjoint
    exception, but generic acoustic evidence and unknown/composite producers do
    not manufacture independence.
    """
    metadata = (
        authority_backend_lineage,
        authority_producer,
        authority_observation_revision,
    )
    if not all(isinstance(value, str) and value.strip() for value in metadata):
        return "unknown"
    tail_family = _tail_backend_family(authority_backend_lineage)
    candidate_families = _candidate_support_families(candidate)
    if tail_family is None or not candidate_families:
        return "unknown"
    if tail_family not in candidate_families:
        return "disjoint"
    direct_family = _single_direct_onset_family(candidate)
    if direct_family is not None and direct_family != tail_family:
        return "direct-disjoint"
    return "overlap"


def _external_tail_current_lineage_relation(
    candidate: TimingCandidate,
    previous_tail: TailRegionEvidence | None,
) -> Literal["intrinsic", "disjoint", "direct-disjoint", "overlap", "unknown"] | None:
    if previous_tail is None:
        return None
    metadata = (
        previous_tail.authority_backend_lineage,
        previous_tail.authority_producer,
        previous_tail.authority_observation_revision,
    )
    # No authority metadata denotes intrinsic predecessor geometry.  Only the
    # validated runtime binder can create external metadata-bearing tails.
    if all(value is None for value in metadata):
        return "intrinsic"
    relation = _current_tail_lineage_relation(candidate, *metadata)
    return relation


def _boundary_grade_opening_can_cross_tail(
    candidate: TimingCandidate,
    *,
    previous_tail: TailRegionEvidence | None = None,
    ownership: OwnershipRelationEvidence | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Whether existing direct opening evidence may cross this tail relation."""
    if not candidate_has_boundary_grade_direct_onset_evidence(
        candidate, occurrence_entries
    ):
        return False
    if previous_tail is not None:
        relation = _external_tail_current_lineage_relation(candidate, previous_tail)
    elif ownership is not None:
        metadata = (
            ownership.previous_tail_authority_backend_lineage,
            ownership.previous_tail_authority_producer,
            ownership.previous_tail_authority_observation_revision,
        )
        relation = (
            "intrinsic"
            if all(value is None for value in metadata)
            else _current_tail_lineage_relation(candidate, *metadata)
        )
    else:
        relation = None
    return relation in {None, "intrinsic", "disjoint", "direct-disjoint"}


def _bounded_independent_consensus_winner(
    evaluations: tuple[CandidateEvaluation, ...],
) -> CandidateEvaluation | None:
    """Resolve a unique tight cluster of independently supported valid candidates.

    This closes a conservative gap in the central evaluator: independent producers
    rarely quantize a real acoustic onset to the exact same centisecond. A cluster
    is therefore allowed to converge within the same tolerance already used for
    cross-backend onset consensus. No timestamp is averaged or fabricated; the
    winner is always one existing valid candidate. Competing equally-supported
    clusters remain unresolved.
    """
    valid = tuple(item for item in evaluations if item.valid)
    if len(valid) < 2:
        return None
    tolerance = RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
    ordered = tuple(sorted(
        valid,
        key=lambda item: (item.candidate.written_time.seconds, item.candidate.candidate_id),
    ))
    clusters: list[tuple[CandidateEvaluation, ...]] = []
    for left in range(len(ordered)):
        members: list[CandidateEvaluation] = []
        start = ordered[left].candidate.written_time.seconds
        for item in ordered[left:]:
            if item.candidate.written_time.seconds - start > tolerance:
                break
            members.append(item)
        if len(members) < 2:
            continue
        cluster = tuple(members)
        families = set().union(*(_candidate_support_families(item.candidate) for item in cluster))
        if len(families) < 2:
            continue
        ids = tuple(item.candidate.candidate_id for item in cluster)
        if not any(tuple(existing.candidate.candidate_id for existing in prior) == ids for prior in clusters):
            clusters.append(cluster)
    if not clusters:
        return None

    def cluster_rank(cluster: tuple[CandidateEvaluation, ...]) -> tuple[int, int, int, float]:
        families = set().union(*(_candidate_support_families(item.candidate) for item in cluster))
        boundary_families = set().union(*(
            _candidate_support_families(item.candidate)
            for item in cluster
            if candidate_has_boundary_grade_direct_onset_evidence(item.candidate)
        )) if any(
            candidate_has_boundary_grade_direct_onset_evidence(item.candidate)
            for item in cluster
        ) else set()
        identity_families = set().union(*(
            _candidate_support_families(item.candidate)
            for item in cluster
            if item.candidate.independent_content_identity
        )) if any(item.candidate.independent_content_identity for item in cluster) else set()
        return (
            len(families),
            len(boundary_families),
            len(identity_families),
            max(item.candidate.confidence for item in cluster),
        )

    best_rank = max(cluster_rank(cluster) for cluster in clusters)
    best_clusters = [cluster for cluster in clusters if cluster_rank(cluster) == best_rank]
    # Two equally strong separated clusters are genuine unresolved evidence.
    if len(best_clusters) != 1:
        return None
    cluster = best_clusters[0]
    center = statistics.median(item.candidate.written_time.seconds for item in cluster)

    def winner_rank(item: CandidateEvaluation) -> tuple[int, int, int, int, int, int, float, float, str, str]:
        vector = _candidate_vector(item)
        return (
            int(candidate_has_boundary_grade_direct_onset_evidence(item.candidate)),
            int(item.candidate.independent_content_identity),
            vector[0], vector[1], vector[2], vector[3], vector[4],
            -abs(item.candidate.written_time.seconds - center),
            item.candidate.source,
            item.candidate.candidate_id,
        )

    numeric_best = max(winner_rank(item)[:-2] for item in cluster)
    strongest = [item for item in cluster if winner_rank(item)[:-2] == numeric_best]
    return min(strongest, key=lambda item: (item.candidate.source, item.candidate.candidate_id))


def _record_current_replacement(
    current: CandidateEvaluation,
    winner: CandidateEvaluation,
    *,
    selection_revision: int,
    peer_graph: PeerGraphPolicy | None,
) -> None:
    """Opt-in observation only; never read an oracle or influence selection."""
    diagnostic_path = os.environ.get("LRC_CURRENT_REPLACEMENT_DIAGNOSTICS")
    if not diagnostic_path:
        return
    record = {
        "schema": 1,
        "entry_index": winner.candidate.entry_index,
        "selection_revision": selection_revision,
        "path": "dominance-over-valid-current" if current.valid else "selection-with-invalid-current",
        "peer_mode": peer_graph.mode if peer_graph is not None else None,
        "current": {
            "candidate_id": current.candidate.candidate_id,
            "generation_revision": current.candidate.generation_revision,
            "source": current.candidate.source,
            "current_flag": current.candidate.current,
            "written_centiseconds": current.candidate.written_time.centiseconds,
            "valid": current.valid,
            "reasons": current.rejection_reasons,
            "vector": _candidate_vector(current),
            "families": sorted(_candidate_support_families(current.candidate)),
        },
        "challenger": {
            "candidate_id": winner.candidate.candidate_id,
            "generation_revision": winner.candidate.generation_revision,
            "source": winner.candidate.source,
            "written_centiseconds": winner.candidate.written_time.centiseconds,
            "valid": winner.valid,
            "vector": _candidate_vector(winner),
            "families": sorted(_candidate_support_families(winner.candidate)),
        },
    }
    try:
        with Path(diagnostic_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        progress_log(f"current replacement diagnostics unavailable: {type(exc).__name__}")


def load_reviewer_validity_evidence(
    path: Path | None, row_count: int,
) -> tuple[dict[str, object] | None, ...]:
    empty = tuple(None for _ in range(row_count))
    if path is None or not Path(path).is_file():
        return empty
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(payload, dict):
        return empty
    reviewers = payload.get("reviewers")
    required = payload.get("required_agreements")
    offsets = payload.get("offsets")
    sample_counts = payload.get("offset_sample_counts")
    rows = payload.get("rows")
    if (
        not isinstance(reviewers, list)
        or not isinstance(required, int)
        or required < 1
        or not isinstance(offsets, dict)
        or not isinstance(sample_counts, dict)
        or not isinstance(rows, list)
    ):
        return empty
    qualified_offsets = {
        str(name): float(value)
        for name, value in offsets.items()
        if isinstance(value, (int, float))
        and isinstance(sample_counts.get(str(name)), int)
        and int(sample_counts[str(name)]) >= 8
    }
    result: list[dict[str, object] | None] = [None] * row_count
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("entry"), int):
            continue
        entry = int(row["entry"])
        if not 1 <= entry <= row_count:
            continue
        expected = row.get("expected_current_seconds")
        reviewer_times = row.get("reviewer_times")
        if not isinstance(expected, (int, float)) or not isinstance(reviewer_times, dict):
            continue
        result[entry - 1] = {
            "expected_current_seconds": float(expected),
            "reviewer_times": {
                str(name): (float(value) if isinstance(value, (int, float)) else None)
                for name, value in reviewer_times.items()
            },
            "offsets": qualified_offsets,
            "reviewers": [str(value) for value in reviewers],
            "required_agreements": int(required),
        }
    return tuple(result)


_REVIEWER_VALIDITY_DIAGNOSTICS_KEY = "reviewer_validity_evidence"


def bind_reviewer_validity_diagnostics(
    diagnostics: dict[str, object], args: argparse.Namespace, row_count: int,
) -> dict[str, object]:
    if str(getattr(args, "arbiter", "off")) != "reviewer-validity":
        return diagnostics
    rows = load_reviewer_validity_evidence(
        getattr(args, "reviewer_evidence", None), row_count
    )
    result = dict(diagnostics)
    result[_REVIEWER_VALIDITY_DIAGNOSTICS_KEY] = {
        "schema": 1,
        "rows": list(rows),
    }
    return result


def reviewer_validity_evidence_from_diagnostics(
    diagnostics: Mapping[str, object], row_count: int,
) -> tuple[dict[str, object] | None, ...]:
    payload = diagnostics.get(_REVIEWER_VALIDITY_DIAGNOSTICS_KEY)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        return tuple(None for _ in range(row_count))
    raw_rows = payload["rows"]
    if len(raw_rows) != row_count:
        return tuple(None for _ in range(row_count))
    return tuple(row if isinstance(row, dict) else None for row in raw_rows)


def _reviewer_validity_endorsement(
    candidate: TimingCandidate,
    reviewer_evidence: dict[str, object] | None,
) -> timing_arbiter.ReviewerValidityDecision | None:
    if not isinstance(reviewer_evidence, dict):
        return None
    expected = reviewer_evidence.get("expected_current_seconds")
    raw_times = reviewer_evidence.get("reviewer_times")
    raw_offsets = reviewer_evidence.get("offsets")
    raw_reviewers = reviewer_evidence.get("reviewers")
    required = reviewer_evidence.get("required_agreements")
    if (
        not isinstance(expected, (int, float))
        or not isinstance(raw_times, dict)
        or not isinstance(raw_offsets, dict)
        or not isinstance(raw_reviewers, list)
        or not isinstance(required, int)
    ):
        return None
    reviewer_times = {
        str(key): (float(value) if isinstance(value, (int, float)) else None)
        for key, value in raw_times.items()
    }
    offsets = {
        str(key): float(value)
        for key, value in raw_offsets.items()
        if isinstance(value, (int, float))
    }
    reviewers = tuple(str(value) for value in raw_reviewers)
    return timing_arbiter.decide_reviewer_validity(
        current_seconds=float(candidate.written_time.centiseconds) / 100.0,
        expected_current_seconds=float(expected),
        reviewer_times=reviewer_times,
        offsets=offsets,
        reviewers=reviewers,
        required_agreements=int(required),
    )


def _reviewer_agreement_count(
    candidate: TimingCandidate,
    reviewer_evidence: dict[str, object] | None,
) -> int | None:
    if not isinstance(reviewer_evidence, dict):
        return None
    raw_times = reviewer_evidence.get("reviewer_times")
    raw_offsets = reviewer_evidence.get("offsets")
    raw_reviewers = reviewer_evidence.get("reviewers")
    if (
        not isinstance(raw_times, dict)
        or not isinstance(raw_offsets, dict)
        or not isinstance(raw_reviewers, list)
    ):
        return None
    reviewer_times = {
        str(key): (float(value) if isinstance(value, (int, float)) else None)
        for key, value in raw_times.items()
    }
    offsets = {
        str(key): float(value)
        for key, value in raw_offsets.items()
        if isinstance(value, (int, float))
    }
    agreeing, _present = timing_arbiter.reviewer_agreements_at(
        candidate_seconds=float(candidate.written_time.centiseconds) / 100.0,
        reviewer_times=reviewer_times,
        offsets=offsets,
        reviewers=tuple(str(value) for value in raw_reviewers),
    )
    return len(agreeing)


# Whole-song CTC path collapse.  A forced path that stretches one lyric line
# over >= 12 s, or squeezes a line into <= 0.8 s, is not a plausible sung
# line; it is the signature of the path giving up on a passage (seen as a run
# of compressed lines next to one line absorbing a 16-42 s gap in Choir,
# My Dearest and ミレニア).  Diagnostic geometry only: it never makes any
# candidate valid or trusted.
CTC_COLLAPSED_LINE_MAX_SECONDS = 12.0


CTC_COLLAPSED_LINE_MIN_SECONDS = 0.8


COLLAPSED_PATH_FALLBACK_SOURCE = "ctc-local-window"


COLLAPSED_PATH_FALLBACK_MIN_SHIFT_SECONDS = 0.5


# Wide-region fallback: the radius-3 local-window replay re-aligns a broken
# passage with more sane context than the radius-1 pass.  Its row is only a
# provisional replacement for a row Central left unresolved, and only when the
# radius-2 replay (or another non-posterior candidate) lands on the same onset
# and no independent onset (recurrence, local Whisper) contradicts it.  On the
# 39-song corpus this rule moved 10-22 rows per run closer to BK and none away.
# The replay's own windows are the region: an admitted row is eligible even
# when Central's radius-3 flag on the whole-song path does not reach it (the
# replay windows run to the nearest sane anchors, past that radius).  Behind
# the flag, 726A4413 kept c04 #5, c17 #32/#34/#35 and c34 #7/#21 on their
# broken whole-song times (up to 3.8 s off) while every admitted wide time on
# an unresolved row was closer to BK.
CTC_WIDE_REGION_FALLBACK_SOURCE = "ctc-local-window-wide"


def ctc_line_extent_collapsed(first_start: float, last_end: float) -> bool:
    extent = float(last_end) - float(first_start)
    return (
        extent >= CTC_COLLAPSED_LINE_MAX_SECONDS
        or extent <= CTC_COLLAPSED_LINE_MIN_SECONDS
    )


def _current_timing_candidate(candidates: tuple[TimingCandidate, ...]) -> TimingCandidate | None:
    return next((item for item in candidates if item.current), None)


def _active_legacy_candidate(candidate: object) -> bool:
    return (
        isinstance(candidate, dict)
        and candidate.get("source") != "selected_path"
        and candidate.get("replaced_by_consensus") is not True
    )


def ctc_spans_collapsed(spans: Sequence[TimingTokenSpan]) -> bool:
    if not spans:
        return False
    return ctc_line_extent_collapsed(spans[0].start, spans[-1].end)


def ctc_collapsed_region_flags(
    row_collapsed: Sequence[bool], radius: int = 1
) -> tuple[bool, ...]:
    """A row is inside a collapsed region when it or a row within ``radius``
    collapsed (radius 1: itself or an adjacent row).

    Shared by Central's unresolved fallback and the local-window generator so
    both stages act on exactly the same rows.  Wider radii are used only by the
    posthoc local-window shadow.
    """
    flags = [bool(value) for value in row_collapsed]
    return tuple(
        any(flags[max(0, row - radius):row + radius + 1]) for row in range(len(flags))
    )


def assignment_ctc_path_collapsed(assignment: object) -> bool:
    """Same judgment as ``current_ctc_path_collapsed`` on the backend row the
    ``ctc-current`` candidate is built from."""
    if not isinstance(assignment, dict):
        return False
    return ctc_spans_collapsed(_timing_spans_from_value(assignment.get("ctc_token_spans")))


def current_ctc_path_collapsed(candidates: tuple[TimingCandidate, ...]) -> bool:
    current = _current_timing_candidate(candidates)
    if current is None:
        return False
    return ctc_spans_collapsed(current.spans)


def current_ctc_path_incoherent(candidates: tuple[TimingCandidate, ...]) -> bool:
    """The ``ctc-current`` row fails Central's own temporal-coherence check
    (a severe gap inside its opening), the same reading the evaluator rejects
    it with as ``temporal-incoherence``."""
    current = _current_timing_candidate(candidates)
    if current is None or not current.spans:
        return False
    return analyze_candidate_regions(current).temporal_coherence == "incoherent"


RAW_ORDER_ANCHOR_SOURCE = "raw-vocal-independent-fusion"


def ctc_current_out_of_raw_order(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> tuple[bool, ...]:
    """Rows whose ``ctc-current`` time contradicts the order of independent
    raw-ASR onsets: later than a later row's onset, or earlier than an
    earlier row's.

    The onsets are the raw-ASR identity rows the vocal stem confirmed
    (``raw-vocal-independent-fusion``).  They only diagnose the whole-song
    path here, exactly like collapse and incoherence; they never become a
    selectable clock.  Read before the timing-source policy strips them.
    c39 #20-#26: seven rows piled into 149-177 s while the raw onsets of #20,
    #22, #24 and #26 sit at 139, 146, 162 and 175 s; only #23's own extent
    looked collapsed.
    """
    anchors = _raw_order_anchors(candidate_sets)
    flags: list[bool] = []
    for index, candidates in enumerate(candidate_sets):
        current = _current_timing_candidate(candidates)
        flags.append(
            current is not None
            and _contradicts_raw_order(index, current.written_time.seconds, anchors)
        )
    return tuple(flags)


def _raw_order_anchors(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> tuple[tuple[int, float], ...]:
    return tuple(
        (index, candidate.written_time.seconds)
        for index, candidates in enumerate(candidate_sets)
        for candidate in candidates
        if candidate.source == RAW_ORDER_ANCHOR_SOURCE
    )


def _contradicts_raw_order(
    index: int, seconds: float, anchors: tuple[tuple[int, float], ...]
) -> bool:
    return any(
        (other > index and value <= seconds) or (other < index and value >= seconds)
        for other, value in anchors
    )


def raw_order_contradicting_candidate_ids(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> tuple[frozenset[str], ...]:
    """Per row, the candidates whose time contradicts the independent raw
    onset order (see ``ctc_current_out_of_raw_order``).  Such a time cannot be
    a valid clock: it puts the row after a later row's heard onset or before
    an earlier one's.  It may still be the provisional time when nothing
    better exists.  c39 #23: a retry of the collapsed path at 173.14 s was
    selected_valid although the raw onsets of #24-#26 sit at 162-175 s, and
    as a valid row it barred their correct wide-region replays.
    """
    anchors = _raw_order_anchors(candidate_sets)
    return tuple(
        frozenset(
            candidate.candidate_id for candidate in candidates
            if candidate.source != RAW_ORDER_ANCHOR_SOURCE
            and _contradicts_raw_order(index, candidate.written_time.seconds, anchors)
        )
        for index, candidates in enumerate(candidate_sets)
    )


def _collapsed_path_fallback(
    evaluations: tuple[CandidateEvaluation, ...],
    eligible_ids: set[str],
    current: CandidateEvaluation,
    previous_candidate: TimingCandidate | None,
    next_bound_seconds: float | None,
) -> CandidateEvaluation | None:
    """Provisional replacement for an unresolved row inside a collapsed CTC path
    (or a row whose own whole-song opening is temporally incoherent).

    Only used when Central found no valid winner.  The whole-song path is
    visibly broken around this row, so its timestamp is the least credible
    unresolved option; a known-lyric CTC re-alignment of the same passage in a
    bounded window is preferred.  The row stays provisional_unresolved.
    """
    lower = (
        previous_candidate.written_time.seconds + HOP_SECONDS
        if previous_candidate is not None
        else float("-inf")
    )
    upper = (
        next_bound_seconds - HOP_SECONDS
        if next_bound_seconds is not None
        else float("inf")
    )
    current_seconds = current.candidate.written_time.seconds
    options = [
        evaluation for evaluation in evaluations
        if evaluation.candidate.source == COLLAPSED_PATH_FALLBACK_SOURCE
        and evaluation.candidate.candidate_id in eligible_ids
        and not evaluation.candidate.window_truncated
        and not evaluation.candidate.right_edge_pileup
        and abs(evaluation.candidate.written_time.seconds - current_seconds)
        > COLLAPSED_PATH_FALLBACK_MIN_SHIFT_SECONDS
        and lower < evaluation.candidate.written_time.seconds < upper
    ]
    if not options:
        return None
    return max(
        options,
        key=lambda item: (item.candidate.confidence, item.candidate.candidate_id),
    )


CORROBORATED_REALIGNMENT_SOURCES = ("ctc-local-window", "ctc-opening-local-retry")


CORROBORATED_REALIGNMENT_AGREEMENT_SECONDS = 0.15


CORROBORATED_REALIGNMENT_RAW_ONSET_SECONDS = 0.30


def raw_onset_seconds_by_row(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Each row's independent raw-ASR onsets, read before the timing-source
    policy strips them (evidence only, never a clock)."""
    return tuple(
        tuple(
            candidate.written_time.seconds for candidate in candidates
            if candidate.source == RAW_ORDER_ANCHOR_SOURCE
        )
        for candidates in candidate_sets
    )


def _corroborated_realignment_fallback(
    evaluations: tuple[CandidateEvaluation, ...],
    eligible_ids: set[str],
    current: CandidateEvaluation,
    previous_candidate: TimingCandidate | None,
    next_bound_seconds: float | None,
    raw_onsets: tuple[float, ...],
) -> CandidateEvaluation | None:
    """Provisional replacement for an unresolved row that is not visibly
    collapsed: the window re-alignment and the opening retry are separate CTC
    passes over different audio spans; when both land on the same onset and
    the independent raw onset is there too, that onset outweighs the
    whole-song time.  The clock is still a CTC row; the row stays
    provisional_unresolved.  c20 #24: whole-song 74.57 s; local window 72.76,
    opening retry 72.77, raw onset 72.78 (BK 72.79).
    """
    if not raw_onsets:
        return None
    lower = (
        previous_candidate.written_time.seconds + HOP_SECONDS
        if previous_candidate is not None
        else float("-inf")
    )
    upper = (
        next_bound_seconds - HOP_SECONDS
        if next_bound_seconds is not None
        else float("inf")
    )
    current_seconds = current.candidate.written_time.seconds
    by_source: dict[str, list[CandidateEvaluation]] = {}
    for evaluation in evaluations:
        candidate = evaluation.candidate
        if (
            candidate.source in CORROBORATED_REALIGNMENT_SOURCES
            and candidate.candidate_id in eligible_ids
            and not candidate.window_truncated
            and not candidate.right_edge_pileup
            and lower < candidate.written_time.seconds < upper
            and abs(candidate.written_time.seconds - current_seconds)
            > COLLAPSED_PATH_FALLBACK_MIN_SHIFT_SECONDS
            and any(
                abs(candidate.written_time.seconds - onset)
                <= CORROBORATED_REALIGNMENT_RAW_ONSET_SECONDS
                for onset in raw_onsets
            )
        ):
            by_source.setdefault(candidate.source, []).append(evaluation)
    if len(by_source) < len(CORROBORATED_REALIGNMENT_SOURCES):
        return None
    options = [
        evaluation
        for evaluation in by_source[COLLAPSED_PATH_FALLBACK_SOURCE]
        if any(
            abs(evaluation.candidate.written_time.seconds - other.candidate.written_time.seconds)
            <= CORROBORATED_REALIGNMENT_AGREEMENT_SECONDS
            for source, others in by_source.items()
            if source != COLLAPSED_PATH_FALLBACK_SOURCE
            for other in others
        )
    ]
    if not options:
        return None
    return max(
        options,
        key=lambda item: (item.candidate.confidence, item.candidate.candidate_id),
    )


def _ctc_wide_region_promotions(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    decisions: list[TimingDecision],
) -> dict[int, str]:
    """Rows whose provisional time yields to their gated wide-region replay.

    Walks the finished worklist in order: an unresolved row carrying an
    admitted ``ctc-local-window-wide`` candidate takes its time when that moves
    it by more than the fallback shift and keeps it strictly between the
    (possibly already promoted) previous row and the next row.  The row stays
    provisional_unresolved; only its written clock changes.
    """
    written = [decision.written_time.seconds for decision in decisions]
    forced: dict[int, str] = {}
    for index, decision in enumerate(decisions):
        if decision.status != "provisional_unresolved":
            continue
        wide = next(
            (
                candidate for candidate in candidate_sets[index]
                if candidate.source == CTC_WIDE_REGION_FALLBACK_SOURCE
            ),
            None,
        )
        if wide is None:
            continue
        seconds = wide.written_time.seconds
        lower = written[index - 1] if index else float("-inf")
        upper = (
            decisions[index + 1].written_time.seconds
            if index + 1 < len(decisions)
            else float("inf")
        )
        if (
            abs(seconds - written[index]) > COLLAPSED_PATH_FALLBACK_MIN_SHIFT_SECONDS
            and lower < seconds < upper
        ):
            forced[index] = wide.candidate_id
            written[index] = seconds
    return forced


def select_timing_decision(
    *,
    entry_index: int,
    candidates: tuple[TimingCandidate, ...],
    previous_candidate: TimingCandidate | None,
    selection_revision: int,
    previous_tail: TailRegionEvidence | None = None,
    peer_graph: PeerGraphPolicy | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
    arbiter_mode: Literal["off", "gross-rescue", "reviewer-validity"] = "off",
    reviewer_evidence: dict[str, object] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
    ctc_path_collapsed: bool = False,
    next_bound_seconds: float | None = None,
    forced_provisional_id: str | None = None,
    raw_order_contradicting_ids: frozenset[str] = frozenset(),
    raw_onsets: tuple[float, ...] = (),
) -> TimingDecision:
    if arbiter_mode not in {"off", "gross-rescue", "reviewer-validity"}:
        raise ValueError(f"unsupported timing arbiter mode: {arbiter_mode}")
    evaluations = tuple(
        evaluate_timing_candidate(
            candidate,
            previous_candidate,
            previous_tail=previous_tail,
            occurrence_entries=occurrence_entries,
        )
        for candidate in sorted(candidates, key=lambda item: item.candidate_id)
    )
    if raw_order_contradicting_ids:
        evaluations = tuple(
            replace(
                evaluation,
                valid=False,
                rejection_reasons=tuple(evaluation.rejection_reasons) + ("contradicts-raw-onset-order",),
            )
            if evaluation.candidate.candidate_id in raw_order_contradicting_ids
            else evaluation
            for evaluation in evaluations
        )
    if timing_source_policy not in {"default", "ctc-first"}:
        raise ValueError(f"unsupported timing-source policy: {timing_source_policy}")
    # Recurrence and the wide-region replay are evidence/fallback rows, never
    # a selectable clock in their own right.
    eligible_ids = {
        item.candidate_id for item in candidates
        if item.source != CTC_WIDE_REGION_FALLBACK_SOURCE
        and (
            (timing_source_policy == "default" and item.source != "mutual-acoustic-recurrence")
            or (item.source != "mutual-acoustic-recurrence" and "whisper" not in item.source.lower() and "raw" not in item.source.lower())
        )
    }
    if not eligible_ids:
        raise LrcError(f"ctc-first: no eligible time candidate at line {entry_index + 1}")
    fallback_current_evaluation = next(
        (
            evaluation for evaluation in evaluations
            if evaluation.candidate.current
            and evaluation.candidate.candidate_id in eligible_ids
        ),
        None,
    )
    current_evaluation = (
        None
        if peer_graph is not None
        and peer_graph.mode == "cross_backend_neutral"
        else fallback_current_evaluation
    )
    reviewer_endorsed_current = False
    reviewer_current_support: int | None = None
    if (
        arbiter_mode == "reviewer-validity"
        and current_evaluation is not None
        and not current_evaluation.valid
    ):
        reviewer_decision = _reviewer_validity_endorsement(
            current_evaluation.candidate, reviewer_evidence
        )
        if (
            reviewer_decision is not None
            and timing_arbiter.reviewer_validity_can_relax(
                current_evaluation.rejection_reasons,
                endorsed=reviewer_decision.endorsed,
            )
        ):
            current_evaluation = replace(
                current_evaluation,
                valid=True,
                rejection_reasons=(),
            )
            reviewer_endorsed_current = True
            reviewer_current_support = len(reviewer_decision.agreeing_reviewers)
            evaluations = tuple(
                current_evaluation
                if evaluation.candidate.candidate_id == current_evaluation.candidate.candidate_id
                else evaluation
                for evaluation in evaluations
            )
    valid = tuple(
        evaluation for evaluation in evaluations
        if evaluation.valid and evaluation.candidate.candidate_id in eligible_ids
    )
    winner: CandidateEvaluation | None = None
    if reviewer_endorsed_current:
        reviewer_challengers = tuple(
            evaluation for evaluation in valid
            if evaluation.candidate.candidate_id != current_evaluation.candidate.candidate_id
        )
        consensus_winner = _bounded_independent_consensus_winner(reviewer_challengers)
        if (
            consensus_winner is not None
            and abs(
                consensus_winner.candidate.written_time.seconds
                - current_evaluation.candidate.written_time.seconds
            ) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            and reviewer_current_support is not None
            and (
                challenger_reviewer_support := _reviewer_agreement_count(
                    consensus_winner.candidate, reviewer_evidence
                )
            ) is not None
            and challenger_reviewer_support > reviewer_current_support
        ):
            winner = consensus_winner
        else:
            winner = current_evaluation
    elif current_evaluation is not None and current_evaluation.valid:
        challengers = tuple(
            evaluation for evaluation in valid
            if evaluation.candidate.candidate_id != current_evaluation.candidate.candidate_id
            and candidate_dominates(evaluation, current_evaluation)
        )
        dominant_challengers = tuple(
            challenger for challenger in challengers
            if all(
                other is challenger or candidate_dominates(challenger, other)
                for other in challengers
            )
        )
        if len(dominant_challengers) == 1:
            winner = dominant_challengers[0]
        else:
            winner = current_evaluation
    elif valid:
        undominated = tuple(
            candidate for candidate in valid
            if not any(
                other.candidate.candidate_id != candidate.candidate.candidate_id
                and candidate_dominates(other, candidate)
                for other in valid
            )
        )
        if len(undominated) == 1:
            winner = undominated[0]
        else:
            winner = _semantically_equivalent_valid_winner(undominated)
            if winner is None:
                # Consensus belongs to the full valid evidence graph.  A support
                # witness may be individually Pareto-dominated by the strongest
                # member of the same tight cluster while still contributing the
                # independent lineage needed to distinguish that cluster from a
                # distant singleton.  Pruning it before consensus discards real
                # evidence.  The winner remains one existing valid candidate; no
                # timestamp is averaged or fabricated.
                winner = _bounded_independent_consensus_winner(valid)

    if (
        winner is not None
        and current_evaluation is not None
        and winner.candidate.candidate_id != current_evaluation.candidate.candidate_id
    ):
        _record_current_replacement(
            current_evaluation, winner,
            selection_revision=selection_revision, peer_graph=peer_graph,
        )

    arbiter_decision: timing_arbiter.GrossRescueDecision | None = None
    if arbiter_mode == "gross-rescue" and current_evaluation is not None:
        central_selection = winner or current_evaluation
        arbiter_decision = timing_arbiter.decide_gross_rescue(
            current_valid=current_evaluation.valid,
            current_centiseconds=current_evaluation.candidate.written_time.centiseconds,
            selected_centiseconds=central_selection.candidate.written_time.centiseconds,
            selected_source=central_selection.candidate.source,
            selected_is_current=(
                winner is not None
                and winner.candidate.candidate_id == current_evaluation.candidate.candidate_id
            ),
        )
        if (
            winner is not None
            and winner.candidate.candidate_id != current_evaluation.candidate.candidate_id
            and arbiter_decision.reverted_to_current
        ):
            winner = current_evaluation

    if winner is None:
        provisional = current_evaluation or fallback_current_evaluation or next(
            evaluation for evaluation in evaluations
            if evaluation.candidate.candidate_id in eligible_ids
        )
        if ctc_path_collapsed and provisional.candidate.current:
            collapsed_fallback = _collapsed_path_fallback(
                evaluations,
                eligible_ids,
                provisional,
                previous_candidate,
                next_bound_seconds,
            )
            if collapsed_fallback is not None:
                provisional = collapsed_fallback
        elif provisional.candidate.current:
            corroborated = _corroborated_realignment_fallback(
                evaluations,
                eligible_ids,
                provisional,
                previous_candidate,
                next_bound_seconds,
                raw_onsets,
            )
            if corroborated is not None:
                provisional = corroborated
        if forced_provisional_id is not None:
            provisional = next(
                (
                    evaluation for evaluation in evaluations
                    if evaluation.candidate.candidate_id == forced_provisional_id
                ),
                provisional,
            )
        status: Literal["selected_valid", "provisional_unresolved"] = "provisional_unresolved"
        selected_id = None
        provisional_id = provisional.candidate.candidate_id
    else:
        provisional = winner
        status = "selected_valid"
        selected_id = winner.candidate.candidate_id
        provisional_id = None

    timestamp_revision = selection_revision
    recovery: RecoveryProvenance | None = None
    if (
        winner is not None
        and not winner.candidate.current
        and current_evaluation is not None
        and not (
            peer_graph is not None
            and peer_graph.mode == "cross_backend_neutral"
        )
    ):
        recovery = RecoveryProvenance(
            candidate_id=winner.candidate.candidate_id,
            candidate_revision=winner.evidence.candidate_revision,
            evidence_revision=selection_revision,
            canonical_written_centiseconds=winner.candidate.written_time.centiseconds,
            old_selection_revision=max(0, selection_revision - 1),
            new_selection_revision=selection_revision,
            resolved_finding_ids=(),
            passed_invariants=(
                "structural-validity",
                "minimum-proof",
                "candidate-dominates-current",
                "revision-consistency",
                "canonical-time-evidence-match",
            ),
            resolution_mode="challenger-recovery",
            evaluator_version="timing-state-v1",
        )
    return TimingDecision(
        entry_index=entry_index,
        selection_revision=selection_revision,
        status=status,
        selected_candidate_id=selected_id,
        provisional_candidate_id=provisional_id,
        written_time=provisional.candidate.written_time,
        timestamp_revision=timestamp_revision,
        timing_evidence_revision=timestamp_revision,
        evidence=provisional.evidence,
        candidate=provisional.candidate,
        rejected_candidates=tuple(
            evaluation for evaluation in evaluations
            if evaluation.candidate.candidate_id != provisional.candidate.candidate_id
        ),
        recovery=recovery,
        arbiter=arbiter_decision,
    )


def global_recurrence_has_independent_corroboration(
    decision: TimingDecision,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Require a nearby, valid, different evidence lineage for recurrence trust.

    Similarity and onset from the same recurrence search are one observation.
    Rejected candidates are retained as diagnostics even under ctc-first.
    """
    if decision.candidate.source != "global-acoustic-recurrence":
        return True
    for evaluation in decision.rejected_candidates:
        witness = evaluation.candidate
        families = _candidate_support_families(witness)
        if (
            evaluation.valid
            and "recurrence" not in families
            and bool(families & {"ctc", "whisper-family", "hubert-ja-known-lyric"})
            and witness.identity_support == "supported"
            and witness.sequence_support == "supported"
            and candidate_has_bound_acoustic_evidence(witness)
            and candidate_has_bound_direct_onset_evidence(witness, occurrence_entries)
            and abs(witness.written_time.seconds - decision.written_time.seconds)
            <= RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
        ):
            return True
    return False


def audit_final_timing_decisions(
    decisions: tuple[TimingDecision, ...],
    unresolved_historical_codes: tuple[tuple[str, ...], ...] | None = None,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
    reviewer_evidence: tuple[dict[str, object] | None, ...] | None = None,
) -> tuple[TimingInvariantAudit, ...]:
    if unresolved_historical_codes is None:
        unresolved_historical_codes = tuple(() for _ in decisions)
    if len(unresolved_historical_codes) != len(decisions):
        raise ValueError("historical timing findings do not match timing decisions")
    if independent_tail_evidence is None:
        independent_tail_evidence = tuple(None for _ in decisions)
    if len(independent_tail_evidence) != len(decisions):
        raise ValueError("independent tail evidence does not match timing decisions")
    if reviewer_evidence is None:
        reviewer_evidence = tuple(None for _ in decisions)
    if len(reviewer_evidence) != len(decisions):
        raise ValueError("reviewer evidence does not match timing decisions")
    trusted = [decision.status == "selected_valid" for decision in decisions]
    failures: list[list[str]] = [[] for _ in decisions]
    for index, decision in enumerate(decisions):
        reviewer_decision = (
            _reviewer_validity_endorsement(decision.candidate, reviewer_evidence[index])
            if decision.candidate.current
            else None
        )
        reviewer_relaxation = bool(
            reviewer_decision is not None
            and reviewer_decision.endorsed
            and set(decision.evidence.structural_failures) <= {"previous-tail-ownership-unknown"}
            and decision.evidence.temporal_coherence != "incoherent"
        )
        if decision.timestamp_revision != decision.timing_evidence_revision:
            trusted[index] = False
            failures[index].append("revision-mismatch")
        if decision.evidence.computed_for_centiseconds != decision.written_time.centiseconds:
            trusted[index] = False
            failures[index].append("canonical-time-evidence-mismatch")
        hard_structural_failures = tuple(
            failure for failure in decision.evidence.structural_failures
            if not (reviewer_relaxation and failure == "previous-tail-ownership-unknown")
        )
        if hard_structural_failures:
            trusted[index] = False
            failures[index].extend(hard_structural_failures)
        strict_direct_onset = candidate_has_bound_direct_onset_evidence(
            decision.candidate, occurrence_entries
        )
        strict_opening_proof = bool(
            decision.evidence.onset_proof_end is not None or strict_direct_onset
        )
        if not strict_opening_proof and not reviewer_relaxation:
            trusted[index] = False
            failures[index].append("no-strict-opening-proof")
        boundary_grade_opening = _boundary_grade_opening_can_cross_tail(
            decision.candidate,
            ownership=decision.evidence.ownership,
            occurrence_entries=occurrence_entries,
        )
        if (
            decision.evidence.ownership.status in {"unknown", "violation"}
            and not boundary_grade_opening
            and not reviewer_relaxation
        ):
            trusted[index] = False
            failures[index].append("tail-conflict-without-independent-current-opening")
        if (
            decision.candidate.direct_onset_support == "supported"
            and not strict_direct_onset
        ):
            trusted[index] = False
            failures[index].append("unbound-or-nonindependent-direct-onset-claim")
        if (
            _candidate_requires_occurrence_evidence(decision.candidate)
            and not candidate_occurrence_evidence_is_valid(
                decision.candidate, occurrence_entries
            )
        ):
            trusted[index] = False
            failures[index].append("occurrence-evidence-invalid")
        if not decision.evidence.minimum_proof_satisfied and not reviewer_relaxation:
            trusted[index] = False
            failures[index].append("insufficient-minimum-proof")
        if not global_recurrence_has_independent_corroboration(
            decision, occurrence_entries
        ):
            trusted[index] = False
            failures[index].append("recurrence-uncorroborated")
        if unresolved_historical_codes[index]:
            trusted[index] = False
            failures[index].extend(unresolved_historical_codes[index])
        expected_tail_revision = None
        if index:
            previous = decisions[index - 1].candidate
            expected_tail_revision = resolve_reliable_tail(
                previous,
                independent_tail_evidence[index - 1],
                tail_evidence_context=tail_evidence_context,
            ).tail_evidence_revision
        if decision.evidence.referenced_tail_evidence_revision != expected_tail_revision:
            trusted[index] = False
            failures[index].append("tail-dependency-revision-mismatch")
    return tuple(
        TimingInvariantAudit(
            entry_index=decision.entry_index,
            timing_trusted=trusted[index],
            status="selected_valid" if trusted[index] else "provisional_unresolved",
            failures=tuple(dict.fromkeys(failures[index])),
        )
        for index, decision in enumerate(decisions)
    )


def build_final_timing_state(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    *,
    worklist_order: Literal["forward", "reverse"] = "forward",
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    peer_graph: PeerGraphPolicy | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
    arbiter_mode: Literal["off", "gross-rescue", "reviewer-validity"] = "off",
    reviewer_evidence: tuple[dict[str, object] | None, ...] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> FinalTimingState:
    if worklist_order not in {"forward", "reverse"}:
        raise ValueError(f"unsupported timing worklist order: {worklist_order}")
    out_of_raw_order = ctc_current_out_of_raw_order(candidate_sets)
    raw_order_contradicting = raw_order_contradicting_candidate_ids(candidate_sets)
    raw_onsets = raw_onset_seconds_by_row(candidate_sets)
    candidate_sets = apply_central_timing_source_policy(
        candidate_sets, timing_source_policy
    )
    if any(not candidates for candidates in candidate_sets):
        empty_index = next(index for index, candidates in enumerate(candidate_sets) if not candidates)
        raise ValueError(f"entry {empty_index + 1} has no provisional timing candidate")
    if independent_tail_evidence is None:
        independent_tail_evidence = tuple(None for _ in candidate_sets)
    if len(independent_tail_evidence) != len(candidate_sets):
        raise ValueError("independent tail evidence does not match candidate sets")
    if reviewer_evidence is None:
        reviewer_evidence = tuple(None for _ in candidate_sets)
    if len(reviewer_evidence) != len(candidate_sets):
        raise ValueError("reviewer evidence does not match candidate sets")
    # Collapse is a property of the immutable whole-song CTC baseline, so it is
    # computed once here and every replay through this owner sees the same view.
    collapsed_rows = tuple(current_ctc_path_collapsed(items) for items in candidate_sets)
    collapsed_region = ctc_collapsed_region_flags(collapsed_rows)
    # A whole-song row whose own opening is fractured is the same broken path
    # seen through its shape instead of its extent; its unresolved fallback is
    # the collapsed one.
    incoherent_rows = tuple(current_ctc_path_incoherent(items) for items in candidate_sets)

    def next_bound(row: int) -> float | None:
        # A sane next baseline row bounds the fallback so output stays ordered;
        # a collapsed next row is itself unreliable and is not a bound.
        if row + 1 >= len(candidate_sets) or collapsed_rows[row + 1]:
            return None
        following = _current_timing_candidate(candidate_sets[row + 1])
        return following.written_time.seconds if following is not None else None

    def run_worklist(forced: dict[int, str]) -> list[TimingDecision]:
        decisions: list[TimingDecision | None] = [None] * len(candidate_sets)
        pending = list(range(len(candidate_sets)))
        if worklist_order == "reverse":
            pending.reverse()
        while pending:
            index = pending.pop(0)
            if index and decisions[index - 1] is None:
                pending.append(index)
                continue
            previous_candidate = decisions[index - 1].candidate if index else None  # type: ignore[union-attr]
            previous_tail = (
                resolve_reliable_tail(
                    previous_candidate,
                    independent_tail_evidence[index - 1],
                    tail_evidence_context=tail_evidence_context,
                )
                if index and previous_candidate is not None
                else None
            )
            new_decision = select_timing_decision(
                entry_index=index,
                candidates=candidate_sets[index],
                previous_candidate=previous_candidate,
                selection_revision=index + 1,
                previous_tail=previous_tail,
                peer_graph=peer_graph,
                occurrence_entries=occurrence_entries,
                arbiter_mode=arbiter_mode,
                reviewer_evidence=reviewer_evidence[index],
                timing_source_policy=timing_source_policy,
                ctc_path_collapsed=(
                    collapsed_region[index] or incoherent_rows[index] or out_of_raw_order[index]
                ),
                next_bound_seconds=next_bound(index),
                forced_provisional_id=forced.get(index),
                raw_order_contradicting_ids=raw_order_contradicting[index],
                raw_onsets=raw_onsets[index],
            )
            old_decision = decisions[index]
            decisions[index] = new_decision
            changed = bool(
                old_decision is None
                or old_decision.candidate.candidate_id != new_decision.candidate.candidate_id
                or old_decision.evidence.candidate_revision != new_decision.evidence.candidate_revision
            )
            if changed and index + 1 < len(candidate_sets) and (index + 1) not in pending:
                pending.append(index + 1)
        return [decision for decision in decisions if decision is not None]

    decisions = run_worklist({})
    forced = _ctc_wide_region_promotions(candidate_sets, decisions)
    while forced:
        promoted = run_worklist(forced)
        # A promotion must survive its own replay: still unresolved, still the
        # forced row, still strictly ordered against the replayed neighbours.
        dropped = {
            index for index, candidate_id in forced.items()
            if promoted[index].candidate.candidate_id != candidate_id
            or (
                index
                and promoted[index - 1].written_time.centiseconds
                >= promoted[index].written_time.centiseconds
            )
            or (
                index + 1 < len(promoted)
                and promoted[index + 1].written_time.centiseconds
                <= promoted[index].written_time.centiseconds
            )
        }
        if not dropped:
            decisions = promoted
            break
        forced = {index: value for index, value in forced.items() if index not in dropped}
    frozen_decisions = tuple(decisions)
    audits = audit_final_timing_decisions(
        frozen_decisions,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        occurrence_entries=occurrence_entries,
        reviewer_evidence=reviewer_evidence,
    )
    return FinalTimingState(
        decisions=frozen_decisions,
        written_times=tuple(decision.written_time for decision in frozen_decisions),
        audits=audits,
    )


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


_GENERATION_DIAGNOSTIC_LOCATOR_KEYS = frozenset({
    "helper_path",
    "alignment_audio_path",
    "ctc_audio_path",
    "audio_path",
    "output_path",
    "report_path",
    "lyrics_path",
    "whisper_cli_path",
    "whisper_model_path",
    "whisperx_python_path",
    "whisperx_helper_path",
    "temp_path",
    "temp_paths",
    "temp_dir",
    "cache_path",
    "cache_dir",
    "cache_root",
    "cache_file",
    "cache_files",
    "transcript_path",
    "result_path",
    "wav_path",
    "alignment_wav_path",
})


def _generation_diagnostics_semantic_projection(value: object) -> object:
    """Remove only known execution locators from diagnostic identity."""
    if isinstance(value, dict):
        return {
            key: _generation_diagnostics_semantic_projection(child)
            for key, child in value.items()
            if key not in _GENERATION_DIAGNOSTIC_LOCATOR_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [
            _generation_diagnostics_semantic_projection(child)
            for child in value
        ]
    return value


def generation_diagnostics_semantic_revision(
    diagnostics: dict[str, object],
) -> str:
    """Digest meaningful diagnostic evidence while retaining full diagnostics."""
    return _timing_content_digest(
        _generation_diagnostics_semantic_projection(diagnostics)
    )


def apply_central_timing_source_policy(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    policy: Literal["default", "ctc-first"] = "default",
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Return the candidates allowed to participate in Central timing choice.

    Full generation diagnostics remain available outside the Central decision
    graph. Under ctc-first, Whisper/RAW timings are evidence-only: they may
    contribute bound row/tail diagnostics, but they cannot become timing
    alternatives, rejected material hypotheses, or REVIEW owners.
    """
    if policy == "default":
        return candidate_sets
    if policy != "ctc-first":
        raise ValueError(f"unsupported timing-source policy: {policy}")
    filtered: list[tuple[TimingCandidate, ...]] = []
    for line, candidates in enumerate(candidate_sets, 1):
        eligible = tuple(
            candidate for candidate in candidates
            if candidate.source != "mutual-acoustic-recurrence"
            and "whisper" not in candidate.source.lower()
            and "raw" not in candidate.source.lower()
        )
        if not eligible:
            raise LrcError(f"ctc-first: no eligible time candidate at line {line}")
        filtered.append(eligible)
    return tuple(filtered)
