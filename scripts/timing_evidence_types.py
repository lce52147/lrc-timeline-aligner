"""Extracted from timing_core.py (split2): timing evidence, candidate, decision and retry dataclasses with their state constants and the generation-diagnostics semantic revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import arbiter as timing_arbiter
from multimodel_consensus import model_identity
from lyrics_io import lyric_identity_revision
from runtime_base import CanonicalWrittenTime, LrcError, _timing_content_digest
from runtime_base import TimingTokenSpan


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
